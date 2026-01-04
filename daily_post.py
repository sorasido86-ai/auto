# -*- coding: utf-8 -*-
"""
daily_post.py
- WordPress REST API로 '오늘의 지표 리포트' 글 자동 발행
- 데이터:
  * USD/KRW, KOSPI: Stooq (CSV)
  * Brent: FRED(DCOILBRENTEU) 우선 + EIA(RBRTED.htm) fallback
- 원인 참고: Google News RSS(한국어) 헤드라인 3개씩
- bot_config.json 없이도 Secrets/환경변수로 동작 가능
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import feedparser


WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"

# -----------------------------
# 공통 HTTP
# -----------------------------
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

def http_get(url: str, params: Optional[dict] = None, timeout: int = 30) -> requests.Response:
    headers = {
        "User-Agent": UA,
        "Accept": "text/csv,application/json,text/plain,*/*",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    r = requests.get(url, params=params, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def http_post(url: str, headers: dict, json_payload: dict, timeout: int = 30) -> requests.Response:
    h = {
        "User-Agent": UA,
        "Accept": "application/json,*/*",
        **headers,
    }
    r = requests.post(url, headers=h, json=json_payload, timeout=timeout)
    r.raise_for_status()
    return r


# -----------------------------
# 설정 로딩 (env -> bot_config.json)
# -----------------------------
ENV_MAP = {
    "wp_base_url": "WP_BASE_URL",
    "wp_user": "WP_USER",
    "wp_app_pass": "WP_APP_PASS",
    "kakao_rest_key": "KAKAO_REST_KEY",
    "kakao_refresh_token": "KAKAO_REFRESH_TOKEN",
    "wp_post_status": "WP_POST_STATUS",  # publish/draft
    "test_mode": "TEST_MODE",            # "1"이면 5분 테스트용으로 매번 새 slug
}

def load_config() -> Dict[str, str]:
    cfg: Dict[str, str] = {}

    # 1) 파일 먼저 (경로 미지정이면 스크립트 옆 bot_config.json)
    cfg_path = os.getenv("CONFIG_PATH")
    if cfg_path:
        p = Path(cfg_path)
    else:
        p = Path(__file__).resolve().with_name("bot_config.json")

    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass

    # 2) env로 덮어쓰기
    for k, env in ENV_MAP.items():
        v = os.getenv(env)
        if v is not None and str(v).strip() != "":
            cfg[k] = str(v).strip()

    return cfg


def must_get(cfg: Dict[str, str], key: str) -> str:
    v = cfg.get(key, "")
    if not v:
        raise ValueError(f"필수 설정 누락: {key}")
    return v


# -----------------------------
# Kakao (선택)
# -----------------------------
def refresh_access_token(cfg: Dict[str, str]) -> str:
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": cfg["kakao_rest_key"],
        "refresh_token": cfg["kakao_refresh_token"],
    }
    r = requests.post(url, data=data, timeout=30, headers={"User-Agent": UA})
    r.raise_for_status()
    tokens = r.json()
    if tokens.get("refresh_token"):
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        try:
            Path(__file__).resolve().with_name("bot_config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
    return tokens["access_token"]

def kakao_send_to_me(cfg: Dict[str, str], text: str) -> None:
    if not (cfg.get("kakao_rest_key") and cfg.get("kakao_refresh_token")):
        return
    access_token = refresh_access_token(cfg)
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": UA}
    template_object = {
        "object_type": "text",
        "text": text[:1000],
        "link": {"web_url": cfg.get("wp_base_url", ""), "mobile_web_url": cfg.get("wp_base_url", "")},
        "button_title": "사이트 열기",
    }
    data = {"template_object": json.dumps(template_object, ensure_ascii=False)}
    r = requests.post(url, headers=headers, data=data, timeout=30)
    r.raise_for_status()


# -----------------------------
# WordPress
# -----------------------------
def wp_posts_api(cfg: Dict[str, str]) -> str:
    return must_get(cfg, "wp_base_url").rstrip("/") + WP_POSTS_API_SUFFIX

def wp_auth_headers(cfg: Dict[str, str]) -> Dict[str, str]:
    user = must_get(cfg, "wp_user").strip()
    app_pass = must_get(cfg, "wp_app_pass").replace(" ", "").strip()
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}

def wp_post_exists(cfg: Dict[str, str], slug: str) -> bool:
    r = requests.get(
        wp_posts_api(cfg),
        params={"slug": slug},
        headers={**wp_auth_headers(cfg), "User-Agent": UA},
        timeout=30,
    )
    r.raise_for_status()
    try:
        return len(r.json()) > 0
    except Exception:
        return False

def wp_create_post(cfg: Dict[str, str], title: str, slug: str, html: str, status: str) -> dict:
    payload = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": status,
    }
    r = http_post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg)},
        json_payload=payload,
        timeout=30,
    )
    return r.json()


# -----------------------------
# 데이터: Stooq / FRED / EIA
# -----------------------------
@dataclass
class Series:
    name: str
    latest: Optional[float]
    prev: Optional[float]
    date: Optional[str]
    source: str
    error: Optional[str] = None

    @property
    def change(self) -> Optional[float]:
        if self.latest is None or self.prev is None:
            return None
        return self.latest - self.prev

    @property
    def change_pct(self) -> Optional[float]:
        if self.latest is None or self.prev is None or self.prev == 0:
            return None
        return (self.latest - self.prev) / self.prev * 100.0


def _is_csv_like(text: str) -> bool:
    head = text.strip()[:300].lower()
    if "<html" in head or "<!doctype" in head:
        return False
    return ("date," in head) or ("date;" in head) or ("," in head and "\n" in head)


def fetch_stooq_last_two_closes(symbol: str) -> Tuple[float, float, str]:
    url = f"https://stooq.com/q/d/l/?s={quote(symbol)}&i=d"
    r = http_get(url, timeout=30)
    text = r.text

    if not _is_csv_like(text):
        raise ValueError(f"Stooq CSV가 아닌 응답(HTML/빈 값) 수신: {symbol}")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"Stooq 데이터가 부족합니다: {symbol}")

    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 5:
            continue
        d = parts[0].strip()
        c = parts[4].strip()
        if not d or not c or c.lower() in ("nan", "null"):
            continue
        try:
            rows.append((d, float(c)))
        except Exception:
            continue

    if len(rows) < 2:
        raise ValueError(f"Stooq 유효 데이터가 부족합니다: {symbol}")

    (d_prev, v_prev), (d_latest, v_latest) = rows[-2], rows[-1]
    return v_latest, v_prev, d_latest


def fetch_fred_last_two(series_id: str) -> Tuple[float, float, str]:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    r = http_get(url, params={"id": series_id}, timeout=30)
    text = r.text

    if not _is_csv_like(text) or "date" not in text[:80].lower():
        raise ValueError(f"FRED CSV 응답이 이상합니다: {series_id}")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"FRED 데이터가 부족합니다: {series_id}")

    rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0].strip(), parts[1].strip()
        if v == "." or v == "":
            continue
        try:
            rows.append((d, float(v)))
        except Exception:
            continue

    if len(rows) < 2:
        raise ValueError(f"FRED 유효 데이터가 부족합니다: {series_id}")

    (d_prev, v_prev), (d_latest, v_latest) = rows[-2], rows[-1]
    return v_latest, v_prev, d_latest


def fetch_eia_brent_last_two() -> Tuple[float, float, str]:
    """
    EIA 일간 Brent(FOB) 테이블 페이지(RBRTED.htm)에서 최근 2개 값을 뽑는다.
    페이지가 '주별 한 줄 + Mon~Fri 값' 형태라서,
    마지막 주 라인의 마지막 값(Fri)과 그 직전 영업일 값을 사용한다.
    """
    url = "https://www.eia.gov/dnav/pet/hist/RBRTED.htm"
    r = http_get(url, timeout=30)
    text = r.text

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    daily: List[Tuple[str, float]] = []

    for ln in lines:
        # 예: "2024 Dec-30 to Jan- 3 74.02 74.88 75.84 76.27 76.52"
        if not ln[:4].isdigit():
            continue
        parts = ln.split()
        if len(parts) < 6:
            continue
        if not parts[0].isdigit() or len(parts[0]) != 4:
            continue

        year = int(parts[0])
        if "-" not in parts[1]:
            continue

        m1_txt, d1_txt = parts[1].split("-", 1)
        m1 = month_map.get(m1_txt.lower()[:3])
        if not m1:
            continue

        d1_txt = d1_txt.strip()
        if d1_txt == "":
            continue

        try:
            d1 = int(d1_txt)
        except Exception:
            continue

        vals: List[float] = []
        for tok in parts:
            try:
                if "." in tok:
                    vals.append(float(tok))
            except Exception:
                pass

        if not vals:
            continue

        start = datetime(year, m1, d1)
        for i, v in enumerate(vals):
            d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            daily.append((d, v))

    if len(daily) < 2:
        raise ValueError("EIA에서 유효 데이터 2개 이상을 찾지 못했습니다.")

    (d_prev, v_prev), (d_latest, v_latest) = daily[-2], daily[-1]
    return v_latest, v_prev, d_latest


# -----------------------------
# 뉴스(원인 참고): Google News RSS (한국어)
# -----------------------------
def hangul_ratio(s: str) -> float:
    if not s:
        return 0.0
    h = sum(1 for ch in s if "가" <= ch <= "힣")
    return h / max(1, len(s))

def google_news_headlines(query: str, limit: int = 3) -> List[Tuple[str, str]]:
    url = (
        "https://news.google.com/rss/search?q="
        + quote(query)
        + "&hl=ko&gl=KR&ceid=KR:ko"
    )
    feed = feedparser.parse(url)
    items: List[Tuple[str, str]] = []
    for e in getattr(feed, "entries", [])[:30]:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        if hangul_ratio(title) < 0.12:
            continue
        items.append((title, link))
        if len(items) >= limit:
            break
    return items

def infer_reasons(kind: str, headlines: List[str]) -> List[str]:
    text = " ".join(headlines).lower()
    reasons: List[str] = []

    def has(*keys: str) -> bool:
        return any(k.lower() in text for k in keys)

    if kind == "fx":
        if has("금리", "연준", "fed", "cpi", "inflation", "물가", "fomc"):
            reasons.append("미국 금리/물가 관련 뉴스가 달러 강세·약세 기대를 바꿨을 수 있어요.")
        if has("위험", "리스크", "안전자산", "risk-off", "risk on", "증시", "주가"):
            reasons.append("위험선호(주식↑)·회피(달러↑) 흐름 변화가 환율에 반영됐을 수 있어요.")
        if not reasons:
            reasons.append("달러 강세/약세, 국내외 수급(외국인), 이벤트(지표/발언) 영향 가능성이 있어요.")
    elif kind == "oil":
        if has("opec", "감산", "증산", "공급", "생산", "중동", "전쟁", "홍해"):
            reasons.append("OPEC/공급 이슈나 지정학 변수가 유가에 영향을 줬을 수 있어요.")
        if has("재고", "inventories", "수요", "경기", "침체", "중국"):
            reasons.append("수요 전망(경기/중국) 또는 재고 지표 변화가 가격에 반영됐을 수 있어요.")
        if not reasons:
            reasons.append("공급·수요·재고·지정학 이슈가 복합적으로 작용하는 경우가 많아요.")
    elif kind == "stock":
        if has("cpi", "fomc", "금리", "연준", "환율"):
            reasons.append("금리/환율 변수에 대한 기대 변화가 지수에 영향을 줬을 수 있어요.")
        if has("실적", "earnings", "반도체", "삼성", "sk", "수출"):
            reasons.append("대형주(특히 반도체) 흐름/실적 기대가 지수에 반영됐을 수 있어요.")
        if not reasons:
            reasons.append("대형주 수급, 금리/환율, 글로벌 증시 흐름이 함께 영향을 주는 경우가 많아요.")
    return reasons[:2]


# -----------------------------
# 리포트 생성(HTML)
# -----------------------------
def build_html(today: str, series: List[Series], news_map: Dict[str, List[Tuple[str, str]]],
               test_badge: Optional[str], failures: List[str]) -> str:
    style = """
<style>
.report-wrap{max-width:920px;margin:0 auto;padding:6px 4px;font-family:system-ui,-apple-system,Segoe UI,Roboto,'Noto Sans KR',sans-serif}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;background:#eef2ff;color:#3730a3;font-weight:700;font-size:12px;margin-left:8px}
.alert{border:1px solid #fecaca;background:#fff1f2;color:#7f1d1d;border-radius:10px;padding:12px 14px;margin:14px 0}
.h2{font-size:22px;margin:24px 0 10px}
.meta{color:#6b7280;font-size:12px;margin-top:6px}
.tbl{width:100%;border-collapse:separate;border-spacing:0;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden}
.tbl th{background:#0f172a;color:#fff;text-align:left;padding:12px 14px;font-size:14px}
.tbl td{padding:12px 14px;border-top:1px solid #e5e7eb;font-size:14px}
.tbl td.num{text-align:right;font-variant-numeric:tabular-nums}
.box{border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;margin:10px 0}
.box h3{margin:0 0 10px;font-size:18px}
.box ul{margin:0;padding-left:18px}
.small{color:#6b7280;font-size:12px;margin-top:6px}
</style>
"""
    alert_html = ""
    if failures:
        items = "".join(f"<li><b>{f}</b></li>" for f in failures)
        alert_html = f"""
<div class="alert">
  <div style="font-weight:800;margin-bottom:6px">⚠️ 일부 데이터 수집 실패</div>
  <ul style="margin:0;padding-left:18px">{items}</ul>
  <div class="small">※ 주말/휴장/사이트 차단(봇 방지) 등으로 값이 비어 있을 수 있어요. 가능한 fallback을 적용했습니다.</div>
</div>
"""

    def fmt_num(x: Optional[float], digits: int = 2) -> str:
        return "-" if x is None else f"{x:,.{digits}f}"

    def fmt_change(x: Optional[float], digits: int = 2) -> str:
        if x is None:
            return "-"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:,.{digits}f}"

    def fmt_pct(x: Optional[float], digits: int = 2) -> str:
        if x is None:
            return "-"
        sign = "+" if x > 0 else ""
        return f"{sign}{x:.{digits}f}%"

    def row(s: Series) -> str:
        latest = "수집 실패" if s.latest is None and s.error else fmt_num(s.latest)
        change = fmt_change(s.change) if s.latest is not None else "-"
        pct = fmt_pct(s.change_pct) if s.latest is not None else "-"
        date = s.date or "-"
        return f"""
<tr>
  <td><b>{s.name}</b></td>
  <td class="num">{latest}</td>
  <td class="num">{change}</td>
  <td class="num">{pct}</td>
  <td class="num">{date}</td>
</tr>
"""

    table_html = f"""
<table class="tbl">
  <thead>
    <tr>
      <th style="width:26%">지표</th>
      <th style="width:19%">현재</th>
      <th style="width:19%">전일대비</th>
      <th style="width:16%">변동률</th>
      <th style="width:20%">기준일(데이터)</th>
    </tr>
  </thead>
  <tbody>
    {''.join(row(s) for s in series)}
  </tbody>
</table>
"""

    def news_box(key: str, header: str, kind: str) -> str:
        items = news_map.get(key, [])
        titles = [t for t, _ in items]
        reasons = infer_reasons(kind, titles)

        if not items:
            li = "<li>관련 헤드라인을 가져오지 못했습니다.</li>"
        else:
            li = "".join(f'<li><a href="{link}" target="_blank" rel="noopener">{title}</a></li>' for title, link in items)

        reason_html = ""
        if reasons:
            reason_html = "<div class='small'><b>요약(참고용):</b><ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul></div>"

        return f"""
<div class="box">
  <h3>{header}</h3>
  <ul>{li}</ul>
  {reason_html}
</div>
"""

    return f"""
{style}
<div class="report-wrap">
  <h1 style="font-size:30px;margin:10px 0 8px">
    오늘의 지표 리포트 ({today})
    {f'<span class="badge">TEST {test_badge}</span>' if test_badge else ''}
  </h1>

  {alert_html}

  <div class="h2">핵심 요약</div>
  <ul>
    <li>원/달러(USD/KRW), 브렌트유, 코스피의 최근 거래일 기준 변동을 한 번에 정리했습니다.</li>
    <li>변동 원인 참고용으로 관련 뉴스 헤드라인을 함께 첨부했습니다(단정 아님).</li>
  </ul>
  <div class="meta">생성 시각: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} (환경 TZ 기준)</div>

  <div class="h2">주요 지표</div>
  {table_html}

  <div class="h2">왜 움직였나? (원인 참고)</div>
  <div class="small">헤드라인 기반 참고입니다. 실제 원인은 복합적일 수 있어요.</div>
  {news_box("usdkrw", "USD/KRW 변동 원인(뉴스)", "fx")}
  {news_box("brent", "Brent 유가 변동 원인(뉴스)", "oil")}
  {news_box("kospi", "KOSPI 변동 원인(뉴스)", "stock")}

  <div class="h2">데이터 출처</div>
  <ul>
    <li>USD/KRW, KOSPI: Stooq (CSV)</li>
    <li>Brent: FRED(DCOILBRENTEU) 우선 + EIA(RBRTED.htm) fallback</li>
    <li>뉴스: Google News RSS (KR)</li>
  </ul>
</div>
"""


# -----------------------------
# 메인: 수집 → 글 발행
# -----------------------------
def fetch_indicators() -> Tuple[List[Series], Dict[str, List[Tuple[str, str]]], List[str]]:
    failures: List[str] = []

    # USD/KRW (Stooq)
    try:
        latest, prev, d = fetch_stooq_last_two_closes("usdkrw")
        usd = Series("USD/KRW", latest, prev, d, source="stooq")
    except Exception as e:
        usd = Series("USD/KRW", None, None, None, source="stooq", error=str(e))
        failures.append(f"USD/KRW 수집 실패: {type(e).__name__}: {e}")

    # Brent (FRED -> EIA)
    try:
        latest, prev, d = fetch_fred_last_two("DCOILBRENTEU")
        brent = Series("Brent Oil", latest, prev, d, source="fred")
    except Exception as e1:
        try:
            latest, prev, d = fetch_eia_brent_last_two()
            brent = Series("Brent Oil", latest, prev, d, source="eia")
        except Exception as e2:
            brent = Series("Brent Oil", None, None, None, source="eia", error=str(e2))
            failures.append(f"Brent 수집 실패: {type(e2).__name__}: {e2}")

    # KOSPI (Stooq)
    try:
        latest, prev, d = fetch_stooq_last_two_closes("^ks11")
        kospi = Series("KOSPI", latest, prev, d, source="stooq")
    except Exception as e:
        kospi = Series("KOSPI", None, None, None, source="stooq", error=str(e))
        failures.append(f"KOSPI 수집 실패: {type(e).__name__}: {e}")

    news_map = {
        "usdkrw": google_news_headlines("원달러 환율 변동 원인", 3),
        "brent": google_news_headlines("브렌트유 유가 변동 원인", 3),
        "kospi": google_news_headlines("코스피 상승 하락 원인", 3),
    }

    return [usd, brent, kospi], news_map, failures


def main() -> None:
    cfg = load_config()
    missing = [k for k in ("wp_base_url", "wp_user", "wp_app_pass") if not cfg.get(k)]
    if missing:
        raise ValueError("필수 설정 누락: " + ", ".join(missing))

    today = datetime.now().strftime("%Y-%m-%d")

    test_mode = str(cfg.get("test_mode", "")).strip() in ("1", "true", "True", "YES", "yes")
    test_badge = str(int(time.time())) if test_mode else None

    status = (cfg.get("wp_post_status") or "publish").strip()

    title = f"오늘의 지표 리포트 ({today})"
    slug = f"daily-indicator-report-{today}"
    if test_mode and test_badge:
        title = f"{title} - {test_badge}"
        slug = f"{slug}-{test_badge}"

    try:
        # 일 단위 운용 시 중복 방지
        if (not test_mode) and wp_post_exists(cfg, slug):
            msg = f"✅ 이미 오늘 글이 있어요 ({today}) → 중복 발행 안 함"
            print(msg)
            kakao_send_to_me(cfg, msg)
            return

        series, news_map, failures = fetch_indicators()
        html = build_html(today, series, news_map, test_badge, failures)

        post = wp_create_post(cfg, title, slug, html, status=status)
        link = post.get("link", cfg.get("wp_base_url", ""))

        ok_msg = f"✅ 글 발행 성공!\n날짜: {today}\n상태: {status}\n링크: {link}"
        if failures:
            ok_msg += "\n\n⚠ 일부 데이터 수집 실패:\n- " + "\n- ".join(failures)

        print(ok_msg)
        kakao_send_to_me(cfg, ok_msg)

    except Exception as e:
        msg = f"❌ 자동발행 실패 ({today})\n{type(e).__name__}: {e}"
        print(msg)
        try:
            kakao_send_to_me(cfg, msg)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
