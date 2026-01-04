# -*- coding: utf-8 -*-
"""
daily_post.py (통합본)
- WordPress에 '오늘의 지표 리포트' 자동 발행
- 데이터:
  * USD/KRW: FRED CSV (DEXKOUS)
  * Brent:   FRED CSV (DCOILBRENTEU)
  * KOSPI:   Stooq CSV (가능하면), 실패해도 글은 발행
- 뉴스(원인 참고): Google News RSS(ko/KR) + 한글 비율 필터
- TEST_MODE=1: 5분마다 새 글 생성(테스트용)
"""

import os
import json
import base64
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import requests
import feedparser


# -----------------------------
# 기본 설정
# -----------------------------
KST = timezone(timedelta(hours=9))
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.getenv("BOT_CONFIG_PATH", str(SCRIPT_DIR / "bot_config.json")))

WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DailyReportBot/1.0; +https://rainsow.com/)"
}


# -----------------------------
# 공통 유틸
# -----------------------------
def now_kst_date_str():
    return datetime.now(tz=KST).strftime("%Y-%m-%d")


def now_kst_ts():
    return datetime.now(tz=KST).strftime("%Y-%m-%d %H:%M:%S %Z")


def must_get(cfg, key: str) -> str:
    v = cfg.get(key)
    if v is None or str(v).strip() == "":
        raise ValueError(f"필수 설정 누락: {key}")
    return str(v).strip()


def load_config():
    """
    1) bot_config.json이 있으면 읽고
    2) 환경변수로 덮어쓰기(있으면)
    """
    cfg = {}
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    # env override (Actions에서 쓰기 편하게)
    env_map = {
        "wp_base_url": "WP_BASE_URL",
        "wp_user": "WP_USER",
        "wp_app_pass": "WP_APP_PASS",
        "kakao_rest_key": "KAKAO_REST_KEY",
        "kakao_refresh_token": "KAKAO_REFRESH_TOKEN",
    }
    for k, envk in env_map.items():
        if os.getenv(envk):
            cfg[k] = os.getenv(envk)

    return cfg


# -----------------------------
# Kakao (선택)
# -----------------------------
def refresh_access_token(cfg):
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": cfg["kakao_rest_key"],
        "refresh_token": cfg["kakao_refresh_token"],
    }
    r = requests.post(url, data=data, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()
    tokens = r.json()

    if tokens.get("refresh_token"):
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        # 갱신된 refresh 토큰 저장(로컬 실행 시만 의미 있음)
        try:
            CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return tokens["access_token"]


def kakao_send_to_me(cfg, text):
    if not (cfg.get("kakao_rest_key") and cfg.get("kakao_refresh_token")):
        return False

    access_token = refresh_access_token(cfg)
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {**HTTP_HEADERS, "Authorization": f"Bearer {access_token}"}

    template_object = {
        "object_type": "text",
        "text": text[:1000],
        "link": {"web_url": cfg["wp_base_url"], "mobile_web_url": cfg["wp_base_url"]},
        "button_title": "사이트 열기"
    }
    data = {"template_object": json.dumps(template_object, ensure_ascii=False)}
    r = requests.post(url, headers=headers, data=data, timeout=30)
    r.raise_for_status()
    return True


# -----------------------------
# WordPress
# -----------------------------
def wp_posts_api(cfg):
    return cfg["wp_base_url"].rstrip("/") + WP_POSTS_API_SUFFIX


def wp_auth_headers(cfg):
    user = cfg["wp_user"].strip()
    app_pass = cfg["wp_app_pass"].replace(" ", "").strip()
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", **HTTP_HEADERS}


def wp_post_exists(cfg, slug):
    r = requests.get(
        wp_posts_api(cfg),
        params={"slug": slug},
        headers=wp_auth_headers(cfg),
        timeout=30,
    )
    r.raise_for_status()
    return len(r.json()) > 0


def wp_create_post(cfg, title, slug, html_content, status="publish"):
    payload = {
        "title": title,
        "slug": slug,
        "content": html_content,
        "status": status,
    }
    r = requests.post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# -----------------------------
# 데이터 수집 (FRED / Stooq)
# -----------------------------
def _last_two_numeric_from_rows(rows, value_key):
    """
    rows: list[dict]
    value_key: 컬럼명
    - 뒤에서부터 숫자 2개 찾아 반환
    """
    found = []
    for row in reversed(rows):
        v = str(row.get(value_key, "")).strip()
        if v in ("", ".", "nan", "None"):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        d = str(row.get("DATE") or row.get("Date") or row.get("date") or "").strip()
        found.append((fv, d))
        if len(found) >= 2:
            break
    if len(found) < 2:
        return None
    latest, prev = found[0], found[1]  # (value,date)
    return latest[0], prev[0], latest[1]


def fetch_fred_last_two(series_id: str):
    """
    FRED CSV 다운로드:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={quote(series_id)}"
    r = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    if not text.startswith("DATE"):
        raise ValueError(f"FRED CSV 응답이 이상합니다: {series_id}")

    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    if not rows:
        raise ValueError(f"FRED 데이터가 비었습니다: {series_id}")

    out = _last_two_numeric_from_rows(rows, series_id)
    if not out:
        raise ValueError(f"FRED 데이터가 부족합니다(최근 2개 수치 없음): {series_id}")

    latest, prev, date_str = out
    return latest, prev, date_str


def fetch_stooq_last_two(symbol: str):
    """
    Stooq CSV:
    https://stooq.com/q/d/l/?s=SYMBOL&i=d
    """
    url = f"https://stooq.com/q/d/l/?s={quote(symbol)}&i=d"
    r = requests.get(url, headers=HTTP_HEADERS, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    if "Date,Open,High,Low,Close" not in text:
        # HTML/빈값/차단 등
        raise ValueError(f"Stooq CSV 수집 실패: {symbol} (CSV가 아닌 응답)")

    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"Stooq 데이터가 부족합니다: {symbol}")

    # Close 기준
    found = []
    for row in reversed(rows):
        v = str(row.get("Close", "")).strip()
        if not v:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        d = str(row.get("Date", "")).strip()
        found.append((fv, d))
        if len(found) >= 2:
            break

    if len(found) < 2:
        raise ValueError(f"Stooq 데이터가 부족합니다(종가 2개 없음): {symbol}")

    latest, prev = found[0], found[1]
    return latest[0], prev[0], latest[1]


def fetch_kospi_last_two():
    # Stooq에서 KOSPI 심볼이 환경에 따라 다르게 막힐 수 있어 여러 개 시도
    candidates = ["^kospi", "kospi", "^ks11", "ks11"]
    last_err = None
    for sym in candidates:
        try:
            return fetch_stooq_last_two(sym)
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"KOSPI 수집 실패: {last_err}")


@dataclass
class Indicator:
    name: str
    latest: float | None
    prev: float | None
    date: str | None
    unit: str
    error: str | None = None

    @property
    def diff(self):
        if self.latest is None or self.prev is None:
            return None
        return self.latest - self.prev

    @property
    def pct(self):
        if self.latest is None or self.prev is None or self.prev == 0:
            return None
        return (self.latest - self.prev) / self.prev * 100.0


def fetch_indicators():
    errors = []

    # USD/KRW (FRED: DEXKOUS)
    try:
        usd_latest, usd_prev, usd_date = fetch_fred_last_two("DEXKOUS")
        usd = Indicator("USD/KRW", usd_latest, usd_prev, usd_date, unit="")
    except Exception as e:
        usd = Indicator("USD/KRW", None, None, None, unit="", error=str(e))
        errors.append(f"USD/KRW 수집 실패: {e}")

    # Brent (FRED: DCOILBRENTEU)
    # ✅ 여기서 “EIA 테이블 못 찾음 / Stooq HTML 응답” 같은 문제를 원천 회피
    try:
        br_latest, br_prev, br_date = fetch_fred_last_two("DCOILBRENTEU")
        br = Indicator("Brent Oil", br_latest, br_prev, br_date, unit="$")
    except Exception as e:
        br = Indicator("Brent Oil", None, None, None, unit="$", error=str(e))
        errors.append(f"Brent 수집 실패: {e}")

    # KOSPI (Stooq, 실패해도 진행)
    try:
        k_latest, k_prev, k_date = fetch_kospi_last_two()
        kospi = Indicator("KOSPI", k_latest, k_prev, k_date, unit="")
    except Exception as e:
        kospi = Indicator("KOSPI", None, None, None, unit="", error=str(e))
        errors.append(f"KOSPI 수집 실패: {e}")

    return [usd, br, kospi], errors


# -----------------------------
# 뉴스(원인 참고) - Google News RSS
# -----------------------------
def hangul_ratio(s: str) -> float:
    if not s:
        return 0.0
    hangul = 0
    letters = 0
    for ch in s:
        # 한글 음절
        if "\uac00" <= ch <= "\ud7a3":
            hangul += 1
            letters += 1
        elif ch.isalpha():
            letters += 1
        # 한자(중국어 등)도 letters에 포함시키면 한글비율이 낮아져 필터링됨
        elif "\u4e00" <= ch <= "\u9fff":
            letters += 1
    return hangul / (letters + 1e-9)


def fetch_news_rss(query: str, max_items: int = 5):
    url = (
        "https://news.google.com/rss/search?q="
        + quote(query)
        + "&hl=ko&gl=KR&ceid=KR:ko"
    )
    feed = feedparser.parse(url)

    items = []
    for entry in feed.entries[: max_items * 3]:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue

        # ✅ 한자/외국어 섞이는 문제 완화: 한글 비율 낮으면 제외
        if hangul_ratio(title) < 0.12:
            continue

        items.append({"title": title, "link": link})
        if len(items) >= max_items:
            break

    return items


def reason_template(ind: Indicator):
    """
    방향(상승/하락)에 따라 '가능한 배경' 문장 템플릿(확정 원인 아님)
    """
    if ind.diff is None:
        return "데이터가 부족해 방향을 판단하기 어려워요."

    up = ind.diff > 0
    if ind.name == "USD/KRW":
        return (
            "달러 강세(미 금리/지표), 위험회피 심리, 수입 결제 수요/외환 수급 등이 영향을 줄 수 있어요."
            if up
            else "달러 약세, 위험선호 회복, 외국인 수급(주식/채권), 수출 네고 물량 등이 영향을 줄 수 있어요."
        )
    if ind.name == "Brent Oil":
        return (
            "공급 이슈(OPEC+ 감산/지정학), 재고 감소, 수요 개선 기대 등이 영향을 줄 수 있어요."
            if up
            else "수요 둔화 우려, 재고 증가, 달러 강세, 공급 증가 기대 등이 영향을 줄 수 있어요."
        )
    if ind.name == "KOSPI":
        return (
            "대형주 강세, 외국인/기관 수급, 금리·달러 안정, 글로벌 증시 분위기 등이 영향을 줄 수 있어요."
            if up
            else "차익 실현, 달러·금리 상승, 대외 리스크, 업종 이슈 등이 영향을 줄 수 있어요."
        )
    return "여러 요인이 복합적으로 작용할 수 있어요."


# -----------------------------
# 렌더링 (HTML)
# -----------------------------
def fmt_num(x: float | None, digits=2):
    if x is None:
        return "-"
    return f"{x:,.{digits}f}"


def fmt_diff(x: float | None, digits=2):
    if x is None:
        return "-"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:,.{digits}f}"


def fmt_pct(x: float | None, digits=2):
    if x is None:
        return "-"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{digits}f}%"


def build_html(title: str, indicators: list[Indicator], errors: list[str], news_map: dict):
    # 인라인 스타일(테마/편집기에서 style 태그가 날아가도 유지됨)
    wrap = "max-width:960px;margin:0 auto;line-height:1.65;"
    h1 = "font-size:32px;margin:0 0 12px 0;"
    badge = "display:inline-block;background:#eef2ff;color:#3730a3;padding:4px 10px;border-radius:999px;font-size:12px;margin-left:10px;"
    card = "border:1px solid #e5e7eb;border-radius:14px;padding:16px 18px;background:#fff;"
    warn = "border:1px solid #fecaca;background:#fff1f2;color:#7f1d1d;border-radius:14px;padding:14px 16px;margin:14px 0;"
    table = "width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border-radius:12px;border:1px solid #e5e7eb;"
    th = "background:#0f172a;color:#fff;text-align:left;padding:12px 12px;font-size:14px;"
    td = "padding:12px;border-top:1px solid #e5e7eb;font-size:14px;"
    small = "color:#6b7280;font-size:12px;"

    # 표 rows
    rows_html = ""
    for ind in indicators:
        latest = fmt_num(ind.latest, 2)
        prevdiff = fmt_diff(ind.diff, 2)
        pct = fmt_pct(ind.pct, 2)
        date = ind.date or "-"
        if ind.error:
            latest = "수집 실패"
            prevdiff = "-"
            pct = "-"
            date = "-"

        rows_html += (
            f"<tr>"
            f"<td style='{td}'><b>{ind.name}</b></td>"
            f"<td style='{td};text-align:right'>{latest}</td>"
            f"<td style='{td};text-align:right'>{prevdiff}</td>"
            f"<td style='{td};text-align:right'>{pct}</td>"
            f"<td style='{td};text-align:center'>{date}</td>"
            f"</tr>"
        )

    # 에러 박스
    err_html = ""
    if errors:
        items = "".join([f"<li><b>{e}</b></li>" for e in errors[:6]])
        err_html = (
            f"<div style='{warn}'>"
            f"<div style='font-weight:700;margin-bottom:8px;'>⚠️ 일부 데이터 수집 실패</div>"
            f"<ul style='margin:0;padding-left:18px;'>{items}</ul>"
            f"<div style='{small};margin-top:8px;'>※ 주말/휴장/차단 등으로 값이 비어 있을 수 있어요. 가능한 fallback을 적용했어요.</div>"
            f"</div>"
        )

    # 원인(뉴스) 섹션
    why_html = ""
    for ind in indicators:
        key = ind.name
        headlines = news_map.get(key, [])
        hl_html = ""
        if headlines:
            hl_items = "".join(
                [f"<li><a href='{h['link']}' target='_blank' rel='noopener'>{h['title']}</a></li>" for h in headlines]
            )
            hl_html = f"<ul style='margin:8px 0 0 0;padding-left:18px;'>{hl_items}</ul>"
        else:
            hl_html = f"<div style='{small};margin-top:6px;'>관련 헤드라인을 찾지 못했어요.</div>"

        why_html += (
            f"<div style='{card};margin-top:12px;'>"
            f"<div style='font-size:18px;font-weight:800;margin-bottom:6px;'>{key} 변동 원인(참고)</div>"
            f"<div style='margin:0 0 6px 0;'><b>가능한 배경:</b> {reason_template(ind)}</div>"
            f"<div style='{small}'>헤드라인 기반 참고입니다. 실제 원인은 복합적일 수 있어요.</div>"
            f"{hl_html}"
            f"</div>"
        )

    html = f"""
<div style="{wrap}">
  <div style="{h1}"><b>{title}</b></div>

  {err_html}

  <div style="{card};margin-top:12px;">
    <div style="font-size:20px;font-weight:800;margin-bottom:8px;">핵심 요약</div>
    <ul style="margin:0;padding-left:18px;">
      <li>원/달러(USD/KRW), 브렌트유, 코스피의 <b>최근 거래일 기준 변동</b>을 한 번에 정리했습니다.</li>
      <li>변동 원인 참고용으로 <b>관련 뉴스 헤드라인</b>을 함께 첨부했습니다.</li>
    </ul>
    <div style="{small};margin-top:10px;">생성 시각: {now_kst_ts()}</div>
  </div>

  <div style="margin-top:18px;font-size:22px;font-weight:900;">주요 지표</div>
  <table style="{table};margin-top:10px;">
    <thead>
      <tr>
        <th style="{th};width:28%;">지표</th>
        <th style="{th};width:18%;text-align:right;">현재</th>
        <th style="{th};width:18%;text-align:right;">전일대비</th>
        <th style="{th};width:18%;text-align:right;">변동률</th>
        <th style="{th};width:18%;text-align:center;">기준일(데이터)</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div style="margin-top:22px;font-size:22px;font-weight:900;">왜 움직였나? (원인 참고)</div>
  <div style="{small};margin-top:6px;">헤드라인 기반 참고입니다. 실제 원인은 복합적일 수 있어요.</div>
  {why_html}

  <div style="{card};margin-top:18px;">
    <div style="font-size:18px;font-weight:900;margin-bottom:6px;">내일 체크포인트</div>
    <ul style="margin:0;padding-left:18px;">
      <li>주요 지표 발표 일정(미국 CPI/고용, 연준 발언 등)</li>
      <li>유가: OPEC+/재고/지정학 이슈 헤드라인</li>
      <li>환율: 위험자산 선호/달러 강세 여부</li>
    </ul>
  </div>
</div>
"""
    return html.strip()


# -----------------------------
# main
# -----------------------------
def main():
    cfg = load_config()

    # 필수 WP 설정
    cfg["wp_base_url"] = must_get(cfg, "wp_base_url")
    cfg["wp_user"] = must_get(cfg, "wp_user")
    cfg["wp_app_pass"] = must_get(cfg, "wp_app_pass")

    today = now_kst_date_str()

    # TEST_MODE=1 이면 계속 새 글 생성(슬러그에 runid)
    test_mode = os.getenv("TEST_MODE", "0").strip() == "1"
    run_id = str(int(time.time())) if test_mode else ""
    test_badge = f" - {run_id}" if test_mode else ""

    title = f"오늘의 지표 리포트 ({today}){test_badge}"
    slug = f"daily-indicator-report-{today}" + (f"-{run_id}" if test_mode else "")

    try:
        # 중복 방지(테스트모드에서는 중복 허용)
        if (not test_mode) and wp_post_exists(cfg, slug):
            msg = f"✅ 이미 오늘 글이 있어요 ({today})\n중복 발행 안 함"
            print(msg)
            kakao_send_to_me(cfg, msg)
            return

        indicators, errors = fetch_indicators()

        # 뉴스(원인 참고)
        news_map = {
            "USD/KRW": fetch_news_rss("원달러 환율 변동 원인", 5),
            "Brent Oil": fetch_news_rss("브렌트유 가격 하락 원인", 5),
            "KOSPI": fetch_news_rss("코스피 변동 원인", 5),
        }

        html = build_html(title, indicators, errors, news_map)
        post = wp_create_post(cfg, title, slug, html, status="publish")
        link = post.get("link", cfg["wp_base_url"])

        ok_msg = f"✅ 글 발행 성공!\n날짜: {today}\n상태: publish\n링크: {link}"
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
