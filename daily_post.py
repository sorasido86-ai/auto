# -*- coding: utf-8 -*-
"""
Daily WordPress Post (No paid API key)
- USD/KRW: Stooq (fallback: exchangerate.host)
- KOSPI  : Stooq (^kospi)
- Brent  : EIA HTML table (fallback: Stooq brn.f / cb.f)
- News   : Google News RSS (ko-KR) + 한글 제목 위주 필터링

MODE
- daily: 같은 날짜 slug면 중복 방지(1일 1회)
- test : 매 실행마다 새로운 slug 생성 (5분 스케줄 테스트용)

ENV (Actions에서 주로 사용)
- MODE=test|daily
- WP_STATUS=draft|publish
- RUN_ID=anything (github.run_id 추천)
"""

import os
import re
import json
import base64
import csv
import io
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
import feedparser


WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"
CONFIG_PATH = Path(__file__).with_name("bot_config.json")

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# -----------------------------
# 공통 유틸
# -----------------------------
def now_kst_date_str() -> str:
    # KST 기준 날짜(서버 TZ에 영향 덜 받게)
    # Actions에서 TZ=Asia/Seoul도 주지만, 혹시 몰라서 로컬 기준으로만 씀
    return datetime.now().strftime("%Y-%m-%d")

def is_hangul_text(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", "")
        if s in ("", "None", "nan", "NaN", "NA", "-"):
            return None
        return float(s)
    except Exception:
        return None

def fmt_num(x: Optional[float], digits=2) -> str:
    if x is None:
        return "-"
    return f"{x:,.{digits}f}"

def pct_change(cur: Optional[float], prev: Optional[float]) -> Optional[float]:
    if cur is None or prev is None or prev == 0:
        return None
    return (cur - prev) / prev * 100.0

def http_get(url: str, timeout=25, retries=2) -> requests.Response:
    last_err = None
    for i in range(retries + 1):
        try:
            r = requests.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": UA,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain,*/*;q=0.8",
                    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
                },
            )
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (i + 1))
    raise last_err


# -----------------------------
# 설정 로드 (파일 + ENV 오버라이드)
# -----------------------------
def load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}

    # ENV 오버라이드 지원
    env_map = {
        "wp_base_url": os.getenv("WP_BASE_URL"),
        "wp_user": os.getenv("WP_USER"),
        "wp_app_pass": os.getenv("WP_APP_PASS"),
        "kakao_rest_key": os.getenv("KAKAO_REST_KEY"),
        "kakao_refresh_token": os.getenv("KAKAO_REFRESH_TOKEN"),
    }
    for k, v in env_map.items():
        if v:
            cfg[k] = v
    return cfg

def must_get(cfg: Dict[str, Any], key: str) -> str:
    v = (cfg.get(key) or "").strip()
    if not v:
        raise ValueError(f"필수 설정 누락: {key}")
    return v


# -----------------------------
# WordPress
# -----------------------------
def wp_posts_api(cfg: Dict[str, Any]) -> str:
    return must_get(cfg, "wp_base_url").rstrip("/") + WP_POSTS_API_SUFFIX

def wp_auth_headers(cfg: Dict[str, Any]) -> Dict[str, str]:
    user = must_get(cfg, "wp_user").strip()
    app_pass = must_get(cfg, "wp_app_pass").replace(" ", "").strip()
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}

def wp_post_exists(cfg: Dict[str, Any], slug: str) -> bool:
    r = requests.get(
        wp_posts_api(cfg),
        params={"slug": slug},
        headers={**wp_auth_headers(cfg), "User-Agent": UA},
        timeout=25,
    )
    r.raise_for_status()
    return len(r.json()) > 0

def wp_create_post(cfg: Dict[str, Any], title: str, slug: str, content: str, status: str) -> Dict[str, Any]:
    payload = {"title": title, "slug": slug, "content": content, "status": status}
    r = requests.post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json", "User-Agent": UA},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=25,
    )
    r.raise_for_status()
    return r.json()


# -----------------------------
# 데이터 수집
# -----------------------------
def fetch_stooq_last_two_closes(symbol: str) -> Tuple[float, float, str]:
    """
    returns: (latest, prev, latest_date)
    """
    sym = quote_plus(symbol)
    urls = [
        f"https://stooq.com/q/d/l/?s={sym}&i=d",
        f"https://stooq.pl/q/d/l/?s={sym}&i=d",
    ]

    last_err = None
    for url in urls:
        try:
            r = http_get(url, timeout=25, retries=1)
            text = (r.text or "").strip()

            # Stooq가 가끔 HTML/차단 페이지를 줌 -> CSV 아니면 스킵
            if text.startswith("<") or "Date,Open,High,Low,Close" not in text:
                raise ValueError("CSV가 아닌 응답(HTML/빈 값) 수신")

            reader = csv.DictReader(io.StringIO(text))
            rows = []
            for row in reader:
                c = safe_float(row.get("Close"))
                d = (row.get("Date") or "").strip()
                if c is not None and d:
                    rows.append((d, c))

            if len(rows) < 2:
                raise ValueError("데이터가 2개 미만")

            # 마지막 2개
            d2, c2 = rows[-1]
            d1, c1 = rows[-2]
            return c2, c1, d2

        except Exception as e:
            last_err = e

    raise ValueError(f"Stooq 데이터 수집 실패: {symbol} ({last_err})")

def fetch_usdkrw() -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    returns (latest, prev, date, error)
    """
    try:
        latest, prev, d = fetch_stooq_last_two_closes("usdkrw")
        return latest, prev, d, None
    except Exception as e:
        # fallback: exchangerate.host (전일값은 없음)
        try:
            r = http_get("https://api.exchangerate.host/latest?base=USD&symbols=KRW", retries=1)
            j = r.json()
            val = safe_float(j.get("rates", {}).get("KRW"))
            if val is None:
                raise ValueError("fallback 값 파싱 실패")
            return val, None, now_kst_date_str(), f"Stooq 실패 → fallback 사용 ({e})"
        except Exception as e2:
            return None, None, None, f"Stooq+fallback 모두 실패 ({e}) / ({e2})"

def fetch_kospi() -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Stooq KOSPI index symbol: ^kospi
    """
    try:
        latest, prev, d = fetch_stooq_last_two_closes("^kospi")
        return latest, prev, d, None
    except Exception as e:
        return None, None, None, f"Stooq 실패 ({e})"

def fetch_brent_from_eia() -> Tuple[float, float, str]:
    """
    EIA Brent Spot Price FOB (RBRTE) HTML table.
    """
    url = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=RBRTE&f=D"
    r = http_get(url, retries=2)
    html = r.text or ""
    soup = BeautifulSoup(html, "lxml")

    # 테이블 찾기(헤더에 Date / Value가 있는 것)
    tables = soup.find_all("table")
    target = None
    for t in tables:
        th_text = " ".join([th.get_text(" ", strip=True) for th in t.find_all("th")])
        if "Date" in th_text and ("Value" in th_text or "Prices" in th_text):
            target = t
            break
    if not target:
        # EIA 페이지 구조가 바뀐 경우
        raise ValueError("EIA 테이블을 찾지 못했습니다")

    rows = []
    for tr in target.find_all("tr"):
        tds = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(tds) >= 2:
            d = tds[0].strip()
            v = safe_float(tds[1])
            if d and v is not None:
                rows.append((d, v))

    if len(rows) < 2:
        raise ValueError("EIA 데이터가 2개 미만입니다")

    d2, c2 = rows[-1]
    d1, c1 = rows[-2]
    return c2, c1, d2

def fetch_brent() -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    try:
        latest, prev, d = fetch_brent_from_eia()
        return latest, prev, d, None
    except Exception as e:
        # fallback: Stooq futures symbols (가끔 되는 경우)
        for sym in ("brn.f", "cb.f"):
            try:
                latest, prev, d = fetch_stooq_last_two_closes(sym)
                return latest, prev, d, f"EIA 실패 → Stooq({sym}) fallback 사용 ({e})"
            except Exception:
                pass
        return None, None, None, f"Brent 수집 실패 (EIA+fallback 모두 실패: {e})"


# -----------------------------
# 뉴스(RSS)
# -----------------------------
def fetch_google_news_rss(query: str, max_items=3) -> List[Tuple[str, str]]:
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=ko&gl=KR&ceid=KR:ko"
    )
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:20]:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        # 한자/영문만 있는 경우가 섞일 때가 있어서 한글 포함 우선
        if is_hangul_text(title):
            out.append((title, link))
        if len(out) >= max_items:
            break

    # 한글이 너무 없으면 그냥 앞에서 채움
    if len(out) < max_items:
        for e in feed.entries:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            if not title or not link:
                continue
            if (title, link) not in out:
                out.append((title, link))
            if len(out) >= max_items:
                break

    return out[:max_items]


# -----------------------------
# 콘텐츠 생성(디자인 개선)
# -----------------------------
def reason_template(name: str, delta: Optional[float]) -> List[str]:
    """
    기사 없이도 쓸 수 있는 '일반론' 템플릿(헤드라인 기반 참고용)
    """
    if delta is None:
        return ["오늘은 휴장/데이터 공백 등으로 전일대비 해석이 제한됩니다."]

    up = delta > 0
    if name == "USD/KRW":
        return [
            "달러 강세(미국 금리/지표/연준 발언)나 위험회피 심리가 커지면 원/달러가 오르는 경향이 있습니다." if up
            else "위험선호(주식 강세)나 달러 약세(금리 기대 완화) 국면이면 원/달러가 내려가는 경향이 있습니다.",
            "한국/미국의 물가·고용·금리 관련 일정, 외국인 자금 흐름이 단기 변동에 영향을 줄 수 있습니다.",
        ]
    if name == "Brent Oil":
        return [
            "공급(OPEC+ 정책, 지정학 이슈)과 수요(경기 전망, 중국/미국 지표)가 유가 방향을 좌우합니다.",
            "재고 지표(EIA/IEA)나 달러 강세는 유가에 압력으로 작용하는 경우가 있습니다." if not up else
            "재고 감소, 공급 차질, 수요 기대가 커지면 유가가 반등하는 경우가 있습니다.",
        ]
    if name == "KOSPI":
        return [
            "미국 증시 흐름, 외국인 수급, 환율이 동반되면 지수 변동이 커질 수 있습니다.",
            "반도체·2차전지 등 대형 업종 뉴스가 지수에 직접 영향을 주는 경우가 많습니다.",
        ]
    return ["관련 이슈(금리/경기/수급/정책)를 함께 보면 이해가 쉬워집니다."]

def build_html_report(today: str, mode: str, run_id: str) -> Tuple[str, str, List[str]]:
    errors: List[str] = []

    usd, usd_prev, usd_date, usd_err = fetch_usdkrw()
    if usd_err:
        errors.append(f"USD/KRW: {usd_err}")

    brent, brent_prev, brent_date, brent_err = fetch_brent()
    if brent_err:
        errors.append(f"Brent: {brent_err}")

    kospi, kospi_prev, kospi_date, kospi_err = fetch_kospi()
    if kospi_err:
        errors.append(f"KOSPI: {kospi_err}")

    usd_delta = None if (usd is None or usd_prev is None) else (usd - usd_prev)
    brent_delta = None if (brent is None or brent_prev is None) else (brent - brent_prev)
    kospi_delta = None if (kospi is None or kospi_prev is None) else (kospi - kospi_prev)

    usd_pct = pct_change(usd, usd_prev)
    brent_pct = pct_change(brent, brent_prev)
    kospi_pct = pct_change(kospi, kospi_prev)

    # 뉴스(헤드라인)
    usd_news = fetch_google_news_rss("원달러 환율 변동 원인", 3)
    brent_news = fetch_google_news_rss("브렌트유 유가 하락 원인", 3)
    kospi_news = fetch_google_news_rss("코스피 상승 하락 원인", 3)

    test_badge = ""
    if mode == "test":
        test_badge = f"""<span style="display:inline-block;margin-left:10px;padding:4px 10px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px;font-weight:700;">
        TEST {run_id}
        </span>"""

    # 상단 요약 문장
    summary_lines = [
        "원/달러(USD/KRW), 브렌트유, 코스피의 최근 거래일 기준 변동을 한 번에 정리했습니다.",
        "변동 원인 참고용으로 관련 뉴스 헤드라인을 함께 첨부했습니다(헤드라인 기반 참고).",
    ]

    # 에러 박스
    err_box = ""
    if errors:
        li = "\n".join([f"<li><b>{e}</b></li>" for e in errors])
        err_box = f"""
        <div style="border:1px solid #fecaca;background:#fff1f2;padding:14px 16px;border-radius:12px;margin:14px 0;">
          <div style="font-weight:800;margin-bottom:6px;">⚠️ 일부 데이터 수집 실패</div>
          <ul style="margin:0 0 0 18px; padding:0;">{li}</ul>
          <div style="margin-top:8px;color:#6b7280;font-size:12px;">
            ※ 주말/휴장/해외사이트 차단 등으로 값이 비어 있을 수 있어요. 가능한 한 fallback을 적용했습니다.
          </div>
        </div>
        """

    def row(name, cur, prev, pct, d):
        cur_txt = "수집 실패" if cur is None else fmt_num(cur)
        prev_txt = "-" if prev is None else (f"{prev:+,.2f}" if name == "USD/KRW" else fmt_num(prev))
        # 전일대비는 값이 있을 때만
        delta_txt = "-"
        if cur is not None and prev is not None:
            delta = cur - prev
            sign = "+" if delta >= 0 else ""
            delta_txt = f"{sign}{delta:,.2f}"
        pct_txt = "-" if pct is None else (f"{pct:+.2f}%")
        d_txt = d or "-"
        return f"""
        <tr>
          <td style="padding:14px 12px;border-top:1px solid #e5e7eb;font-weight:700;">{name}</td>
          <td style="padding:14px 12px;border-top:1px solid #e5e7eb;text-align:right;">{cur_txt}</td>
          <td style="padding:14px 12px;border-top:1px solid #e5e7eb;text-align:right;">{delta_txt}</td>
          <td style="padding:14px 12px;border-top:1px solid #e5e7eb;text-align:right;">{pct_txt}</td>
          <td style="padding:14px 12px;border-top:1px solid #e5e7eb;text-align:center;color:#374151;">{d_txt}</td>
        </tr>
        """

    def news_block(title, items):
        if not items:
            return "<div style='color:#6b7280;'>- 관련 뉴스 수집 실패(또는 결과 없음)</div>"
        lis = "\n".join([f"<li style='margin:6px 0;'><a href='{link}' target='_blank' rel='noopener noreferrer'>{t}</a></li>"
                         for t, link in items])
        return f"<ul style='margin:8px 0 0 18px;padding:0;'>{lis}</ul>"

    def reason_block(name, delta):
        pts = reason_template(name, delta)
        lis = "\n".join([f"<li style='margin:6px 0;'>{p}</li>" for p in pts])
        return f"<ul style='margin:8px 0 0 18px;padding:0;'>{lis}</ul>"

    html = f"""
<div style="max-width:860px;margin:0 auto;font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;line-height:1.6;">
  <h1 style="font-size:28px;margin:0 0 10px 0;">오늘의 지표 리포트 ({today}){test_badge}</h1>

  {err_box}

  <h2 style="font-size:20px;margin:18px 0 8px 0;">핵심 요약</h2>
  <ul style="margin:0 0 0 18px;padding:0;">
    {''.join([f"<li style='margin:6px 0;'>{s}</li>" for s in summary_lines])}
  </ul>

  <h2 style="font-size:20px;margin:22px 0 10px 0;">주요 지표</h2>
  <div style="border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;">
    <table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="background:#0f172a;color:#fff;">
          <th style="padding:14px 12px;text-align:left;">지표</th>
          <th style="padding:14px 12px;text-align:right;">현재</th>
          <th style="padding:14px 12px;text-align:right;">전일대비</th>
          <th style="padding:14px 12px;text-align:right;">변동률</th>
          <th style="padding:14px 12px;text-align:center;">기준일(데이터)</th>
        </tr>
      </thead>
      <tbody style="background:#fff;">
        {row("USD/KRW", usd, usd_prev, usd_pct, usd_date)}
        {row("Brent Oil", brent, brent_prev, brent_pct, brent_date)}
        {row("KOSPI", kospi, kospi_prev, kospi_pct, kospi_date)}
      </tbody>
    </table>
  </div>

  <h2 style="font-size:20px;margin:24px 0 8px 0;">왜 움직였나? (원인 참고)</h2>
  <div style="color:#6b7280;font-size:13px;margin-bottom:10px;">
    헤드라인 기반 참고입니다. 실제 원인은 복합적일 수 있어요.
  </div>

  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;margin:10px 0;">
    <div style="font-size:18px;font-weight:800;margin-bottom:6px;">USD/KRW 변동 원인(뉴스)</div>
    {news_block("USD/KRW", usd_news)}
    <div style="margin-top:12px;font-weight:800;">가능한 요인(일반론)</div>
    {reason_block("USD/KRW", usd_delta)}
  </div>

  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;margin:10px 0;">
    <div style="font-size:18px;font-weight:800;margin-bottom:6px;">Brent 유가 변동 원인(뉴스)</div>
    {news_block("Brent", brent_news)}
    <div style="margin-top:12px;font-weight:800;">가능한 요인(일반론)</div>
    {reason_block("Brent Oil", brent_delta)}
  </div>

  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px 16px;margin:10px 0;">
    <div style="font-size:18px;font-weight:800;margin-bottom:6px;">KOSPI 변동 원인(뉴스)</div>
    {news_block("KOSPI", kospi_news)}
    <div style="margin-top:12px;font-weight:800;">가능한 요인(일반론)</div>
    {reason_block("KOSPI", kospi_delta)}
  </div>

  <h2 style="font-size:20px;margin:24px 0 8px 0;">내일 체크포인트</h2>
  <ul style="margin:0 0 0 18px;padding:0;">
    <li style="margin:6px 0;">미국 CPI/고용/연준 발언 등 주요 일정</li>
    <li style="margin:6px 0;">유가: OPEC/재고/지정학 헤드라인</li>
    <li style="margin:6px 0;">환율: 위험자산 선호/달러 강세 여부</li>
  </ul>

  <div style="margin-top:18px;color:#6b7280;font-size:12px;">
    ※ 데이터는 각 소스의 “최신 거래일” 기준입니다. 주말/휴장일에는 기준일이 이전 날짜로 표시될 수 있습니다.
  </div>
</div>
"""

    # test 모드면 제목/슬러그에 run_id를 붙여 매번 새 글 생성
    if mode == "test":
        title = f"오늘의 지표 리포트 ({today}) - {run_id}"
        slug = f"daily-indicator-report-{today}-{run_id}"
    else:
        title = f"오늘의 지표 리포트 ({today})"
        slug = f"daily-indicator-report-{today}"

    return title, slug, errors, html


def main():
    cfg = load_config()

    # 실행 모드
    mode = (os.getenv("MODE") or "daily").strip().lower()
    if mode not in ("daily", "test"):
        mode = "daily"

    # WP 상태
    wp_status = (os.getenv("WP_STATUS") or ("draft" if mode == "test" else "publish")).strip().lower()
    if wp_status not in ("draft", "publish", "private", "pending"):
        wp_status = "draft" if mode == "test" else "publish"

    run_id = (os.getenv("RUN_ID") or str(int(time.time()))).strip()
    today = now_kst_date_str()

    # 필수 설정 체크
    must_get(cfg, "wp_base_url")
    must_get(cfg, "wp_user")
    must_get(cfg, "wp_app_pass")

    title, slug, errors, html = build_html_report(today=today, mode=mode, run_id=run_id)

    # daily 모드만 중복 방지
    if mode == "daily":
        if wp_post_exists(cfg, slug):
            print(f"[SKIP] 이미 오늘 글이 존재합니다: {slug}")
            return

    post = wp_create_post(cfg, title=title, slug=slug, content=html, status=wp_status)
    link = post.get("link") or (must_get(cfg, "wp_base_url").rstrip("/") + "/" + slug + "/")

    print("✅ 글 생성 완료")
    print(" - mode:", mode)
    print(" - status:", wp_status)
    print(" - slug:", slug)
    print(" - link:", link)

    if errors:
        print("⚠️ 일부 데이터 수집 실패:")
        for e in errors:
            print(" -", e)


if __name__ == "__main__":
    main()
