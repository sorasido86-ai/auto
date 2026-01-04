# -*- coding: utf-8 -*-
"""
daily_post.py (통합본)
- WordPress REST API로 '오늘의 지표 리포트' 자동 발행
- 카테고리: 팁(카테고리 ID=8)로 업로드
- 데이터: 무료(키 없이) 기반으로 최대한 안정적으로 수집(Stooq + Google News RSS)

필수 파일: bot_config.json (daily_post.py와 같은 폴더에 두는 걸 권장)
예)
{
  "wp_base_url": "https://rainsow.com",
  "wp_user": "아이디",
  "wp_app_pass": "앱비밀번호"
}

선택(카톡 알림)
{
  "kakao_rest_key": "...",
  "kakao_refresh_token": "..."
}

필수 패키지:
pip install requests beautifulsoup4 lxml feedparser
"""

import os
import re
import json
import csv
import time
import base64
from io import StringIO
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import requests
import feedparser
from bs4 import BeautifulSoup


# =========================
# 설정
# =========================
DEFAULT_CATEGORY_ID = 8       # ✅ "팁" 카테고리 ID
POST_STATUS = "publish"       # ✅ 실행하면 즉시 발행(publish)
REQUEST_TIMEOUT = 25
STOOQ_RETRY = 3
STOOQ_SLEEP = 1.2

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

STOOQ_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
}

GOOGLE_NEWS_BASE = "https://news.google.com/rss/search"


# =========================
# 공통 유틸
# =========================
def script_dir() -> Path:
    return Path(__file__).resolve().parent

def config_path() -> Path:
    # env로 경로 지정 가능: BOT_CONFIG_PATH
    env = os.getenv("BOT_CONFIG_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return script_dir() / "bot_config.json"

def load_config() -> Dict[str, Any]:
    p = config_path()
    if not p.exists():
        raise FileNotFoundError(
            f"bot_config.json을 찾을 수 없습니다.\n"
            f"- 현재 찾는 위치: {p}\n"
            f"- 해결: daily_post.py와 같은 폴더에 bot_config.json을 두거나, "
            f"환경변수 BOT_CONFIG_PATH로 경로를 지정하세요."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def now_date_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", "")
        if s == "" or s.lower() in ("nan", "none", "-"):
            return None
        return float(s)
    except:
        return None

def fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:,.{digits}f}"

def fmt_signed(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:,.{digits}f}"

def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.{digits}f}%"

def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#039;")
    )


# =========================
# Kakao (선택)
# =========================
def refresh_access_token(cfg: Dict[str, Any]) -> str:
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": cfg["kakao_rest_key"],
        "refresh_token": cfg["kakao_refresh_token"],
    }
    r = requests.post(url, data=data, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    tokens = r.json()
    if "refresh_token" in tokens and tokens["refresh_token"]:
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        with open(config_path(), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    return tokens["access_token"]

def kakao_send_to_me(cfg: Dict[str, Any], text: str) -> None:
    if not cfg.get("kakao_rest_key") or not cfg.get("kakao_refresh_token"):
        return
    access_token = refresh_access_token(cfg)
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {"Authorization": f"Bearer {access_token}"}
    template_object = {
        "object_type": "text",
        "text": text[:1000],
        "link": {"web_url": cfg["wp_base_url"], "mobile_web_url": cfg["wp_base_url"]},
        "button_title": "사이트 열기"
    }
    data = {"template_object": json.dumps(template_object, ensure_ascii=False)}
    r = requests.post(url, headers=headers, data=data, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()


# =========================
# WordPress
# =========================
def wp_api_posts(cfg: Dict[str, Any]) -> str:
    return cfg["wp_base_url"].rstrip("/") + "/wp-json/wp/v2/posts"

def wp_headers(cfg: Dict[str, Any]) -> Dict[str, str]:
    user = str(cfg["wp_user"]).strip()
    app_pass = str(cfg["wp_app_pass"]).replace(" ", "").strip()
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "User-Agent": UA,
    }

def wp_post_exists(cfg: Dict[str, Any], slug: str) -> bool:
    r = requests.get(
        wp_api_posts(cfg),
        params={"slug": slug, "per_page": 1},
        headers=wp_headers(cfg),
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return len(r.json()) > 0

def wp_create_post(
    cfg: Dict[str, Any],
    title: str,
    slug: str,
    content_html: str,
    status: str = POST_STATUS,
) -> Dict[str, Any]:
    # ✅ 카테고리 "팁"(ID=8) 적용
    cat_id = cfg.get("wp_category_id", DEFAULT_CATEGORY_ID)
    try:
        cat_id = int(cat_id)
    except:
        cat_id = DEFAULT_CATEGORY_ID

    payload = {
        "title": title,
        "slug": slug,
        "content": content_html,
        "status": status,
        "categories": [cat_id],  # ✅ 여기서 팁 카테고리 지정됨
    }

    r = requests.post(
        wp_api_posts(cfg),
        headers=wp_headers(cfg),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# =========================
# 데이터 수집: Stooq (무료)
# =========================
def fetch_stooq_csv(symbol: str) -> str:
    """
    Stooq CSV 다운로드. 가끔 봇 방지/차단으로 HTML이 올 수 있어서 검증 필요.
    """
    url = "https://stooq.com/q/d/l/"
    params = {"s": symbol, "i": "d"}

    last_err = None
    for i in range(STOOQ_RETRY):
        try:
            r = requests.get(url, params=params, headers=STOOQ_HEADERS, timeout=REQUEST_TIMEOUT)
            # 200인데도 HTML이 올 수 있음
            txt = (r.text or "").strip()
            if not txt:
                raise ValueError("빈 응답")
            # CSV 헤더가 "Date"로 시작하는지 체크
            if not txt.lower().startswith("date,"):
                # HTML 응답일 확률이 높음
                raise ValueError("CSV가 아닌 응답(HTML/빈 값) 수신")
            return txt
        except Exception as e:
            last_err = e
            time.sleep(STOOQ_SLEEP * (i + 1))

    raise ValueError(f"Stooq CSV 수집 실패: {symbol} ({type(last_err).__name__}: {last_err})")

def parse_stooq_last_two_closes(csv_text: str) -> Tuple[float, float, str]:
    """
    CSV에서 마지막 2개 종가(close)를 반환: (latest, prev, latest_date)
    """
    f = StringIO(csv_text)
    reader = csv.DictReader(f)
    rows = []
    for row in reader:
        # Date, Open, High, Low, Close, Volume
        c = safe_float(row.get("Close"))
        d = (row.get("Date") or "").strip()
        if c is not None and d:
            rows.append((d, c))

    if len(rows) < 2:
        raise ValueError("데이터가 부족합니다(2개 미만)")

    latest_d, latest_c = rows[-1]
    prev_d, prev_c = rows[-2]
    return latest_c, prev_c, latest_d

def fetch_last_two_closes_candidates(symbol_candidates: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    여러 후보 심볼을 순차 시도해 성공하는 값을 반환.
    return: latest, prev, date, used_symbol
    """
    last_err = None
    for sym in symbol_candidates:
        try:
            txt = fetch_stooq_csv(sym)
            latest, prev, d = parse_stooq_last_two_closes(txt)
            return latest, prev, d, sym
        except Exception as e:
            last_err = e
            continue
    return None, None, None, f"FAIL({type(last_err).__name__}: {last_err})" if last_err else "FAIL(unknown)"

def fetch_indicators_free() -> Tuple[Dict[str, Any], List[str]]:
    """
    무료로 가능한 범위에서 최대한 안정적으로 수집.
    - USD/KRW: usdkrw
    - Brent: brent.f / brn.f / cb.f 등 후보 시도
    - KOSPI: ^ks11 / ks11 후보 시도
    """
    errors = []

    # USD/KRW
    usd_latest, usd_prev, usd_date, usd_used = fetch_last_two_closes_candidates(
        ["usdkrw", "usdkpw"]  # 혹시 몰라 후보
    )
    if usd_latest is None:
        errors.append(f"USD/KRW 수집 실패: {usd_used}")

    # Brent (Stooq에서 심볼이 가끔 바뀌거나 막힐 수 있어 후보를 넉넉히)
    brent_latest, brent_prev, brent_date, brent_used = fetch_last_two_closes_candidates(
        ["brent.f", "brn.f", "cb.f", "co.f"]
    )
    if brent_latest is None:
        errors.append(f"Brent 수집 실패: {brent_used}")

    # KOSPI
    kospi_latest, kospi_prev, kospi_date, kospi_used = fetch_last_two_closes_candidates(
        ["^ks11", "ks11"]
    )
    if kospi_latest is None:
        errors.append(f"KOSPI 수집 실패: {kospi_used}")

    out = {
        "usdkrw": {"latest": usd_latest, "prev": usd_prev, "date": usd_date, "source": usd_used},
        "brent": {"latest": brent_latest, "prev": brent_prev, "date": brent_date, "source": brent_used},
        "kospi": {"latest": kospi_latest, "prev": kospi_prev, "date": kospi_date, "source": kospi_used},
    }
    return out, errors


# =========================
# 뉴스: Google News RSS (ko-KR)
# =========================
def google_news_rss(query: str, max_items: int = 3) -> List[Dict[str, str]]:
    params = {
        "q": query,
        "hl": "ko",
        "gl": "KR",
        "ceid": "KR:ko",
    }
    r = requests.get(GOOGLE_NEWS_BASE, params=params, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    feed = feedparser.parse(r.text)
    items = []
    for e in (feed.entries or [])[:max_items]:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if title and link:
            # title이 너무 길면 조금 줄이기
            if len(title) > 90:
                title = title[:90] + "…"
            items.append({"title": title, "link": link})
    return items

def build_reason_section(ind: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    # 최대한 한국어 결과가 나오게 키워드도 한국어 중심
    return {
        "usdkrw": google_news_rss("원달러 환율 변동 원인", 3),
        "brent": google_news_rss("브렌트유 유가 하락 원인", 3),
        "kospi": google_news_rss("코스피 상승 하락 원인", 3),
    }


# =========================
# 리포트 HTML 생성
# =========================
def calc_change(latest: Optional[float], prev: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if latest is None or prev is None:
        return None, None
    chg = latest - prev
    pct = (chg / prev * 100.0) if prev != 0 else None
    return chg, pct

def heuristics_comment(name: str, chg: Optional[float]) -> str:
    """
    아주 짧은 '가능한 요인' 템플릿(단정 X)
    """
    if chg is None:
        return "데이터가 부족해 변동 요인을 추정하기 어렵습니다."
    up = chg > 0
    if name == "usdkrw":
        return "달러 강세/위험회피 심리, 외국인 수급, 대외 이벤트(미국 지표·연준 발언 등) 영향 가능성이 있습니다." if up \
            else "달러 약세/위험선호 회복, 수급 완화, 대외 불확실성 완화 등의 영향 가능성이 있습니다."
    if name == "brent":
        return "수요 둔화 우려, 재고 증가, OPEC+ 공급 이슈, 달러 강세 등의 영향 가능성이 있습니다." if not up \
            else "공급 차질 우려, OPEC+ 감산, 지정학 리스크, 수요 회복 기대 등의 영향 가능성이 있습니다."
    if name == "kospi":
        return "대형주 수급, 환율/금리/반도체 사이클, 해외 증시 흐름 등 복합 요인일 수 있습니다." if up \
            else "차익실현, 환율/금리 부담, 대외 불확실성 등 복합 요인일 수 있습니다."
    return "복합 요인일 수 있습니다."

def build_html_report(today: str, ind: Dict[str, Any], errors: List[str], news: Dict[str, List[Dict[str, str]]]) -> str:
    # 계산
    usd = ind["usdkrw"]
    brent = ind["brent"]
    kospi = ind["kospi"]

    usd_chg, usd_pct = calc_change(usd["latest"], usd["prev"])
    br_chg, br_pct = calc_change(brent["latest"], brent["prev"])
    ko_chg, ko_pct = calc_change(kospi["latest"], kospi["prev"])

    run_id = os.getenv("TEST_RUN_ID", "").strip()
    badge = f"TEST {html_escape(run_id)}" if run_id else ""

    # 표 행
    def row(label: str, latest: Optional[float], chg: Optional[float], pct: Optional[float], date: Optional[str]) -> str:
        return f"""
        <tr>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; font-weight:600;">{html_escape(label)}</td>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; text-align:right;">{fmt_num(latest)}</td>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; text-align:right;">{fmt_signed(chg)}</td>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; text-align:right;">{fmt_pct(pct)}</td>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; text-align:center;">{html_escape(date or "-")}</td>
        </tr>
        """

    # 에러 박스
    err_html = ""
    if errors:
        li = "".join([f"<li style='margin:6px 0'>{html_escape(e)}</li>" for e in errors])
        err_html = f"""
        <div style="border:1px solid #ffb3b3; background:#fff3f3; padding:14px 16px; border-radius:10px; margin:18px 0;">
          <div style="font-weight:700; margin-bottom:6px;">⚠️ 일부 데이터 수집 실패</div>
          <ul style="margin:8px 0 0 18px; padding:0;">{li}</ul>
          <div style="color:#666; font-size:13px; margin-top:10px;">
            ※ 주말/휴장/봇 방지 차단 등으로 값이 비어 있을 수 있어요. (가능한 범위에서 재시도/후보 심볼 fallback 적용)
          </div>
        </div>
        """

    # 뉴스 리스트
    def news_list(items: List[Dict[str, str]]) -> str:
        if not items:
            return "<div style='color:#666'>- 관련 뉴스 결과 없음</div>"
        out = []
        for it in items:
            t = html_escape(it["title"])
            l = html_escape(it["link"])
            out.append(f"<li style='margin:7px 0;'><a href='{l}' target='_blank' rel='noopener'>{t}</a></li>")
        return "<ul style='margin:10px 0 0 18px; padding:0;'>" + "".join(out) + "</ul>"

    # 가능한 요인 코멘트(단정X)
    usd_reason = heuristics_comment("usdkrw", usd_chg)
    br_reason = heuristics_comment("brent", br_chg)
    ko_reason = heuristics_comment("kospi", ko_chg)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
    <div style="max-width:860px; margin:0 auto; font-family:system-ui,-apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif; color:#111;">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <h1 style="margin:0; font-size:34px; line-height:1.15;">오늘의 지표 리포트 ({html_escape(today)})</h1>
        {f"<div style='padding:8px 12px; background:#eef2ff; color:#1e3a8a; border-radius:999px; font-weight:700; font-size:13px;'>{badge}</div>" if badge else ""}
      </div>

      {err_html}

      <h2 style="margin:18px 0 10px; font-size:22px;">핵심 요약</h2>
      <ul style="margin:8px 0 0 18px;">
        <li style="margin:6px 0;">원/달러(USD/KRW), 브렌트유, 코스피의 <b>최근 거래일 기준 변동</b>을 한 번에 정리했습니다.</li>
        <li style="margin:6px 0;">변동 원인(참고용)으로 <b>관련 뉴스 헤드라인</b>을 함께 첨부했습니다(단정 아님).</li>
      </ul>

      <div style="color:#666; font-size:13px; margin-top:10px;">
        생성 시각: {html_escape(generated_at)} (환경 TZ 기준)
      </div>

      <h2 style="margin:22px 0 10px; font-size:22px;">주요 지표</h2>
      <div style="border:1px solid #e6e6e6; border-radius:12px; overflow:hidden;">
        <table style="width:100%; border-collapse:collapse;">
          <thead>
            <tr style="background:#0b1220; color:#fff;">
              <th style="padding:14px 12px; text-align:left;">지표</th>
              <th style="padding:14px 12px; text-align:right;">현재</th>
              <th style="padding:14px 12px; text-align:right;">전일대비</th>
              <th style="padding:14px 12px; text-align:right;">변동률</th>
              <th style="padding:14px 12px; text-align:center;">기준일(데이터)</th>
            </tr>
          </thead>
          <tbody style="background:#fff;">
            {row("USD/KRW", usd["latest"], usd_chg, usd_pct, usd["date"])}
            {row("Brent Oil", brent["latest"], br_chg, br_pct, brent["date"])}
            {row("KOSPI", kospi["latest"], ko_chg, ko_pct, kospi["date"])}
          </tbody>
        </table>
      </div>

      <h2 style="margin:26px 0 8px; font-size:22px;">왜 움직였나? (원인 참고)</h2>
      <div style="color:#666; font-size:13px; margin-bottom:14px;">헤드라인 기반 참고입니다. 실제 원인은 복합적일 수 있어요.</div>

      <div style="border:1px solid #e6e6e6; border-radius:12px; padding:16px; margin-bottom:12px;">
        <div style="font-size:18px; font-weight:800;">USD/KRW 변동 원인(뉴스)</div>
        <div style="margin-top:8px; color:#333;">- 가능 요인(단정X): {html_escape(usd_reason)}</div>
        {news_list(news.get("usdkrw", []))}
      </div>

      <div style="border:1px solid #e6e6e6; border-radius:12px; padding:16px; margin-bottom:12px;">
        <div style="font-size:18px; font-weight:800;">Brent 유가 변동 원인(뉴스)</div>
        <div style="margin-top:8px; color:#333;">- 가능 요인(단정X): {html_escape(br_reason)}</div>
        {news_list(news.get("brent", []))}
      </div>

      <div style="border:1px solid #e6e6e6; border-radius:12px; padding:16px; margin-bottom:12px;">
        <div style="font-size:18px; font-weight:800;">KOSPI 변동 원인(뉴스)</div>
        <div style="margin-top:8px; color:#333;">- 가능 요인(단정X): {html_escape(ko_reason)}</div>
        {news_list(news.get("kospi", []))}
      </div>

      <div style="color:#666; font-size:12px; margin-top:16px;">
        데이터 출처: Stooq(가격), Google News RSS(헤드라인)
      </div>
    </div>
    """
    return html


# =========================
# main
# =========================
def main():
    cfg = load_config()

    # wp 필수값 체크
    for k in ("wp_base_url", "wp_user", "wp_app_pass"):
        if not cfg.get(k):
            raise ValueError(f"필수 설정 누락: {k}")

    today = now_date_str()

    # 기본은 "하루 1개" (slug 중복 방지)
    # 테스트용(5분마다 발행 등)일 때는 TEST_RUN_ID를 넣으면 slug에 붙어서 중복 회피됨
    test_id = os.getenv("TEST_RUN_ID", "").strip()
    if test_id:
        slug = f"daily-indicator-report-{today}-{test_id}"
        title = f"오늘의 지표 리포트 ({today}) - {test_id}"
    else:
        slug = f"daily-indicator-report-{today}"
        title = f"오늘의 지표 리포트 ({today})"

    try:
        # 중복 발행 방지(테스트 모드가 아닐 때)
        if not test_id and wp_post_exists(cfg, slug):
            msg = f"✅ 이미 오늘 글이 있어요 ({today})\n중복 발행 안 함\nslug={slug}\n카테고리=팁(ID={DEFAULT_CATEGORY_ID})"
            print(msg)
            kakao_send_to_me(cfg, msg)
            return

        indicators, errors = fetch_indicators_free()
        news = build_reason_section(indicators)
        html = build_html_report(today, indicators, errors, news)

        post = wp_create_post(
            cfg=cfg,
            title=title,
            slug=slug,
            content_html=html,
            status=POST_STATUS,
        )
        link = post.get("link", cfg["wp_base_url"])

        msg = (
            f"✅ 글 발행 성공!\n"
            f"날짜: {today}\n"
            f"상태: {POST_STATUS}\n"
            f"카테고리: 팁(ID={DEFAULT_CATEGORY_ID})\n"
            f"링크: {link}"
        )
        print(msg)
        kakao_send_to_me(cfg, msg)

    except Exception as e:
        msg = f"❌ 자동발행 실패 ({today})\n{type(e).__name__}: {e}"
        print(msg)
        try:
            kakao_send_to_me(cfg, msg)
        except Exception as e2:
            print("카톡 알림까지 실패:", type(e2).__name__, e2)
        raise


if __name__ == "__main__":
    main()
