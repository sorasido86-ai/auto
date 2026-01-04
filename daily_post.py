# -*- coding: utf-8 -*-
"""
daily_post.py
- USD/KRW, Brent(cb.f), KOSPI(^kospi) 일간 변동 요약을 워드프레스에 자동 발행
- 데이터 소스: Stooq "HTML quote page" 파싱(= CSV가 HTML로 떨어지는 문제 회피)
- 뉴스(원인 참고): Google News RSS(ko) 기반 헤드라인 몇 개 첨부
"""

from __future__ import annotations

import base64
import html as htmlmod
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
import feedparser


KST = timezone(timedelta(hours=9))
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bot_config.json"

# ✅ 기본 카테고리(팁) ID
DEFAULT_WP_CATEGORY_ID = 8

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DailyReportBot/1.0; +https://github.com/)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ko-KR;q=0.8,ko;q=0.7",
    "Connection": "close",
}


MONTH_MAP = {
    # English
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    # Polish (혹시 페이지에 섞여 나올 때 대비)
    "sty": 1, "lut": 2, "mar": 3, "kwi": 4, "maj": 5, "cze": 6,
    "lip": 7, "sie": 8, "wrz": 9, "paź": 10, "paz": 10, "lis": 11, "gru": 12,
}

@dataclass
class Quote:
    name: str
    symbol: str
    last: Optional[float]
    change: Optional[float]
    pct: Optional[float]
    date: Optional[str]  # YYYY-MM-DD
    error: Optional[str] = None


def load_config() -> Dict[str, str]:
    """
    1) bot_config.json(스크립트와 같은 폴더) 있으면 그걸 사용
    2) 없으면 환경변수(WP_BASE_URL/WP_USER/WP_APP_PASS)로 구성
    + (선택) WP_CATEGORY_ID 또는 bot_config.json의 wp_category_id 사용 가능
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "wp_base_url": os.getenv("WP_BASE_URL", "").rstrip("/"),
            "wp_user": os.getenv("WP_USER", ""),
            "wp_app_pass": os.getenv("WP_APP_PASS", ""),
        }
        if os.getenv("KAKAO_REST_KEY") and os.getenv("KAKAO_REFRESH_TOKEN"):
            cfg["kakao_rest_key"] = os.environ["KAKAO_REST_KEY"]
            cfg["kakao_refresh_token"] = os.environ["KAKAO_REFRESH_TOKEN"]

    missing = [k for k in ["wp_base_url", "wp_user", "wp_app_pass"] if not cfg.get(k)]
    if missing:
        raise ValueError("필수 설정 누락: " + ", ".join(missing))

    # ✅ 카테고리 ID: bot_config.json의 wp_category_id > 환경변수 WP_CATEGORY_ID > 기본(8)
    cat_env = os.getenv("WP_CATEGORY_ID", "").strip()
    if not cfg.get("wp_category_id"):
        if cat_env.isdigit():
            cfg["wp_category_id"] = int(cat_env)
        else:
            cfg["wp_category_id"] = DEFAULT_WP_CATEGORY_ID
    else:
        try:
            cfg["wp_category_id"] = int(cfg["wp_category_id"])
        except Exception:
            cfg["wp_category_id"] = DEFAULT_WP_CATEGORY_ID

    return cfg


def wp_headers(cfg: Dict[str, str]) -> Dict[str, str]:
    token = base64.b64encode(f"{cfg['wp_user']}:{cfg['wp_app_pass']}".encode("utf-8")).decode("ascii")
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": HEADERS["User-Agent"],
    }


def wp_find_post_id_by_slug(cfg: Dict[str, str], slug: str) -> Optional[int]:
    url = f"{cfg['wp_base_url']}/wp-json/wp/v2/posts"
    r = requests.get(url, headers=wp_headers(cfg), params={"slug": slug, "per_page": 1}, timeout=30)
    if r.status_code == 200:
        arr = r.json()
        if arr:
            return int(arr[0]["id"])
        return None
    raise RuntimeError(f"WP 조회 실패: {r.status_code} {r.text[:200]}")


def wp_create_post(cfg: Dict[str, str], title: str, slug: str, content_html: str) -> str:
    url = f"{cfg['wp_base_url']}/wp-json/wp/v2/posts"

    # ✅ 여기만 추가/수정: categories=[8]
    cat_id = int(cfg.get("wp_category_id", DEFAULT_WP_CATEGORY_ID))

    payload = {
        "title": title,
        "slug": slug,
        "content": content_html,
        "status": "publish",
        "categories": [cat_id],  # ✅ 팁(8)
    }

    r = requests.post(url, headers=wp_headers(cfg), data=json.dumps(payload), timeout=30)
    if r.status_code in (200, 201):
        return r.json().get("link", "")
    raise RuntimeError(f"WP 발행 실패: {r.status_code} {r.text[:300]}")


def _to_float(s: str) -> float:
    s = s.strip().replace(",", "")
    return float(s)


def fetch_stooq_quote_html(symbol: str) -> Tuple[float, float, float, str]:
    """
    Stooq quote 페이지(HTML)에서:
    - last(종가/최근값), change(전일대비), pct, 기준일(YYYY-MM-DD)
    를 뽑아온다.
    """
    url = f"https://stooq.com/q/?s={quote(symbol)}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    text = r.text

    # 연도(페이지 어딘가에 20xx가 들어있음)
    y = None
    m_year = re.search(r"(20\d{2})", text)
    if m_year:
        y = int(m_year.group(1))
    else:
        y = datetime.now(KST).year

    # "2 Jan, 7:00 1443.71 -1.26 (-0.09%)" 같은 라인을 찾는다
    # (공백이 들쭉날쭉할 수 있어 최대한 유연하게)
    pattern = re.compile(
        r"(?m)^\s*(\d{1,2})\s+([A-Za-ząćęłńóśźż]{3,4})\s*,\s*\d{1,2}:\d{2}\s+([0-9.,]+)\s+([+-][0-9.,]+)\s+\(([+-]?[0-9.,]+)%\)"
    )
    m = pattern.search(text)
    if not m:
        # BeautifulSoup로 텍스트만 뽑아 재시도(사이트가 줄바꿈을 바꾸는 경우)
        soup = BeautifulSoup(text, "lxml")
        plain = soup.get_text("\n")
        m = pattern.search(plain)
        if not m:
            raise ValueError(f"Stooq 파싱 실패: {symbol} (quote 라인을 찾지 못함)")

    day = int(m.group(1))
    mon_str = m.group(2)
    mon_str = mon_str.replace("ą", "a").replace("ć", "c").replace("ę", "e").replace("ł", "l") \
                     .replace("ń", "n").replace("ó", "o").replace("ś", "s").replace("ź", "z").replace("ż", "z")
    mon_str = mon_str[:3].title() if mon_str[:3].isalpha() else mon_str[:3]
    mon = MONTH_MAP.get(mon_str, MONTH_MAP.get(mon_str.lower()))
    if not mon:
        raise ValueError(f"월 파싱 실패: {symbol} month={mon_str}")

    last = _to_float(m.group(3))
    chg = _to_float(m.group(4))
    pct = _to_float(m.group(5))

    date_iso = f"{y:04d}-{mon:02d}-{day:02d}"
    return last, chg, pct, date_iso


def fetch_all_quotes() -> Tuple[List[Quote], List[str]]:
    items = [
        ("USD/KRW", "usdkrw"),
        ("Brent Oil", "cb.f"),
        ("KOSPI", "^kospi"),
    ]

    quotes: List[Quote] = []
    errors: List[str] = []

    for name, sym in items:
        try:
            last, chg, pct, d = fetch_stooq_quote_html(sym)
            quotes.append(Quote(name=name, symbol=sym, last=last, change=chg, pct=pct, date=d))
        except Exception as e:
            msg = f"{name} 수집 실패: {e}"
            errors.append(msg)
            quotes.append(Quote(name=name, symbol=sym, last=None, change=None, pct=None, date=None, error=str(e)))

    return quotes, errors


def cjk_heavy(title: str) -> bool:
    if not title:
        return True
    han = sum(1 for ch in title if "\u4e00" <= ch <= "\u9fff")
    hangul = sum(1 for ch in title if "\uac00" <= ch <= "\ud7a3")
    letters = sum(1 for ch in title if ("a" <= ch.lower() <= "z"))
    # 한글이 없고, 한자가 과도하면 제외
    if hangul == 0 and han > 0 and han / max(1, len(title)) >= 0.25 and letters < 5:
        return True
    return False


def google_news_rss(query: str, max_items: int = 3) -> List[Tuple[str, str]]:
    url = (
        "https://news.google.com/rss/search?q="
        + quote(query)
        + "&hl=ko&gl=KR&ceid=KR:ko"
    )
    d = feedparser.parse(url)
    out: List[Tuple[str, str]] = []
    for e in getattr(d, "entries", [])[:15]:
        title = htmlmod.unescape(getattr(e, "title", "")).strip()
        link = getattr(e, "link", "").strip()
        if not title or not link:
            continue
        if cjk_heavy(title):
            continue
        out.append((title, link))
        if len(out) >= max_items:
            break
    return out


def direction_text(x: Optional[float]) -> str:
    if x is None:
        return "변동"
    if x > 0:
        return "상승"
    if x < 0:
        return "하락"
    return "보합"


def build_post_html(quotes: List[Quote], errors: List[str], run_tag: str) -> str:
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S (KST)")

    # 뉴스 쿼리(방향에 따라 문구만 바꿈)
    usd = next((q for q in quotes if q.name == "USD/KRW"), None)
    brent = next((q for q in quotes if q.name == "Brent Oil"), None)
    kospi = next((q for q in quotes if q.name == "KOSPI"), None)

    usd_q = f"원달러 환율 {direction_text(usd.change if usd else None)} 원인"
    brent_q = f"브렌트유 {direction_text(brent.change if brent else None)} 원인"
    kospi_q = f"코스피 {direction_text(kospi.change if kospi else None)} 원인"

    usd_news = google_news_rss(usd_q, 3)
    brent_news = google_news_rss(brent_q, 3)
    kospi_news = google_news_rss(kospi_q, 3)

    def fmt(v: Optional[float], kind: str = "num") -> str:
        if v is None:
            return "-"
        if kind == "pct":
            return f"{v:+.2f}%"
        if kind == "chg":
            return f"{v:+,.2f}"
        return f"{v:,.2f}"

    # 표 rows
    rows = []
    for q in quotes:
        if q.last is None:
            rows.append(f"""
              <tr>
                <td><b>{q.name}</b></td>
                <td class="muted">수집 실패</td>
                <td>-</td><td>-</td><td>-</td>
              </tr>
            """)
        else:
            rows.append(f"""
              <tr>
                <td><b>{q.name}</b></td>
                <td>{fmt(q.last)}</td>
                <td>{fmt(q.change, "chg")}</td>
                <td>{fmt(q.pct, "pct")}</td>
                <td>{q.date}</td>
              </tr>
            """)

    def news_block(title: str, items: List[Tuple[str, str]]) -> str:
        if not items:
            return f"<div class='card'><h3>{title}</h3><p class='muted'>- 관련 헤드라인 수집 실패(또는 결과 없음)</p></div>"
        lis = "\n".join([f"<li><a href='{link}' target='_blank' rel='noopener'>{htmlmod.escape(t)}</a></li>" for t, link in items])
        return f"<div class='card'><h3>{title}</h3><ul>{lis}</ul></div>"

    err_html = ""
    if errors:
        li = "\n".join([f"<li>{htmlmod.escape(e)}</li>" for e in errors])
        err_html = f"""
        <div class="alert">
          <b>⚠️ 일부 데이터 수집 실패</b>
          <ul>{li}</ul>
          <div class="muted">※ 주말/휴장/사이트 차단(봇 방지) 등으로 값이 비어 있을 수 있어요.</div>
        </div>
        """

    html = f"""
    <div class="wrap">
      <h1>오늘의 지표 리포트 ({datetime.now(KST).date()}) <span class="tag">{htmlmod.escape(run_tag)}</span></h1>

      {err_html}

      <div class="card">
        <h2>핵심 요약</h2>
        <ul>
          <li>원/달러(USD/KRW), 브렌트유, 코스피의 최근 거래일 기준 변동을 한 번에 정리했습니다.</li>
          <li>변동 원인 참고용으로 관련 뉴스 헤드라인을 함께 첨부했습니다(단정 아님).</li>
        </ul>
        <div class="muted">생성 시각: {now}</div>
      </div>

      <h2>주요 지표</h2>
      <div class="tablebox">
        <table>
          <thead>
            <tr>
              <th>지표</th>
              <th>현재</th>
              <th>전일대비</th>
              <th>변동률</th>
              <th>기준일(데이터)</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>

      <h2>왜 움직였나? (원인 참고)</h2>
      <p class="muted">헤드라인 기반 참고입니다. 실제 원인은 복합적일 수 있어요.</p>

      <div class="grid">
        {news_block("USD/KRW 변동 원인(뉴스)", usd_news)}
        {news_block("Brent 유가 변동 원인(뉴스)", brent_news)}
        {news_block("KOSPI 변동 원인(뉴스)", kospi_news)}
      </div>

      <div class="muted" style="margin-top:14px;">
        데이터: Stooq(최근 거래일 기준). 주말/휴장일엔 마지막 거래일이 표시됩니다.
      </div>
    </div>

    <style>
      .wrap {{ max-width: 920px; margin: 0 auto; padding: 8px 10px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Apple SD Gothic Neo", "Noto Sans KR", Arial, sans-serif; }}
      h1 {{ font-size: 30px; margin: 10px 0 16px; }}
      h2 {{ font-size: 22px; margin: 18px 0 10px; }}
      .tag {{ display:inline-block; margin-left:8px; padding:6px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; vertical-align:middle; }}
      .muted {{ color:#6b7280; font-size: 13px; }}
      .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,.03); }}
      .alert {{ background:#fff5f5; border:1px solid #fecaca; color:#7f1d1d; border-radius:14px; padding:12px 14px; margin: 10px 0 14px; }}
      .tablebox {{ border:1px solid #e5e7eb; border-radius:14px; overflow:hidden; }}
      table {{ width:100%; border-collapse: collapse; }}
      thead th {{ background:#0b1220; color:#fff; text-align:left; padding:12px 12px; font-size:14px; }}
      tbody td {{ padding:12px 12px; border-top:1px solid #eef2f7; font-size:14px; }}
      .grid {{ display:grid; grid-template-columns: 1fr; gap: 12px; }}
      @media (min-width: 860px) {{ .grid {{ grid-template-columns: 1fr 1fr 1fr; }} }}
      a {{ color:#2563eb; text-decoration:none; }}
      a:hover {{ text-decoration:underline; }}
      ul {{ margin: 8px 0 0 18px; }}
    </style>
    """
    return html.strip()


def main():
    cfg = load_config()

    test_mode = os.getenv("TEST_MODE", "0").strip() == "1"
    today = datetime.now(KST).strftime("%Y-%m-%d")
    run_tag = f"TEST {int(time.time())}" if test_mode else "DAILY"

    # 슬러그: 기본은 날짜 1개만(중복발행 방지)
    slug = f"daily-report-{today}" if not test_mode else f"daily-report-{today}-{int(time.time())}"
    title = f"오늘의 지표 리포트 ({today})" + (f" - {int(time.time())}" if test_mode else "")

    # 중복 발행 방지(DAILY 모드)
    if not test_mode:
        existed = wp_find_post_id_by_slug(cfg, slug)
        if existed:
            print(f"이미 발행됨(중복 방지): slug={slug}, post_id={existed}")
            return

    quotes, errors = fetch_all_quotes()
    content_html = build_post_html(quotes, errors, run_tag=run_tag)

    link = wp_create_post(cfg, title=title, slug=slug, content_html=content_html)
    print("✅ 발행 완료:", link)


if __name__ == "__main__":
    main()
