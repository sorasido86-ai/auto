# -*- coding: utf-8 -*-
"""
Daily indicator -> WordPress auto post
- Free sources: Stooq (prices) + Google News RSS (headlines)
- Robust:
  * Uses LAST available close dates (handles weekends/holidays)
  * Shows each indicator's own "기준일"(data date), not "today"
  * If fetch fails, shows reason instead of '-' silently
- Local run:
  * bot_config.json is searched relative to this script file
"""

import os
import json
import csv
import base64
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import quote

import requests
import feedparser

# -----------------------------
# Paths / timezone
# -----------------------------
KST = timezone(timedelta(hours=9))
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bot_config.json"

WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# -----------------------------
# Config
# -----------------------------
def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"bot_config.json not found at: {CONFIG_PATH}\n"
            f"→ daily_post.py 와 같은 폴더에 bot_config.json 을 두세요."
        )
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def now_date_str():
    return datetime.now(KST).strftime("%Y-%m-%d")

def now_run_id():
    # 테스트 시 중복 발행 방지용
    return str(int(datetime.now(KST).timestamp())) + str(random.randint(100, 999))

# -----------------------------
# WordPress
# -----------------------------
def wp_posts_api(cfg):
    return cfg["wp_base_url"].rstrip("/") + WP_POSTS_API_SUFFIX

def wp_auth_headers(cfg):
    user = cfg["wp_user"].strip()
    app_pass = cfg["wp_app_pass"].replace(" ", "").strip()
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}

def wp_create_post(cfg, title, slug, html_content, status="publish"):
    payload = {"title": title, "slug": slug, "content": html_content, "status": status}
    r = requests.post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def wp_post_exists(cfg, slug):
    r = requests.get(
        wp_posts_api(cfg),
        params={"slug": slug},
        headers=wp_auth_headers(cfg),
        timeout=30,
    )
    r.raise_for_status()
    return len(r.json()) > 0

# -----------------------------
# Data model
# -----------------------------
@dataclass
class IndicatorRow:
    name: str
    latest: float | None
    prev: float | None
    unit: str
    date_latest: str | None
    fail_reason: str | None = None

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

# -----------------------------
# Stooq CSV fetch (with fallback domain + HTML-block detection)
# -----------------------------
def _fetch_stooq_csv(symbol: str) -> str:
    sym = symbol.strip()
    sym_q = quote(sym, safe="")
    urls = [
        f"https://stooq.com/q/d/l/?s={sym_q}&i=d",
        f"https://stooq.pl/q/d/l/?s={sym_q}&i=d",
    ]
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            text = r.text.strip()
            # Stooq가 차단/오류면 HTML이 오거나 CSV 헤더가 없음
            if not text.lower().startswith("date,open,high,low,close,volume"):
                raise ValueError("CSV가 아닌 응답(HTML/빈 값) 수신")
            return text
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Stooq CSV 수집 실패: {symbol} ({type(last_err).__name__}: {last_err})")

def fetch_stooq_last_two_closes(symbol: str) -> tuple[float, float, str]:
    """
    Returns: (latest_close, prev_close, latest_date_str)
    - Uses last two valid close rows (handles weekends/holidays)
    """
    text = _fetch_stooq_csv(symbol)
    rows = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        c = (row.get("Close") or "").strip()
        d = (row.get("Date") or "").strip()
        if not d or not c:
            continue
        try:
            close_val = float(c)
        except:
            continue
        rows.append((d, close_val))

    if len(rows) < 2:
        raise ValueError(f"Stooq 데이터가 부족합니다: {symbol}")

    # last two
    d2, v2 = rows[-1]
    d1, v1 = rows[-2]
    return v2, v1, d2

# -----------------------------
# Google News RSS (Korean filtering)
# -----------------------------
def is_korean_title(s: str) -> bool:
    if not s:
        return False
    hangul = sum(1 for ch in s if "가" <= ch <= "힣")
    # 한글 비중이 너무 낮으면 제외
    return hangul >= 4

def google_news_rss(query: str, max_items=5):
    # 한국/한국어 뉴스로 고정
    q = quote(query, safe="")
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    d = feedparser.parse(url)
    items = []
    for e in d.entries[: max_items * 2]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        if not title or not link:
            continue
        items.append((title, link))
    # 한글 우선 필터
    ko = [it for it in items if is_korean_title(it[0])]
    return (ko if len(ko) >= 2 else items)[:max_items]

# -----------------------------
# Reason helper (lightweight, non-claimy)
# -----------------------------
def reason_hints(ind_name: str, pct: float | None, headlines: list[str]) -> list[str]:
    """
    너무 단정하지 않게 '자주 언급되는 요인' 형태로 힌트 생성
    """
    hints = []
    if pct is None:
        return ["관련 헤드라인을 참고해 원인을 확인하세요."]

    up = pct > 0
    if ind_name == "USD/KRW":
        hints.append("환율은 보통 **달러 강세/약세(미 금리·연준), 위험선호/회피, 외국인 수급** 영향을 많이 받아요.")
    elif ind_name == "Brent Oil":
        hints.append("유가는 보통 **수요 전망(경기), OPEC+ 공급, 원유재고/지정학 리스크** 이슈에 민감해요.")
    elif ind_name == "KOSPI":
        hints.append("코스피는 보통 **반도체 등 대형주 실적 기대, 외국인 수급, 환율/금리** 영향이 커요.")

    # 헤드라인 키워드 기반 보조(아주 약하게)
    joined = " ".join(headlines)
    keyword_map = [
        ("금리", "금리 이슈"),
        ("연준", "연준/통화정책"),
        ("CPI", "물가(CPI)"),
        ("고용", "고용지표"),
        ("OPEC", "OPEC+"),
        ("재고", "원유재고"),
        ("중동", "지정학(중동)"),
        ("중국", "중국발 수요/정책"),
        ("반도체", "반도체 업종"),
        ("외국인", "외국인 수급"),
    ]
    picked = [tag for k, tag in keyword_map if k in joined]
    if picked:
        hints.append("헤드라인에서 많이 보이는 키워드: " + ", ".join(sorted(set(picked))))

    if up:
        hints.append("오늘 기준으로는 **상승 방향** 변동이었어요(자세한 맥락은 뉴스 링크 참고).")
    else:
        hints.append("오늘 기준으로는 **하락 방향** 변동이었어요(자세한 맥락은 뉴스 링크 참고).")
    return hints[:3]

# -----------------------------
# Report build
# -----------------------------
def fmt_num(x: float | None, digits=2):
    if x is None:
        return "수집 실패"
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
    return f"{sign}{x:,.{digits}f}%"

def build_post_html(today: str, test_label: str, rows: list[IndicatorRow], news_map: dict):
    # 실패 요약
    fails = [r for r in rows if r.latest is None or r.prev is None]
    fail_html = ""
    if fails:
        lis = "".join(
            f"<li><b>{r.name}</b>: {r.fail_reason or '값을 가져오지 못했습니다.'}</li>"
            for r in fails
        )
        fail_html = f"""
        <div style="border:1px solid #f2c2c2;background:#fff5f5;padding:14px;border-radius:10px;margin:16px 0;">
          <b>⚠️ 일부 데이터 수집 실패</b>
          <ul style="margin:10px 0 0 18px;">{lis}</ul>
        </div>
        """

    # 표
    tr = []
    for r in rows:
        tr.append(
            f"""
            <tr>
              <td style="padding:12px;border-top:1px solid #e5e7eb;">{r.name}</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">
                {fmt_num(r.latest, 2)}{(' ' + r.unit) if r.latest is not None else ''}
              </td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">
                {fmt_diff(r.diff, 2)}
              </td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">
                {fmt_pct(r.pct, 2)}
              </td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:center;">
                {r.date_latest or '-'}
              </td>
            </tr>
            """
        )
    table_html = f"""
    <div style="border:1px solid #e5e7eb;border-radius:14px;overflow:hidden;margin:12px 0 22px;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr style="background:#0f172a;color:#fff;">
            <th style="padding:12px;text-align:left;">지표</th>
            <th style="padding:12px;text-align:right;">현재</th>
            <th style="padding:12px;text-align:right;">전일대비</th>
            <th style="padding:12px;text-align:right;">변동률</th>
            <th style="padding:12px;text-align:center;">기준일(데이터)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(tr)}
        </tbody>
      </table>
    </div>
    """

    # 원인(뉴스+힌트)
    reason_blocks = []
    for r in rows:
        headlines = [t for (t, _) in news_map.get(r.name, [])][:3]
        hints = reason_hints(r.name, r.pct, headlines)
        links = ""
        if news_map.get(r.name):
            links = "<ul style='margin:10px 0 0 18px;'>" + "".join(
                [f"<li><a href='{link}' target='_blank' rel='noopener'>{title}</a></li>"
                 for title, link in news_map[r.name][:3]]
            ) + "</ul>"
        else:
            links = "<div style='color:#6b7280;margin-top:10px;'>관련 뉴스 결과가 없어요.</div>"

        hint_html = "<ul style='margin:10px 0 0 18px;'>" + "".join(f"<li>{h}</li>" for h in hints) + "</ul>"

        reason_blocks.append(f"""
        <div style="border:1px solid #e5e7eb;border-radius:14px;padding:14px;margin:10px 0;">
          <div style="font-weight:800;font-size:18px;margin-bottom:6px;">{r.name} 변동 원인(참고)</div>
          {hint_html}
          <div style="margin-top:10px;font-weight:700;">관련 뉴스</div>
          {links}
        </div>
        """)

    # 최종
    test_badge = f"<span style='display:inline-block;margin-left:8px;padding:3px 8px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px;font-weight:700;'>{test_label}</span>" if test_label else ""
    html = f"""
    <div style="max-width:860px;margin:0 auto;line-height:1.65;">
      <h2 style="margin:0 0 10px;">오늘의 지표 리포트 ({today}){test_badge}</h2>

      {fail_html}

      <h3 style="margin:16px 0 8px;">핵심 요약</h3>
      <ul style="margin:8px 0 0 18px;">
        <li>USD/KRW, 브렌트유, 코스피의 <b>최근 거래일 기준</b> 변동을 한 번에 정리했습니다.</li>
        <li>변동 원인 참고용으로 관련 뉴스 헤드라인을 함께 첨부했습니다.</li>
      </ul>

      <h3 style="margin:22px 0 8px;">주요 지표</h3>
      {table_html}

      <h3 style="margin:22px 0 8px;">왜 움직였나? (원인 참고)</h3>
      {''.join(reason_blocks)}

      <h3 style="margin:22px 0 8px;">내일 체크포인트</h3>
      <ul style="margin:8px 0 0 18px;">
        <li>미국 CPI/고용/연준 발언 등 주요 이벤트 일정</li>
        <li>유가: OPEC+/재고/지정학 헤드라인</li>
        <li>환율: 달러 강세(미 금리)·위험선호 흐름</li>
      </ul>

      <div style="color:#6b7280;font-size:12px;margin-top:18px;">
        데이터: Stooq(가격/지수), Google News RSS(헤드라인)
      </div>
    </div>
    """
    return html

# -----------------------------
# Fetch indicators (Stooq)
# -----------------------------
def fetch_indicators():
    out = []

    # Stooq symbols:
    # - USD/KRW: usdkrw  
    # - Brent (CO1 futures on Stooq): cb.f :contentReference[oaicite:1]{index=1}
    # - KOSPI index: ^KOSPI :contentReference[oaicite:2]{index=2}
    specs = [
        ("USD/KRW", "usdkrw", ""),
        ("Brent Oil", "cb.f", "USD"),
        ("KOSPI", "^kospi", ""),
    ]

    for name, sym, unit in specs:
        try:
            latest, prev, d_latest = fetch_stooq_last_two_closes(sym)
            out.append(IndicatorRow(name=name, latest=latest, prev=prev, unit=unit, date_latest=d_latest))
        except Exception as e:
            out.append(IndicatorRow(name=name, latest=None, prev=None, unit=unit, date_latest=None,
                                   fail_reason=f"{type(e).__name__}: {e}"))
    return out

def fetch_news(rows: list[IndicatorRow]):
    news_map = {}
    queries = {
        "USD/KRW": "원달러 환율 변동 원인",
        "Brent Oil": "브렌트유 하락 원인",
        "KOSPI": "코스피 변동 원인",
    }
    for r in rows:
        q = queries.get(r.name, r.name)
        items = google_news_rss(q, max_items=5)
        news_map[r.name] = items
    return news_map

# -----------------------------
# Main
# -----------------------------
def main():
    cfg = load_config()

    today = now_date_str()

    # 환경변수로 동작 제어(Workflow에서 설정 가능)
    test_mode = os.getenv("TEST_MODE", "0").strip() == "1"
    wp_status = os.getenv("WP_STATUS", "").strip() or ("draft" if test_mode else "publish")

    run_id = now_run_id() if test_mode else ""
    test_label = f"TEST {run_id}" if test_mode else ""

    title = f"오늘의 지표 리포트 ({today})" + (f" - {run_id}" if test_mode else "")
    slug = f"daily-indicator-report-{today}" + (f"-{run_id}" if test_mode else "")

    # 일반 모드에서는 중복 방지
    if not test_mode and wp_post_exists(cfg, slug):
        print(f"[SKIP] already exists: {slug}")
        return

    rows = fetch_indicators()
    news_map = fetch_news(rows)
    html = build_post_html(today, test_label, rows, news_map)

    post = wp_create_post(cfg, title, slug, html, status=wp_status)
    link = post.get("link", cfg["wp_base_url"])

    print(f"[OK] posted: {title} ({wp_status})")
    print(f"[LINK] {link}")

if __name__ == "__main__":
    main()
