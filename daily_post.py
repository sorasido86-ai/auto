# -*- coding: utf-8 -*-
"""
daily_post.py (통합 안정화 - Brent 실패 시 WP에서 최근값 재사용)

✅ 기능
- WordPress에 '오늘의 지표 리포트' 자동 발행
- 카테고리: 팁(카테고리 ID=8)
- 데이터:
  - USD/KRW : 네이버
  - KOSPI   : 네이버
  - Brent   : 네이버 -> FRED -> Stooq 순 fallback
            -> 그래도 실패하면 "최근 발행된 내 WP 글"에서 Brent 값 2개를 찾아 전일대비까지 계산해 채움

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
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple

import requests
import feedparser
from bs4 import BeautifulSoup


# =========================
# 설정
# =========================
DEFAULT_CATEGORY_ID = 8          # ✅ 팁 카테고리 ID
POST_STATUS = "publish"
REQUEST_TIMEOUT = 25

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

COMMON_HEADERS = {
    "User-Agent": UA,
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

NAVER_HEADERS = {**COMMON_HEADERS, "Referer": "https://finance.naver.com/"}

# Naver
NAVER_USDKRW_URL = "https://finance.naver.com/marketindex/exchangeDailyQuote.naver?marketindexCd=FX_USDKRW&page=1"
NAVER_KOSPI_URL  = "https://finance.naver.com/sise/sise_index_day.naver?code=KOSPI&page=1"
NAVER_BRENT_DAILY_URL  = "https://finance.naver.com/marketindex/worldOilDailyQuote.naver?marketindexCd=OIL_BRT&page=1"
NAVER_BRENT_DETAIL_URL = "https://finance.naver.com/marketindex/worldOilDetail.naver?marketindexCd=OIL_BRT"

# FRED (키 없이)
FRED_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/csv,*/*;q=0.8",
    "Referer": "https://fred.stlouisfed.org/",
}
FRED_BRENT_SERIES = "DCOILBRENTEU"
FRED_GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# Stooq
STOOQ_RETRY = 3
STOOQ_SLEEP = 1.3
STOOQ_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/csv,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7",
}
STOOQ_DOMAINS = ["https://stooq.com", "https://stooq.pl"]

# Google News RSS
GOOGLE_NEWS_BASE = "https://news.google.com/rss/search"


# =========================
# 공통 유틸
# =========================
def script_dir() -> Path:
    return Path(__file__).resolve().parent

def config_path() -> Path:
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
            f"- 해결: daily_post.py와 같은 폴더에 bot_config.json을 두거나,\n"
            f"  환경변수 BOT_CONFIG_PATH로 경로를 지정하세요."
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

def parse_dot_date(s: str) -> Optional[str]:
    m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", s)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

def iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        y, m, d = s.split("-")
        return date(int(y), int(m), int(d))
    except:
        return None

def http_get(url: str, headers: Optional[Dict[str, str]] = None, tries: int = 3) -> str:
    h = COMMON_HEADERS.copy()
    if headers:
        h.update(headers)

    last_err = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=h, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            # 레이트/차단 흔한 코드들
            if r.status_code in (403, 429, 503):
                raise ValueError(f"HTTP {r.status_code} (차단/레이트리밋 가능)")
            r.raise_for_status()
            txt = (r.text or "").strip()
            if not txt:
                raise ValueError("빈 응답")
            return txt
        except Exception as e:
            last_err = e
            time.sleep(0.8 + i * 1.2)
    raise last_err


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

def wp_create_post(cfg: Dict[str, Any], title: str, slug: str, content_html: str, status: str = POST_STATUS) -> Dict[str, Any]:
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
        "categories": [cat_id],  # ✅ 팁 카테고리
    }
    r = requests.post(
        wp_api_posts(cfg),
        headers=wp_headers(cfg),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()

def wp_fetch_recent_posts_public(cfg: Dict[str, Any], per_page: int = 12) -> List[Dict[str, Any]]:
    # 공개 글은 인증 없이도 조회되는 경우가 대부분이지만,
    # 혹시 모를 경우를 대비해 실패하면 인증 헤더로 1번 더 시도
    url = wp_api_posts(cfg)
    params = {
        "search": "오늘의 지표 리포트",
        "per_page": per_page,
        "orderby": "date",
        "order": "desc",
        "status": "publish",
    }
    try:
        r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except:
        r = requests.get(url, params=params, headers=wp_headers(cfg), timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()


# =========================
# Naver 파서(USD/KOSPI/Brent 공용)
# =========================
def naver_extract_last_two_from_table(url: str) -> Tuple[float, float, str]:
    html = http_get(url, headers=NAVER_HEADERS, tries=3)
    soup = BeautifulSoup(html, "lxml")

    rows: List[Tuple[str, float]] = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        d_txt = re.sub(r"\s+", " ", tds[0].get_text(strip=True))
        iso = parse_dot_date(d_txt)
        if not iso:
            continue

        v_txt = re.sub(r"\s+", " ", tds[1].get_text(strip=True))
        v = safe_float(v_txt)
        if v is None:
            continue
        rows.append((iso, v))

    # 중복 제거
    dedup: List[Tuple[str, float]] = []
    seen = set()
    for d, v in rows:
        if d in seen:
            continue
        seen.add(d)
        dedup.append((d, v))

    if len(dedup) < 2:
        raise ValueError("네이버 테이블 데이터가 부족합니다(2개 미만/휴장/차단 가능)")

    latest_d, latest_v = dedup[0]
    prev_d, prev_v = dedup[1]
    return latest_v, prev_v, latest_d

def fetch_usdkrw_naver() -> Tuple[float, float, str]:
    return naver_extract_last_two_from_table(NAVER_USDKRW_URL)

def fetch_kospi_naver() -> Tuple[float, float, str]:
    return naver_extract_last_two_from_table(NAVER_KOSPI_URL)

def fetch_brent_naver() -> Tuple[float, float, str]:
    # dailyQuote가 막히면 detail로 1번 더
    try:
        return naver_extract_last_two_from_table(NAVER_BRENT_DAILY_URL)
    except:
        return naver_extract_last_two_from_table(NAVER_BRENT_DETAIL_URL)


# =========================
# FRED (키 없이 CSV)
# =========================
def fetch_fred_last_two(series: str) -> Tuple[float, float, str]:
    url = FRED_GRAPH_CSV.format(series=series)
    txt = http_get(url, headers=FRED_HEADERS, tries=3)
    if not txt.lower().startswith("date,"):
        # HTML/차단일 때 이런 식으로 옴
        raise ValueError("FRED CSV 응답이 CSV가 아닙니다(차단/HTML 가능)")

    lines = txt.splitlines()
    data: List[Tuple[str, float]] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        d = parts[0].strip()
        v = parts[1].strip()
        if not d or not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            continue
        if v in (".", "", "nan", "NaN"):
            continue
        fv = safe_float(v)
        if fv is None:
            continue
        data.append((d, fv))

    if len(data) < 2:
        raise ValueError("FRED 데이터가 부족합니다(2개 미만)")

    latest_d, latest_v = data[-1]
    prev_d, prev_v = data[-2]
    return latest_v, prev_v, latest_d


# =========================
# Stooq (보조)
# =========================
def fetch_stooq_csv(symbol: str) -> str:
    last_err = None
    for base in STOOQ_DOMAINS:
        for i in range(STOOQ_RETRY):
            try:
                url = f"{base}/q/d/l/"
                params = {"s": symbol, "i": "d"}
                r = requests.get(url, params=params, headers=STOOQ_HEADERS, timeout=REQUEST_TIMEOUT)
                txt = (r.text or "").strip()
                if not txt:
                    raise ValueError("빈 응답")
                if not txt.lower().startswith("date,"):
                    raise ValueError("CSV가 아닌 응답(HTML/빈 값) 수신")
                return txt
            except Exception as e:
                last_err = e
                time.sleep(STOOQ_SLEEP * (i + 1))
                continue
    raise ValueError(f"Stooq CSV 수집 실패: {symbol} ({type(last_err).__name__}: {last_err})")

def parse_stooq_last_two_closes(csv_text: str) -> Tuple[float, float, str]:
    f = StringIO(csv_text)
    reader = csv.DictReader(f)
    rows = []
    for row in reader:
        d = (row.get("Date") or "").strip()
        c = safe_float(row.get("Close"))
        if d and c is not None:
            rows.append((d, c))
    if len(rows) < 2:
        raise ValueError("Stooq 데이터가 부족합니다(2개 미만)")
    latest_d, latest_c = rows[-1]
    prev_d, prev_c = rows[-2]
    return latest_c, prev_c, latest_d


# =========================
# Brent Best Effort (네이버 -> FRED -> Stooq) + 실패 시 WP 재사용
# =========================
def fetch_brent_best_effort_raw() -> Tuple[Optional[float], Optional[float], Optional[str], str]:
    # 성공한 후보 중 가장 최신 날짜 선택
    candidates: List[Tuple[Optional[float], Optional[float], Optional[str], str]] = []

    # 1) Naver
    try:
        a, b, d = fetch_brent_naver()
        candidates.append((a, b, d, "NAVER"))
    except Exception as e:
        candidates.append((None, None, None, f"NAVER_FAIL({type(e).__name__})"))

    # 2) FRED
    try:
        a, b, d = fetch_fred_last_two(FRED_BRENT_SERIES)
        candidates.append((a, b, d, f"FRED:{FRED_BRENT_SERIES}"))
    except Exception as e:
        candidates.append((None, None, None, f"FRED_FAIL({type(e).__name__})"))

    # 3) Stooq (심볼 여러개 시도)
    for sym in ["brn.f", "brent.f", "brent.o", "co.f", "cb.f"]:
        try:
            txt = fetch_stooq_csv(sym)
            a, b, d = parse_stooq_last_two_closes(txt)
            candidates.append((a, b, d, f"STOOQ:{sym}"))
            break
        except:
            continue

    best = None
    best_date = None
    for a, b, d, src in candidates:
        if a is None or b is None or not d:
            continue
        dd = iso_to_date(d)
        if dd is None:
            continue
        if best is None or best_date is None or dd > best_date:
            best = (a, b, d, src)
            best_date = dd

    if best is None:
        return None, None, None, "FAIL(네이버+FRED+Stooq 모두 실패)"
    return best


def extract_value_from_report_html(html: str, label_contains: str) -> Optional[Tuple[float, str]]:
    """
    발행된 글 HTML에서 표의 특정 행(label)에 있는 (현재값, 기준일) 추출
    """
    soup = BeautifulSoup(html or "", "lxml")
    # 모든 tr 중에서 첫 td가 label 포함하는 행 찾기
    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if len(tds) < 5:
            continue
        label = tds[0].get_text(" ", strip=True)
        if label_contains.lower() in label.lower():
            val_txt = tds[1].get_text(" ", strip=True)
            date_txt = tds[4].get_text(" ", strip=True)
            v = safe_float(val_txt)
            if v is None:
                return None
            # 날짜는 YYYY-MM-DD만 뽑기
            m = re.search(r"(\d{4}-\d{2}-\d{2})", date_txt)
            d = m.group(1) if m else date_txt
            return (v, d)
    return None


def fetch_brent_from_wp_cache(cfg: Dict[str, Any]) -> Optional[Tuple[float, float, str, str]]:
    """
    브렌트가 외부에서 다 실패하면, 최근 발행된 내 글에서 Brent 2개 값을 찾아 (latest, prev, date, source) 만들기
    """
    posts = wp_fetch_recent_posts_public(cfg, per_page=15)
    found: List[Tuple[float, str]] = []
    for p in posts:
        content = ((p.get("content") or {}).get("rendered") or "")
        got = extract_value_from_report_html(content, "Brent")
        if got:
            found.append(got)
        if len(found) >= 2:
            break

    if len(found) >= 2:
        latest_v, latest_d = found[0]
        prev_v, prev_d = found[1]
        # 기준일은 최신값의 날짜 사용
        return latest_v, prev_v, latest_d, "WP_CACHE"
    return None


def fetch_brent_best_effort(cfg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str], str]:
    # 1) 외부 소스 시도
    latest, prev, d, src = fetch_brent_best_effort_raw()
    if latest is not None and prev is not None and d:
        return latest, prev, d, src

    # 2) 그래도 실패면 WP 캐시 재사용
    cached = fetch_brent_from_wp_cache(cfg)
    if cached:
        a, b, d, src2 = cached
        return a, b, d, src2

    return None, None, None, src


# =========================
# Google News RSS
# =========================
def google_news_rss(query: str, max_items: int = 3) -> List[Dict[str, str]]:
    params = {"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    r = requests.get(GOOGLE_NEWS_BASE, params=params, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    items = []
    for e in (feed.entries or [])[:max_items]:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if title and link:
            if len(title) > 90:
                title = title[:90] + "…"
            items.append({"title": title, "link": link})
    return items

def build_reason_section() -> Dict[str, List[Dict[str, str]]]:
    return {
        "usdkrw": google_news_rss("원달러 환율 변동 원인", 3),
        "brent": google_news_rss("브렌트유 유가 하락 원인", 3),
        "kospi": google_news_rss("코스피 상승 하락 원인", 3),
    }


# =========================
# 리포트 HTML
# =========================
def calc_change(latest: Optional[float], prev: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if latest is None or prev is None:
        return None, None
    chg = latest - prev
    pct = (chg / prev * 100.0) if prev != 0 else None
    return chg, pct

def heuristics_comment(name: str, chg: Optional[float]) -> str:
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

def build_html_report(
    today: str,
    usd: Dict[str, Any],
    brent: Dict[str, Any],
    kospi: Dict[str, Any],
    errors: List[str],
    news: Dict[str, List[Dict[str, str]]],
) -> str:
    usd_chg, usd_pct = calc_change(usd["latest"], usd["prev"])
    br_chg, br_pct = calc_change(brent["latest"], brent["prev"])
    ko_chg, ko_pct = calc_change(kospi["latest"], kospi["prev"])

    run_id = os.getenv("TEST_RUN_ID", "").strip()
    badge = f"TEST {html_escape(run_id)}" if run_id else ""

    def row(label: str, latest: Optional[float], chg: Optional[float], pct: Optional[float], d: Optional[str]) -> str:
        return f"""
        <tr>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; font-weight:600;">{html_escape(label)}</td>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; text-align:right;">{fmt_num(latest)}</td>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; text-align:right;">{fmt_signed(chg)}</td>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; text-align:right;">{fmt_pct(pct)}</td>
          <td style="padding:14px 12px; border-top:1px solid #e6e6e6; text-align:center;">{html_escape(d or "-")}</td>
        </tr>
        """

    err_html = ""
    if errors:
        li = "".join([f"<li style='margin:6px 0'>{html_escape(e)}</li>" for e in errors])
        err_html = f"""
        <div style="border:1px solid #ffb3b3; background:#fff3f3; padding:14px 16px; border-radius:10px; margin:18px 0;">
          <div style="font-weight:700; margin-bottom:6px;">⚠️ 일부 데이터 수집 실패</div>
          <ul style="margin:8px 0 0 18px; padding:0;">{li}</ul>
          <div style="color:#666; font-size:13px; margin-top:10px;">
            ※ 잦은 실행/공용IP(깃허브 액션)/휴장 등으로 차단되면 값이 비어 있을 수 있어요.
            (Brent는 네이버→FRED→Stooq→WP최근글 재사용 순)
          </div>
        </div>
        """

    def news_list(items: List[Dict[str, str]]) -> str:
        if not items:
            return "<div style='color:#666'>- 관련 뉴스 결과 없음</div>"
        out = []
        for it in items:
            t = html_escape(it["title"])
            l = html_escape(it["link"])
            out.append(f"<li style='margin:7px 0;'><a href='{l}' target='_blank' rel='noopener'>{t}</a></li>")
        return "<ul style='margin:10px 0 0 18px; padding:0;'>" + "".join(out) + "</ul>"

    usd_reason = heuristics_comment("usdkrw", usd_chg)
    br_reason = heuristics_comment("brent", br_chg)
    ko_reason = heuristics_comment("kospi", ko_chg)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""
    <div style="max-width:860px; margin:0 auto; font-family:system-ui,-apple-system,'Apple SD Gothic Neo','Malgun Gothic',sans-serif; color:#111;">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <h1 style="margin:0; font-size:34px; line-height:1.15;">오늘의 지표 리포트 ({html_escape(today)})</h1>
        {f"<div style='padding:8px 12px; background:#eef2ff; color:#1e3a8a; border-radius:999px; font-weight:700; font-size:13px;'>{badge}</div>" if badge else ""}
      </div>

      {err_html}

      <h2 style="margin:18px 0 10px; font-size:22px;">핵심 요약</h2>
      <ul style="margin:8px 0 0 18px;">
        <li style="margin:6px 0;">원/달러(USD/KRW), 브렌트유, 코스피의 <b>최근 제공일 기준 변동</b>을 정리했습니다.</li>
        <li style="margin:6px 0;">변동 원인(참고용)으로 <b>관련 뉴스 헤드라인</b>을 첨부했습니다(단정 아님).</li>
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
        데이터 출처: Naver Finance(USD/KRW,KOSPI), Brent(Naver/FRED/Stooq/WP최근글재사용), Google News RSS
      </div>
    </div>
    """


# =========================
# 지표 수집(최종)
# =========================
def fetch_indicators_stable(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []

    # USD/KRW
    try:
        latest, prev, d = fetch_usdkrw_naver()
        usd = {"latest": latest, "prev": prev, "date": d, "source": "NAVER"}
    except Exception as e:
        usd = {"latest": None, "prev": None, "date": None, "source": "NAVER_FAIL"}
        errors.append(f"USD/KRW 수집 실패: {type(e).__name__}: {e}")

    # KOSPI
    try:
        latest, prev, d = fetch_kospi_naver()
        kospi = {"latest": latest, "prev": prev, "date": d, "source": "NAVER"}
    except Exception as e:
        kospi = {"latest": None, "prev": None, "date": None, "source": "NAVER_FAIL"}
        errors.append(f"KOSPI 수집 실패: {type(e).__name__}: {e}")

    # Brent (외부->WP재사용)
    try:
        latest, prev, d, src = fetch_brent_best_effort(cfg)
        brent = {"latest": latest, "prev": prev, "date": d, "source": src}
        if latest is None:
            errors.append(f"Brent 수집 실패: {src}")
        elif src == "WP_CACHE":
            # 재사용이면 알려주기
            errors.append(f"Brent 외부 수집 실패 → 최근 발행글 값 재사용(WP_CACHE, 기준일 {d})")
    except Exception as e:
        brent = {"latest": None, "prev": None, "date": None, "source": "FAIL"}
        errors.append(f"Brent 수집 실패: {type(e).__name__}: {e}")

    return {"usdkrw": usd, "kospi": kospi, "brent": brent}, errors


# =========================
# main
# =========================
def main():
    cfg = load_config()
    for k in ("wp_base_url", "wp_user", "wp_app_pass"):
        if not cfg.get(k):
            raise ValueError(f"필수 설정 누락: {k}")

    today = now_date_str()

    test_id = os.getenv("TEST_RUN_ID", "").strip()
    if test_id:
        slug = f"daily-indicator-report-{today}-{test_id}"
        title = f"오늘의 지표 리포트 ({today}) - {test_id}"
    else:
        slug = f"daily-indicator-report-{today}"
        title = f"오늘의 지표 리포트 ({today})"

    # 중복 발행 방지(테스트가 아닐 때)
    if not test_id and wp_post_exists(cfg, slug):
        print(f"✅ 이미 오늘 글이 있어요 ({today}) - 중복 발행 안 함")
        return

    # 잦은 실행으로 차단 줄이기: 요청 사이 짧은 텀
    time.sleep(1.2)

    ind, errors = fetch_indicators_stable(cfg)
    news = build_reason_section()

    html = build_html_report(
        today=today,
        usd=ind["usdkrw"],
        brent=ind["brent"],
        kospi=ind["kospi"],
        errors=errors,
        news=news,
    )

    post = wp_create_post(cfg, title, slug, html, status=POST_STATUS)
    link = post.get("link", cfg["wp_base_url"])
    print(f"✅ 글 발행 성공: {link}")


if __name__ == "__main__":
    main()
