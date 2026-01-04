# -*- coding: utf-8 -*-
"""
hot_keyword_naver_shop_to_wp.py (완전 통합본)

- Google 핫 키워드(Trends) 수집: RSS 여러 후보 + dailytrends JSON 폴백
- 각 키워드 → 네이버 쇼핑 검색 API(shop.json) → 상품 TOP N개 추천
- WordPress REST API로 발행(AM/PM 슬롯별 글을 따로 생성/업데이트)
- SQLite 저장(전일 traffic 비교용)

✅ GitHub Secrets(환경변수) 이름:
  - NAVER_CLIENT_ID
  - NAVER_CLIENT_SECRET
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

(호환용 fallback: XNAVERCLIENT_ID/XNAVERCLIENT_SECRET 도 읽긴 함)
"""

from __future__ import annotations

import base64
import html as htmlmod
import json
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

KST = timezone(timedelta(hours=9))
NAVER_SHOP_ENDPOINT = "https://openapi.naver.com/v1/search/shop.json"


# -----------------------------
# env helpers
# -----------------------------
def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _parse_int_list(raw: str, default: List[int]) -> List[int]:
    out: List[int] = []
    for x in re.split(r"[,\s]+", (raw or "").strip()):
        if x.isdigit():
            out.append(int(x))
    return out if out else default


def _parse_bool(v: str) -> bool:
    return (v or "").strip().lower() in ("1", "true", "yes", "y", "on")


# -----------------------------
# Config
# -----------------------------
@dataclass
class GoogleCfg:
    geo: str = "KR"
    limit: int = 20
    timeout: int = 25


@dataclass
class NaverCfg:
    client_id: str = ""
    client_secret: str = ""
    per_keyword: int = 5
    sort: str = "sim"
    exclude: str = "used:rental:cbshop"
    timeout: int = 25
    throttle: float = 0.25


@dataclass
class WPCfg:
    base_url: str = ""
    user: str = ""
    app_pass: str = ""
    status: str = "publish"
    category_ids: List[int] = None

    def __post_init__(self):
        if self.category_ids is None:
            self.category_ids = [31]


@dataclass
class AppCfg:
    google: GoogleCfg
    naver: NaverCfg
    wp: WPCfg
    sqlite_path: str
    slot: str
    dry_run: bool
    debug: bool


def load_cfg() -> AppCfg:
    # ✅ 너의 이름으로 우선 읽기 (fallback로 XNAVER도 읽어줌)
    naver_id = _env("NAVER_CLIENT_ID") or _env("XNAVERCLIENT_ID")
    naver_secret = _env("NAVER_CLIENT_SECRET") or _env("XNAVERCLIENT_SECRET")

    wp_base = (_env("WP_BASE_URL") or _env("WP_SITE_URL")).rstrip("/")
    wp_user = _env("WP_USER") or _env("WP_USERNAME")
    wp_pass = _env("WP_APP_PASS") or _env("WP_APP_PASSWORD")

    slot = (_env("RUN_SLOT", "am") or "am").lower().strip()
    if slot not in ("am", "pm"):
        slot = "am"

    dry_run = _parse_bool(_env("DRY_RUN", "0"))
    debug = _parse_bool(_env("DEBUG", "0"))

    geo = (_env("TRENDS_GEO", "KR") or "KR").upper()
    hot_limit = int(_env("HOT_KEYWORDS_LIMIT", "20") or "20")
    g_timeout = int(_env("GOOGLE_TIMEOUT", "25") or "25")

    per_kw = int(_env("NAVER_PER_KEYWORD", "5") or "5")
    sort = _env("NAVER_SORT", "sim") or "sim"
    exclude = _env("NAVER_EXCLUDE", "used:rental:cbshop")
    n_timeout = int(_env("NAVER_TIMEOUT", "25") or "25")
    throttle = float(_env("NAVER_THROTTLE_SEC", "0.25") or "0.25")

    wp_status = _env("WP_STATUS", "publish") or "publish"
    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "31"), [31])

    sqlite_path = _env("SQLITE_PATH", "data/hot_keyword_naver_shop.sqlite3")

    return AppCfg(
        google=GoogleCfg(geo=geo, limit=hot_limit, timeout=g_timeout),
        naver=NaverCfg(
            client_id=naver_id,
            client_secret=naver_secret,
            per_keyword=per_kw,
            sort=sort,
            exclude=exclude,
            timeout=n_timeout,
            throttle=throttle,
        ),
        wp=WPCfg(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=cat_ids,
        ),
        sqlite_path=sqlite_path,
        slot=slot,
        dry_run=dry_run,
        debug=debug,
    )


def validate_cfg(cfg: AppCfg) -> None:
    missing = []
    if not cfg.naver.client_id:
        missing.append("NAVER_CLIENT_ID")
    if not cfg.naver.client_secret:
        missing.append("NAVER_CLIENT_SECRET")
    if not cfg.wp.base_url:
        missing.append("WP_BASE_URL")
    if not cfg.wp.user:
        missing.append("WP_USER")
    if not cfg.wp.app_pass:
        missing.append("WP_APP_PASS")
    if missing:
        raise RuntimeError("필수 설정 누락:\n- " + "\n- ".join(missing))


def print_safe_cfg(cfg: AppCfg) -> None:
    def ok(v: str) -> str:
        return f"OK(len={len(v)})" if v else "MISSING"

    print("[CFG] slot:", cfg.slot, "dry_run:", cfg.dry_run, "debug:", cfg.debug)
    print("[CFG] NAVER_CLIENT_ID:", ok(cfg.naver.client_id))
    print("[CFG] NAVER_CLIENT_SECRET:", ok(cfg.naver.client_secret))
    print("[CFG] naver per_keyword/sort:", cfg.naver.per_keyword, cfg.naver.sort)
    print("[CFG] WP_BASE_URL:", cfg.wp.base_url)
    print("[CFG] WP_USER:", ok(cfg.wp.user))
    print("[CFG] WP_APP_PASS:", ok(cfg.wp.app_pass))
    print("[CFG] WP category_ids:", cfg.wp.category_ids)
    print("[CFG] sqlite:", cfg.sqlite_path)


# -----------------------------
# SQLite
# -----------------------------
def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS posts(
          date TEXT,
          slot TEXT,
          wp_post_id INTEGER,
          wp_link TEXT,
          created_at TEXT,
          PRIMARY KEY(date, slot)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS keywords(
          date TEXT,
          slot TEXT,
          rank INTEGER,
          keyword TEXT,
          traffic_str TEXT,
          traffic_int INTEGER,
          link TEXT,
          PRIMARY KEY(date, slot, rank)
        )
        """
    )
    con.commit()
    con.close()


def save_post_meta(db_path: str, date_str: str, slot: str, post_id: int, link: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO posts(date, slot, wp_post_id, wp_link, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (date_str, slot, post_id, link, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


def get_existing_post(db_path: str, date_str: str, slot: str) -> Optional[Tuple[int, str]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT wp_post_id, wp_link FROM posts WHERE date=? AND slot=?", (date_str, slot))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return int(row[0]), str(row[1] or "")


def get_prev_traffic(db_path: str, prev_date: str, keyword: str) -> Optional[int]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT traffic_int FROM keywords WHERE date=? AND keyword=? ORDER BY traffic_int DESC LIMIT 1",
        (prev_date, keyword),
    )
    row = cur.fetchone()
    con.close()
    return int(row[0]) if row and row[0] is not None else None


def upsert_keywords(db_path: str, date_str: str, slot: str, items: List[Dict[str, Any]]) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    for i, it in enumerate(items, start=1):
        cur.execute(
            """
            INSERT OR REPLACE INTO keywords(date, slot, rank, keyword, traffic_str, traffic_int, link)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date_str,
                slot,
                i,
                str(it.get("keyword", "")),
                str(it.get("traffic_str", "")),
                int(it.get("traffic_int", 0) or 0),
                str(it.get("link", "")),
            ),
        )
    con.commit()
    con.close()


# -----------------------------
# Google Trends fetch (RSS + fallback JSON)
# -----------------------------
def _parse_traffic_to_int(s: str) -> int:
    if not s:
        return 0
    t = s.strip().upper().replace("+", "").replace(",", "")
    m = re.match(r"^(\d+(?:\.\d+)?)([KM])?$", t)
    if not m:
        t = re.sub(r"\D", "", t)
        return int(t or 0)
    num = float(m.group(1))
    unit = m.group(2)
    if unit == "K":
        num *= 1_000
    elif unit == "M":
        num *= 1_000_000
    return int(num)


def _parse_rss(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    ch = root.find("channel")
    if ch is None:
        return []
    out: List[Dict[str, Any]] = []
    for item in ch.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        traffic_str = ""
        for child in list(item):
            if child.tag.endswith("approx_traffic"):
                traffic_str = (child.text or "").strip()
                break
        keyword = title.split(" - ", 1)[0].strip() if title else ""
        if keyword:
            out.append(
                {
                    "keyword": keyword,
                    "traffic_str": traffic_str,
                    "traffic_int": _parse_traffic_to_int(traffic_str),
                    "link": link,
                }
            )
    return out


def _fetch_dailytrends_json(geo: str, timeout: int) -> List[Dict[str, Any]]:
    geo = (geo or "KR").upper()
    url = f"https://trends.google.com/trends/api/dailytrends?hl=ko&tz=-540&geo={geo}&ns=15"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    if r.status_code != 200:
        return []
    text = re.sub(r"^\)\]\}',?\s*", "", (r.text or "").strip())
    try:
        data = json.loads(text)
    except Exception:
        return []
    days = (((data or {}).get("default") or {}).get("trendingSearchesDays") or [])
    if not days:
        return []
    day0 = days[0] or {}
    searches = day0.get("trendingSearches") or []
    out: List[Dict[str, Any]] = []
    for s in searches:
        q = (((s.get("title") or {}).get("query")) or "").strip()
        traffic_str = (s.get("formattedTraffic") or "").strip()
        traffic_int = _parse_traffic_to_int(traffic_str)
        link = ""
        arts = s.get("articles") or []
        if isinstance(arts, list) and arts:
            link = str((arts[0] or {}).get("url") or "").strip()
        if q:
            out.append({"keyword": q, "traffic_str": traffic_str, "traffic_int": traffic_int, "link": link})
    return out


def fetch_google_hot_keywords(geo: str, limit: int, timeout: int, debug: bool) -> List[Dict[str, Any]]:
    geo = (geo or "KR").upper()
    candidates = [
        f"https://trends.google.com/trending/rss?geo={geo}",
        f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}",
        f"https://trends.google.co.kr/trending/rss?geo={geo}",
        f"https://trends.google.co.kr/trends/trendingsearches/daily/rss?geo={geo}",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; hotkw/1.0)"}

    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if debug:
                print("[DEBUG] RSS try:", url, "status:", r.status_code)
            if r.status_code != 200:
                continue
            txt = (r.text or "").strip()
            if "<rss" not in txt.lower():
                continue
            items = _parse_rss(txt)
            # dedupe
            seen = set()
            out = []
            for it in items:
                k = it["keyword"]
                if k in seen:
                    continue
                seen.add(k)
                out.append(it)
                if len(out) >= limit:
                    break
            if out:
                return out
        except Exception as e:
            if debug:
                print("[DEBUG] RSS error:", repr(e))
            continue

    # 마지막 폴백: dailytrends json
    items = _fetch_dailytrends_json(geo, timeout)
    if items:
        return items[:limit]
    raise RuntimeError("Google Trends fetch failed (RSS + dailytrends JSON 모두 실패)")


# -----------------------------
# Naver Shopping Search
# -----------------------------
def _clean_title(t: str) -> str:
    t = htmlmod.unescape(t or "")
    t = re.sub(r"</?b>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def naver_shop_search(session: requests.Session, cfg: NaverCfg, query: str) -> List[Dict[str, Any]]:
    headers = {
        "X-Naver-Client-Id": cfg.client_id,
        "X-Naver-Client-Secret": cfg.client_secret,
        "User-Agent": "Mozilla/5.0 (compatible; hotkw/1.0)",
    }
    params: Dict[str, Any] = {
        "query": query,
        "display": cfg.per_keyword,
        "start": 1,
        "sort": cfg.sort,
        "exclude": cfg.exclude,
    }

    r = session.get(NAVER_SHOP_ENDPOINT, headers=headers, params=params, timeout=cfg.timeout)
    if r.status_code != 200:
        head = (r.text or "")[:300]
        if '"errorCode":"024"' in head or "Scope Status Invalid" in head:
            raise RuntimeError(
                "Naver Shop API 401/024(인증 실패)입니다.\n"
                "✅ 네이버 개발자센터에서 해당 앱에 '검색(쇼핑)' API 사용 설정이 켜져있는지 확인하세요.\n"
                f"- 응답: {r.status_code} body={head}"
            )
        raise RuntimeError(f"Naver Shop API failed: {r.status_code} body={head}")

    data = r.json()
    items = data.get("items", [])
    if not isinstance(items, list):
        return []
    for it in items:
        it["title"] = _clean_title(str(it.get("title", "")))
    return items


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "hotkw/1.0"}


def wp_create_post(cfg: WPCfg, title: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "content": html,
        "status": cfg.status,
        "categories": cfg.category_ids,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:300]}")
    j = r.json()
    return int(j["id"]), str(j.get("link") or "")


def wp_update_post(cfg: WPCfg, post_id: int, title: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "content": html,
        "status": cfg.status,
        "categories": cfg.category_ids,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:300]}")
    j = r.json()
    return int(j["id"]), str(j.get("link") or "")


# -----------------------------
# Rendering
# -----------------------------
def pct_change(today: int, prev: Optional[int]) -> str:
    if prev is None or prev <= 0 or today <= 0:
        return "-"
    ch = (today - prev) / prev * 100.0
    sign = "+" if ch >= 0 else ""
    return f"{sign}{ch:.1f}%"


def bar_width(value: int, max_value: int) -> int:
    if max_value <= 0 or value <= 0:
        return 0
    w = int(round((value / max_value) * 100))
    return max(1, min(100, w))


def build_overview(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "<p>(키워드 없음)</p>"
    maxv = max(int(x.get("traffic_int", 0) or 0) for x in items) or 0
    rows = []
    for i, it in enumerate(items, start=1):
        kw = htmlmod.escape(str(it.get("keyword", "")))
        link = str(it.get("link", ""))
        traffic = htmlmod.escape(str(it.get("traffic_str", "") or "-"))
        change = htmlmod.escape(str(it.get("change", "-")))
        v = int(it.get("traffic_int", 0) or 0)
        w = bar_width(v, maxv)
        kw_html = f'<a href="{htmlmod.escape(link)}" target="_blank" rel="noopener noreferrer">{kw}</a>' if link else kw
        bar = (
            '<div style="height:10px;background:#eee;border-radius:999px;overflow:hidden;">'
            f'<div style="height:10px;width:{w}%;background:#111;"></div></div>'
        )
        rows.append(
            f"""
            <tr>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">{kw_html}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;">{traffic}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;">{change}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">{bar}</td>
            </tr>
            """
        )

    return f"""
    <h2>오늘의 핫 키워드 TOP {len(items)}</h2>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">순위</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">키워드</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">대략 트래픽</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">전일 대비(대략)</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">그래프</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def fmt_krw(n: int) -> str:
    try:
        return f"{int(n):,}원"
    except Exception:
        return str(n)


def build_products_table(keyword: str, items: List[Dict[str, Any]]) -> str:
    rows = []
    for i, it in enumerate(items, start=1):
        name = htmlmod.escape(str(it.get("title", "")))
        link = str(it.get("link", ""))
        img = str(it.get("image", ""))
        lprice = int(it.get("lprice", 0) or 0)
        mall = htmlmod.escape(str(it.get("mallName", "")))

        img_html = (
            f'<img src="{htmlmod.escape(img)}" alt="{name}" style="width:72px;height:auto;border-radius:10px;">'
            if img
            else ""
        )
        name_html = (
            f'<a href="{htmlmod.escape(link)}" target="_blank" rel="nofollow sponsored noopener">{name}</a>'
            if link
            else name
        )

        rows.append(
            f"""
            <tr>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">{img_html}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">
                {name_html}
                <div style="font-size:12px;opacity:.75;margin-top:4px;">{mall}</div>
              </td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;">{fmt_krw(lprice)}</td>
            </tr>
            """
        )

    return f"""
    <h3 style="margin-top:18px;">{htmlmod.escape(keyword)} → 네이버 쇼핑 추천 TOP {len(items)}</h3>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">순위</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">이미지</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">상품</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">최저가</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows) if rows else '<tr><td colspan="4" style="padding:10px;border:1px solid #e5e5e5;">결과 없음</td></tr>'}
      </tbody>
    </table>
    """


def build_post_html(date_str: str, slot: str, geo: str, items: List[Dict[str, Any]], kw_to_products: Dict[str, List[Dict[str, Any]]]) -> str:
    slot_label = "오전" if slot == "am" else "오후"
    disclosure = (
        '<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">'
        "※ 이 포스팅은 자료모으는 용도입니다."
        "</p>"
    )
    head = f"<p>기준일: <b>{htmlmod.escape(date_str)}</b> / 슬롯: <b>{slot_label}</b> / 지역: <b>{htmlmod.escape(geo)}</b></p>"
    note = '<p style="font-size:13px;opacity:.8;">키워드 출처: Google Trends. 상품 출처: 네이버 쇼핑 검색 API.</p>'

    parts = [disclosure, head, note, build_overview(items), "<hr/>", "<h2>키워드별 추천 상품</h2>"]
    for it in items:
        kw = str(it.get("keyword", ""))
        parts.append(build_products_table(kw, kw_to_products.get(kw, [])))
    return "".join(parts)


# -----------------------------
# Main run
# -----------------------------
def run(cfg: AppCfg) -> None:
    now_kst = datetime.now(KST)
    date_str = now_kst.strftime("%Y-%m-%d")

    init_db(cfg.sqlite_path)

    hot = fetch_google_hot_keywords(cfg.google.geo, cfg.google.limit, cfg.google.timeout, cfg.debug)

    prev_date = (now_kst - timedelta(days=1)).strftime("%Y-%m-%d")
    for it in hot:
        prev = get_prev_traffic(cfg.sqlite_path, prev_date, str(it.get("keyword", "")))
        it["change"] = pct_change(int(it.get("traffic_int", 0) or 0), prev)

    upsert_keywords(cfg.sqlite_path, date_str, cfg.slot, hot)

    kw_to_products: Dict[str, List[Dict[str, Any]]] = {}
    with requests.Session() as s:
        for idx, it in enumerate(hot, start=1):
            kw = str(it.get("keyword", "")).strip()
            if not kw:
                continue
            if cfg.debug:
                print(f"[DEBUG] ({idx}/{len(hot)}) NAVER:", kw)
            items = naver_shop_search(s, cfg.naver, kw)
            kw_to_products[kw] = items
            time.sleep(cfg.naver.throttle)

    title = f"{date_str} {'오전' if cfg.slot=='am' else '오후'} 핫 키워드 → 네이버 쇼핑 추천 TOP상품"
    html = build_post_html(date_str, cfg.slot, cfg.google.geo, hot, kw_to_products)

    if cfg.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기(앞 2000자):")
        print(html[:2000])
        return

    existing = get_existing_post(cfg.sqlite_path, date_str, cfg.slot)
    if existing:
        post_id, old_link = existing
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html)
        save_post_meta(cfg.sqlite_path, date_str, cfg.slot, wp_post_id, wp_link or old_link)
        print("OK(updated):", wp_post_id, wp_link or old_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, html)
        save_post_meta(cfg.sqlite_path, date_str, cfg.slot, wp_post_id, wp_link)
        print("OK(created):", wp_post_id, wp_link)


def parse_args(argv: List[str]) -> Dict[str, Any]:
    out = {"debug": False, "dry_run": False, "slot": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--debug":
            out["debug"] = True
            i += 1
            continue
        if a == "--dry-run":
            out["dry_run"] = True
            i += 1
            continue
        if a in ("--slot", "-s") and i + 1 < len(argv):
            out["slot"] = (argv[i + 1] or "").lower().strip()
            i += 2
            continue
        i += 1
    return out


def main():
    args = parse_args(sys.argv[1:])
    cfg = load_cfg()

    if args["debug"]:
        cfg.debug = True
    if args["dry_run"]:
        cfg.dry_run = True
    if args["slot"] in ("am", "pm"):
        cfg.slot = args["slot"]

    validate_cfg(cfg)
    print_safe_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
