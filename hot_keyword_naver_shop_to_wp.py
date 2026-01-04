# -*- coding: utf-8 -*-
"""
hot_keyword_naver_shop_to_wp.py (완전 통합본)
- Google 트렌드(핫 키워드) 수집: RSS 우선 + 다중 폴백 + (최후) hidden JSON dailytrends 폴백
- 각 키워드별 네이버 쇼핑 TOP상품(기본 5개) 추천
- 워드프레스 REST API로 글 발행(AM/PM 슬롯별로 글을 따로 생성)
- SQLite에 키워드/상품/포스트ID 저장(전일 대비 % 계산용)

✅ GitHub Actions Secrets 이름(너가 쓰는 그대로)
  - XNAVERCLIENT_ID
  - XNAVERCLIENT_SECRET
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

✅ 카테고리 기본값: 31
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


# -----------------------------
# Timezone
# -----------------------------
KST = timezone(timedelta(hours=9))


# -----------------------------
# Config
# -----------------------------
def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


@dataclass
class GoogleTrendsConfig:
    geo: str = "KR"
    limit: int = 20  # hot keywords count
    request_timeout: int = 25


@dataclass
class NaverShopConfig:
    client_id: str = ""
    client_secret: str = ""
    display: int = 5  # products per keyword
    sort: str = "sim"  # sim/date/asc/dsc
    exclude: str = "used:rental:cbshop"
    request_timeout: int = 25
    throttle_sec: float = 0.25


@dataclass
class WordPressConfig:
    base_url: str = ""  # e.g. https://your-site.com
    user: str = ""
    app_pass: str = ""
    status: str = "publish"
    category_ids: List[int] = None  # default [31]

    def __post_init__(self):
        if self.category_ids is None:
            self.category_ids = [31]


@dataclass
class StorageConfig:
    sqlite_path: str = "data/hot_keyword_naver_shop.sqlite3"


@dataclass
class RunConfig:
    slot: str = "am"  # am / pm
    dry_run: bool = False
    debug: bool = False


@dataclass
class AppConfig:
    google: GoogleTrendsConfig
    naver: NaverShopConfig
    wp: WordPressConfig
    store: StorageConfig
    run: RunConfig


def load_cfg_from_env() -> AppConfig:
    # ✅ 너가 바꿔쓴 이름 우선
    naver_id = _env("XNAVERCLIENT_ID") or _env("NAVER_CLIENT_ID")
    naver_secret = _env("XNAVERCLIENT_SECRET") or _env("NAVER_CLIENT_SECRET")

    wp_base = (_env("WP_BASE_URL") or _env("WP_SITE_URL")).rstrip("/")
    wp_user = _env("WP_USER") or _env("WP_USERNAME")
    wp_pass = _env("WP_APP_PASS") or _env("WP_APP_PASSWORD")

    # run
    slot = (_env("RUN_SLOT", "am") or "am").lower()
    dry_run = _env("DRY_RUN", "0").lower() in ("1", "true", "yes")
    debug = _env("DEBUG", "0").lower() in ("1", "true", "yes")

    # google
    geo = (_env("TRENDS_GEO", "KR") or "KR").upper()
    hot_limit = int(_env("HOT_KEYWORDS_LIMIT", "20") or "20")
    g_timeout = int(_env("GOOGLE_TIMEOUT", "25") or "25")

    # naver
    p_per_kw = int(_env("NAVER_PER_KEYWORD", "5") or "5")
    sort = _env("NAVER_SORT", "sim") or "sim"
    exclude = _env("NAVER_EXCLUDE", "used:rental:cbshop")
    n_timeout = int(_env("NAVER_TIMEOUT", "25") or "25")
    throttle = float(_env("NAVER_THROTTLE_SEC", "0.25") or "0.25")

    # wp
    wp_status = _env("WP_STATUS", "publish") or "publish"
    cat_ids_raw = _env("WP_CATEGORY_IDS", "31") or "31"
    cat_ids = []
    for x in re.split(r"[,\s]+", cat_ids_raw.strip()):
        if x.strip().isdigit():
            cat_ids.append(int(x.strip()))
    if not cat_ids:
        cat_ids = [31]

    # storage
    sqlite_path = _env("SQLITE_PATH", "data/hot_keyword_naver_shop.sqlite3")

    return AppConfig(
        google=GoogleTrendsConfig(geo=geo, limit=hot_limit, request_timeout=g_timeout),
        naver=NaverShopConfig(
            client_id=naver_id,
            client_secret=naver_secret,
            display=p_per_kw,
            sort=sort,
            exclude=exclude,
            request_timeout=n_timeout,
            throttle_sec=throttle,
        ),
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=cat_ids,
        ),
        store=StorageConfig(sqlite_path=sqlite_path),
        run=RunConfig(slot=slot, dry_run=dry_run, debug=debug),
    )


def validate_cfg(cfg: AppConfig) -> None:
    missing = []
    if not cfg.naver.client_id:
        missing.append("XNAVERCLIENT_ID (or NAVER_CLIENT_ID)")
    if not cfg.naver.client_secret:
        missing.append("XNAVERCLIENT_SECRET (or NAVER_CLIENT_SECRET)")
    if not cfg.wp.base_url:
        missing.append("WP_BASE_URL")
    if not cfg.wp.user:
        missing.append("WP_USER")
    if not cfg.wp.app_pass:
        missing.append("WP_APP_PASS")
    if cfg.run.slot not in ("am", "pm"):
        missing.append("RUN_SLOT must be am or pm")
    if missing:
        raise RuntimeError("필수 설정 누락/오류:\n- " + "\n- ".join(missing))


def print_safe_cfg(cfg: AppConfig) -> None:
    def ok(v: str) -> str:
        return f"OK(len={len(v)})" if v else "MISSING"

    print("[CFG] RUN_SLOT:", cfg.run.slot)
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "DEBUG:", cfg.run.debug)
    print("[CFG] GOOGLE geo/limit:", cfg.google.geo, cfg.google.limit)
    print("[CFG] NAVER id/secret:", ok(cfg.naver.client_id), ok(cfg.naver.client_secret))
    print("[CFG] NAVER per_keyword/sort:", cfg.naver.display, cfg.naver.sort)
    print("[CFG] WP_BASE_URL:", cfg.wp.base_url)
    print("[CFG] WP_USER/WP_APP_PASS:", ok(cfg.wp.user), ok(cfg.wp.app_pass))
    print("[CFG] WP categories:", cfg.wp.category_ids)
    print("[CFG] SQLITE:", cfg.store.sqlite_path)


# -----------------------------
# SQLite
# -----------------------------
def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
          date TEXT,
          slot TEXT,
          wp_post_id INTEGER,
          wp_link TEXT,
          created_at TEXT,
          PRIMARY KEY (date, slot)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS keywords (
          date TEXT,
          slot TEXT,
          rank INTEGER,
          keyword TEXT,
          traffic_str TEXT,
          traffic_int INTEGER,
          link TEXT,
          PRIMARY KEY (date, slot, rank)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
          date TEXT,
          slot TEXT,
          keyword TEXT,
          rank INTEGER,
          title TEXT,
          link TEXT,
          image TEXT,
          lprice INTEGER,
          mall_name TEXT,
          product_id TEXT,
          PRIMARY KEY (date, slot, keyword, rank)
        )
        """
    )

    con.commit()
    con.close()


def save_post_meta(db_path: str, date_str: str, slot: str, wp_post_id: int, wp_link: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO posts(date, slot, wp_post_id, wp_link, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (date_str, slot, wp_post_id, wp_link, datetime.utcnow().isoformat()),
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


def upsert_products(db_path: str, date_str: str, slot: str, keyword: str, products: List[Dict[str, Any]]) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    for i, p in enumerate(products, start=1):
        cur.execute(
            """
            INSERT OR REPLACE INTO products(date, slot, keyword, rank, title, link, image, lprice, mall_name, product_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                date_str,
                slot,
                keyword,
                i,
                str(p.get("title", "")),
                str(p.get("link", "")),
                str(p.get("image", "")),
                int(p.get("lprice", 0) or 0),
                str(p.get("mallName", "")),
                str(p.get("productId", "")),
            ),
        )
    con.commit()
    con.close()


def get_prev_traffic(db_path: str, prev_date: str, keyword: str) -> Optional[int]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    # 전날(AM/PM 상관 없이)에서 같은 키워드 traffic_int 하나라도 있으면 가져옴
    cur.execute(
        """
        SELECT traffic_int
        FROM keywords
        WHERE date=? AND keyword=?
        ORDER BY traffic_int DESC
        LIMIT 1
        """,
        (prev_date, keyword),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return int(row[0] or 0)


# -----------------------------
# Google Trends (RSS + fallback)
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


def _parse_google_trends_rss(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []

    out: List[Dict[str, Any]] = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()

        approx_traffic = ""
        for child in list(item):
            if child.tag.endswith("approx_traffic"):
                approx_traffic = (child.text or "").strip()
                break

        keyword = title.split(" - ", 1)[0].strip() if title else ""
        if keyword:
            out.append(
                {
                    "keyword": keyword,
                    "traffic_str": approx_traffic,
                    "traffic_int": _parse_traffic_to_int(approx_traffic),
                    "link": link,
                }
            )
    return out


def _fetch_google_rss_candidates(geo: str) -> List[str]:
    geo = (geo or "KR").upper().strip()
    return [
        # ✅ 많이 쓰이는 RSS
        f"https://trends.google.com/trending/rss?geo={geo}",
        # ✅ 일별 트렌드 RSS
        f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}",
        # (레거시; 일부 환경에서 404/변동 가능)
        f"https://trends.google.co.kr/trending/rss?geo={geo}",
        f"https://trends.google.co.kr/trends/trendingsearches/daily/rss?geo={geo}",
    ]


def _fetch_google_dailytrends_json(geo: str, timeout: int) -> List[Dict[str, Any]]:
    # hidden endpoint (프론트에서 쓰는 형태로 널리 알려짐)
    # 예시: /trends/api/dailytrends?hl=en-US&tz=300&geo=US&ns=15 :contentReference[oaicite:3]{index=3}
    geo = (geo or "KR").upper().strip()
    url = f"https://trends.google.com/trends/api/dailytrends?hl=ko&tz=-540&geo={geo}&ns=15"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    if r.status_code != 200:
        return []
    text = r.text.strip()
    # 응답 앞에 )]}', 같은 프리픽스가 붙는 경우 제거
    text = re.sub(r"^\)\]\}',?\s*", "", text)
    try:
        data = json.loads(text)
    except Exception:
        return []

    days = (((data or {}).get("default") or {}).get("trendingSearchesDays") or [])
    if not days:
        return []
    # 가장 최신 day
    day0 = days[0] or {}
    searches = day0.get("trendingSearches") or []
    out: List[Dict[str, Any]] = []
    for s in searches:
        q = (((s.get("title") or {}).get("query")) or "").strip()
        traffic_str = (s.get("formattedTraffic") or "").strip()
        traffic_int = _parse_traffic_to_int(traffic_str)
        # 대표 기사 링크 하나
        link = ""
        arts = s.get("articles") or []
        if arts and isinstance(arts, list):
            link = str((arts[0] or {}).get("url") or "").strip()
        if q:
            out.append({"keyword": q, "traffic_str": traffic_str, "traffic_int": traffic_int, "link": link})
    return out


def fetch_google_hot_keywords(cfg: GoogleTrendsConfig, debug: bool) -> List[Dict[str, Any]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; hot-keyword-bot/1.0)",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }

    # 1) RSS 우선
    for url in _fetch_google_rss_candidates(cfg.geo):
        try:
            r = requests.get(url, headers=headers, timeout=cfg.request_timeout, allow_redirects=True)
            if debug:
                print("[DEBUG] Google RSS try:", url, "status=", r.status_code)
            if r.status_code != 200:
                continue
            txt = (r.text or "").strip()
            if "<rss" not in txt.lower():
                continue
            items = _parse_google_trends_rss(txt)
            # 중복 제거 + limit
            seen = set()
            uniq = []
            for it in items:
                k = it["keyword"]
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(it)
                if len(uniq) >= cfg.limit:
                    break
            if uniq:
                return uniq
        except Exception as e:
            if debug:
                print("[DEBUG] Google RSS error:", repr(e))
            continue

    # 2) 최후 폴백: dailytrends JSON
    items = _fetch_google_dailytrends_json(cfg.geo, cfg.request_timeout)
    if items:
        return items[: cfg.limit]

    raise RuntimeError("Google Trends fetch failed (all RSS candidates + dailytrends JSON failed)")


# -----------------------------
# Naver Shop Search API (shop.json)
# -----------------------------
NAVER_SHOP_ENDPOINT = "https://openapi.naver.com/v1/search/shop.json"  # 공식 문서 :contentReference[oaicite:4]{index=4}


def _clean_title(t: str) -> str:
    t = htmlmod.unescape(t or "")
    t = re.sub(r"</?b>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def naver_shop_search(session: requests.Session, cfg: NaverShopConfig, query: str) -> List[Dict[str, Any]]:
    headers = {
        "X-Naver-Client-Id": cfg.client_id,
        "X-Naver-Client-Secret": cfg.client_secret,
        "User-Agent": "Mozilla/5.0 (compatible; hot-keyword-bot/1.0)",
    }
    params: Dict[str, Any] = {
        "query": query,
        "display": cfg.display,
        "start": 1,
        "sort": cfg.sort,
        "exclude": cfg.exclude,
    }

    r = session.get(NAVER_SHOP_ENDPOINT, headers=headers, params=params, timeout=cfg.request_timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Naver Shop API failed: {r.status_code} body={r.text[:300]}")
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
    return {"Authorization": f"Basic {token}", "User-Agent": "hot-keyword-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, html: str) -> Tuple[int, str]:
    endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "content": html,
        "status": cfg.status,
        "categories": cfg.category_ids,
    }
    r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:300]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str) -> Tuple[int, str]:
    endpoint = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "content": html,
        "status": cfg.status,
        "categories": cfg.category_ids,
    }
    r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:300]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


# -----------------------------
# Rendering
# -----------------------------
def fmt_krw(n: int) -> str:
    try:
        return f"{int(n):,}원"
    except Exception:
        return str(n)


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


def build_keyword_overview_table(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "<p>(키워드 없음)</p>"
    maxv = max(int(x.get("traffic_int", 0) or 0) for x in items) or 0

    rows = []
    for i, it in enumerate(items, start=1):
        kw = htmlmod.escape(str(it.get("keyword", "")))
        link = str(it.get("link", ""))
        traffic = htmlmod.escape(str(it.get("traffic_str", "") or "-"))
        ch = htmlmod.escape(str(it.get("change", "-")))
        v = int(it.get("traffic_int", 0) or 0)
        w = bar_width(v, maxv)

        kw_html = f'<a href="{htmlmod.escape(link)}" target="_blank" rel="noopener noreferrer">{kw}</a>' if link else kw
        bar = f'<div style="height:10px;background:#eee;border-radius:999px;overflow:hidden;"><div style="height:10px;width:{w}%;background:#111;"></div></div>'

        rows.append(
            f"""
            <tr>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">{kw_html}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;">{traffic}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;">{ch}</td>
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
        "※ 이 포스팅은 제휴마케팅 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받을 수 있습니다."
        "</p>"
    )
    head = f"<p>기준일: <b>{htmlmod.escape(date_str)}</b> / 슬롯: <b>{slot_label}</b> / 지역: <b>{htmlmod.escape(geo)}</b></p>"
    note = '<p style="font-size:13px;opacity:.8;">키워드 출처: Google Trends (RSS/일별 트렌드 폴백). 상품 출처: 네이버 쇼핑 검색 API.</p>'

    overview = build_keyword_overview_table(items)

    parts = [disclosure, head, note, overview, "<hr/>", "<h2>키워드별 추천 상품</h2>"]
    for it in items:
        kw = str(it.get("keyword", ""))
        prods = kw_to_products.get(kw, [])
        parts.append(build_products_table(kw, prods))

    return "".join(parts)


# -----------------------------
# Main run
# -----------------------------
def parse_args(argv: List[str]) -> Dict[str, Any]:
    out = {"dry_run": False, "debug": False, "slot": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry-run":
            out["dry_run"] = True
            i += 1
            continue
        if a == "--debug":
            out["debug"] = True
            i += 1
            continue
        if a in ("--slot", "-s") and i + 1 < len(argv):
            out["slot"] = (argv[i + 1] or "").lower().strip()
            i += 2
            continue
        i += 1
    return out


def run(cfg: AppConfig) -> None:
    # 날짜는 KST 기준
    now_kst = datetime.now(KST)
    date_str = now_kst.strftime("%Y-%m-%d")
    slot = cfg.run.slot

    init_db(cfg.store.sqlite_path)

    # 1) Google hot keywords
    hot = fetch_google_hot_keywords(cfg.google, cfg.run.debug)
    hot = hot[: cfg.google.limit]

    # 2) 전일 대비(대략) 계산(traffic_int 기반)
    prev_date = (now_kst - timedelta(days=1)).strftime("%Y-%m-%d")
    for it in hot:
        prev = get_prev_traffic(cfg.store.sqlite_path, prev_date, str(it.get("keyword", "")))
        it["change"] = pct_change(int(it.get("traffic_int", 0) or 0), prev)

    upsert_keywords(cfg.store.sqlite_path, date_str, slot, hot)

    # 3) 각 키워드 -> 네이버 쇼핑 TOP 상품
    kw_to_products: Dict[str, List[Dict[str, Any]]] = {}
    with requests.Session() as s:
        for idx, it in enumerate(hot, start=1):
            kw = str(it.get("keyword", "")).strip()
            if not kw:
                continue
            if cfg.run.debug:
                print(f"[DEBUG] ({idx}/{len(hot)}) Naver search:", kw)
            items = naver_shop_search(s, cfg.naver, kw)
            kw_to_products[kw] = items
            upsert_products(cfg.store.sqlite_path, date_str, slot, kw, items)
            time.sleep(cfg.naver.throttle_sec)

    # 4) 글 생성/업데이트
    slot_label = "오전" if slot == "am" else "오후"
    title = f"{date_str} {slot_label} 핫 키워드 → 네이버 쇼핑 추천 TOP상품"
    html = build_post_html(date_str, slot, cfg.google.geo, hot, kw_to_products)

    if cfg.run.dry_run:
        print("[DRY_RUN] Posting skipped.\n")
        print(html[:2000] + ("\n...(truncated)" if len(html) > 2000 else ""))
        return

    existing = get_existing_post(cfg.store.sqlite_path, date_str, slot)
    if existing:
        post_id, old_link = existing
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html)
        save_post_meta(cfg.store.sqlite_path, date_str, slot, wp_post_id, wp_link or old_link)
        print(f"OK(updated): {wp_post_id} {wp_link or old_link}")
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, html)
        save_post_meta(cfg.store.sqlite_path, date_str, slot, wp_post_id, wp_link)
        print(f"OK(created): {wp_post_id} {wp_link}")


def main():
    args = parse_args(sys.argv[1:])
    cfg = load_cfg_from_env()

    if args["dry_run"]:
        cfg.run.dry_run = True
    if args["debug"]:
        cfg.run.debug = True
    if args["slot"] in ("am", "pm"):
        cfg.run.slot = args["slot"]

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
