# -*- coding: utf-8 -*-
"""
hot_keyword_naver_shop_to_wp.py (완전 통합)
- 구글 트렌드 "일별 급상승" RSS에서 핫 키워드 N개 수집
- 각 키워드로 네이버 쇼핑 검색 API(shop.json) 호출 → 상품 TOP M개 추천
- 오전/오후 RUN_SLOT 별로 글을 "따로" 생성(슬롯별 slug 고정)
- WordPress REST API로 생성/업데이트
- SQLite로 슬롯별 발행 이력 저장(같은 슬롯은 업데이트)

✅ GitHub Secrets(환경변수) 이름 그대로 사용:
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

✅ 네이버 API 키 (둘 중 하나만 있어도 됨)
  - NAVER_CLIENT_ID / NAVER_CLIENT_SECRET
  - XNAVERCLIENT_ID / XNAVERCLIENT_SECRET   (네가 쓰는 이름)

✅ 옵션 환경변수:
  - WP_STATUS: publish (기본 publish)
  - WP_CATEGORY_IDS: "31" (기본 31)
  - WP_TAG_IDS: "1,2,3" (선택)
  - SQLITE_PATH: data/hot_keyword_shop.sqlite3
  - RUN_SLOT: am / pm
  - HOT_KEYWORDS_LIMIT: 20 (기본 20)
  - PRODUCTS_PER_KEYWORD: 5 (기본 5)
  - GOOGLE_TRENDS_GEO: KR (기본 KR)
  - NAVER_SORT: sim (기본 sim)
  - DRY_RUN: 1이면 워드프레스 발행 안하고 미리보기 출력
  - DEBUG: 1이면 상세 로그
"""

from __future__ import annotations

import base64
import hashlib
import html as htmlmod
import json
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests


KST = timezone(timedelta(hours=9))
NAVER_SHOP_ENDPOINT = "https://openapi.naver.com/v1/search/shop.json"


# -----------------------------
# Utils / env
# -----------------------------
def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name, str(default))
    try:
        return int(v)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name, "1" if default else "0").lower()
    return v in ("1", "true", "yes", "y", "on")


def _parse_int_list(csv: str) -> List[int]:
    out: List[int] = []
    for x in (csv or "").split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(int(x))
        except Exception:
            pass
    return out


def clean_title(t: str) -> str:
    t = htmlmod.unescape(t or "")
    t = re.sub(r"</?b>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fmt_krw(n: int) -> str:
    try:
        return f"{int(n):,}원"
    except Exception:
        return str(n)


# -----------------------------
# Config
# -----------------------------
@dataclass
class NaverConfig:
    client_id: str
    client_secret: str
    sort: str = "sim"


@dataclass
class WordPressConfig:
    base_url: str
    user: str
    app_pass: str
    status: str = "publish"
    category_ids: List[int] = field(default_factory=list)
    tag_ids: List[int] = field(default_factory=list)


@dataclass
class RunConfig:
    run_slot: str = "am"  # am/pm
    hot_keywords_limit: int = 20
    products_per_keyword: int = 5
    trends_geo: str = "KR"
    dry_run: bool = False
    debug: bool = False


@dataclass
class AppConfig:
    naver: NaverConfig
    wp: WordPressConfig
    run: RunConfig
    sqlite_path: str


def load_cfg() -> AppConfig:
    # Naver keys: 네가 쓰는 이름(XNAVER...) 우선 + 일반 이름도 fallback
    naver_id = _env("XNAVERCLIENT_ID") or _env("NAVER_CLIENT_ID")
    naver_secret = _env("XNAVERCLIENT_SECRET") or _env("NAVER_CLIENT_SECRET")
    naver_sort = _env("NAVER_SORT", "sim") or "sim"

    wp_base = (_env("WP_BASE_URL") or _env("WP_SITE_URL")).rstrip("/")
    wp_user = _env("WP_USER") or _env("WP_USERNAME")
    wp_pass = _env("WP_APP_PASS") or _env("WP_APP_PASSWORD")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    wp_cats = _parse_int_list(_env("WP_CATEGORY_IDS", "31"))
    wp_tags = _parse_int_list(_env("WP_TAG_IDS", ""))

    run_slot = (_env("RUN_SLOT", "am") or "am").lower()
    if run_slot not in ("am", "pm"):
        run_slot = "am"

    hot_limit = _env_int("HOT_KEYWORDS_LIMIT", 20)
    prod_per = _env_int("PRODUCTS_PER_KEYWORD", 5)
    geo = (_env("GOOGLE_TRENDS_GEO", "KR") or "KR").upper()

    sqlite_path = _env("SQLITE_PATH", "data/hot_keyword_shop.sqlite3")

    dry_run = _env_bool("DRY_RUN", False)
    debug = _env_bool("DEBUG", False)

    return AppConfig(
        naver=NaverConfig(client_id=naver_id, client_secret=naver_secret, sort=naver_sort),
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=wp_cats,
            tag_ids=wp_tags,
        ),
        run=RunConfig(
            run_slot=run_slot,
            hot_keywords_limit=hot_limit,
            products_per_keyword=prod_per,
            trends_geo=geo,
            dry_run=dry_run,
            debug=debug,
        ),
        sqlite_path=sqlite_path,
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
    if missing:
        raise RuntimeError("필수 설정 누락:\n- " + "\n- ".join(missing))


def print_safe_cfg(cfg: AppConfig) -> None:
    def ok(v: str) -> str:
        return f"OK(len={len(v)})" if v else "MISSING"

    print("[CFG] XNAVERCLIENT_ID:", ok(cfg.naver.client_id))
    print("[CFG] XNAVERCLIENT_SECRET:", ok(cfg.naver.client_secret))
    print("[CFG] NAVER_SORT:", cfg.naver.sort)
    print("[CFG] WP_BASE_URL:", cfg.wp.base_url or "MISSING")
    print("[CFG] WP_USER:", ok(cfg.wp.user))
    print("[CFG] WP_APP_PASS:", ok(cfg.wp.app_pass))
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CFG] WP_TAG_IDS:", cfg.wp.tag_ids)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| GEO:", cfg.run.trends_geo,
          "| HOT_LIMIT:", cfg.run.hot_keywords_limit, "| PROD_PER:", cfg.run.products_per_keyword)
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)


# -----------------------------
# SQLite (slot post history)
# -----------------------------
def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_posts (
          date_slot TEXT PRIMARY KEY,
          wp_post_id INTEGER,
          wp_link TEXT,
          created_at TEXT
        )
        """
    )
    con.commit()
    con.close()


def get_existing_post(path: str, date_slot: str) -> Optional[Tuple[int, str]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT wp_post_id, wp_link FROM daily_posts WHERE date_slot=?", (date_slot,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return int(row[0]), str(row[1] or "")


def save_post_meta(path: str, date_slot: str, post_id: int, link: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_slot, wp_post_id, wp_link, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (date_slot, post_id, link, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "hot-keyword-shop-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}

    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}

    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


# -----------------------------
# Google Trends Daily RSS
# -----------------------------
def fetch_google_trends_daily(geo: str, limit: int, debug: bool = False) -> List[Dict[str, Any]]:
    # 널리 쓰이는 일별 RSS 엔드포인트 (KR 가능) :contentReference[oaicite:2]{index=2}
    url = f"https://trends.google.co.kr/trends/trendingsearches/daily/rss?geo={quote(geo)}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; hot-keyword-shop-bot/1.0)"}

    r = requests.get(url, headers=headers, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Google Trends RSS failed: {r.status_code} body={r.text[:300]}")

    try:
        root = ET.fromstring(r.text)
    except Exception as e:
        raise RuntimeError(f"Google Trends RSS parse error: {e}")

    out: List[Dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue

        # ht:approx_traffic 같은 namespaced tag를 와일드카드로 찾기
        approx = ""
        for ch in list(item):
            tag = ch.tag.lower()
            if tag.endswith("approx_traffic"):
                approx = (ch.text or "").strip()
                break

        pub = (item.findtext("pubDate") or "").strip()
        dt = None
        if pub:
            try:
                dt = parsedate_to_datetime(pub).astimezone(KST)
            except Exception:
                dt = None
        if dt is None:
            dt = datetime.now(tz=KST)

        out.append({"keyword": title, "approx_traffic": approx, "published_at": dt.isoformat()})

        if len(out) >= limit:
            break

    if debug:
        print(f"[DEBUG] Google Trends keywords={len(out)} from {url}")

    return out


# -----------------------------
# Naver Shopping Search
# -----------------------------
def naver_shop_search(
    session: requests.Session,
    ncfg: NaverConfig,
    query: str,
    display: int,
    sort: str,
    retry: int = 3,
) -> List[Dict[str, Any]]:
    headers = {
        "X-Naver-Client-Id": ncfg.client_id,
        "X-Naver-Client-Secret": ncfg.client_secret,
        "User-Agent": "Mozilla/5.0 (compatible; hot-keyword-shop-bot/1.0)",
    }
    params: Dict[str, Any] = {"query": query, "display": display, "start": 1, "sort": sort}

    last_err = None
    for attempt in range(1, retry + 1):
        try:
            r = session.get(NAVER_SHOP_ENDPOINT, headers=headers, params=params, timeout=25)
            if r.status_code == 200:
                data = r.json()
                items = data.get("items", [])
                return items if isinstance(items, list) else []
            # 429/5xx는 재시도
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"{r.status_code} {r.text[:200]}"
                time.sleep(1.2 * attempt)
                continue
            raise RuntimeError(f"Naver API failed: {r.status_code} body={r.text[:300]}")
        except Exception as e:
            last_err = repr(e)
            time.sleep(1.0 * attempt)

    raise RuntimeError(f"Naver API retry exhausted: {last_err}")


# -----------------------------
# Rendering
# -----------------------------
DISCLOSURE = "※ 이 포스팅은 제휴마케팅 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받을 수 있습니다."
NOTE = "핫 키워드(구글 트렌드) 기준으로 네이버 쇼핑 검색 결과 상위 상품을 자동 추천합니다."


def build_product_table(items: List[Dict[str, Any]]) -> str:
    rows = []
    for i, it in enumerate(items, start=1):
        name = clean_title(str(it.get("title", "")))
        link = str(it.get("link", ""))
        img = str(it.get("image", ""))
        lprice = int(it.get("lprice", 0) or 0)
        mall = htmlmod.escape(str(it.get("mallName", "")))

        img_html = f'<img src="{img}" alt="{htmlmod.escape(name)}" style="width:64px;height:auto;border-radius:10px;">' if img else ""
        name_html = (
            f'<a href="{htmlmod.escape(link)}" target="_blank" rel="nofollow sponsored noopener">{htmlmod.escape(name)}</a>'
            if link else htmlmod.escape(name)
        )

        rows.append(
            f"""
            <tr>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{img_html}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">
                {name_html}
                <div style="font-size:12px;opacity:.75;margin-top:4px;">{mall}</div>
              </td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;">{fmt_krw(lprice)}</td>
            </tr>
            """
        )

    return f"""
    <table style="border-collapse:collapse;width:100%;font-size:14px;margin-top:8px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">#</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">이미지</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">상품</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">최저가</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows) if rows else '<tr><td colspan="4" style="padding:10px;border:1px solid #e5e5e5;">검색 결과 없음</td></tr>'}
      </tbody>
    </table>
    """


def build_post_html(now: datetime, slot_label: str, keywords: List[Dict[str, Any]], products_map: Dict[str, List[Dict[str, Any]]]) -> str:
    disclosure = f'<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{htmlmod.escape(DISCLOSURE)}</p>'
    head = f"<p>기준시각: <b>{htmlmod.escape(now.strftime('%Y-%m-%d %H:%M'))}</b> / 슬롯: <b>{htmlmod.escape(slot_label)}</b></p>"
    note = f'<p style="font-size:13px;opacity:.8;">{htmlmod.escape(NOTE)}</p>'

    sections = []
    for idx, k in enumerate(keywords, start=1):
        kw = k["keyword"]
        traffic = k.get("approx_traffic", "")
        badge = f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:#f1f1f1;font-size:12px;opacity:.85;margin-left:8px;">{htmlmod.escape(traffic)}</span>' if traffic else ""
        naver_search_link = f"https://search.shopping.naver.com/search/all?query={quote(kw)}"
        table = build_product_table(products_map.get(kw, []))

        sections.append(
            f"""
            <details {'open' if idx <= 3 else ''} style="margin:14px 0;">
              <summary style="cursor:pointer;font-size:16px;">
                <b>{idx}. {htmlmod.escape(kw)}</b>{badge}
                <span style="font-size:12px;opacity:.7;margin-left:10px;">
                  <a href="{htmlmod.escape(naver_search_link)}" target="_blank" rel="nofollow noopener">네이버쇼핑 검색</a>
                </span>
              </summary>
              {table}
            </details>
            """
        )

    return disclosure + head + note + "".join(sections)


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else "오후"
    date_slot = f"{date_str}_{slot}"

    init_db(cfg.sqlite_path)

    # 1) hot keywords
    keywords = fetch_google_trends_daily(cfg.run.trends_geo, cfg.run.hot_keywords_limit, cfg.run.debug)
    if not keywords:
        raise RuntimeError("핫 키워드가 0개입니다(구글 트렌드 RSS 파싱 실패 가능).")

    # 2) map keyword -> products
    products_map: Dict[str, List[Dict[str, Any]]] = {}
    with requests.Session() as session:
        for k in keywords:
            kw = k["keyword"]
            items = naver_shop_search(
                session=session,
                ncfg=cfg.naver,
                query=kw,
                display=cfg.run.products_per_keyword,
                sort=cfg.naver.sort,
                retry=3,
            )
            products_map[kw] = items
            # 과도한 호출 방지(가벼운 딜레이)
            time.sleep(0.15)

    title = f"{date_str} 핫키워드 쇼핑 추천 TOP{len(keywords)} ({slot_label})"
    slug = f"hotkeyword-shop-{date_str}-{slot}"
    html = build_post_html(now, slot_label, keywords, products_map)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. HTML 미리보기 ↓\n")
        print(html)
        return

    existing = get_existing_post(cfg.sqlite_path, date_slot)
    if existing:
        post_id, old_link = existing
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html)
        save_post_meta(cfg.sqlite_path, date_slot, wp_post_id, wp_link)
        print("OK(updated):", wp_post_id, wp_link or old_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, html)
        save_post_meta(cfg.sqlite_path, date_slot, wp_post_id, wp_link)
        print("OK(created):", wp_post_id, wp_link)


def parse_args(argv: List[str]) -> Dict[str, Any]:
    out = {"slot": None, "dry_run": False, "debug": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--slot" and i + 1 < len(argv):
            out["slot"] = (argv[i + 1] or "").lower()
            i += 2
            continue
        if a == "--dry-run":
            out["dry_run"] = True
            i += 1
            continue
        if a == "--debug":
            out["debug"] = True
            i += 1
            continue
        i += 1
    return out


def main():
    args = parse_args(sys.argv[1:])
    cfg = load_cfg()

    if args["slot"] in ("am", "pm"):
        cfg.run.run_slot = args["slot"]
    if args["dry_run"]:
        cfg.run.dry_run = True
    if args["debug"]:
        cfg.run.debug = True

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
