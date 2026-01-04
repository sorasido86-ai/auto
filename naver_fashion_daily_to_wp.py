# -*- coding: utf-8 -*-
"""
naver_fashion_daily_to_wp.py
- 네이버 쇼핑 검색 API로 남/여 의류 검색 상단 TOP20을 매일 수집
- SQLite에 날짜별 랭킹 누적 저장
- 워드프레스 REST API로 데일리 포스트 발행(이미 발행된 날짜면 업데이트)

필요:
  pip install requests
"""

from __future__ import annotations

import base64
import html as htmlmod
import json
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from pathlib import Path


# -----------------------------
# Config
# -----------------------------
@dataclass
class NaverConfig:
    client_id: str
    client_secret: str
    display: int = 20
    sort: str = "sim"              # sim/date/asc/dsc
    filter: str = ""               # naverpay 등
    exclude: str = "used:rental:cbshop"
    queries: Dict[str, str] = field(default_factory=lambda: {"women": "여성의류", "men": "남성의류"})

@dataclass
class WordPressConfig:
    site_url: str
    username: str
    app_password: str
    status: str = "publish"
    category_ids: List[int] = field(default_factory=list)
    tag_ids: List[int] = field(default_factory=list)

@dataclass
class PostConfig:
    title_template: str = "{date} 네이버 쇼핑 남성/여성 의류 TOP20 (데일리)"
    disclosure: str = "※ 이 포스팅은 제휴마케팅 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받을 수 있습니다."
    note: str = "데이터 출처: 네이버 쇼핑 검색 API(정렬 기준: sim)."

@dataclass
class StorageConfig:
    sqlite_path: str = "naver_fashion_rankings.sqlite3"

@dataclass
class RunConfig:
    dry_run: bool = False

@dataclass
class AppConfig:
    naver: NaverConfig
    wordpress: WordPressConfig
    post: PostConfig
    storage: StorageConfig
    run: RunConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    n = raw["naver"]
    w = raw["wordpress"]
    p = raw.get("post", {})
    s = raw.get("storage", {})
    r = raw.get("run", {})

    return AppConfig(
        naver=NaverConfig(
            client_id=n["client_id"],
            client_secret=n["client_secret"],
            display=int(n.get("display", 20)),
            sort=n.get("sort", "sim"),
            filter=n.get("filter", ""),
            exclude=n.get("exclude", "used:rental:cbshop"),
            queries=n.get("queries", {"women": "여성의류", "men": "남성의류"}),
        ),
        wordpress=WordPressConfig(
            site_url=w["site_url"].rstrip("/"),
            username=w["username"],
            app_password=w["app_password"],
            status=w.get("status", "publish"),
            category_ids=w.get("category_ids", []),
            tag_ids=w.get("tag_ids", []),
        ),
        post=PostConfig(
            title_template=p.get("title_template", "{date} 네이버 쇼핑 남성/여성 의류 TOP20 (데일리)"),
            disclosure=p.get("disclosure", "※ 이 포스팅은 제휴마케팅 활동의 일환으로, 이에 따른 일정액의 수수료를 제공받을 수 있습니다."),
            note=p.get("note", "데이터 출처: 네이버 쇼핑 검색 API(정렬 기준: sim)."),
        ),
        storage=StorageConfig(sqlite_path=s.get("sqlite_path", "naver_fashion_rankings.sqlite3")),
        run=RunConfig(dry_run=bool(r.get("dry_run", False))),
    )


# -----------------------------
# Naver Shopping Search API
# -----------------------------
NAVER_SHOP_ENDPOINT = "https://openapi.naver.com/v1/search/shop.json"

def naver_shop_search(session: requests.Session, cfg: NaverConfig, query: str) -> List[Dict[str, Any]]:
    headers = {
        "X-Naver-Client-Id": cfg.client_id,
        "X-Naver-Client-Secret": cfg.client_secret,
        "User-Agent": "Mozilla/5.0 (compatible; naver-fashion-bot/1.0)",
    }
    params: Dict[str, Any] = {
        "query": query,
        "display": cfg.display,
        "start": 1,
        "sort": cfg.sort,
    }
    if cfg.filter:
        params["filter"] = cfg.filter
    if cfg.exclude:
        params["exclude"] = cfg.exclude

    resp = session.get(NAVER_SHOP_ENDPOINT, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("items", [])
    return items if isinstance(items, list) else []


def clean_title(t: str) -> str:
    # title에 <b>태그가 들어올 수 있어서 제거
    t = htmlmod.unescape(t or "")
    t = re.sub(r"</?b>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -----------------------------
# Storage (SQLite)
# -----------------------------
def init_db(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_posts (
      date TEXT PRIMARY KEY,
      wp_post_id INTEGER,
      wp_link TEXT,
      created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rankings (
      date TEXT,
      gender TEXT,
      rank INTEGER,
      product_id TEXT,
      title TEXT,
      link TEXT,
      image TEXT,
      lprice INTEGER,
      mall_name TEXT,
      brand TEXT,
      maker TEXT,
      category1 TEXT,
      category2 TEXT,
      category3 TEXT,
      category4 TEXT,
      PRIMARY KEY (date, gender, rank)
    )
    """)
    con.commit()
    con.close()


def upsert_rankings(db_path: str, date_str: str, gender: str, items: List[Dict[str, Any]]) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    for idx, it in enumerate(items, start=1):
        title = clean_title(str(it.get("title", "")))
        cur.execute("""
        INSERT OR REPLACE INTO rankings
        (date, gender, rank, product_id, title, link, image, lprice, mall_name, brand, maker, category1, category2, category3, category4)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_str,
            gender,
            idx,
            str(it.get("productId", "")),
            title,
            str(it.get("link", "")),
            str(it.get("image", "")),
            int(it.get("lprice", 0) or 0),
            str(it.get("mallName", "")),
            str(it.get("brand", "")),
            str(it.get("maker", "")),
            str(it.get("category1", "")),
            str(it.get("category2", "")),
            str(it.get("category3", "")),
            str(it.get("category4", "")),
        ))

    con.commit()
    con.close()


def get_existing_post(db_path: str, date_str: str) -> Optional[Tuple[int, str]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT wp_post_id, wp_link FROM daily_posts WHERE date = ?", (date_str,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return int(row[0]), str(row[1] or "")


def save_post_meta(db_path: str, date_str: str, wp_post_id: int, wp_link: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO daily_posts(date, wp_post_id, wp_link, created_at)
    VALUES (?, ?, ?, ?)
    """, (date_str, wp_post_id, wp_link, datetime.utcnow().isoformat()))
    con.commit()
    con.close()


# -----------------------------
# WordPress REST API
# -----------------------------
def wp_auth_header(username: str, app_password: str) -> Dict[str, str]:
    token = base64.b64encode(f"{username}:{app_password}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


def wp_create_post(cfg: WordPressConfig, title: str, html: str) -> Tuple[int, str]:
    endpoint = cfg.site_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.username, cfg.app_password), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str) -> Tuple[int, str]:
    endpoint = cfg.site_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.username, cfg.app_password), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    # WP는 보통 POST로도 업데이트가 동작함(환경에 따라 PUT도 가능)
    r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
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

def build_table(title: str, items: List[Dict[str, Any]]) -> str:
    rows = []
    for i, it in enumerate(items, start=1):
        name = clean_title(str(it.get("title", "")))
        link = str(it.get("link", ""))
        img = str(it.get("image", ""))
        lprice = int(it.get("lprice", 0) or 0)
        mall = htmlmod.escape(str(it.get("mallName", "")))

        img_html = f'<img src="{img}" alt="{htmlmod.escape(name)}" style="width:72px;height:auto;border-radius:10px;">' if img else ""
        name_html = f'<a href="{link}" target="_blank" rel="nofollow sponsored noopener">{htmlmod.escape(name)}</a>' if link else htmlmod.escape(name)

        rows.append(f"""
        <tr>
          <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
          <td style="padding:8px;border:1px solid #e5e5e5;">{img_html}</td>
          <td style="padding:8px;border:1px solid #e5e5e5;">{name_html}<div style="font-size:12px;opacity:.75;margin-top:4px;">{mall}</div></td>
          <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;">{fmt_krw(lprice)}</td>
        </tr>
        """)

    return f"""
    <h2>{htmlmod.escape(title)}</h2>
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
        {''.join(rows)}
      </tbody>
    </table>
    """

def build_post_html(date_str: str, post_cfg: PostConfig, women_items: List[Dict[str, Any]], men_items: List[Dict[str, Any]], sort: str) -> str:
    disclosure = f'<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{htmlmod.escape(post_cfg.disclosure)}</p>'
    note = f'<p style="font-size:13px;opacity:.8;">{htmlmod.escape(post_cfg.note).replace("sim", sort)}</p>'
    head = f"<p>기준일: <b>{date_str}</b></p>"

    women_html = build_table("여성의류 TOP 20", women_items)
    men_html = build_table("남성의류 TOP 20", men_items)

    return f"""
    {disclosure}
    {head}
    {note}
    {women_html}
    <hr/>
    {men_html}
    """


# -----------------------------
# Main (Daily)
# -----------------------------
def run_daily(cfg: AppConfig) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")

    init_db(cfg.storage.sqlite_path)

    with requests.Session() as session:
        women_q = cfg.naver.queries.get("women", "여성의류")
        men_q = cfg.naver.queries.get("men", "남성의류")

        women_items = naver_shop_search(session, cfg.naver, women_q)
        men_items = naver_shop_search(session, cfg.naver, men_q)

    # DB 누적
    upsert_rankings(cfg.storage.sqlite_path, date_str, "women", women_items)
    upsert_rankings(cfg.storage.sqlite_path, date_str, "men", men_items)

    # 포스트 HTML
    title = cfg.post.title_template.format(date=date_str)
    html = build_post_html(date_str, cfg.post, women_items, men_items, cfg.naver.sort)

    if cfg.run.dry_run:
        print(html)
        return

    existing = get_existing_post(cfg.storage.sqlite_path, date_str)
    if existing:
        post_id, _ = existing
        wp_post_id, wp_link = wp_update_post(cfg.wordpress, post_id, title, html)
        save_post_meta(cfg.storage.sqlite_path, date_str, wp_post_id, wp_link)
        print(f"OK(updated): {wp_post_id} {wp_link}")
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wordpress, title, html)
        save_post_meta(cfg.storage.sqlite_path, date_str, wp_post_id, wp_link)
        print(f"OK(created): {wp_post_id} {wp_link}")


def main():
    # 1) 인자 있으면 그걸 쓰고
    # 2) 없으면 "이 .py 파일이 있는 폴더"의 bot_config.json을 찾음
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = str(Path(__file__).resolve().parent / "bot_config.json")

    cfg = load_config(config_path)
    run_daily(cfg)

if __name__ == "__main__":
    main()
