# -*- coding: utf-8 -*-
"""
naver_fashion_daily_to_wp.py (완전 통합)
- 네이버 쇼핑 검색 API로 남/여 의류 TOP20 수집
- SQLite 누적 저장(월간 결산용)
- 워드프레스 REST API로 데일리 포스트 생성/업데이트
- GitHub Actions에서 bot_config.json 없이 Secrets(환경변수)로 실행 가능

✅ 너의 Secrets 이름 그대로 사용:
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

✅ 워드프레스 카테고리 기본값: ID 30
"""

from __future__ import annotations

import base64
import html as htmlmod
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------------
# Config
# -----------------------------
@dataclass
class NaverConfig:
    client_id: str
    client_secret: str
    display: int = 20
    sort: str = "sim"  # sim/date/asc/dsc
    filter: str = ""
    exclude: str = "used:rental:cbshop"
    queries: Dict[str, str] = field(default_factory=lambda: {"women": "여성의류", "men": "남성의류"})


@dataclass
class WordPressConfig:
    site_url: str
    username: str
    app_password: str
    status: str = "publish"
    # ✅ 기본 카테고리 ID 30
    category_ids: List[int] = field(default_factory=lambda: [30])
    tag_ids: List[int] = field(default_factory=list)


@dataclass
class PostConfig:
    title_template: str = "{date} 네이버 쇼핑 남성/여성 의류 TOP20 (데일리)"
    disclosure: str = "※ 이 포스팅은 데이터 순위를 기록하기 위한 자료입니다."
    note: str = "데이터 출처: 네이버 쇼핑 검색 API(정렬 기준: {sort})."


@dataclass
class StorageConfig:
    sqlite_path: str = "data/naver_fashion_rankings.sqlite3"


@dataclass
class RunConfig:
    dry_run: bool = False
    debug: bool = False


@dataclass
class AppConfig:
    naver: NaverConfig
    wordpress: WordPressConfig
    post: PostConfig
    storage: StorageConfig
    run: RunConfig


def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def cfg_from_env() -> AppConfig:
    """
    GitHub Actions(Secrets → env)용
    ✅ 너의 Secrets 이름:
      NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
      WP_BASE_URL, WP_USER, WP_APP_PASS

    (호환용 fallback도 포함: WP_SITE_URL/WP_USERNAME/WP_APP_PASSWORD)
    """
    naver_client_id = _env("NAVER_CLIENT_ID")
    naver_client_secret = _env("NAVER_CLIENT_SECRET")

    # ✅ 너의 Secrets 이름 우선
    wp_site_url = (_env("WP_BASE_URL") or _env("WP_SITE_URL")).rstrip("/")
    wp_username = _env("WP_USER") or _env("WP_USERNAME")
    wp_app_password = _env("WP_APP_PASS") or _env("WP_APP_PASSWORD")

    display = int(_env("NAVER_DISPLAY", "20") or "20")
    sort = _env("NAVER_SORT", "sim") or "sim"
    exclude = _env("NAVER_EXCLUDE", "used:rental:cbshop")
    filt = _env("NAVER_FILTER", "")

    women_q = _env("NAVER_WOMEN_QUERY", "여성의류") or "여성의류"
    men_q = _env("NAVER_MEN_QUERY", "남성의류") or "남성의류"

    sqlite_path = _env("SQLITE_PATH", "data/naver_fashion_rankings.sqlite3")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    dry_run = _env("DRY_RUN", "0") in ("1", "true", "True", "YES", "yes")
    debug = _env("DEBUG", "0") in ("1", "true", "True", "YES", "yes")

    return AppConfig(
        naver=NaverConfig(
            client_id=naver_client_id,
            client_secret=naver_client_secret,
            display=display,
            sort=sort,
            filter=filt,
            exclude=exclude,
            queries={"women": women_q, "men": men_q},
        ),
        wordpress=WordPressConfig(
            site_url=wp_site_url,
            username=wp_username,
            app_password=wp_app_password,
            status=wp_status,
            # ✅ 카테고리 30 고정
            category_ids=[30],
            tag_ids=[],
        ),
        post=PostConfig(),
        storage=StorageConfig(sqlite_path=sqlite_path),
        run=RunConfig(dry_run=dry_run, debug=debug),
    )


def load_config_from_file(path: str) -> AppConfig:
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
            # ✅ 파일에 category_ids가 있으면 그걸 쓰고, 없으면 기본값([30])
            category_ids=w.get("category_ids", [30]),
            tag_ids=w.get("tag_ids", []),
        ),
        post=PostConfig(
            title_template=p.get("title_template", PostConfig.title_template),
            disclosure=p.get("disclosure", PostConfig.disclosure),
            note=p.get("note", PostConfig.note),
        ),
        storage=StorageConfig(sqlite_path=s.get("sqlite_path", "data/naver_fashion_rankings.sqlite3")),
        run=RunConfig(
            dry_run=bool(r.get("dry_run", False)),
            debug=bool(r.get("debug", False)),
        ),
    )


def load_config_auto(cli_config_path: Optional[str] = None) -> AppConfig:
    """
    우선순위:
    1) CLI로 --config 지정
    2) 스크립트 폴더에 bot_config.json 있으면 로드
    3) 환경변수로 구성(GitHub Actions)
    """
    if cli_config_path:
        return load_config_from_file(cli_config_path)

    script_dir = Path(__file__).resolve().parent
    local_cfg = script_dir / "bot_config.json"
    if local_cfg.exists():
        return load_config_from_file(str(local_cfg))

    return cfg_from_env()


def validate_cfg(cfg: AppConfig) -> None:
    missing = []
    if not cfg.naver.client_id:
        missing.append("NAVER_CLIENT_ID")
    if not cfg.naver.client_secret:
        missing.append("NAVER_CLIENT_SECRET")

    if not cfg.wordpress.site_url:
        missing.append("WP_BASE_URL (또는 WP_SITE_URL / bot_config.json wordpress.site_url)")
    if not cfg.wordpress.username:
        missing.append("WP_USER (또는 WP_USERNAME / bot_config.json wordpress.username)")
    if not cfg.wordpress.app_password:
        missing.append("WP_APP_PASS (또는 WP_APP_PASSWORD / bot_config.json wordpress.app_password)")

    if missing:
        raise RuntimeError("필수 설정 누락:\n- " + "\n- ".join(missing))


def print_safe_settings(cfg: AppConfig) -> None:
    def ok(v: str) -> str:
        return f"OK(len={len(v)})" if v else "MISSING"

    print("[CONFIG] NAVER_CLIENT_ID:", ok(cfg.naver.client_id))
    print("[CONFIG] NAVER_CLIENT_SECRET:", ok(cfg.naver.client_secret))
    print("[CONFIG] WP_BASE_URL:", cfg.wordpress.site_url or "MISSING")
    print("[CONFIG] WP_USER:", ok(cfg.wordpress.username))
    print("[CONFIG] WP_APP_PASS:", ok(cfg.wordpress.app_password))
    print("[CONFIG] WP category_ids:", cfg.wordpress.category_ids)
    print("[CONFIG] queries:", cfg.naver.queries)
    print("[CONFIG] sort:", cfg.naver.sort, "display:", cfg.naver.display)
    print("[CONFIG] sqlite:", cfg.storage.sqlite_path)
    print("[CONFIG] dry_run:", cfg.run.dry_run, "debug:", cfg.run.debug)


# -----------------------------
# Naver Shopping Search API
# -----------------------------
NAVER_SHOP_ENDPOINT = "https://openapi.naver.com/v1/search/shop.json"


def clean_title(t: str) -> str:
    t = htmlmod.unescape(t or "")
    t = re.sub(r"</?b>", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def naver_shop_search(session: requests.Session, cfg: NaverConfig, query: str) -> List[Dict[str, Any]]:
    headers = {
        "X-Naver-Client-Id": cfg.client_id,
        "X-Naver-Client-Secret": cfg.client_secret,
        "User-Agent": "Mozilla/5.0 (compatible; naver-fashion-bot/1.0)",
    }
    params: Dict[str, Any] = {"query": query, "display": cfg.display, "start": 1, "sort": cfg.sort}
    if cfg.filter:
        params["filter"] = cfg.filter
    if cfg.exclude:
        params["exclude"] = cfg.exclude

    r = session.get(NAVER_SHOP_ENDPOINT, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Naver API failed: {r.status_code} body={r.text[:300]}")
    data = r.json()
    items = data.get("items", [])
    return items if isinstance(items, list) else []


# -----------------------------
# Storage (SQLite)
# -----------------------------
def init_db(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS daily_posts (
      date TEXT PRIMARY KEY,
      wp_post_id INTEGER,
      wp_link TEXT,
      created_at TEXT
    )
    """
    )
    cur.execute(
        """
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
    """
    )
    con.commit()
    con.close()


def upsert_rankings(db_path: str, date_str: str, gender: str, items: List[Dict[str, Any]]) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    for idx, it in enumerate(items, start=1):
        title = clean_title(str(it.get("title", "")))
        cur.execute(
            """
        INSERT OR REPLACE INTO rankings
        (date, gender, rank, product_id, title, link, image, lprice, mall_name, brand, maker, category1, category2, category3, category4)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
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
            ),
        )

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
    cur.execute(
        """
    INSERT OR REPLACE INTO daily_posts(date, wp_post_id, wp_link, created_at)
    VALUES (?, ?, ?, ?)
    """,
        (date_str, wp_post_id, wp_link, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


# -----------------------------
# WordPress REST API
# -----------------------------
def wp_auth_header(username: str, app_password: str) -> Dict[str, str]:
    token = base64.b64encode(f"{username}:{app_password}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "naver-fashion-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, html: str) -> Tuple[int, str]:
    endpoint = cfg.site_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.username, cfg.app_password), "Content-Type": "application/json"}

    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}
    # ✅ 카테고리/태그 적용
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:300]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str) -> Tuple[int, str]:
    endpoint = cfg.site_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.username, cfg.app_password), "Content-Type": "application/json"}

    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}
    # ✅ 카테고리/태그 적용(업데이트에도 반영)
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

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


def build_table(title: str, items: List[Dict[str, Any]]) -> str:
    rows = []
    for i, it in enumerate(items, start=1):
        name = clean_title(str(it.get("title", "")))
        link = str(it.get("link", ""))
        img = str(it.get("image", ""))
        lprice = int(it.get("lprice", 0) or 0)
        mall = htmlmod.escape(str(it.get("mallName", "")))

        img_html = f'<img src="{img}" alt="{htmlmod.escape(name)}" style="width:72px;height:auto;border-radius:10px;">' if img else ""
        name_html = (
            f'<a href="{link}" target="_blank" rel="nofollow sponsored noopener">{htmlmod.escape(name)}</a>'
            if link
            else htmlmod.escape(name)
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
          <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;">{fmt_krw(lprice)}</td>
        </tr>
        """
        )

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
    head = f"<p>기준일: <b>{date_str}</b></p>"
    note = f'<p style="font-size:13px;opacity:.8;">{htmlmod.escape(post_cfg.note.format(sort=sort))}</p>'

    women_html = build_table("여성의류 TOP 20", women_items)
    men_html = build_table("남성의류 TOP 20", men_items)

    return f"{disclosure}{head}{note}{women_html}<hr/>{men_html}"


# -----------------------------
# Debug self-test
# -----------------------------
def debug_test_naver(cfg: AppConfig) -> None:
    print("[DEBUG] Testing Naver API...")
    with requests.Session() as s:
        items = naver_shop_search(s, cfg.naver, "여성의류")
    print(f"[DEBUG] Naver OK. items={len(items)}")


def debug_test_wp(cfg: AppConfig) -> None:
    print("[DEBUG] Testing WordPress REST...")
    site = cfg.wordpress.site_url.rstrip("/")
    r = requests.get(site + "/wp-json/", timeout=20)
    print("[DEBUG] wp-json status:", r.status_code)

    headers = wp_auth_header(cfg.wordpress.username, cfg.wordpress.app_password)
    r2 = requests.get(site + "/wp-json/wp/v2/users/me", headers=headers, timeout=20)
    print("[DEBUG] users/me status:", r2.status_code)
    if r2.status_code != 200:
        print("[DEBUG] users/me body head:", r2.text[:300])
        raise RuntimeError("WordPress 인증 실패(401/403 가능). WP_USER/WP_APP_PASS 또는 REST 차단(보안플러그인) 확인 필요.")
    print("[DEBUG] WordPress auth OK.")


# -----------------------------
# Main logic
# -----------------------------
def run_daily(cfg: AppConfig) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")

    init_db(cfg.storage.sqlite_path)

    with requests.Session() as session:
        women_q = cfg.naver.queries.get("women", "여성의류")
        men_q = cfg.naver.queries.get("men", "남성의류")

        women_items = naver_shop_search(session, cfg.naver, women_q)
        men_items = naver_shop_search(session, cfg.naver, men_q)

    upsert_rankings(cfg.storage.sqlite_path, date_str, "women", women_items)
    upsert_rankings(cfg.storage.sqlite_path, date_str, "men", men_items)

    title = cfg.post.title_template.format(date=date_str)
    html = build_post_html(date_str, cfg.post, women_items, men_items, cfg.naver.sort)

    if cfg.run.dry_run:
        print("[DRY_RUN] Posting skipped. HTML preview below:\n")
        print(html)
        return

    existing = get_existing_post(cfg.storage.sqlite_path, date_str)
    if existing:
        post_id, old_link = existing
        wp_post_id, wp_link = wp_update_post(cfg.wordpress, post_id, title, html)
        save_post_meta(cfg.storage.sqlite_path, date_str, wp_post_id, wp_link)
        print(f"OK(updated): {wp_post_id} {wp_link or old_link}")
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wordpress, title, html)
        save_post_meta(cfg.storage.sqlite_path, date_str, wp_post_id, wp_link)
        print(f"OK(created): {wp_post_id} {wp_link}")


def parse_args(argv: List[str]) -> Dict[str, Any]:
    out = {"config": None, "dry_run": False, "debug": False}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--config", "-c") and i + 1 < len(argv):
            out["config"] = argv[i + 1]
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
    cfg = load_config_auto(args["config"])

    if args["dry_run"]:
        cfg.run.dry_run = True
    if args["debug"]:
        cfg.run.debug = True

    validate_cfg(cfg)
    print_safe_settings(cfg)

    if cfg.run.debug:
        debug_test_naver(cfg)
        debug_test_wp(cfg)

    run_daily(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
