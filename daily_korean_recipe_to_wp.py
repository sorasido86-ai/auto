# -*- coding: utf-8 -*-
"""daily_korean_recipe_to_wp.py (완전 통합 / 안정화 / 이미지 보장 / 글 정돈)

핵심 개선
- MFDS(식품안전나라) OpenAPI가 느리거나 타임아웃 나도 **절대 크래시하지 않고** 로컬 폴백으로 진행
- MFDS 호출 시간 예산(MFDS_MAX_SECONDS) + 타임아웃(MFDS_TIMEOUT_SEC)로 GitHub Actions에서 무한 대기 방지
- **이미지 항상 포함**: MFDS 대표/과정이미지 → DEFAULT_THUMB_URL → 자동 이미지(Unsplash/Pexels)
  + 가능하면 WP 미디어로 업로드 후 featured_media 설정
- 본문: 불필요 문구 제거(자동생성/기준시각/출처/지난 링크) + 중복 문장 방지 + 길이 축소
- 레시피/사진/본문 싱크 유지(선택된 레시피 제목으로 사진 검색)

필수 Secrets
- WP_BASE_URL, WP_USER, WP_APP_PASS

권장
- MFDS_API_KEY (없으면 로컬 레시피 폴백)
- DEFAULT_THUMB_URL (완전 보장용 기본 이미지)

옵션
- MFDS_TIMEOUT_SEC=12 (기본 12)
- MFDS_MAX_SECONDS=25 (기본 25)
- MFDS_ONLY=0 (1이면 MFDS 못 가져오면 "발행 스킵"하고 성공 종료)
- AUTO_IMAGE=1 (기본 1)  : 기본이미지도 없을 때 자동 이미지 생성(Unsplash/Pexels)
- PEXELS_API_KEY=... (있으면 Pexels 우선)

- UPLOAD_THUMB=1, SET_FEATURED=1, EMBED_IMAGE_IN_BODY=1
- NAVER_STYLE=1 (스타일 최소화)

- WP_STATUS=publish
- WP_CATEGORY_IDS=7
- WP_TAG_IDS=...

- SQLITE_PATH=data/daily_korean_recipe.sqlite3

- RUN_SLOT=day|am|pm
- FORCE_NEW=0|1
- DRY_RUN=0|1
- DEBUG=0|1
- AVOID_REPEAT_DAYS=90
- MAX_TRIES=20
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import random
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

KST = timezone(timedelta(hours=9))

# -----------------------------
# 내장 한식 레시피(폴백)
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "kimchi-jjigae",
        "title": "돼지고기 김치찌개",
        "ingredients": [
            ("신김치", "2컵"),
            ("돼지고기(앞다리/삼겹)", "200g"),
            ("양파", "1/2개"),
            ("대파", "1대"),
            ("두부", "1/2모"),
            ("고춧가루", "1큰술"),
            ("다진마늘", "1큰술"),
            ("국간장", "1큰술"),
            ("멸치다시마 육수(또는 물)", "700ml"),
        ],
        "steps": [
            "냄비에 돼지고기를 넣고 중불에서 기름이 살짝 돌 때까지 볶아주세요",
            "신김치를 넣고 2~3분 더 볶아 김치의 신맛을 한 번 눌러줘요",
            "고춧가루 다진마늘 국간장을 넣고 30초만 볶아 향을 내요",
            "육수를 붓고 10~12분 끓여요",
            "양파를 넣고 3분 두부를 넣고 2분 더 끓인 뒤 대파로 마무리해요",
        ],
        "image_url": "",
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1.5큰술"),
            ("고추장(선택)", "1/2큰술"),
            ("애호박", "1/3개"),
            ("양파", "1/3개"),
            ("두부", "1/2모"),
            ("대파", "1/2대"),
            ("다진마늘", "1작은술"),
            ("멸치다시마 육수(또는 물)", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 5분 끓여요",
            "양파 애호박 두부를 넣고 5~6분 더 끓여요",
            "대파 넣고 한 번만 더 끓인 뒤 간을 보고 마무리해요",
        ],
        "image_url": "",
    },
    {
        "id": "bulgogi",
        "title": "간장 불고기",
        "ingredients": [
            ("소고기 불고기용", "300g"),
            ("양파", "1/2개"),
            ("대파", "1대"),
            ("간장", "4큰술"),
            ("설탕", "1큰술"),
            ("다진마늘", "1큰술"),
            ("참기름", "1큰술"),
            ("후추", "약간"),
            ("물(또는 배즙)", "3큰술"),
        ],
        "steps": [
            "간장 설탕 다진마늘 참기름 물 후추로 양념장을 섞어요",
            "고기에 양념장을 넣고 15분 이상 재워둬요",
            "팬에 고기를 볶고 양파 대파를 넣어 숨이 죽을 때까지 볶아요",
        ],
        "image_url": "",
    },
]

KOREAN_NEGATIVE_KEYWORDS = ["파스타", "피자", "타코", "스시", "커리", "샌드위치", "버거", "샐러드"]


# -----------------------------
# Env helpers
# -----------------------------
def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
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


# -----------------------------
# Config
# -----------------------------
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
    run_slot: str = "day"  # day/am/pm
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    mfds_timeout_sec: int = 12
    mfds_max_seconds: int = 25
    mfds_only: bool = False


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True


@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""
    reuse_media_by_search: bool = True
    auto_image: bool = True
    pexels_api_key: str = ""


@dataclass
class ContentConfig:
    naver_style: bool = True
    hashtag_count: int = 8


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    recipe: RecipeSourceConfig
    img: ImageConfig
    content: ContentConfig
    sqlite_path: str


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=cat_ids,
            tag_ids=tag_ids,
        ),
        run=RunConfig(
            run_slot=run_slot,
            force_new=_env_bool("FORCE_NEW", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 20),
            mfds_timeout_sec=_env_int("MFDS_TIMEOUT_SEC", 12),
            mfds_max_seconds=_env_int("MFDS_MAX_SECONDS", 25),
            mfds_only=_env_bool("MFDS_ONLY", False),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
        ),
        img=ImageConfig(
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
            auto_image=_env_bool("AUTO_IMAGE", True),
            pexels_api_key=_env("PEXELS_API_KEY", ""),
        ),
        content=ContentConfig(
            naver_style=_env_bool("NAVER_STYLE", True),
            hashtag_count=_env_int("HASHTAG_COUNT", 8),
        ),
        sqlite_path=_env("SQLITE_PATH", "data/daily_korean_recipe.sqlite3"),
    )


def validate_cfg(cfg: AppConfig) -> None:
    missing = []
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

    print("[CFG] WP_BASE_URL:", cfg.wp.base_url or "MISSING")
    print("[CFG] WP_USER:", ok(cfg.wp.user))
    print("[CFG] WP_APP_PASS:", ok(cfg.wp.app_pass))
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CFG] WP_TAG_IDS:", cfg.wp.tag_ids)
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| STRICT_KOREAN:", cfg.recipe.strict_korean)
    print("[CFG] MFDS_TIMEOUT_SEC:", cfg.run.mfds_timeout_sec, "| MFDS_MAX_SECONDS:", cfg.run.mfds_max_seconds, "| MFDS_ONLY:", int(cfg.run.mfds_only))
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] AUTO_IMAGE:", int(cfg.img.auto_image), "| PEXELS_API_KEY:", ok(cfg.img.pexels_api_key))
    print("[CFG] UPLOAD_THUMB:", cfg.img.upload_thumb, "| SET_FEATURED:", cfg.img.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.img.embed_image_in_body)
    print("[CFG] REUSE_MEDIA_BY_SEARCH:", cfg.img.reuse_media_by_search)
    print("[CFG] NAVER_STYLE:", cfg.content.naver_style, "| HASHTAG_COUNT:", cfg.content.hashtag_count)


# -----------------------------
# SQLite
# -----------------------------
REQUIRED_COLS = {
    "date_slot": "TEXT PRIMARY KEY",
    "recipe_source": "TEXT",
    "recipe_id": "TEXT",
    "recipe_title": "TEXT",
    "wp_post_id": "INTEGER",
    "wp_link": "TEXT",
    "media_id": "INTEGER",
    "media_url": "TEXT",
    "created_at": "TEXT",
}


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_posts (
          date_slot TEXT PRIMARY KEY,
          recipe_source TEXT,
          recipe_id TEXT,
          recipe_title TEXT,
          wp_post_id INTEGER,
          wp_link TEXT,
          media_id INTEGER,
          media_url TEXT,
          created_at TEXT
        )
        """
    )
    con.commit()

    cur.execute("PRAGMA table_info(daily_posts)")
    existing = {row[1] for row in cur.fetchall()}
    for col, coldef in REQUIRED_COLS.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE daily_posts ADD COLUMN {col} {coldef}")
    con.commit()
    con.close()


def get_today_post(path: str, date_slot: str) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at
        FROM daily_posts WHERE date_slot = ?
        """,
        (date_slot,),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "date_slot": row[0],
        "recipe_source": row[1] or "",
        "recipe_id": row[2] or "",
        "recipe_title": row[3] or "",
        "wp_post_id": int(row[4] or 0),
        "wp_link": row[5] or "",
        "media_id": int(row[6] or 0),
        "media_url": row[7] or "",
        "created_at": row[8] or "",
    }


def save_post_meta(path: str, meta: Dict[str, Any]) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.get("date_slot", ""),
            meta.get("recipe_source", ""),
            meta.get("recipe_id", ""),
            meta.get("recipe_title", ""),
            int(meta.get("wp_post_id", 0) or 0),
            meta.get("wp_link", ""),
            int(meta.get("media_id", 0) or 0),
            meta.get("media_url", ""),
            meta.get("created_at", datetime.utcnow().isoformat()),
        ),
    )
    con.commit()
    con.close()


def get_recent_recipe_ids(path: str, days: int) -> List[Tuple[str, str]]:
    since = datetime.utcnow() - timedelta(days=days)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT recipe_source, recipe_id
        FROM daily_posts
        WHERE created_at IS NOT NULL AND created_at != '' AND created_at >= ?
        """,
        (since.isoformat(),),
    )
    rows = cur.fetchall()
    con.close()
    out: List[Tuple[str, str]] = []
    for s, rid in rows:
        if s and rid:
            out.append((str(s), str(rid)))
    return out


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/1.3"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, excerpt: str = "") -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, featured_media: int = 0, excerpt: str = "") -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if featured_media:
        payload["featured_media"] = featured_media
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_find_media_by_search(cfg: WordPressConfig, search: str) -> Optional[Tuple[int, str]]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass)
    params = {"search": search, "per_page": 10}
    r = requests.get(url, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        return None
    try:
        items = r.json()
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    it = items[0]
    mid = int(it.get("id") or 0)
    src = str(it.get("source_url") or "")
    if mid and src:
        return mid, src
    return None


def wp_upload_media_from_url(cfg: WordPressConfig, image_url: str, filename: str) -> Tuple[int, str]:
    r = requests.get(image_url, timeout=45)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code}")

    content = r.content
    ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()

    if not ctype:
        if filename.lower().endswith(".png"):
            ctype = "image/png"
        elif filename.lower().endswith((".jpg", ".jpeg")):
            ctype = "image/jpeg"
        else:
            ctype = "application/octet-stream"

    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
    }

    rr = requests.post(url, headers=headers, data=content, timeout=60)
    if rr.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {rr.status_code} body={rr.text[:500]}")
    data = rr.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# Recipe model / MFDS provider
# -----------------------------
@dataclass
class Recipe:
    source: str  # mfds|local
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    image_url: str = ""
    step_images: List[str] = field(default_factory=list)

    def uid(self) -> str:
        s = f"{self.source}|{self.recipe_id}|{self.title}"
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _has_hangul(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))


def _is_korean_recipe_name(name: str, strict: bool = True) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if strict and not _has_hangul(n):
        return False
    for bad in KOREAN_NEGATIVE_KEYWORDS:
        if bad in n:
            return False
    return True


def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int, end: int, timeout_sec: int, debug: bool) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    try:
        r = requests.get(url, timeout=timeout_sec)
    except requests.RequestException as e:
        if debug:
            print("[MFDS] request failed:", repr(e))
        return []
    if r.status_code != 200:
        if debug:
            print("[MFDS] bad status:", r.status_code)
        return []
    try:
        data = r.json()
    except Exception:
        if debug:
            print("[MFDS] json parse failed")
        return []
    co = data.get("COOKRCP01") or {}
    rows = co.get("row") or []
    return rows if isinstance(rows, list) else []


def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or ""
    title = str(row.get("RCP_NM") or "").strip()
    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()

    ingredients: List[str] = []
    for p in re.split(r"\s*,\s*", parts):
        p = p.strip()
        if p:
            ingredients.append(p)

    steps: List[str] = []
    step_imgs: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        img = str(row.get(f"MANUAL_IMG{str(i).zfill(2)}") or "").strip()
        if s:
            s = re.sub(r"[a-zA-Z]\s*$", "", s).strip()
            steps.append(s)
            if img.startswith("http"):
                step_imgs.append(img)

    img_main = str(row.get("ATT_FILE_NO_MAIN") or "").strip()
    if not img_main:
        img_main = str(row.get("ATT_FILE_NO_MK") or "").strip()

    return Recipe(
        source="mfds",
        recipe_id=rid or hashlib.sha1(title.encode("utf-8")).hexdigest()[:8],
        title=title,
        ingredients=ingredients,
        steps=steps,
        image_url=img_main if img_main.startswith("http") else "",
        step_images=step_imgs,
    )


def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡"]

    t0 = time.time()
    tries = 0

    while tries < cfg.run.max_tries and (time.time() - t0) < cfg.run.mfds_max_seconds:
        tries += 1
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(
            cfg.recipe.mfds_api_key,
            "RCP_NM",
            kw,
            start=1,
            end=60,
            timeout_sec=cfg.run.mfds_timeout_sec,
            debug=cfg.run.debug,
        )
        if not rows:
            continue
        random.shuffle(rows)
        for row in rows:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if cfg.recipe.strict_korean and not _is_korean_recipe_name(rcp.title, strict=True):
                continue
            if (rcp.source, rcp.recipe_id) in used:
                continue
            if not rcp.title or not rcp.steps:
                continue
            return rcp

    return None


def pick_recipe_local(recent_pairs: List[Tuple[str, str]]) -> Recipe:
    used = set(recent_pairs)
    pool = [x for x in LOCAL_KOREAN_RECIPES if ("local", str(x["id"])) not in used]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    pick = random.choice(pool)
    ing = [f"{a} - {b}".strip() for a, b in pick.get("ingredients", [])]
    steps = [str(s).strip() for s in pick.get("steps", []) if str(s).strip()]
    return Recipe(
        source="local",
        recipe_id=str(pick["id"]),
        title=str(pick["title"]),
        ingredients=ing,
        steps=steps,
        image_url=str(pick.get("image_url") or "").strip(),
        step_images=[],
    )


def get_recipe_by_id(cfg: AppConfig, source: str, recipe_id: str) -> Optional[Recipe]:
    if source == "local":
        for x in LOCAL_KOREAN_RECIPES:
            if str(x.get("id")) == recipe_id:
                ing = [f"{a} - {b}".strip() for a, b in x.get("ingredients", [])]
                steps = [str(s).strip() for s in x.get("steps", []) if str(s).strip()]
                return Recipe(
                    source="local",
                    recipe_id=recipe_id,
                    title=str(x.get("title") or ""),
                    ingredients=ing,
                    steps=steps,
                    image_url=str(x.get("image_url") or "").strip(),
                )
        return None

    if source == "mfds" and cfg.recipe.mfds_api_key:
        rows = mfds_fetch_by_param(
            cfg.recipe.mfds_api_key,
            "RCP_SEQ",
            recipe_id,
            start=1,
            end=5,
            timeout_sec=cfg.run.mfds_timeout_sec,
            debug=cfg.run.debug,
        )
        for row in rows:
            rcp = mfds_row_to_recipe(row)
            if rcp.recipe_id == recipe_id:
                return rcp
        return None

    return None


# -----------------------------
# Content helpers
# -----------------------------
def _esc(s: str) -> str:
    return html.escape(s or "")


def _no_punct(s: str) -> str:
    # 마침표 계열 제거 + 느낌표/물음표 제거
    if not s:
        return ""
    for ch in [".", "!", "?", "…", "。", ";", ":"]:
        s = s.replace(ch, "")
    return s


def _clean_title_tokens(title: str) -> List[str]:
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", title or "")
    toks = [x.strip() for x in t.split() if x.strip()]
    out: List[str] = []
    seen = set()
    for x in toks:
        if len(x) <= 1:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def build_hashtags(cfg: AppConfig, recipe: Recipe) -> str:
    base = ["한식레시피", "집밥", "오늘뭐먹지", "간단요리", "초간단레시피", "자취요리", "밥도둑", "요리기록"]
    for tok in _clean_title_tokens(recipe.title)[:4]:
        base.append(tok.replace(" ", ""))

    uniq: List[str] = []
    seen = set()
    for x in base:
        x = re.sub(r"\s+", "", x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    n = max(5, min(15, cfg.content.hashtag_count))
    return " ".join([f"#{x}" for x in uniq[:n]])


def build_post_title(date_str: str, slot_label: str, recipe_title: str) -> str:
    # 클릭 유도는 하되 레시피와 싱크 유지
    hooks = [
        f"{recipe_title} 오늘처럼 바쁜 날에 진짜 잘 먹히는 버전",
        f"{recipe_title} 간 맞추기 헷갈릴 때 딱 이 흐름만 기억",
        f"{recipe_title} 한 번만 이렇게 해보면 다음부터 훨씬 편해져요",
        f"{recipe_title} 재료 적게 써도 맛이 나는 포인트만 모아봤어요",
    ]
    hook = random.choice(hooks)
    return hook


# -----------------------------
# Image helpers
# -----------------------------
def _safe_slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "recipe"


def pexels_search_image(api_key: str, query: str, timeout_sec: int, debug: bool) -> Optional[str]:
    if not api_key:
        return None
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key, "User-Agent": "daily-korean-recipe-bot/1.3"}
    params = {"query": query, "per_page": 5, "orientation": "landscape"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout_sec)
    except requests.RequestException as e:
        if debug:
            print("[PEXELS] request failed:", repr(e))
        return None
    if r.status_code != 200:
        if debug:
            print("[PEXELS] bad status:", r.status_code)
        return None
    try:
        data = r.json()
    except Exception:
        return None
    photos = data.get("photos") or []
    if not photos:
        return None
    src = (photos[0].get("src") or {})
    # 가능한 큰 이미지
    for k in ("large2x", "large", "original"):
        u = str(src.get(k) or "").strip()
        if u.startswith("http"):
            return u
    return None


def auto_image_url(cfg: AppConfig, recipe_title: str) -> str:
    # 1) Pexels(있을 때)
    q = f"{recipe_title} korean food"
    u = pexels_search_image(cfg.img.pexels_api_key, q, timeout_sec=min(12, cfg.run.mfds_timeout_sec + 2), debug=cfg.run.debug)
    if u:
        return u
    # 2) Unsplash source
    return f"https://source.unsplash.com/1200x800/?korean-food,{quote(recipe_title)}"


def choose_best_image_url(cfg: AppConfig, recipe: Recipe) -> str:
    # MFDS 대표 → MFDS 과정이미지 → DEFAULT → 자동
    if (recipe.image_url or "").strip().startswith("http"):
        return recipe.image_url.strip()
    for u in (recipe.step_images or [])[:3]:
        if (u or "").strip().startswith("http"):
            return u.strip()
    if (cfg.img.default_thumb_url or "").strip().startswith("http"):
        return cfg.img.default_thumb_url.strip()
    if cfg.img.auto_image:
        return auto_image_url(cfg, recipe.title)
    return ""


def ensure_media(cfg: AppConfig, image_url: str, stable_name: str) -> Tuple[int, str]:
    if not image_url:
        return 0, ""

    h = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    filename = f"{stable_name}_{h}.jpg"

    if cfg.img.reuse_media_by_search:
        found = wp_find_media_by_search(cfg.wp, search=f"{stable_name}_{h}")
        if found:
            return found

    mid, murl = wp_upload_media_from_url(cfg.wp, image_url, filename)
    return mid, murl


# -----------------------------
# Render body
# -----------------------------
def build_body_html(cfg: AppConfig, recipe: Recipe, display_img_url: str) -> Tuple[str, str]:
    title = recipe.title.strip()

    # 짧은 도입부(홈피드 느낌)
    intro_lines = [
        f"{title} 생각나면 괜히 마음이 바빠지죠",
        "오늘은 길게 설명 안 하고 바로 따라할 수 있게 흐름만 딱 정리해둘게요",
        "저도 이런 글 하나 저장해두면 다음번엔 훨씬 편해져서 자주 꺼내보게 되더라고요",
    ]
    intro = " ".join(intro_lines)
    intro = _no_punct(intro)

    # 중복 방지용 팁 풀
    tips_pool = [
        "불은 강불로 오래 끌고 가기보다 중불로 안정적으로 가는 게 결과가 좋아요",
        "간은 중간에 확정하지 말고 마지막에만 살짝 잡는 게 안전해요",
        "향은 마지막에 살아나니까 완성 직전에 한 번만 맛을 보면 돼요",
        "양념은 한 번에 많이 넣지 말고 조금씩 추가하는 쪽이 실패가 덜해요",
        "뭔가 아쉬우면 소금부터 올리기보다 향부터 살려보면 깔끔해져요",
    ]
    tips = random.sample(tips_pool, k=3)
    tips = [_no_punct(x) for x in tips]

    # 재료/순서
    ing_lines = recipe.ingredients or []
    if not ing_lines:
        ing_lines = ["재료 정보가 비어 있어요"]

    step_lines = recipe.steps or []
    if not step_lines:
        step_lines = ["조리 과정 정보가 비어 있어요"]

    # 이미지(상단 고정)
    img_html = ""
    if cfg.img.embed_image_in_body and display_img_url:
        img_html = f'<p><img src="{_esc(display_img_url)}" alt="{_esc(title)}"/></p>'

    # 3개 굵은 소제목(정돈)
    sec1 = "".join(
        [
            "<h3><b>오늘의 포인트</b></h3>",
            f"<p>{_esc(intro)}</p>",
            "<ul>" + "".join([f"<li>{_esc(t)}</li>" for t in tips]) + "</ul>",
        ]
    )

    sec2_intro = _no_punct("재료는 복잡하게 늘리지 말고 집에 있는 걸 중심으로 가도 충분해요")
    sec2 = "".join(
        [
            "<h3><b>재료 준비</b></h3>",
            f"<p>{_esc(sec2_intro)}</p>",
            "<ul>" + "".join([f"<li>{_esc(_no_punct(x))}</li>" for x in ing_lines]) + "</ul>",
        ]
    )

    sec3_intro = _no_punct("순서는 길게 외울 필요 없어요 그대로 따라만 가면 돼요")
    sec3 = "".join(
        [
            "<h3><b>만드는 순서</b></h3>",
            f"<p>{_esc(sec3_intro)}</p>",
            "<ol>" + "".join([f"<li>{_esc(_no_punct(s))}</li>" for s in step_lines]) + "</ol>",
        ]
    )

    # 마무리
    closing_lines = [
        _no_punct("저장해두면 다음에 오늘 뭐 먹지 할 때 바로 꺼내 쓰기 좋아요"),
        _no_punct(f"{title} 만들 때 여러분은 어떤 재료를 추가하는 편이세요 버섯 두부 대파 조합도 좋아요"),
    ]
    closing = "".join([f"<p>{_esc(x)}</p>" for x in closing_lines])

    hashtags = build_hashtags(cfg, recipe)
    tag_block = f"<p>{_esc(hashtags)}</p>" if hashtags else ""

    body = img_html + sec1 + sec2 + sec3 + closing + "<hr/>" + tag_block

    excerpt = _no_punct(f"{title} 레시피 재료와 순서를 짧게 정리했어요")
    excerpt = excerpt[:140]

    return body, excerpt


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = {"day": "오늘", "am": "오전", "pm": "오후"}.get(slot, "오늘")
    date_slot = f"{date_str}_{slot}"

    init_db(cfg.sqlite_path)

    today_meta = get_today_post(cfg.sqlite_path, date_slot)
    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days)

    print(f"[RUN] slot={slot} force_new={int(cfg.run.force_new)} date_slot={date_slot}")

    chosen: Optional[Recipe] = None
    if today_meta and not cfg.run.force_new and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

    if not chosen:
        chosen = pick_recipe_mfds(cfg, recent_pairs)
        if not chosen:
            if cfg.run.mfds_only:
                print("[SKIP] MFDS_ONLY=1 이고 MFDS 레시피를 가져오지 못해서 오늘 발행을 건너뜁니다")
                return
            chosen = pick_recipe_local(recent_pairs)

    assert chosen is not None

    if cfg.run.debug:
        print(f"[RECIPE] source={chosen.source} id={chosen.recipe_id} title={chosen.title} img={'Y' if chosen.image_url else 'N'} step_imgs={len(chosen.step_images)}")

    title = build_post_title(date_str, slot_label, chosen.title)
    slug = f"korean-recipe-{date_str}-{slot}"

    # 이미지 선정 + 미디어 업로드
    best_img = choose_best_image_url(cfg, chosen)

    media_id = 0
    media_url = ""

    if cfg.img.upload_thumb and best_img:
        try:
            stable_name = f"korean_recipe_{_safe_slug(chosen.uid())}"
            media_id, media_url = ensure_media(cfg, best_img, stable_name=stable_name)
        except Exception as e:
            if cfg.run.debug:
                print("[IMG] upload failed:", repr(e))
            media_id, media_url = 0, ""

    display_img_url = (media_url or best_img or "").strip()

    body_html, excerpt = build_body_html(cfg, chosen, display_img_url)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략 미리보기 ↓")
        print(body_html[:2000])
        print("... (truncated)")
        return

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    try:
        if today_meta and today_meta.get("wp_post_id"):
            post_id = int(today_meta["wp_post_id"])
            wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, featured_media=featured_id, excerpt=excerpt)
            print("OK(updated):", wp_post_id, wp_link)
        else:
            wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt)
            if featured_id:
                try:
                    wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, excerpt=excerpt)
                except Exception:
                    pass
            print("OK(created):", wp_post_id, wp_link)
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] post create/update failed, fallback to create:", repr(e))
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt)
        if featured_id:
            try:
                wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, excerpt=excerpt)
            except Exception:
                pass
        print("OK(created-fallback):", wp_post_id, wp_link)

    save_post_meta(
        cfg.sqlite_path,
        {
            "date_slot": date_slot,
            "recipe_source": chosen.source,
            "recipe_id": chosen.recipe_id,
            "recipe_title": chosen.title,
            "wp_post_id": wp_post_id,
            "wp_link": wp_link,
            "media_id": media_id,
            "media_url": media_url,
            "created_at": datetime.utcnow().isoformat(),
        },
    )


def main() -> None:
    cfg = load_cfg()
    validate_cfg(cfg)
    print_safe_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
