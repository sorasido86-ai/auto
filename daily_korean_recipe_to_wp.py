# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp.py (홈피드형 레시피 템플릿 + 안정화 + "목록/태그" 강화)
- 도입부 200~300자(텍스트 기준) + 굵은 소제목 3개
- 각 소제목 섹션 최소 글자수: SECTION_MIN_CHARS(기본 1200)  ← 사용자 요청 반영
- 친구에게 진심 담아 수다떠는 존댓말 톤
- 마침표(.) 없이, 여백/줄바꿈으로 호흡
- ✅ 레시피 "목록" 포함(재료/만드는 순서) + 앞에 특수문자(불릿/번호/체크) 없이 출력
- ✅ 해시태그(본문) 다수 삽입 (기본 18개)
- ✅ (선택) WP 태그 자동 생성/연결(AUTO_WP_TAGS=1, WP_TAG_NAMES=...)
- MFDS(식품안전나라) OpenAPI 우선, 느리면 로컬 폴백
- 이미지: MFDS 이미지 > DEFAULT_THUMB_URL > (없으면) AUTO_IMAGE(Unsplash)
- "자동생성/기준시각/출처/지난 레시피 링크" 기본 비노출
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
            ("돼지고기", "200g"),
            ("양파", "반 개"),
            ("대파", "한 대"),
            ("두부", "반 모"),
            ("고춧가루", "한 큰술"),
            ("다진마늘", "한 큰술"),
            ("국간장", "한 큰술"),
            ("멸치다시마 육수", "700ml"),
        ],
        "steps": [
            "냄비에 돼지고기를 넣고 중불에서 기름이 살짝 돌 때까지 볶아주세요",
            "신김치를 넣고 조금 더 볶아서 김치의 신맛을 한 번 눌러주세요",
            "고춧가루 다진마늘 국간장을 넣고 향이 올라올 때까지만 가볍게 볶아주세요",
            "육수를 붓고 한 번 끓인 뒤 약불로 줄여서 더 끓여주세요",
            "양파를 넣고 두부를 넣고 마지막에 대파로 마무리해요",
        ],
        "image_url": "",
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "한 큰술 반"),
            ("고추장", "반 큰술 선택"),
            ("애호박", "삼분의 일 개"),
            ("양파", "삼분의 일 개"),
            ("두부", "반 모"),
            ("대파", "반 대"),
            ("다진마늘", "한 작은술"),
            ("멸치다시마 육수", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 잠깐 끓여주세요",
            "양파 애호박 두부를 넣고 더 끓여주세요",
            "대파를 넣고 한 번 더 끓인 뒤 간을 보고 마무리해요",
        ],
        "image_url": "",
    },
    {
        "id": "bulgogi",
        "title": "간장 불고기",
        "ingredients": [
            ("소고기 불고기용", "300g"),
            ("양파", "반 개"),
            ("대파", "한 대"),
            ("간장", "네 큰술"),
            ("설탕", "한 큰술"),
            ("다진마늘", "한 큰술"),
            ("참기름", "한 큰술"),
            ("후추", "약간"),
            ("물 또는 배즙", "세 큰술"),
        ],
        "steps": [
            "간장 설탕 다진마늘 참기름 물 후추를 섞어서 양념장을 만들어주세요",
            "고기에 양념장을 넣고 재워두면 맛이 더 안정적으로 들어가요",
            "팬에 고기를 볶다가 양파 대파를 넣고 숨이 죽을 때까지 볶아주세요",
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

def _parse_str_list(csv: str) -> List[str]:
    out: List[str] = []
    for x in (csv or "").split(","):
        x = x.strip()
        if not x:
            continue
        out.append(x)
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
    auto_wp_tags: bool = False
    tag_names: List[str] = field(default_factory=list)

@dataclass
class RunConfig:
    run_slot: str = "day"       # day / am / pm
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 25

@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""      # foodsafetykorea openapi key (optional)
    strict_korean: bool = True
    mfds_timeout: int = 12
    mfds_budget_seconds: int = 20
    mfds_max_fails: int = 2

@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""
    reuse_media_by_search: bool = True
    auto_image: bool = True
    extra_image_count: int = 0

@dataclass
class ContentConfig:
    hashtag_count: int = 18
    # 길이 제어
    intro_min: int = 200
    intro_max: int = 300
    section_min_chars: int = 1200
    # 내부 링크(지난 레시피) 제거 기본
    add_internal_links: bool = False

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
    auto_wp_tags = _env_bool("AUTO_WP_TAGS", False)
    tag_names = _parse_str_list(_env("WP_TAG_NAMES", ""))

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
            auto_wp_tags=auto_wp_tags,
            tag_names=tag_names,
        ),
        run=RunConfig(
            run_slot=run_slot,
            force_new=_env_bool("FORCE_NEW", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 25),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout=_env_int("MFDS_TIMEOUT", 12),
            mfds_budget_seconds=_env_int("MFDS_BUDGET_SECONDS", 20),
            mfds_max_fails=_env_int("MFDS_MAX_FAILS", 2),
        ),
        img=ImageConfig(
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
            auto_image=_env_bool("AUTO_IMAGE", True),
            extra_image_count=_env_int("EXTRA_IMAGE_COUNT", 0),
        ),
        content=ContentConfig(
            hashtag_count=_env_int("HASHTAG_COUNT", 18),
            intro_min=_env_int("INTRO_MIN_CHARS", 200),
            intro_max=_env_int("INTRO_MAX_CHARS", 300),
            section_min_chars=_env_int("SECTION_MIN_CHARS", 1200),
            add_internal_links=_env_bool("ADD_INTERNAL_LINKS", False),
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
    print("[CFG] AUTO_WP_TAGS:", int(cfg.wp.auto_wp_tags), "| WP_TAG_NAMES:", cfg.wp.tag_names[:10])
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| STRICT_KOREAN:", cfg.recipe.strict_korean)
    print("[CFG] MFDS_TIMEOUT:", cfg.recipe.mfds_timeout, "| MFDS_BUDGET_SECONDS:", cfg.recipe.mfds_budget_seconds, "| MFDS_MAX_FAILS:", cfg.recipe.mfds_max_fails)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] UPLOAD_THUMB:", cfg.img.upload_thumb, "| SET_FEATURED:", cfg.img.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.img.embed_image_in_body)
    print("[CFG] AUTO_IMAGE:", cfg.img.auto_image, "| EXTRA_IMAGE_COUNT:", cfg.img.extra_image_count)
    print("[CFG] HASHTAG_COUNT:", cfg.content.hashtag_count, "| INTRO:", cfg.content.intro_min, "~", cfg.content.intro_max, "| SECTION_MIN_CHARS:", cfg.content.section_min_chars)
    print("[CFG] ADD_INTERNAL_LINKS:", int(cfg.content.add_internal_links))

# -----------------------------
# SQLite history
# -----------------------------
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/2.1"}

def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, excerpt: str = "", tags: Optional[List[int]] = None) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    tag_ids = tags if tags is not None else cfg.tag_ids
    if tag_ids:
        payload["tags"] = tag_ids
    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")

def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, featured_media: int = 0, excerpt: str = "", tags: Optional[List[int]] = None) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if featured_media:
        payload["featured_media"] = featured_media
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    tag_ids = tags if tags is not None else cfg.tag_ids
    if tag_ids:
        payload["tags"] = tag_ids
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
    r = requests.get(image_url, timeout=35)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code}")
    content = r.content
    ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
    if not ctype:
        if filename.lower().endswith(".png"):
            ctype = "image/png"
        else:
            ctype = "image/jpeg"
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

# --- WP Tag helpers (optional)
def wp_find_tag_id(cfg: WordPressConfig, name: str) -> int:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/tags"
    headers = wp_auth_header(cfg.user, cfg.app_pass)
    params = {"search": name, "per_page": 100}
    r = requests.get(url, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        return 0
    try:
        items = r.json()
    except Exception:
        return 0
    if not isinstance(items, list):
        return 0
    nlow = name.strip().lower()
    for it in items:
        if str(it.get("name") or "").strip().lower() == nlow:
            return int(it.get("id") or 0)
    # fallback first
    if items:
        return int(items[0].get("id") or 0)
    return 0

def wp_create_tag(cfg: WordPressConfig, name: str) -> int:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/tags"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload = {"name": name}
    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        return 0
    try:
        data = r.json()
        return int(data.get("id") or 0)
    except Exception:
        return 0

def wp_ensure_tag_ids(cfg: WordPressConfig, names: List[str]) -> List[int]:
    out: List[int] = []
    for nm in names:
        nm = nm.strip()
        if not nm:
            continue
        tid = wp_find_tag_id(cfg, nm)
        if not tid:
            tid = wp_create_tag(cfg, nm)
        if tid and tid not in out:
            out.append(tid)
    return out

# -----------------------------
# Recipe model / MFDS provider
# -----------------------------
@dataclass
class Recipe:
    source: str          # "mfds" or "local"
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    image_url: str = ""
    step_images: List[str] = field(default_factory=list)

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

def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int, end: int, timeout: int) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json()
        co = data.get("COOKRCP01") or {}
        rows = co.get("row") or []
        return rows if isinstance(rows, list) else []
    except Exception:
        return []

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
    start_ts = datetime.now(tz=KST)
    fails = 0

    for _ in range(cfg.run.max_tries):
        elapsed = (datetime.now(tz=KST) - start_ts).total_seconds()
        if elapsed >= cfg.recipe.mfds_budget_seconds:
            return None
        if fails >= cfg.recipe.mfds_max_fails:
            return None

        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, start=1, end=60, timeout=cfg.recipe.mfds_timeout)
        if not rows:
            fails += 1
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

def pick_recipe_local(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Recipe:
    used = set(recent_pairs)
    pool = [x for x in LOCAL_KOREAN_RECIPES if ("local", str(x["id"])) not in used]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    pick = random.choice(pool)
    ing = [f"{a} {b}".strip() for a, b in pick.get("ingredients", [])]
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

# -----------------------------
# Content helpers (homefeed)
# -----------------------------
def _esc(s: str) -> str:
    return html.escape(s or "")

def _strip_punct_for_homefeed(s: str) -> str:
    # 마침표/강한 문장부호 제거 + 불필요한 특수문자 완화
    s = (s or "").replace(".", " ")
    s = s.replace("!", " ").replace("?", " ")
    s = re.sub(r"[•●◆■▶✅✔︎✓\u2022]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _text_len(html_str: str) -> int:
    # 대충 태그 제거 후 길이
    txt = re.sub(r"<[^>]+>", "", html_str or "")
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return len(txt)

def _make_paragraphs(lines: List[str]) -> str:
    # 여백/호흡: 줄바꿈을 <br/><br/>로
    safe = [_esc(_strip_punct_for_homefeed(x)) for x in lines if (x or "").strip()]
    safe = [x for x in safe if x]
    if not safe:
        return ""
    return "<p>" + "<br/><br/>".join(safe) + "</p>"

def _ordinal_korean(i: int) -> str:
    words = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덟째", "아홉째", "열째",
             "열한째", "열두째", "열셋째", "열넷째", "열다섯째", "열여섯째", "열일곱째", "열여덟째", "열아홉째", "스무째"]
    return words[i-1] if 1 <= i <= len(words) else f"{i}번째"

def _render_recipe_list_no_bullets(recipe: Recipe) -> str:
    # "레시피 목록"을 요구대로 제공하되, 앞에 특수문자/불릿 없이
    ing_lines = []
    for x in recipe.ingredients[:30]:
        x = _strip_punct_for_homefeed(x)
        if x:
            ing_lines.append(x)
    step_lines = []
    for idx, s in enumerate(recipe.steps[:20], start=1):
        s = _strip_punct_for_homefeed(s)
        if s:
            step_lines.append(f"{_ordinal_korean(idx)} {s}")

    ing_html = "<p>" + "<br/>".join([_esc(x) for x in ing_lines]) + "</p>" if ing_lines else "<p>재료 정보가 비어있어요</p>"
    step_html = "<p>" + "<br/><br/>".join([_esc(x) for x in step_lines]) + "</p>" if step_lines else "<p>과정 정보가 비어있어요</p>"

    return (
        "<h3><strong>레시피 목록</strong></h3>"
        "<h4><strong>재료</strong></h4>"
        f"{ing_html}"
        "<h4><strong>만드는 순서</strong></h4>"
        f"{step_html}"
    )

def _clean_title_tokens(title: str) -> List[str]:
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", title or "")
    toks = [x.strip() for x in t.split() if x.strip()]
    out, seen = [], set()
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
    base = [
        "한식레시피", "집밥", "오늘뭐먹지", "간단요리", "초간단레시피",
        "자취요리", "밥도둑", "국물요리", "반찬", "요리기록",
        "한끼", "집들이", "레시피추천", "냉장고털기", "한식",
        "요리팁", "요리초보", "맛있는집밥",
    ]
    for tok in _clean_title_tokens(recipe.title)[:6]:
        base.append(tok.replace(" ", ""))

    # 재료 기반 태그 조금 추가(너무 과하지 않게)
    key_ing = []
    for x in recipe.ingredients[:12]:
        x = re.sub(r"\(.*?\)", " ", x)
        x = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", x)
        w = x.split()
        if w:
            key_ing.append(w[0])
    for w in key_ing[:6]:
        base.append(re.sub(r"\s+", "", w))

    uniq, seen = [], set()
    for x in base:
        x = re.sub(r"\s+", "", x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    n = max(10, min(30, cfg.content.hashtag_count))
    return " ".join([f"#{x}" for x in uniq[:n]])

# -----------------------------
# Image helpers (sync)
# -----------------------------
def choose_thumb_url(cfg: AppConfig, recipe: Recipe) -> str:
    if (recipe.image_url or "").strip():
        return recipe.image_url.strip()
    if (cfg.img.default_thumb_url or "").strip():
        return cfg.img.default_thumb_url.strip()
    if cfg.img.auto_image:
        q = recipe.title.strip()
        # Unsplash Source: 음식 키워드로 자동
        return f"https://source.unsplash.com/1200x800/?{quote(q)},korean,food"
    return ""

def ensure_media(cfg: AppConfig, image_url: str, stable_name: str) -> Tuple[int, str]:
    if not image_url:
        return 0, ""
    h = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    ext = ".jpg"
    u = image_url.lower()
    if u.endswith(".png"):
        ext = ".png"
    elif u.endswith(".jpeg"):
        ext = ".jpg"
    filename = f"{stable_name}_{h}{ext}"
    if cfg.img.reuse_media_by_search:
        found = wp_find_media_by_search(cfg.wp, search=f"{stable_name}_{h}")
        if found:
            return found
    mid, murl = wp_upload_media_from_url(cfg.wp, image_url, filename)
    return mid, murl

# -----------------------------
# Title templates
# -----------------------------
TITLE_BENEFIT = [
    "{t} 이번에 해먹었더니 돈 아끼는 느낌이라서 웃음 나왔어요",
    "{t} 집에서 해보면 생각보다 쉬워서 놀라실 거예요",
    "{t} 이렇게만 하면 실패 확률이 확 줄어요",
]
TITLE_CURIOSITY = [
    "{t} 왜 자꾸 당기는지 오늘에서야 알겠더라고요",
    "{t} 딱 한 가지 때문에 맛이 달라져요",
    "{t} 이 방법이 제일 무난하다는 말이 진짜였어요",
]
TITLE_THREAT = [
    "{t} 여기서 한 번만 실수하면 맛이 확 무너져요",
    "{t} 이거 모르고 하면 짜게 끝날 확률이 높아요",
    "{t} 불 조절을 놓치면 아쉬운 맛이 되더라고요",
]
TITLE_COMPARE = [
    "{t} 밖에서 사먹는 맛이랑 비교해보니 결론이 났어요",
    "{t} 배달 대신 해먹어보면 차이가 꽤 크게 느껴져요",
    "{t} 자주 먹는 메뉴일수록 집밥이 이득이더라고요",
]

def build_post_title(date_str: str, slot_label: str, recipe_title: str, style: str) -> str:
    t = recipe_title.strip()
    pools = {
        "benefit": TITLE_BENEFIT,
        "curiosity": TITLE_CURIOSITY,
        "threat": TITLE_THREAT,
        "compare": TITLE_COMPARE,
        "random": TITLE_BENEFIT + TITLE_CURIOSITY + TITLE_THREAT + TITLE_COMPARE,
    }
    pool = pools.get(style, pools["random"])
    return random.choice(pool).format(t=t) + f" {date_str} {slot_label}"

# -----------------------------
# Homefeed body generator
# -----------------------------
INTRO_TEMPLATES = [
    "요즘 하루가 좀 빠르게 지나가서 밥 챙기는 것도 은근 부담되더라고요 그래서 오늘은 {t}로 마음을 좀 다독이듯이 해먹어봤어요  뭔가 거창하게 하려는 날이 아니라  그냥 따뜻하고 익숙한 맛으로 정리하고 싶은 날 있잖아요  그런 날에 딱 좋아요  제가 해보면서 느꼈던 포인트도 같이 적어둘게요",
    "진짜 솔직히요  오늘은 머리가 복잡해서 간단하게라도 맛있는 걸 먹고 싶었어요 그래서 {t}를 꺼냈는데  해보니까 생각보다 어렵지 않아서 마음이 좀 놓이더라고요  괜히 요리 하나 잘 되면 하루가 정리되는 느낌 있잖아요  그 느낌 그대로 담아서 적어볼게요",
    "오늘은 {t}로 가자  이렇게 마음을 딱 정해두면 편하더라고요  재료도 크게 복잡하지 않고  불 조절만 잘 잡으면 거의 실패가 없어서요  저도 예전에 여러 번 삐끗했는데  오늘은 그 실수를 안 하게끔 흐름을 잡아드릴게요",
]

def _fit_intro(text: str, min_len: int, max_len: int) -> str:
    t = _strip_punct_for_homefeed(text)
    # 너무 짧으면 덧붙이기
    adds = [
        "괜히 과하게 욕심내지 말고  딱 먹고 싶은 만큼만 해도 충분하더라고요",
        "이렇게 적어두면 다음에 또 해먹을 때 정말 편해요  그래서 저도 메모처럼 남겨요",
        "오늘은 제 기준으로 가장 무난한 흐름만 잡았어요  너무 어렵게 안 갈게요",
    ]
    i = 0
    while len(t) < min_len and i < len(adds):
        t = (t + "  " + adds[i]).strip()
        i += 1
    # 너무 길면 컷
    if len(t) > max_len:
        t = t[:max_len].rstrip()
    return t

def _section_blocks(recipe: Recipe) -> Dict[str, List[str]]:
    t = recipe.title.strip()
    # 섹션용 블록: 중복 방지 위해 다양하게 준비
    blocks1 = [
        f"{t}를 좋아하시는 분들은 아마  첫맛이 확 들어오고  뒤에 여운이 남는 그 느낌 때문에 자꾸 찾게 되실 거예요  저도 딱 그래요",
        "요리할 때 제일 힘든 건  사실 손질이 아니라  중간에 간과 불을 어디서 잡느냐더라고요  오늘은 그 두 가지만 같이 잡아볼게요",
        "저는 요리할 때 마음이 급하면 꼭 실수를 하더라고요  그래서 오늘은 천천히  순서만 지키는 느낌으로 가는 게 핵심이에요",
        "한 번만 잘 해두면  다음엔 진짜 편해요  손이 기억하는 흐름이 생겨서요",
        "오늘은 맛을 과하게 꾸미지 말고  기본을 안정적으로 잡는 버전으로 갈게요",
    ]
    blocks2 = [
        "재료는 있는 걸로 충분히 되는데  딱 하나만 챙기면 맛이 확 안정되는 게 있더라고요  그 포인트를 같이 적어둘게요",
        "저는 재료를 완벽하게 맞추는 날보다  대체재로 정리해두는 날이 더 도움되더라고요  그래서 대체 방법도 같이 써둘게요",
        "불은 중불로 시작해서  뭔가 향이 올라오면  그때부터는 약불로 눌러주는 느낌이 좋아요  그게 제일 안전했어요",
        "간은 중간에 성급하게 맞추면  나중에 꼭 후회해요  마지막 한 번만 조절하는 게 훨씬 편해요",
        "딱 한 숟갈씩  조금씩만 움직여도 맛이 달라져요  저는 그게 요리의 재미라고 느껴요",
    ]
    blocks3 = [
        "다 하고 나면  그릇에 담는 순간에 기분이 좋아져요  내가 나를 챙겼다  이런 느낌이요",
        "남으면 다음날이 더 맛있을 때도 있잖아요  그래서 보관과 재가열도 같이 적어둘게요",
        "혹시 처음 하시다가 맛이 애매하면  그건 실패가 아니라  조절할 지점이 있다는 뜻이더라고요  그 지점을 알려드릴게요",
        "이거 한 번 저장해두면  다음에 고민 줄어들어요  저는 진짜 그게 제일 큰 이득이었어요",
        "먹는 사람 입맛은 다 다르니까  마지막에 내 입맛으로 살짝만 맞추는 게 정답이더라고요",
    ]
    return {"s1": blocks1, "s2": blocks2, "s3": blocks3}

def _expand_to_min(lines: List[str], min_chars: int, pool: List[str]) -> List[str]:
    # 중복 없이 확장
    used = set(lines)
    i = 0
    while True:
        cur = " ".join(lines)
        cur = re.sub(r"\s+", " ", cur).strip()
        if len(cur) >= min_chars:
            break
        if i >= len(pool):
            # 동적 문장 생성(중복 방지)
            dyn = f"제가 해보니까  같은 {random.choice(['순서', '불 조절', '간 조절', '타이밍'])}이라도  내 주방에서는 조금씩 다르게 느껴지더라고요  그래서 한 번 해보시고  다음엔 그 느낌만 메모해두시면 훨씬 편해요"
            if dyn not in used:
                lines.append(dyn)
                used.add(dyn)
            continue
        cand = pool[i]
        i += 1
        if cand in used:
            continue
        lines.append(cand)
        used.add(cand)
    return lines

def _recipe_sync_paragraph(recipe: Recipe) -> str:
    # 재료/과정을 대화형으로 한 번 더 풀어 쓰기(섹션2 핵심)
    t = recipe.title.strip()
    ing = [re.sub(r"\s+", " ", _strip_punct_for_homefeed(x)) for x in recipe.ingredients[:10]]
    stp = [re.sub(r"\s+", " ", _strip_punct_for_homefeed(x)) for x in recipe.steps[:6]]
    ing_sent = "  ".join([x for x in ing if x])
    stp_sent = "  ".join([x for x in stp if x])
    lines = [
        f"재료는 이렇게 생각하시면 편해요  {ing_sent}",
        f"흐름은 이렇게만 잡으셔도 돼요  {stp_sent}",
        f"그리고 저는 마지막에 간을 딱 한 번만 봐요  그게 제일 덜 흔들리더라고요",
    ]
    return "  ".join([x for x in lines if x])

def build_body_html(cfg: AppConfig, recipe: Recipe, display_img_url: str, slot_label: str) -> Tuple[str, str, List[str]]:
    # intro
    intro = random.choice(INTRO_TEMPLATES).format(t=recipe.title.strip())
    intro = _fit_intro(intro, cfg.content.intro_min, cfg.content.intro_max)

    # headings
    h1 = f"{recipe.title.strip()}  오늘은 이렇게 해보면 좋아요"
    h2 = f"재료랑 순서  제가 해보면서 제일 편했던 흐름이에요"
    h3 = f"맛이 흔들릴 때  제가 잡는 방법이 있어요"

    blocks = _section_blocks(recipe)

    # section contents (no bullet, no heavy punctuation)
    s1_lines = _expand_to_min([intro] + blocks["s1"], cfg.content.section_min_chars, blocks["s1"] + blocks["s2"])
    s2_seed = [_recipe_sync_paragraph(recipe), "그리고 아래에 레시피 목록을 한 번에 정리해둘게요  앞에 점 같은 건 안 붙이고 그냥 줄로만 갈게요"]
    s2_lines = _expand_to_min(s2_seed + blocks["s2"], cfg.content.section_min_chars, blocks["s2"] + blocks["s3"])
    s3_seed = ["마지막은 너무 거창하게 말 안 할게요  그냥 다음번에 더 편해지도록 정리하는 느낌이에요", "보관은 냉장 기준으로 잡고  다시 데울 때는 마지막에 간만 살짝 보시면 돼요"]
    s3_lines = _expand_to_min(s3_seed + blocks["s3"], cfg.content.section_min_chars, blocks["s3"] + blocks["s1"])

    # recipe list block (no bullets)
    recipe_list_html = _render_recipe_list_no_bullets(recipe)

    # image
    img_html = ""
    if cfg.img.embed_image_in_body and display_img_url:
        img_html = f'<p><img src="{_esc(display_img_url)}" alt="{_esc(recipe.title.strip())}"/></p>'

    # body composition (only 3 bold headings)
    body = ""
    body += img_html
    body += f"<h3><strong>{_esc(_strip_punct_for_homefeed(h1))}</strong></h3>"
    body += _make_paragraphs(s1_lines[1:])  # intro already inside list; keep flow
    body += f"<h3><strong>{_esc(_strip_punct_for_homefeed(h2))}</strong></h3>"
    body += _make_paragraphs(s2_lines)
    body += recipe_list_html
    body += f"<h3><strong>{_esc(_strip_punct_for_homefeed(h3))}</strong></h3>"
    body += _make_paragraphs(s3_lines)

    # tags (body)
    tags = build_hashtags(cfg, recipe)
    body += "<p><br/><br/>" + _esc(tags) + "</p>"

    # excerpt
    excerpt = f"{recipe.title.strip()} 레시피  오늘은 집에서 편하게 해먹는 흐름으로 정리했어요"
    excerpt = _strip_punct_for_homefeed(excerpt)[:140]

    # WP tag names suggestion list (optional)
    tag_names = _clean_title_tokens(recipe.title)[:6]
    tag_names += ["한식레시피", "집밥", "간단요리", "오늘뭐먹지", "요리팁", "자취요리"]
    # de-dup
    seen = set()
    out_names = []
    for x in tag_names:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out_names.append(x)

    return body, excerpt, out_names

# -----------------------------
# MFDS / local recipe getter
# -----------------------------
def get_recipe_by_id(cfg: AppConfig, source: str, recipe_id: str) -> Optional[Recipe]:
    if source == "local":
        for x in LOCAL_KOREAN_RECIPES:
            if str(x.get("id")) == recipe_id:
                ing = [f"{a} {b}".strip() for a, b in x.get("ingredients", [])]
                steps = [str(s).strip() for s in x.get("steps", []) if str(s).strip()]
                return Recipe(source="local", recipe_id=recipe_id, title=str(x.get("title") or ""), ingredients=ing, steps=steps, image_url=str(x.get("image_url") or "").strip())
        return None
    if source == "mfds" and cfg.recipe.mfds_api_key:
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_SEQ", recipe_id, start=1, end=5, timeout=cfg.recipe.mfds_timeout)
        for row in rows:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if rcp.recipe_id == recipe_id:
                return rcp
    return None

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

    title_style = (_env("TITLE_STYLE", "random") or "random").lower()
    if title_style not in ("benefit", "curiosity", "threat", "compare", "random"):
        title_style = "random"

    chosen: Optional[Recipe] = None
    if today_meta and not cfg.run.force_new and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

    if not chosen:
        chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(cfg, recent_pairs)

    assert chosen is not None

    # title/slug
    title = build_post_title(date_str, slot_label, chosen.title, title_style)
    slug = f"korean-recipe-{date_str}-{slot}"

    # image
    thumb_url = choose_thumb_url(cfg, chosen)

    media_id = 0
    media_url = ""
    if cfg.img.upload_thumb and thumb_url:
        try:
            media_id, media_url = ensure_media(cfg, thumb_url, stable_name="korean_recipe_thumb")
        except Exception:
            media_id, media_url = 0, ""

    display_img_url = (media_url or thumb_url or "").strip()

    # body
    body_html, excerpt, suggested_tag_names = build_body_html(cfg, chosen, display_img_url, slot_label)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. HTML 일부")
        print(body_html[:1500])
        return

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    # WP tags: explicit IDs + optional auto tag creation
    final_tag_ids = cfg.wp.tag_ids[:]
    if cfg.wp.auto_wp_tags:
        names = cfg.wp.tag_names[:] if cfg.wp.tag_names else []
        # title-based suggestions
        names += suggested_tag_names
        # de-dup
        seen = set()
        uniq = []
        for x in names:
            k = x.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(x.strip())
        try:
            auto_ids = wp_ensure_tag_ids(cfg.wp, uniq[:25])
            for tid in auto_ids:
                if tid and tid not in final_tag_ids:
                    final_tag_ids.append(tid)
        except Exception:
            pass

    try:
        if today_meta and today_meta.get("wp_post_id"):
            post_id = int(today_meta["wp_post_id"])
            wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, featured_media=featured_id, excerpt=excerpt, tags=final_tag_ids)
            print("OK(updated):", wp_post_id, wp_link, "| source=", chosen.source)
        else:
            wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt, tags=final_tag_ids)
            if featured_id:
                try:
                    wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, excerpt=excerpt, tags=final_tag_ids)
                except Exception:
                    pass
            print("OK(created):", wp_post_id, wp_link, "| source=", chosen.source)
    except Exception as e:
        # fallback create
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt, tags=final_tag_ids)
        if featured_id:
            try:
                wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, excerpt=excerpt, tags=final_tag_ids)
            except Exception:
                pass
        print("OK(created-fallback):", wp_post_id, wp_link, "| source=", chosen.source, "| err=", repr(e))

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

def main():
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
