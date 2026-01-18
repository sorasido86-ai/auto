# -*- coding: utf-8 -*-
"""daily_korean_recipe_to_wp.py (홈피드형 + 이미지 안정화 + 중복발행 지원)

- 한식 레시피 1건을 WordPress에 자동 발행
- 1순위: 식품안전나라(식약처) COOKRCP01 OpenAPI (MFDS_API_KEY)
- 2순위: 내장(폴백) 한식 레시피

핵심 개선
- MFDS API 타임아웃/장애 시에도 "즉시" 폴백(크래시 방지)
- 이미지: MFDS 이미지 > DEFAULT_THUMB_URL > OpenAI 이미지(선택) > Unsplash(최후)
  - 가능하면 WP Media 업로드 + featured_media 설정 + 본문 상단 <img> 삽입
  - https 강제(혼합콘텐츠로 엑박 뜨는 케이스 완화)
- 글 구성: 홈피드형 도입부(200~300자) + 굵은 소제목 3개 + 총 글자수 MIN_TOTAL_CHARS(기본 1200)
  - 마침표( . ) 없이, 친구에게 수다 떠는 존댓말 느낌
  - 불릿/점찍기 특수문자(•, -, ✅ 등) 없이 레시피 목록 제공
  - "자동 생성", "기준시각/슬롯", "출처", "지난 글 링크" 문구 제거
- 중복 발행: ALLOW_DUPLICATE_POSTS=1 이면 같은 날/같은 슬롯에도 매번 새 글 생성(슬러그에 run_id 부여)

필수 Secrets
- WP_BASE_URL (예: https://example.com)
- WP_USER
- WP_APP_PASS

권장
- DEFAULT_THUMB_URL (기본 대표이미지)

선택(이미지 생성)
- USE_OPENAI_IMAGE=1
- OPENAI_API_KEY
- OPENAI_IMAGE_MODEL (기본 gpt-image-1)

기타
- WP_STATUS=publish (기본 publish)
- WP_CATEGORY_IDS=7
- WP_TAG_IDS=1,2,3 (옵션)
- AUTO_WP_TAGS=1 (기본 1)  # 태그 이름을 자동 생성/연결
- SQLITE_PATH=data/daily_korean_recipe.sqlite3

- RUN_SLOT=day|am|pm
- FORCE_NEW=0|1  # 같은 슬롯 글이 있어도 "새 레시피"로 교체(업데이트)할지
- ALLOW_DUPLICATE_POSTS=0|1  # 1이면 교체가 아니라 "새 글"로 중복 발행

- MIN_TOTAL_CHARS=1200

MFDS 안정화
- MFDS_TIMEOUT_SEC=8
- MFDS_BUDGET_SEC=15
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
from requests.exceptions import RequestException

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
            ("양파", "반개"),
            ("대파", "1대"),
            ("두부", "반모"),
            ("고춧가루", "1큰술"),
            ("다진마늘", "1큰술"),
            ("국간장", "1큰술"),
            ("육수 또는 물", "700ml"),
        ],
        "steps": [
            "냄비에 돼지고기를 넣고 중불에서 살짝 볶아 기름이 돌게 해요",
            "김치를 넣고 2분 정도 더 볶아서 신맛을 한 번 눌러줘요",
            "고춧가루 다진마늘 국간장을 넣고 30초만 볶아서 향을 올려요",
            "육수를 붓고 10분 정도 끓여요",
            "양파를 넣고 3분 더 끓인 뒤 두부를 넣고 2분만 더  마지막에 대파로 마무리해요",
        ],
        "image_url": "",
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1과 1 2큰술"),
            ("고추장", "반큰술"),
            ("애호박", "3분의 1개"),
            ("양파", "3분의 1개"),
            ("두부", "반모"),
            ("대파", "반대"),
            ("다진마늘", "1작은술"),
            ("육수 또는 물", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 5분 정도만 먼저 끓여요",
            "양파 애호박을 넣고 4분 정도 끓여요",
            "두부를 넣고 2분 더  대파와 다진마늘을 넣고 한 번만 더 끓여요",
            "간을 보고 부족하면 된장을 아주 조금만 추가해요",
        ],
        "image_url": "",
    },
    {
        "id": "bulgogi",
        "title": "간장 불고기",
        "ingredients": [
            ("소고기 불고기용", "300g"),
            ("양파", "반개"),
            ("대파", "1대"),
            ("간장", "4큰술"),
            ("설탕", "1큰술"),
            ("다진마늘", "1큰술"),
            ("참기름", "1큰술"),
            ("후추", "약간"),
            ("물 또는 배즙", "3큰술"),
        ],
        "steps": [
            "간장 설탕 다진마늘 참기름 물 후추를 섞어서 양념장을 만들어요",
            "고기에 양념장을 넣고 15분 정도만 재워요",
            "팬을 중불로 달군 뒤 고기를 볶고  양파 대파를 넣어 숨이 죽을 때까지 더 볶아요",
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
    auto_wp_tags: bool = True


@dataclass
class RunConfig:
    run_slot: str = "day"  # day/am/pm
    force_new: bool = False
    allow_duplicate_posts: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True
    mfds_timeout: int = 8
    mfds_budget: int = 15


@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""
    force_https_media: bool = True

    # OpenAI image
    use_openai_image: bool = False
    openai_api_key: str = ""
    openai_image_model: str = "gpt-image-1"
    auto_image: bool = True  # if no mfds/default, try openai then unsplash


@dataclass
class ContentConfig:
    min_total_chars: int = 1200
    intro_min_chars: int = 200
    intro_max_chars: int = 300


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

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=_env("WP_STATUS", "publish") or "publish",
            category_ids=_parse_int_list(_env("WP_CATEGORY_IDS", "7")),
            tag_ids=_parse_int_list(_env("WP_TAG_IDS", "")),
            auto_wp_tags=_env_bool("AUTO_WP_TAGS", True),
        ),
        run=RunConfig(
            run_slot=run_slot,
            force_new=_env_bool("FORCE_NEW", False),
            allow_duplicate_posts=_env_bool("ALLOW_DUPLICATE_POSTS", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout=_env_int("MFDS_TIMEOUT_SEC", 8),
            mfds_budget=_env_int("MFDS_BUDGET_SEC", 15),
        ),
        img=ImageConfig(
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            force_https_media=_env_bool("FORCE_HTTPS_MEDIA", True),
            use_openai_image=_env_bool("USE_OPENAI_IMAGE", False),
            openai_api_key=_env("OPENAI_API_KEY", ""),
            openai_image_model=_env("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            auto_image=_env_bool("AUTO_IMAGE", True),
        ),
        content=ContentConfig(
            min_total_chars=_env_int("MIN_TOTAL_CHARS", 1200),
            intro_min_chars=_env_int("INTRO_MIN_CHARS", 200),
            intro_max_chars=_env_int("INTRO_MAX_CHARS", 300),
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


def _ok(v: str) -> str:
    return f"OK(len={len(v)})" if v else "EMPTY"


def print_safe_cfg(cfg: AppConfig) -> None:
    print("[CFG] WP_BASE_URL:", "OK" if cfg.wp.base_url else "EMPTY")
    print("[CFG] WP_USER:", _ok(cfg.wp.user))
    print("[CFG] WP_APP_PASS:", _ok(cfg.wp.app_pass))
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CFG] WP_TAG_IDS:", cfg.wp.tag_ids)
    print("[CFG] AUTO_WP_TAGS:", cfg.wp.auto_wp_tags)
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new), "| ALLOW_DUPLICATE_POSTS:", int(cfg.run.allow_duplicate_posts))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", _ok(cfg.recipe.mfds_api_key), "| STRICT_KOREAN:", cfg.recipe.strict_korean)
    print("[CFG] MFDS_TIMEOUT_SEC:", cfg.recipe.mfds_timeout, "| MFDS_BUDGET_SEC:", cfg.recipe.mfds_budget)
    print("[CFG] DEFAULT_THUMB_URL:", _ok(cfg.img.default_thumb_url))
    print("[CFG] UPLOAD_THUMB:", cfg.img.upload_thumb, "| SET_FEATURED:", cfg.img.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.img.embed_image_in_body)
    print("[CFG] FORCE_HTTPS_MEDIA:", cfg.img.force_https_media)
    print("[CFG] USE_OPENAI_IMAGE:", cfg.img.use_openai_image, "| AUTO_IMAGE:", cfg.img.auto_image, "| OPENAI_API_KEY:", _ok(cfg.img.openai_api_key))
    print("[CFG] MIN_TOTAL_CHARS:", cfg.content.min_total_chars)


# -----------------------------
# SQLite (history)
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/2.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, excerpt: str = "", tags: Optional[List[int]] = None) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if tags:
        payload["tags"] = tags
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
    if tags:
        payload["tags"] = tags
    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media_bytes(cfg: WordPressConfig, image_bytes: bytes, filename: str, mime: str) -> Tuple[Optional[int], Optional[str]]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": mime,
    }
    r = requests.post(url, headers=headers, data=image_bytes, timeout=70)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data.get("id") or 0), str(data.get("source_url") or "")


def wp_find_tag_id(cfg: WordPressConfig, name: str) -> Optional[int]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/tags"
    headers = wp_auth_header(cfg.user, cfg.app_pass)
    r = requests.get(url, headers=headers, params={"search": name, "per_page": 20}, timeout=25)
    if r.status_code != 200:
        return None
    try:
        items = r.json()
    except Exception:
        return None
    for it in items if isinstance(items, list) else []:
        if str(it.get("name") or "").strip().lower() == name.strip().lower():
            tid = int(it.get("id") or 0)
            if tid:
                return tid
    # fallback: first result
    if isinstance(items, list) and items:
        tid = int(items[0].get("id") or 0)
        return tid or None
    return None


def wp_create_tag(cfg: WordPressConfig, name: str) -> Optional[int]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/tags"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json={"name": name}, timeout=25)
    if r.status_code not in (200, 201):
        return None
    try:
        return int((r.json() or {}).get("id") or 0) or None
    except Exception:
        return None


def wp_ensure_tag_ids(cfg: WordPressConfig, names: List[str]) -> List[int]:
    out: List[int] = []
    for name in names:
        n = (name or "").strip()
        if not n:
            continue
        tid = wp_find_tag_id(cfg, n)
        if not tid:
            tid = wp_create_tag(cfg, n)
        if tid and tid not in out:
            out.append(tid)
    return out


# -----------------------------
# Recipe model / MFDS provider
# -----------------------------

@dataclass
class Recipe:
    source: str
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    image_url: str = ""


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
    except RequestException:
        return []
    except Exception:
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
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        if s:
            # 끝에 영문 표기 등 정리
            s = re.sub(r"[a-zA-Z]\s*$", "", s).strip()
            steps.append(s)

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
    )


def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡"]

    t0 = time.monotonic()
    while time.monotonic() - t0 < max(3, cfg.recipe.mfds_budget):
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, start=1, end=60, timeout=cfg.recipe.mfds_timeout)
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
            if not rcp.title or len(rcp.steps) < 2:
                continue
            return rcp

    return None


def pick_recipe_local(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Recipe:
    used = set(recent_pairs)
    pool = [x for x in LOCAL_KOREAN_RECIPES if ("local", str(x.get("id"))) not in used]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    pick = random.choice(pool)

    ing = [f"{a} {b}".strip() for a, b in pick.get("ingredients", [])]
    steps = [str(s).strip() for s in pick.get("steps", []) if str(s).strip()]

    return Recipe(
        source="local",
        recipe_id=str(pick.get("id")),
        title=str(pick.get("title")),
        ingredients=ing,
        steps=steps,
        image_url=str(pick.get("image_url") or "").strip(),
    )


# -----------------------------
# Content builder (홈피드)
# -----------------------------

def _esc(s: str) -> str:
    return html.escape(s or "")


def _strip_ai_smell(s: str) -> str:
    # 마침표 제거 + 불릿/체크 특수문자 제거
    s = (s or "").replace(".", " ")
    s = re.sub(r"[•\u2022\u00b7\u25cf\u25aa\u25a0\u25a1\u2713\u2705\-\*]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _k_len(s: str) -> int:
    # HTML 태그 제외한 대략 글자수
    s2 = re.sub(r"<[^>]+>", "", s or "")
    s2 = re.sub(r"\s+", "", s2)
    return len(s2)


def _make_paragraphs(lines: List[str]) -> str:
    out = []
    for ln in lines:
        ln = _strip_ai_smell(ln)
        if not ln:
            continue
        out.append(f"<p>{_esc(ln)}</p>")
    return "".join(out)


def _intro_200_300(title: str) -> str:
    # 200~300자 정도로 맞추기
    base = (
        f"{title}는 이상하게요  바쁜 날에도 갑자기 생각나서 마음이 먼저 움직이잖아요  "
        f"저는 그런 날에 대충 때우면 꼭 후회가 남더라고요  그래서 오늘은 진짜 집에서 해먹는 흐름으로 "
        f"재료랑 순서만 딱 잡아드릴게요  읽다가 바로 장보고 바로 끓이게끔요"
    )
    # 길이 보정(마침표 없음)
    base = _strip_ai_smell(base)
    # 너무 길면 잘라내기
    if len(base) > 310:
        base = base[:300].rstrip()
    return base


def _expand_to_min(lines: List[str], min_chars: int, pool: List[str]) -> List[str]:
    out = [x for x in lines if x and x.strip()]
    seed = pool[:]
    random.shuffle(seed)
    i = 0
    while len("".join(out)) < min_chars and i < 200:
        out.append(seed[i % len(seed)])
        i += 1
    return out


def _title_variants(style: str, recipe_title: str) -> str:
    benefit = ["실패 줄이는 포인트", "간 딱 맞는 흐름", "집에서 끝내는 방법", "손 덜 가는 순서", "맛이 확 살아나는 타이밍"]
    b = random.choice(benefit)
    if style == "benefit":
        return f"{recipe_title} 레시피  {b}만 잡아드릴게요"
    if style == "threat":
        return f"{recipe_title}  여기서만 실수 안 하면 맛이 확 달라져요"
    if style == "curiosity":
        return f"{recipe_title}  왜 집에서 하면 더 맛있게 느껴질까요"
    if style == "compare":
        return f"{recipe_title}  집에서 한 버전과 밖에서 먹던 맛 차이"
    return f"{recipe_title} 레시피  오늘은 집밥으로 딱 정리해요"


def build_hashtags(recipe_title: str) -> str:
    base = [
        "한식레시피",
        "집밥",
        "오늘뭐먹지",
        "간단요리",
        "자취요리",
        "요리팁",
        "국물요리",
        "밥도둑",
        "집밥기록",
        "요리메모",
        "집에서요리",
        "한끼해결",
    ]
    # 제목에서 토큰 추가
    toks = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", recipe_title or "").split()
    for t in toks[:4]:
        t = t.strip()
        if len(t) >= 2:
            base.append(t)

    uniq = []
    seen = set()
    for x in base:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    return " ".join([f"#{x}" for x in uniq[:16]])


def render_recipe_list_no_bullets(recipe: Recipe) -> str:
    # 재료/레시피 목록을 불릿 없이  줄바꿈으로
    ing_lines = [f"{_strip_ai_smell(x)}" for x in (recipe.ingredients or []) if str(x).strip()]
    step_lines = [f"{_strip_ai_smell(s)}" for s in (recipe.steps or []) if str(s).strip()]

    ing_html = "<br/>".join([_esc(x) for x in ing_lines[:25]]) if ing_lines else _esc("재료 정보가 비어 있어요")

    step_render: List[str] = []
    for i, s in enumerate(step_lines[:12], start=1):
        # 숫자만으로 단계 표시
        step_render.append(f"{i} {s}")
    step_html = "<br/>".join([_esc(x) for x in step_render]) if step_render else _esc("조리 과정 정보가 비어 있어요")

    return (
        "<p><b>재료</b><br/>" + ing_html + "</p>"
        "<p><b>레시피 순서</b><br/>" + step_html + "</p>"
    )


def build_body_html(cfg: AppConfig, recipe: Recipe, display_img_url: str, title_style: str) -> Tuple[str, str, List[str]]:
    title = recipe.title.strip()

    # intro
    intro = _intro_200_300(title)

    # 3 headings
    h1 = f"{title}  오늘은 이 느낌으로 가요"
    h2 = "재료랑 순서만 깔끔하게 잡아드릴게요"
    h3 = "먹어보고 나서  다음번이 더 쉬워지는 포인트"

    # seed pools
    pool1 = [
        "저는 요리할 때 복잡한 말이 제일 힘들더라고요  그래서 오늘은 말 줄이고 흐름만 딱 잡아볼게요",
        "처음부터 완벽하게 하려는 마음이 오히려 손을 묶어요  한 번만 해보면 감이 바로 오거든요",
        "재료가 조금 모자라도 괜찮아요  중요한 건 마지막에 간을 맞추는 타이밍이더라고요",
        "한 숟갈 먹었을 때 아  이게 집밥이지 싶은 순간이 있어요  그 느낌이 오늘 목표예요",
        "혹시 오늘 기분이 좀 처지는 날이면요  뜨끈한 한 그릇이 생각보다 큰 위로가 되더라고요",
    ]
    pool2 = [
        "재료는 있는 것부터 꺼내두시면 마음이 진짜 편해져요  중간에 찾느라 흐름 끊기면 맛도 흔들려요",
        "불은 크게 안 써도 돼요  중불로만 차분하게 가면 실패 확률이 확 내려가요",
        "간은 중간에 자꾸 손대면 오히려 헷갈려요  마지막에 한 번만 딱 잡아도 충분해요",
        "냄새가 올라오는 순간이 있거든요  그때까지는 조급해하지 말고 기다리는 게 포인트예요",
        "오늘은 레시피대로만 따라가도 맛이 나게끔  순서를 최대한 단순하게 정리했어요",
    ]
    pool3 = [
        "남은 건 냉장에 넣어두고 다음날 데우면 더 맛있을 때가 있어요  특히 국물요리는요",
        "다음번에는 오늘 먹고 느낀 걸 한 줄만 메모해두세요  간을 조금 더  매운맛은 조금 덜  이런 식으로요",
        "제가 제일 자주 하는 실수는 마지막에 욕심내서 양념을 더 넣는 거예요  그 순간만 참으면 맛이 안정돼요",
        "먹고 나서 부담이 덜한 느낌을 원하시면요  기름은 조금 줄이고 채소를 조금만 더 넣어도 좋아요",
        "혹시 다른 재료 넣고 싶으시면 댓글로 알려주세요  저는 버섯이나 대파 많이 넣는 편이거든요",
    ]

    # build sections
    sec1 = [intro] + _expand_to_min(pool1[:2], 350, pool1)
    recipe_list = render_recipe_list_no_bullets(recipe)
    sec2 = _expand_to_min(pool2[:3], 350, pool2)
    sec3 = _expand_to_min(pool3[:3], 350, pool3)

    # compose
    body = ""

    if cfg.img.embed_image_in_body and display_img_url:
        body += f'<p><img src="{_esc(display_img_url)}" alt="{_esc(title)}"/></p>'

    body += f"<h3><strong>{_esc(_strip_ai_smell(h1))}</strong></h3>"
    body += _make_paragraphs(sec1)

    body += f"<h3><strong>{_esc(_strip_ai_smell(h2))}</strong></h3>"
    body += _make_paragraphs(sec2)
    body += recipe_list

    body += f"<h3><strong>{_esc(_strip_ai_smell(h3))}</strong></h3>"
    body += _make_paragraphs(sec3)

    tags = build_hashtags(title)
    body += f"<p><br/><br/>{_esc(tags)}</p>"

    # excerpt
    excerpt = _strip_ai_smell(_title_variants(title_style, title))[:140]

    # ensure min_total_chars by expanding sec3 a bit
    min_total = max(600, int(cfg.content.min_total_chars))
    if _k_len(body) < min_total:
        extra_pool = pool1 + pool2 + pool3
        random.shuffle(extra_pool)
        extra_lines = _expand_to_min([], min_total - _k_len(body) + 40, extra_pool)
        body += _make_paragraphs(extra_lines)

    # suggested tag names (for WP tag creation)
    name_tokens = [t for t in re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", title).split() if len(t) >= 2]
    suggested = (name_tokens[:6] + ["한식레시피", "집밥", "간단요리", "요리팁", "오늘뭐먹지", "자취요리"])[:12]

    # final clean (remove '.' just in case)
    body = body.replace(".", " ")

    return body, excerpt, suggested


# -----------------------------
# Image pipeline
# -----------------------------


def _guess_mime_and_ext(url: str, content_type: str) -> Tuple[str, str]:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in ("image/jpeg", "image/jpg"):
        return "image/jpeg", "jpg"
    if ct == "image/png":
        return "image/png", "png"
    if ct == "image/webp":
        return "image/webp", "webp"

    u = (url or "").lower()
    if u.endswith(".png"):
        return "image/png", "png"
    if u.endswith(".webp"):
        return "image/webp", "webp"
    return "image/jpeg", "jpg"


def fetch_image_bytes(url: str, timeout: int = 20) -> Optional[Tuple[bytes, str, str]]:
    """return (bytes, mime, ext)"""
    if not url:
        return None
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200 or not r.content:
            return None
        mime, ext = _guess_mime_and_ext(url, r.headers.get("Content-Type", ""))
        if not mime.startswith("image/"):
            # sometimes missing content-type; still accept common signatures
            pass
        return r.content, mime, ext
    except RequestException:
        return None


def build_unsplash_url(recipe_title: str) -> str:
    q = re.sub(r"\s+", ",", (recipe_title or "").strip())
    q = re.sub(r"[^0-9가-힣a-zA-Z,]", "", q)
    q = q or "korean-food"
    # featured는 랜덤성이 있어도 이미지가 거의 항상 뜸
    return f"https://source.unsplash.com/featured/1200x800/?{quote(q)},korean,food"


def openai_generate_food_image(cfg: AppConfig, recipe_title: str) -> Optional[bytes]:
    """OpenAI 이미지 생성(b64_json) -> bytes"""
    if not (cfg.img.use_openai_image and cfg.img.openai_api_key):
        return None
    prompt = (
        f"Korean home-cooked food photo, {recipe_title}, appetizing, warm lighting, realistic,"
        f" served in a clean bowl on a wooden table, no text, no watermark"
    )
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {cfg.img.openai_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": cfg.img.openai_image_model or "gpt-image-1",
        "prompt": prompt,
        "size": "1024x1024",
        "response_format": "b64_json",
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return None
        data = r.json()
        arr = data.get("data") or []
        if not arr:
            return None
        b64 = (arr[0] or {}).get("b64_json")
        if not b64:
            return None
        return base64.b64decode(b64)
    except Exception:
        return None


def choose_image_for_recipe(cfg: AppConfig, recipe: Recipe) -> Tuple[Optional[bytes], Optional[str], Optional[str], str]:
    """returns bytes, mime, ext, source_label"""
    # 1) MFDS image url
    if recipe.image_url and recipe.image_url.startswith("http"):
        got = fetch_image_bytes(recipe.image_url, timeout=20)
        if got:
            b, mime, ext = got
            return b, mime, ext, "mfds"

    # 2) default thumb
    if cfg.img.default_thumb_url:
        got = fetch_image_bytes(cfg.img.default_thumb_url, timeout=20)
        if got:
            b, mime, ext = got
            return b, mime, ext, "default"

    # 3) OpenAI image
    if cfg.img.auto_image:
        b = openai_generate_food_image(cfg, recipe.title)
        if b:
            return b, "image/png", "png", "openai"

    # 4) Unsplash
    if cfg.img.auto_image:
        u = build_unsplash_url(recipe.title)
        got = fetch_image_bytes(u, timeout=25)
        if got:
            b, mime, ext = got
            return b, mime, ext, "unsplash"

    return None, None, None, "none"


def normalize_media_url(cfg: AppConfig, media_url: str) -> str:
    u = (media_url or "").strip()
    if not u:
        return ""
    # mixed content 완화: site가 https인데 media가 http로 내려오는 케이스
    if cfg.img.force_https_media and cfg.wp.base_url.lower().startswith("https://") and u.lower().startswith("http://"):
        u = "https://" + u[len("http://") :]
    return u


# -----------------------------
# Main
# -----------------------------


def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = {"day": "오늘", "am": "오전", "pm": "오후"}.get(slot, "오늘")

    run_id = _env("RUN_ID", "").strip()
    if not run_id:
        run_id = now.strftime("%H%M%S") + f"{random.randint(100,999)}"

    # 중복 발행이면 슬롯에 run_id를 붙여 고유화
    base_date_slot = f"{date_str}_{slot}"
    date_slot = f"{base_date_slot}_{run_id}" if cfg.run.allow_duplicate_posts else base_date_slot

    init_db(cfg.sqlite_path)

    today_meta = None
    if not cfg.run.allow_duplicate_posts:
        today_meta = get_today_post(cfg.sqlite_path, date_slot)

    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days)

    title_style = (_env("TITLE_STYLE", "random") or "random").lower()
    if title_style not in ("benefit", "curiosity", "threat", "compare", "random"):
        title_style = "random"

    # choose recipe
    chosen: Optional[Recipe] = None
    if today_meta and (not cfg.run.force_new) and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        # 이 모드에서는 같은 글 계속 업데이트하는 구조라면 여기서 복원 가능
        pass

    if not chosen:
        chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(cfg, recent_pairs)

    assert chosen is not None

    # title + slug
    title = _title_variants(title_style if title_style != "random" else random.choice(["benefit", "curiosity", "threat", "compare"]), chosen.title)

    base_slug = f"korean-recipe-{date_str}-{slot}"
    slug = f"{base_slug}-{run_id}" if (cfg.run.allow_duplicate_posts or cfg.run.force_new) else base_slug

    if cfg.run.debug:
        print(f"[RUN] date_slot={date_slot} slot={slot} run_id={run_id} slug={slug} title_style={title_style}")

    # image -> upload to WP
    media_id = 0
    media_url = ""
    display_img_url = ""

    if cfg.img.upload_thumb:
        b, mime, ext, src = choose_image_for_recipe(cfg, chosen)
        if cfg.run.debug:
            print("[IMG] source=", src, "bytes=", (len(b) if b else 0), "mime=", mime, "ext=", ext)
        if b and mime and ext:
            try:
                filename = f"{slug}.{ext}"
                mid, murl = wp_upload_media_bytes(cfg.wp, b, filename, mime)
                media_id = int(mid or 0)
                media_url = normalize_media_url(cfg, str(murl or ""))
            except Exception as e:
                if cfg.run.debug:
                    print("[IMG] wp upload failed:", repr(e))

    display_img_url = (media_url or "").strip()

    # body
    body_html, excerpt, suggested_tag_names = build_body_html(cfg, chosen, display_img_url, title_style)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print(body_html[:1800])
        return

    # tags
    final_tag_ids = cfg.wp.tag_ids[:]
    if cfg.wp.auto_wp_tags:
        try:
            # 너무 길거나 특수문자 많은 태그는 정리
            clean_names = []
            for nm in suggested_tag_names:
                nm = re.sub(r"[^0-9가-힣a-zA-Z]", "", (nm or "").strip())
                if 1 < len(nm) <= 20:
                    clean_names.append(nm)
            auto_ids = wp_ensure_tag_ids(cfg.wp, clean_names[:20])
            for tid in auto_ids:
                if tid and tid not in final_tag_ids:
                    final_tag_ids.append(tid)
        except Exception:
            pass

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    # publish
    if cfg.run.allow_duplicate_posts:
        # 무조건 새 글
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt, tags=final_tag_ids)
        if featured_id:
            try:
                wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, excerpt=excerpt, tags=final_tag_ids)
            except Exception:
                pass
        print("OK(created-dup):", wp_post_id, wp_link, "| recipe=", chosen.title, "| img=", bool(media_id))
    else:
        # 기존 글 업데이트(기본 모드)
        try:
            if today_meta and today_meta.get("wp_post_id"):
                post_id = int(today_meta["wp_post_id"])
                wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, featured_media=featured_id, excerpt=excerpt, tags=final_tag_ids)
                print("OK(updated):", wp_post_id, wp_link, "| recipe=", chosen.title, "| img=", bool(media_id))
            else:
                wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt, tags=final_tag_ids)
                if featured_id:
                    try:
                        wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, excerpt=excerpt, tags=final_tag_ids)
                    except Exception:
                        pass
                print("OK(created):", wp_post_id, wp_link, "| recipe=", chosen.title, "| img=", bool(media_id))
        except Exception as e:
            # fallback create
            wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt, tags=final_tag_ids)
            if featured_id:
                try:
                    wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, excerpt=excerpt, tags=final_tag_ids)
                except Exception:
                    pass
            print("OK(created-fallback):", wp_post_id, wp_link, "| err=", repr(e))

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
