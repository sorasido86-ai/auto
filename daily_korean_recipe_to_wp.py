# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp.py (완전 통합  깔끔한 레시피 글 + 이미지 싱크 + 안정화)

무엇이 바뀌었나
- MFDS(식품안전나라) API가 느리거나 타임아웃이어도 예외로 죽지 않고 즉시 폴백
- 글자수 과도한 강제(섹션 1500자 등) 제거  기본은 짧고 정돈된 레시피 글
- 반복 문장 방지  섹션 보강 문장은 중복 없이 생성
- 본문에 "자동 생성" "기준시각/슬롯" "출처" 문구 기본 제거
- 사진이 없을 때도 레시피 제목 기반으로 자동 이미지(Unsplash Source) 생성
- 지난 레시피 링크 기본 제거(ADD_INTERNAL_LINKS=0 기본)

필수 환경변수(Secrets)
- WP_BASE_URL
- WP_USER
- WP_APP_PASS

권장 환경변수
- WP_STATUS=publish
- WP_CATEGORY_IDS=7
- WP_TAG_IDS= (옵션)
- SQLITE_PATH=data/daily_korean_recipe.sqlite3

MFDS(식품안전나라 OpenAPI)
- MFDS_API_KEY=... (없으면 내장 레시피만 사용)
- MFDS_REQ_TIMEOUT=10 (초)
- MFDS_BUDGET_SECONDS=20 (초)
- MFDS_MAX_FAILS=2 (연속 실패 허용)

이미지
- DEFAULT_THUMB_URL=... (선택)
- AUTO_IMAGE=1 (기본 1)  기본이미지도 없을 때 Unsplash Source로 자동 생성
- UPLOAD_THUMB=1 (기본 1)  WP 미디어 업로드 시도
- SET_FEATURED=1 (기본 1)
- EMBED_IMAGE_IN_BODY=1 (기본 1)

글 길이(짧게)
- INTRO_CHARS=160 (기본 160)
- SECTION_MIN_CHARS=420 (기본 420)
- TOTAL_MIN_CHARS=1200 (기본 1200)
- HASHTAG_COUNT=8 (기본 8)

동작
- RUN_SLOT=day|am|pm (기본 day)
- FORCE_NEW=0|1 (기본 0)
- DRY_RUN=0|1 (기본 0)
- DEBUG=0|1 (기본 0)
- AVOID_REPEAT_DAYS=90 (기본 90)
- MAX_TRIES=20 (기본 20)

OpenAI(선택)
- USE_OPENAI=0|1
- OPENAI_API_KEY=...
- OPENAI_MODEL=gpt-4.1-mini

주의
- 본문은 마침표를 최대한 빼고  줄바꿈과 여백으로 호흡을 둡니다
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
            "신김치를 넣고 2~3분 더 볶아 김치의 신맛을 한 번 눌러줍니다",
            "고춧가루 다진마늘 국간장을 넣고 30초만 볶아 향을 내요",
            "육수를 붓고 10~12분 끓입니다",
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
            "양파 애호박 두부 넣고 5~6분 더 끓입니다",
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
            "고기에 양념장을 넣고 15분 이상 재워둡니다",
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


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True
    mfds_req_timeout: int = 10
    mfds_budget_seconds: int = 20
    mfds_max_fails: int = 2


@dataclass
class ImageConfig:
    default_thumb_url: str = ""
    auto_image: bool = True
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    reuse_media_by_search: bool = True


@dataclass
class ContentConfig:
    schema_mode: str = "comment"  # comment|script|off
    hashtag_count: int = 8
    intro_chars: int = 160
    section_min_chars: int = 420
    total_min_chars: int = 1200
    add_internal_links: bool = False
    title_style: str = "random"  # benefit|threat|curiosity|compare|random


@dataclass
class OpenAIConfig:
    use_openai: bool = False
    api_key: str = ""
    model: str = "gpt-4.1-mini"


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    recipe: RecipeSourceConfig
    img: ImageConfig
    content: ContentConfig
    openai: OpenAIConfig
    sqlite_path: str


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    schema_mode = (_env("SCHEMA_MODE", "comment") or "comment").lower()
    if schema_mode not in ("comment", "script", "off"):
        schema_mode = "comment"

    title_style = (_env("TITLE_STYLE", "random") or "random").lower()
    if title_style not in ("benefit", "threat", "curiosity", "compare", "random"):
        title_style = "random"

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=_env("WP_STATUS", "publish") or "publish",
            category_ids=_parse_int_list(_env("WP_CATEGORY_IDS", "7")),
            tag_ids=_parse_int_list(_env("WP_TAG_IDS", "")),
        ),
        run=RunConfig(
            run_slot=run_slot,
            force_new=_env_bool("FORCE_NEW", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 20),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_req_timeout=_env_int("MFDS_REQ_TIMEOUT", 10),
            mfds_budget_seconds=_env_int("MFDS_BUDGET_SECONDS", 20),
            mfds_max_fails=_env_int("MFDS_MAX_FAILS", 2),
        ),
        img=ImageConfig(
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            auto_image=_env_bool("AUTO_IMAGE", True),
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
        ),
        content=ContentConfig(
            schema_mode=schema_mode,
            hashtag_count=_env_int("HASHTAG_COUNT", 8),
            intro_chars=_env_int("INTRO_CHARS", 160),
            section_min_chars=_env_int("SECTION_MIN_CHARS", 420),
            total_min_chars=_env_int("TOTAL_MIN_CHARS", 1200),
            add_internal_links=_env_bool("ADD_INTERNAL_LINKS", False),
            title_style=title_style,
        ),
        openai=OpenAIConfig(
            use_openai=_env_bool("USE_OPENAI", False),
            api_key=_env("OPENAI_API_KEY", ""),
            model=_env("OPENAI_MODEL", "gpt-4.1-mini"),
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
    print("[CFG] MFDS_REQ_TIMEOUT:", cfg.recipe.mfds_req_timeout, "MFDS_BUDGET_SECONDS:", cfg.recipe.mfds_budget_seconds)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] AUTO_IMAGE:", int(cfg.img.auto_image), "UPLOAD_THUMB:", int(cfg.img.upload_thumb), "SET_FEATURED:", int(cfg.img.set_featured))
    print("[CFG] INTRO_CHARS:", cfg.content.intro_chars, "SECTION_MIN_CHARS:", cfg.content.section_min_chars, "TOTAL_MIN_CHARS:", cfg.content.total_min_chars)
    print("[CFG] ADD_INTERNAL_LINKS:", int(cfg.content.add_internal_links), "HASHTAG_COUNT:", cfg.content.hashtag_count)
    print("[CFG] TITLE_STYLE:", cfg.content.title_style)
    print("[CFG] USE_OPENAI:", int(cfg.openai.use_openai), "OPENAI_API_KEY:", ok(cfg.openai.api_key), "OPENAI_MODEL:", cfg.openai.model)


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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/2.0"}


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
    # 이미지 다운로드(리다이렉트 포함)  실패하면 예외
    r = requests.get(image_url, timeout=20, allow_redirects=True)
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


# -----------------------------
# Recipe model + MFDS provider
# -----------------------------

@dataclass
class Recipe:
    source: str  # mfds|local
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    image_url: str = ""

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


def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int, end: int, timeout_s: int) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    try:
        r = requests.get(url, timeout=timeout_s)
        if r.status_code != 200:
            return []
        data = r.json()
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

    t0 = time.time()
    fails = 0
    for _ in range(cfg.run.max_tries):
        if time.time() - t0 > max(5, cfg.recipe.mfds_budget_seconds):
            break
        if fails >= max(1, cfg.recipe.mfds_max_fails):
            break

        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(
            cfg.recipe.mfds_api_key,
            "RCP_NM",
            kw,
            start=1,
            end=60,
            timeout_s=max(3, cfg.recipe.mfds_req_timeout),
        )
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
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_SEQ", recipe_id, start=1, end=5, timeout_s=max(3, cfg.recipe.mfds_req_timeout))
        for row in rows:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if rcp.recipe_id == recipe_id:
                return rcp
    return None


# -----------------------------
# Rendering helpers
# -----------------------------


def _esc(s: str) -> str:
    return html.escape(s or "")


def _no_period(s: str) -> str:
    # 본문 텍스트용  마침표 제거
    s = (s or "").replace(".", "").replace("。", "")
    return s


def _plain_len(html_s: str) -> int:
    t = re.sub(r"<[^>]+>", "", html_s or "")
    t = html.unescape(t)
    return len(t)


def _wrap_p(txt: str) -> str:
    return f"<p>{_esc(_no_period(txt))}</p>"


def _wrap_lines(lines: List[str]) -> str:
    safe = "<br/>".join(_esc(_no_period(x)) for x in lines if x)
    return f"<p>{safe}</p>"


def _top_tokens_from_recipe(recipe: Recipe) -> List[str]:
    # 제목 + 재료에서 대표 토큰 추출
    toks: List[str] = []
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", recipe.title or "")
    toks += [x.strip() for x in t.split() if len(x.strip()) >= 2]

    for ing in (recipe.ingredients or [])[:8]:
        s = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", str(ing))
        toks += [x.strip() for x in s.split() if len(x.strip()) >= 2]

    # 중복 제거
    out: List[str] = []
    seen = set()
    for x in toks:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out[:10]


def build_hashtags(cfg: AppConfig, recipe: Recipe) -> str:
    base = ["한식레시피", "집밥", "오늘뭐먹지", "간단요리", "초간단레시피", "자취요리", "밥도둑", "요리기록"]
    for tok in _top_tokens_from_recipe(recipe)[:4]:
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


# -----------------------------
# 이미지 싱크(레시피 제목 -> 검색쿼리)
# -----------------------------

_IMAGE_QUERY_MAP = [
    ("김치찌개", ["kimchi jjigae", "kimchi stew", "korean stew"]),
    ("된장찌개", ["doenjang jjigae", "soybean paste stew", "korean stew"]),
    ("순두부", ["sundubu jjigae", "soft tofu stew", "korean tofu stew"]),
    ("불고기", ["bulgogi", "korean bbq beef", "korean beef"]),
    ("비빔밥", ["bibimbap", "korean mixed rice"]),
    ("잡채", ["japchae", "korean glass noodles"]),
    ("떡볶이", ["tteokbokki", "spicy rice cake"]),
    ("미역국", ["miyeokguk", "seaweed soup"]),
    ("제육", ["jeyuk bokkeum", "spicy pork stir fry"]),
    ("김밥", ["gimbap", "korean rice roll"]),
    ("갈비", ["galbi", "korean short ribs"]),
    ("부침", ["korean pancake", "jeon"]),
]


def infer_image_query(recipe_title: str) -> str:
    t = (recipe_title or "").strip()
    for k, qs in _IMAGE_QUERY_MAP:
        if k in t:
            return ",".join(qs)

    # 일반 폴백  제목 토큰 + korean food
    # 한국어 제목도 같이 넣으면 검색 힌트로 도움이 될 때가 있음
    tok = re.sub(r"\s+", " ", re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", t)).strip()
    tok = tok[:60]
    if tok:
        return f"korean food,{tok}"
    return "korean food"


def build_unsplash_source_url(recipe: Recipe) -> str:
    q = infer_image_query(recipe.title)
    # source.unsplash.com 은 랜덤이지만 키워드 기반으로 관련 이미지를 주는 편
    # 고정 크기 지정
    return f"https://source.unsplash.com/1600x900/?{quote(q)}"


def choose_image_url(cfg: AppConfig, recipe: Recipe) -> str:
    # 1) 레시피 자체 이미지
    if (recipe.image_url or "").strip():
        return recipe.image_url.strip()

    # 2) 기본 이미지
    if (cfg.img.default_thumb_url or "").strip():
        return cfg.img.default_thumb_url.strip()

    # 3) 자동 이미지
    if cfg.img.auto_image:
        return build_unsplash_source_url(recipe)

    return ""


# -----------------------------
# Schema(옵션)
# -----------------------------


def build_recipe_schema_json(recipe: Recipe, image_url: str, now: datetime) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "Recipe",
        "name": (recipe.title or "").strip(),
        "image": [image_url] if image_url else [],
        "datePublished": now.astimezone(KST).isoformat(),
        "recipeCuisine": "Korean",
        "recipeIngredient": recipe.ingredients or [],
        "recipeInstructions": [{"@type": "HowToStep", "text": s} for s in (recipe.steps or [])],
    }
    return json.dumps(data, ensure_ascii=False)


def build_schema_block(cfg: AppConfig, recipe: Recipe, image_url: str, now: datetime) -> str:
    mode = cfg.content.schema_mode
    if mode == "off":
        return ""
    schema_json = build_recipe_schema_json(recipe, image_url, now)
    if mode == "script":
        return f'<script type="application/ld+json">{_esc(schema_json)}</script>'
    return f"<!-- RECIPE_SCHEMA_JSONLD: {_esc(schema_json)} -->"


# -----------------------------
# Title templates
# -----------------------------


def _title_templates(keyword: str) -> Dict[str, List[str]]:
    kw = _no_period(keyword).strip()
    return {
        "benefit": [
            f"{kw} 이번에 해먹은 사람만  맛이 달라지는 포인트 듣고 놀랐다",
            f"{kw} 한 번만 이렇게 했더니  집밥이 갑자기 업그레이드됐다",
            f"{kw} 생각날 때  실패 확률 낮게 가는 순서가 따로 있더라",
        ],
        "threat": [
            f"{kw} 이 순서 놓치면  맛이 반으로 내려가더라  저도 처음엔 몰랐어요",
            f"{kw} 간을 중간에 잡으면  마지막에 망하기 쉬워요  이 부분만 조심해요",
            f"{kw} 불 조절 잘못하면  향이 확 죽어요  딱 한 단계만 바꿔봐요",
        ],
        "curiosity": [
            f"{kw} 왜 밖에서 먹는 맛이 안 날까  한 가지 차이가 있더라고요",
            f"{kw} 맛이 확 살아나는 타이밍  오늘은 그 부분만 콕 잡아드릴게요",
            f"{kw} 해먹을수록 쉬워지는 이유  알고 보니 흐름이 딱 정해져 있어요",
        ],
        "compare": [
            f"{kw} 집에서 만든 맛 vs 밖에서 먹는 맛  결국 차이는 여기였어요",
            f"{kw} 간단 버전 vs 정석 버전  저는 이쪽이 더 안정적이더라고요",
            f"{kw} 빠른 버전으로도 충분한 이유  오늘은 그 방법으로 정리해요",
        ],
    }


def build_post_title(date_str: str, slot_label: str, keyword: str, title_style: str) -> str:
    tpls = _title_templates(keyword)
    style = title_style
    if style == "random":
        style = random.choice(["benefit", "threat", "curiosity", "compare"])
    base = random.choice(tpls.get(style, tpls["curiosity"]))
    return _no_period(f"{base} ({date_str} {slot_label})")


# -----------------------------
# Body builder(깔끔)
# -----------------------------


def _build_intro(keyword: str, target_chars: int) -> str:
    kw = _no_period(keyword).strip()
    lines = [
        f"요즘 {kw} 생각나면  괜히 마음이 좀 바빠지지 않으세요",
        "오늘은 어렵게 말고  진짜 해먹을 수 있게  재료랑 흐름만 딱 정리해볼게요",
        "저도 바쁜 날엔 이 정도 정리만 있으면  바로 따라 하게 되더라고요",
    ]
    txt = " ".join(lines)

    # 너무 짧으면 1줄만 추가
    if len(txt) < max(100, target_chars - 30):
        txt += " " + "간은 마지막에만 살짝 잡고  불은 중불로만 가면  실패가 확 줄어요"

    # 길이 컷
    if len(txt) > target_chars:
        txt = txt[:target_chars].rstrip()

    return _no_period(txt)


def _limit_list(items: List[str], n: int) -> List[str]:
    out: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        out.append(x)
        if len(out) >= n:
            break
    return out


def _format_ingredients(recipe: Recipe, limit_n: int = 14) -> List[str]:
    if not recipe.ingredients:
        return ["재료 정보가 비어있어요"]
    ings = _limit_list([str(x).strip() for x in recipe.ingredients], limit_n)
    return [f"- {x}" for x in ings]


def _format_steps(recipe: Recipe, limit_n: int = 12) -> List[str]:
    if not recipe.steps:
        return ["1 과정 정보가 비어있어요"]
    st = _limit_list([str(x).strip() for x in recipe.steps], limit_n)
    out = []
    for i, s in enumerate(st, start=1):
        out.append(f"{i} {s}")
    return out


def _dish_type_hint(title: str) -> str:
    t = title or ""
    if any(k in t for k in ["찌개", "국", "탕"]):
        return "국물"
    if any(k in t for k in ["볶음", "불고기", "제육"]):
        return "볶음"
    if any(k in t for k in ["전", "부침"]):
        return "부침"
    return "집밥"


def _make_unique_fillers(recipe: Recipe, used: set, count: int) -> List[str]:
    # 레시피와 싱크를 맞추는 보강 문장들
    title = _no_period(recipe.title)
    dish_type = _dish_type_hint(title)
    toks = _top_tokens_from_recipe(recipe)
    tok1 = toks[0] if toks else "재료"
    tok2 = toks[1] if len(toks) > 1 else "양념"

    templates = [
        f"제가 {title} 할 때 제일 신경 쓰는 건  {dish_type}의 밸런스예요  짠맛이 앞서면 끝이 좀 피곤해지니까요",
        f"{tok1} 들어가면 맛이 확 살아나더라고요  대신 양은 과하게 말고  마지막에 한 번만 조절해요",
        f"{tok2}는 처음부터 많이 넣기보다  한 번 끓이고 난 뒤에 조금씩 맞추는 게 훨씬 안전해요",
        f"불은 강불로 밀어붙이기보다  중불로 가다가  마지막에만 살짝 올리는 느낌이 결과가 좋아요",
        f"맛이 애매하면  소금부터 더 넣지 말고  향이 살아있는 재료를 아주 조금만 더해보세요",
        f"저는 이런 레시피는  한 번 성공하면  다음부터 마음이 확 편해져서  자주 돌려보게 되더라고요",
    ]

    out: List[str] = []
    random.shuffle(templates)
    for s in templates:
        k = s.strip()
        if not k:
            continue
        if k in used:
            continue
        used.add(k)
        out.append(k)
        if len(out) >= count:
            break

    # 그래도 부족하면 동적 생성
    while len(out) < count:
        seed = f"{title}|{tok1}|{tok2}|{len(out)}|{random.randint(1,9999)}"
        h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:6]
        s = f"오늘은 {title}를 {dish_type} 기준으로 정리해두는 날이라고 생각해요  저장해두면 다음엔 훨씬 빨라져요 {h}"
        s = _no_period(s)
        if s in used:
            continue
        used.add(s)
        out.append(s)

    return out


def _ensure_lengths(body_html: str, recipe: Recipe, cfg: AppConfig) -> str:
    # 전체 글자수 보강  중복 없이
    used: set = set()
    # 이미 포함된 본문 텍스트를 used에 넣어 중복 방지
    plain = re.sub(r"<[^>]+>", "\n", body_html)
    for line in [x.strip() for x in html.unescape(plain).splitlines() if x.strip()]:
        used.add(_no_period(line))

    while _plain_len(body_html) < max(500, cfg.content.total_min_chars):
        fillers = _make_unique_fillers(recipe, used, count=1)
        body_html += _wrap_p(fillers[0])
        # 안전장치
        if _plain_len(body_html) > cfg.content.total_min_chars + 600:
            break

    return body_html


def build_body_html(cfg: AppConfig, now: datetime, run_slot_label: str, recipe: Recipe, display_img_url: str) -> Tuple[str, str]:
    keyword = _no_period((recipe.title or "").strip())

    # intro
    intro_txt = _build_intro(keyword, cfg.content.intro_chars)
    excerpt = intro_txt[:140].strip()

    schema_block = build_schema_block(cfg, recipe, display_img_url, now)

    # 상단 이미지
    img_html = ""
    if cfg.img.embed_image_in_body and display_img_url:
        img_html = f'<p><img src="{_esc(display_img_url)}" alt="{_esc(keyword)}"/></p>'

    # 3개 섹션
    used: set = set()

    # Section 1
    s1_title = f"{keyword}  재료 준비랑 맛 포인트"
    s1 = f"<h3><strong>{_esc(_no_period(s1_title))}</strong></h3>"
    s1 += _wrap_p(f"오늘은 {keyword}를 어렵게 말고  바로 해먹을 수 있게 정리해볼게요")
    s1 += _wrap_lines(["재료는 이렇게만 준비해도 충분해요", *_format_ingredients(recipe, 14)])

    dish_type = _dish_type_hint(keyword)
    s1_points = [
        "간은 중간에 확정하지 말고  마지막에만 살짝 잡아요",
        "불은 강불로 오래 가기보다  중불로 안정적으로 가요",
        f"{dish_type}은 끝에서 한 번 더 살아나니까  완성 직전에 맛을 한 번만 보면 돼요",
    ]
    s1 += _wrap_lines(["맛 포인트는 이 세 가지만 기억하면 돼요", *s1_points])

    # 섹션 길이 보강
    while _plain_len(s1) < cfg.content.section_min_chars:
        s1 += _wrap_p(_make_unique_fillers(recipe, used, 1)[0])

    # Section 2
    s2_title = f"{keyword}  만드는 순서"
    s2 = f"<h3><strong>{_esc(_no_period(s2_title))}</strong></h3>"
    s2 += _wrap_p("순서는 길게 외울 필요 없어요  그대로 따라만 가면 돼요")
    s2 += _wrap_lines(["조리 순서", *_format_steps(recipe, 12)])
    s2_tips = [
        "양념은 한 번에 많이 넣지 말고  조금씩 추가하는 쪽이 안전해요",
        "맛이 애매하면  소금부터 더 넣지 말고  향을 먼저 살려보세요",
    ]
    s2 += _wrap_lines(["실패 줄이는 팁", *s2_tips])

    while _plain_len(s2) < cfg.content.section_min_chars:
        s2 += _wrap_p(_make_unique_fillers(recipe, used, 1)[0])

    # Section 3
    s3_title = f"{keyword}  먹는 법  보관  그리고 한 마디"
    s3 = f"<h3><strong>{_esc(_no_period(s3_title))}</strong></h3>"
    s3 += _wrap_p("저는 이 메뉴는 다음날 다시 데워 먹을 때 더 만족스러울 때가 많더라고요")
    s3 += _wrap_lines([
        "보관과 재가열",
        "냉장 보관은 밀폐 용기에 담아 하루 이틀 안에 먹는 걸 추천해요",
        "재가열은 한 번 끓인 다음에  간은 마지막에만 살짝 보세요",
    ])
    s3 += _wrap_lines([
        "응용 아이디어",
        "더 칼칼하게는 마지막에 고춧가루를 아주 조금만 추가해요",
        "아이 버전은 매운 양념을 줄이고  간장과 육수 비율로 부드럽게 가요",
    ])

    # 댓글 유도  마지막 고정
    s3 += _wrap_p(f"댓글로 하나만 알려주세요  {keyword} 만들 때 여러분은 어떤 재료를 추가하는 편이세요  버섯 두부 대파 같은 조합도 좋아요")

    # 해시태그  마지막 고정
    hashtags = build_hashtags(cfg, recipe)
    s3 += "<hr/>" + _wrap_p(hashtags)

    # 섹션 길이 보강  단 댓글 이후로는 보강하지 않음
    # 그래서 보강은 해시태그 전에 넣는 방식으로 처리
    # 여기서는 필요한 만큼 위쪽에만 추가
    while _plain_len(s3) < cfg.content.section_min_chars:
        # 해시태그 앞에 끼워넣기
        insert = _wrap_p(_make_unique_fillers(recipe, used, 1)[0])
        s3 = s3.replace("<hr/>", insert + "<hr/>", 1)

    body = schema_block + _wrap_p(intro_txt) + img_html + s1 + s2 + s3
    body = _ensure_lengths(body, recipe, cfg)
    return body, excerpt


# -----------------------------
# OpenAI upgrade (optional)
# -----------------------------


def generate_with_openai(cfg: AppConfig, recipe: Recipe, base_html: str) -> Optional[str]:
    if not (cfg.openai.use_openai and cfg.openai.api_key):
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        client = OpenAI(api_key=cfg.openai.api_key)
        prompt = f"""
너는 한국 요리 블로그 전문 에디터야
아래 레시피 제목 재료 과정은 내용이 바뀌면 안 돼  그대로 유지해
대신 문장만 더 자연스럽게 정돈해줘
문장 끝 마침표는 가능한 쓰지 말고  띄어쓰기와 줄바꿈으로 호흡을 둬
굵은 소제목은 정확히 3개만 유지해
글은 너무 길지 않게  깔끔하게 정리해
HTML만 출력해  코드블럭 금지

[레시피 제목]
{recipe.title}

[재료]
- """ + "\n- ".join(recipe.ingredients) + """

[과정]
""" + "\n".join([f"{i+1}) {s}" for i, s in enumerate(recipe.steps)]) + f"""

[현재 HTML 초안]
{base_html}
"""

        resp = client.responses.create(model=cfg.openai.model, input=prompt)
        out_text = getattr(resp, "output_text", None)
        if out_text and "<" in out_text:
            return out_text.strip()
    except Exception:
        return None
    return None


# -----------------------------
# Media helpers
# -----------------------------


def ensure_media(cfg: AppConfig, image_url: str, stable_name: str) -> Tuple[int, str]:
    if not image_url:
        return 0, ""

    h = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    ext = ".jpg"
    u = image_url.lower()
    if ".png" in u:
        ext = ".png"
    filename = f"{stable_name}_{h}{ext}"

    if cfg.img.reuse_media_by_search:
        found = wp_find_media_by_search(cfg.wp, search=f"{stable_name}_{h}")
        if found:
            return found

    mid, murl = wp_upload_media_from_url(cfg.wp, image_url, filename)
    return mid, murl


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
    if today_meta and (not cfg.run.force_new) and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

    if not chosen:
        chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(recent_pairs)

    assert chosen is not None

    title = build_post_title(date_str, slot_label, chosen.title, title_style=cfg.content.title_style)
    slug = f"korean-recipe-{date_str}-{slot}"

    # 이미지 후보 선정
    thumb_url = choose_image_url(cfg, chosen)
    if not thumb_url:
        print("[WARN] image candidate is empty  본문에 이미지가 안 들어갈 수 있어요")

    media_id = 0
    media_url = ""

    if cfg.img.upload_thumb and thumb_url:
        try:
            media_id, media_url = ensure_media(cfg, thumb_url, stable_name="korean_recipe_thumb")
            if cfg.run.debug:
                print("[IMG] media:", media_id, media_url)
        except Exception as e:
            if cfg.run.debug:
                print("[IMG] upload failed:", repr(e))
            media_id, media_url = 0, ""

    display_img_url = (media_url or thumb_url or "").strip()

    body_html, excerpt = build_body_html(cfg, now, slot_label, chosen, display_img_url=display_img_url)

    upgraded = generate_with_openai(cfg, chosen, body_html)
    if upgraded:
        body_html = upgraded

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략  미리보기 일부")
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
        # 업데이트 실패 시 새로 생성
        if cfg.run.debug:
            print("[WARN] post create/update failed  fallback to create:", repr(e))
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
