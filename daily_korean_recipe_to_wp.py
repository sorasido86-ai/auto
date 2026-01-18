# -*- coding: utf-8 -*-
"""daily_korean_recipe_to_wp.py

한식 레시피 자동 포스팅 (WordPress)

요구사항/개선사항 반영
- MFDS(OpenAPI) 응답 지연/타임아웃이 있어도 워크플로우가 오래 멈추지 않도록
  - MFDS_TIMEOUT_SEC(요청 타임아웃)
  - MFDS_BUDGET_SEC(전체 예산)
  를 적용해 빠르게 폴백합니다.

- 이미지가 안 나오는 문제를 해결하기 위해 "이미지 1장 보장" 전략으로 바꿨습니다.
  - MFDS 이미지가 있으면: 다운로드 -> WP 미디어 업로드 -> featured_media + 본문 상단 삽입
  - MFDS 이미지가 없으면:
      1) DEFAULT_THUMB_URL
      2) AUTO_IMAGE=1이면 Unsplash Source(키 불필요)로 이미지 가져오기
      3) (선택) USE_OPENAI_IMAGE=1이면 OpenAI 이미지 생성

- 본문은 홈피드형으로 정돈
  - 도입부 200~300자
  - 굵은 소제목 3개(재료/조리순서 포함)
  - 최소 글자수 MIN_TOTAL_CHARS(기본 1200자)
  - 항목 앞 불릿/점 특수문자 제거: <ul>/<ol> 대신 줄바꿈 텍스트 사용
  - "자동 생성"/"기준시각"/"슬롯"/"지난 레시피 링크" 기본 제거

필수 환경변수(Secrets)
- WP_BASE_URL
- WP_USER
- WP_APP_PASS

권장
- MFDS_API_KEY (있으면 MFDS 우선)
- SQLITE_PATH (기본 data/daily_korean_recipe.sqlite3)

이미지 옵션
- DEFAULT_THUMB_URL (없어도 됨)
- AUTO_IMAGE=1 (기본 1)
- AUTO_IMAGE_PROVIDER=unsplash|openai (기본 unsplash)
- USE_OPENAI_IMAGE=0|1 (기본 0)
- OPENAI_API_KEY (USE_OPENAI_IMAGE=1이면 필요)
- OPENAI_IMAGE_MODEL (기본 gpt-image-1)
- OPENAI_IMAGE_SIZE (기본 1024x1024)

태그 옵션
- AUTO_TAGS=1 (기본 1)
- TAG_NAMES="한식레시피,집밥,오늘뭐먹지" (추가 태그명)
- WP_TAG_IDS="1,2,3" (이미 WP에 있는 태그 ID)

기타
- MIN_TOTAL_CHARS=1200 (기본 1200)
- MAX_TOTAL_CHARS=1900 (기본 1900)
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
# Local fallback recipes
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1.5큰술"),
            ("고추장", "선택 1/2큰술"),
            ("애호박", "1/3개"),
            ("양파", "1/3개"),
            ("두부", "1/2모"),
            ("대파", "1/2대"),
            ("다진마늘", "1작은술"),
            ("멸치다시마 육수(또는 물)", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 5분 정도 끓여요",
            "양파 애호박을 넣고 4분 정도 더 끓입니다",
            "두부를 넣고 2분 정도 끓인 뒤 간을 보고 마무리해요",
            "대파를 넣고 한 번만 더 끓이면 끝이에요",
        ],
    },
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
            "냄비에 돼지고기를 먼저 볶아 기름을 살짝 내요",
            "김치를 넣고 2분 정도 같이 볶아 신맛을 눌러줍니다",
            "고춧가루 다진마늘 국간장을 넣고 30초만 볶아 향을 내요",
            "육수를 붓고 10분 정도 끓입니다",
            "양파 두부를 넣고 3분 더 끓인 뒤 대파로 마무리해요",
        ],
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
            ("물 또는 배즙", "3큰술"),
        ],
        "steps": [
            "간장 설탕 다진마늘 참기름 후추 물을 섞어 양념을 만들어요",
            "고기에 양념을 넣고 15분 이상 재워둡니다",
            "팬에 고기를 볶다가 양파 대파를 넣어 숨이 죽을 때까지 볶아요",
        ],
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
        if x:
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


@dataclass
class TagConfig:
    auto_tags: bool = True
    tag_names: List[str] = field(default_factory=list)
    max_auto_tags: int = 10


@dataclass
class RunConfig:
    run_slot: str = "day"  # day/am/pm
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 18


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True
    mfds_timeout_sec: int = 10
    mfds_budget_sec: int = 18


@dataclass
class ImageConfig:
    default_thumb_url: str = ""
    upload_thumb: bool = True
    set_featured: bool = True
    embed_in_body: bool = True
    reuse_media_by_search: bool = True
    auto_image: bool = True
    auto_image_provider: str = "unsplash"  # unsplash|openai


@dataclass
class ContentConfig:
    title_style: str = "random"  # random|benefit|threat|curiosity|compare
    min_total_chars: int = 1200
    max_total_chars: int = 1900


@dataclass
class OpenAIImageConfig:
    use_openai_image: bool = False
    api_key: str = ""
    model: str = "gpt-image-1"
    size: str = "1024x1024"


@dataclass
class AppConfig:
    wp: WordPressConfig
    tags: TagConfig
    run: RunConfig
    recipe: RecipeSourceConfig
    img: ImageConfig
    content: ContentConfig
    openai_img: OpenAIImageConfig
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

    title_style = (_env("TITLE_STYLE", "random") or "random").lower()
    if title_style not in ("random", "benefit", "threat", "curiosity", "compare"):
        title_style = "random"

    auto_image_provider = (_env("AUTO_IMAGE_PROVIDER", "unsplash") or "unsplash").lower()
    if auto_image_provider not in ("unsplash", "openai"):
        auto_image_provider = "unsplash"

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=cat_ids,
            tag_ids=tag_ids,
        ),
        tags=TagConfig(
            auto_tags=_env_bool("AUTO_TAGS", True),
            tag_names=_parse_str_list(_env("TAG_NAMES", "한식레시피,집밥,오늘뭐먹지,간단요리,집밥메뉴,요리")),
            max_auto_tags=_env_int("MAX_AUTO_TAGS", 10),
        ),
        run=RunConfig(
            run_slot=run_slot,
            force_new=_env_bool("FORCE_NEW", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 18),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout_sec=_env_int("MFDS_TIMEOUT_SEC", 10),
            mfds_budget_sec=_env_int("MFDS_BUDGET_SEC", 18),
        ),
        img=ImageConfig(
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
            auto_image=_env_bool("AUTO_IMAGE", True),
            auto_image_provider=auto_image_provider,
        ),
        content=ContentConfig(
            title_style=title_style,
            min_total_chars=_env_int("MIN_TOTAL_CHARS", 1200),
            max_total_chars=_env_int("MAX_TOTAL_CHARS", 1900),
        ),
        openai_img=OpenAIImageConfig(
            use_openai_image=_env_bool("USE_OPENAI_IMAGE", False),
            api_key=_env("OPENAI_API_KEY", ""),
            model=_env("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            size=_env("OPENAI_IMAGE_SIZE", "1024x1024"),
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
        return f"OK(len={len(v)})" if v else "EMPTY"

    print("[CFG] WP_BASE_URL:", "OK" if cfg.wp.base_url else "EMPTY")
    print("[CFG] WP_USER:", ok(cfg.wp.user))
    print("[CFG] WP_APP_PASS:", ok(cfg.wp.app_pass))
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CFG] WP_TAG_IDS:", cfg.wp.tag_ids)
    print("[CFG] AUTO_TAGS:", cfg.tags.auto_tags, "| TAG_NAMES:", len(cfg.tags.tag_names))
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| TIMEOUT:", cfg.recipe.mfds_timeout_sec, "| BUDGET:", cfg.recipe.mfds_budget_sec)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] AUTO_IMAGE:", cfg.img.auto_image, "| PROVIDER:", cfg.img.auto_image_provider)
    print("[CFG] USE_OPENAI_IMAGE:", cfg.openai_img.use_openai_image, "| OPENAI_API_KEY:", ok(cfg.openai_img.api_key), "| MODEL:", cfg.openai_img.model)
    print("[CFG] MIN_TOTAL_CHARS:", cfg.content.min_total_chars, "| MAX_TOTAL_CHARS:", cfg.content.max_total_chars)


# -----------------------------
# SQLite history
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
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
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


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, tag_ids: List[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    tags = [x for x in (cfg.tag_ids + (tag_ids or [])) if int(x) > 0]
    if tags:
        payload["tags"] = sorted(list({int(x) for x in tags}))

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, featured_media: int, tag_ids: List[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status}
    if featured_media:
        payload["featured_media"] = int(featured_media)
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    tags = [x for x in (cfg.tag_ids + (tag_ids or [])) if int(x) > 0]
    if tags:
        payload["tags"] = sorted(list({int(x) for x in tags}))

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


def wp_upload_media_bytes(cfg: WordPressConfig, content: bytes, filename: str, content_type: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": content_type or "application/octet-stream",
    }
    rr = requests.post(url, headers=headers, data=content, timeout=60)
    if rr.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {rr.status_code} body={rr.text[:500]}")
    data = rr.json()
    return int(data["id"]), str(data.get("source_url") or "")


def wp_get_or_create_tag_id(cfg: WordPressConfig, tag_name: str) -> int:
    name = (tag_name or "").strip()
    if not name:
        return 0

    headers = {**wp_auth_header(cfg.user, cfg.app_pass)}

    # search
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/tags"
    r = requests.get(url, headers=headers, params={"search": name, "per_page": 50}, timeout=25)
    if r.status_code == 200:
        try:
            items = r.json()
            if isinstance(items, list):
                for it in items:
                    if str(it.get("name") or "").strip() == name:
                        return int(it.get("id") or 0)
        except Exception:
            pass

    # create
    rr = requests.post(url, headers={**headers, "Content-Type": "application/json"}, json={"name": name}, timeout=25)
    if rr.status_code not in (200, 201):
        # 충돌 등은 무시
        return 0
    try:
        data = rr.json()
        return int(data.get("id") or 0)
    except Exception:
        return 0


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


def mfds_fetch_by_param(cfg: AppConfig, param: str, value: str, start: int = 1, end: int = 60) -> List[Dict[str, Any]]:
    api_key = cfg.recipe.mfds_api_key
    if not api_key:
        return []

    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"

    try:
        # timeout: (connect, read)
        r = requests.get(url, timeout=(5, max(3, cfg.recipe.mfds_timeout_sec)))
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
    # MFDS 재료는 긴 텍스트가 많아 줄바꿈 기준도 섞어 처리
    parts = parts.replace("\r", "\n")
    for p in re.split(r"\n|\s*,\s*", parts):
        p = p.strip()
        if not p:
            continue
        ingredients.append(p)

    steps: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s)
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

    candidates_with_img: List[Recipe] = []
    candidates_no_img: List[Recipe] = []

    for _ in range(cfg.run.max_tries):
        if (time.monotonic() - t0) > max(5, cfg.recipe.mfds_budget_sec):
            break

        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg, "RCP_NM", kw, start=1, end=60)
        if not rows:
            continue

        random.shuffle(rows)
        for row in rows[:30]:
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

            if rcp.image_url:
                candidates_with_img.append(rcp)
            else:
                candidates_no_img.append(rcp)

        if candidates_with_img:
            return random.choice(candidates_with_img)

    if candidates_no_img:
        return random.choice(candidates_no_img)

    return None


def pick_recipe_local(recent_pairs: List[Tuple[str, str]]) -> Recipe:
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
        image_url="",
    )


# -----------------------------
# Image pipeline
# -----------------------------

def _guess_ext_and_type(content_type: str) -> Tuple[str, str]:
    c = (content_type or "").split(";")[0].strip().lower()
    if c in ("image/jpeg", "image/jpg"):
        return ".jpg", "image/jpeg"
    if c == "image/png":
        return ".png", "image/png"
    if c == "image/webp":
        return ".webp", "image/webp"
    return ".jpg", "image/jpeg"


def _looks_like_image_bytes(b: bytes) -> bool:
    if not b or len(b) < 12:
        return False
    # JPEG
    if b[:3] == b"\xFF\xD8\xFF":
        return True
    # PNG
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    # GIF
    if b[:6] in (b"GIF87a", b"GIF89a"):
        return True
    # WEBP (RIFF....WEBP)
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return True
    return False


def download_image_bytes(url: str, timeout_sec: int = 25) -> Tuple[bytes, str]:
    """이미지 URL을 다운로드해서 (bytes, content-type) 반환

    - 핫링크/차단 페이지(HTML) 다운로드를 이미지로 착각해 업로드하는 문제를 방지
    - User-Agent 지정
    - content-type + 매직바이트로 이미지 여부 검증
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; daily-korean-recipe-bot/2.2)",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=(5, max(5, timeout_sec)), allow_redirects=True)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"image download failed: {r.status_code}")

    ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
    b = r.content

    # 1) content-type이 image/*이면 OK (단, HTML일 가능성 대비 매직바이트도 한번 확인)
    if ctype.startswith('image/'):
        if _looks_like_image_bytes(b) or len(b) > 1024:
            return b, ctype

    # 2) content-type이 없거나 text/*라도, 매직바이트가 이미지면 OK
    if _looks_like_image_bytes(b):
        return b, ctype or "image/jpeg"

    # 3) 여기까지 오면 거의 확실히 HTML/오류페이지
    head = b[:200].decode("utf-8", errors="ignore").strip().replace("\n", " ")[:200]
    raise RuntimeError(f"downloaded content is not an image (ctype={ctype or 'unknown'} head={head!r})")



def unsplash_source_url(keyword: str, size: int = 1024) -> str:
    kw = re.sub(r"\s+", " ", (keyword or "한식").strip())
    # Source endpoint는 키 없이 랜덤 이미지 제공
    return f"https://source.unsplash.com/{size}x{size}/?korean,food,{quote(kw)}"


def openai_generate_image_bytes(cfg: AppConfig, keyword: str) -> Optional[Tuple[bytes, str]]:
    if not (cfg.openai_img.use_openai_image and cfg.openai_img.api_key):
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    prompt = (
        f"A clean, appetizing, photorealistic Korean dish photo of {keyword}. "
        "Top-down angle, natural lighting, no text, no watermark, minimal background."
    )

    try:
        client = OpenAI(api_key=cfg.openai_img.api_key)
        resp = client.images.generate(
            model=cfg.openai_img.model,
            prompt=prompt,
            size=cfg.openai_img.size,
        )

        # openai python은 보통 resp.data[0].b64_json 제공
        data0 = None
        if hasattr(resp, "data") and resp.data:
            data0 = resp.data[0]

        b64 = None
        img_url = None
        if data0 is not None:
            b64 = getattr(data0, "b64_json", None) or (data0.get("b64_json") if isinstance(data0, dict) else None)
            img_url = getattr(data0, "url", None) or (data0.get("url") if isinstance(data0, dict) else None)

        if b64:
            return base64.b64decode(b64), "image/png"
        if img_url and str(img_url).startswith("http"):
            b, ct = download_image_bytes(str(img_url), timeout_sec=40)
            return b, ct

    except Exception:
        return None

    return None


def _normalize_public_url(base_url: str, u: str) -> str:
    """WP가 source_url을 http로 돌려주는 경우(혼합콘텐츠) 보정"""
    u = (u or '').strip()
    if not u:
        return ''
    if base_url.startswith('https://') and u.startswith('http://'):
        try:
            from urllib.parse import urlsplit, urlunsplit
            b = urlsplit(base_url)
            x = urlsplit(u)
            if x.netloc == b.netloc:
                return urlunsplit(('https', x.netloc, x.path, x.query, x.fragment))
        except Exception:
            pass
    return u



def ensure_featured_media(cfg: AppConfig, keyword: str, primary_image_url: str) -> Tuple[int, str]:
    """가능하면 반드시 1장의 이미지를 얻어 WP media로 업로드합니다

    - 1순위: MFDS 원본 이미지(있으면)
    - 2순위: DEFAULT_THUMB_URL(있으면)
    - 3순위: OpenAI 이미지(USE_OPENAI_IMAGE=1이면)
    - 4순위: Unsplash Source(항상 최후 fallback, 키 불필요)

    주의: 외부 URL을 본문에 직접 박으면 핫링크 차단/혼합콘텐츠로 엑박이 잘 뜹니다
          그래서 '다운로드 -> WP 업로드'로 내 사이트에 올린 URL만 쓰도록 강제합니다
    """

    def _normalize_media_url(u: str) -> str:
        u = (u or '').strip()
        if not u:
            return ''
        # WP가 http로 반환하면 https 사이트에서 엑박(혼합콘텐츠) 발생 가능
        if cfg.wp.base_url.startswith('https://') and u.startswith('http://'):
            try:
                from urllib.parse import urlsplit, urlunsplit
                b = urlsplit(cfg.wp.base_url)
                x = urlsplit(u)
                if x.netloc == b.netloc:
                    u = urlunsplit(('https', x.netloc, x.path, x.query, x.fragment))
            except Exception:
                pass
        return u

    # 1) URL 후보(원본/기본 썸네일)
    url_candidates: list[tuple[str, str]] = []

    def _add_url(label: str, u: str) -> None:
        u = (u or '').strip()
        if not u:
            return
        url_candidates.append((label, u))
        if u.startswith('http://'):
            # https도 되는 경우가 많아서 2안으로 같이 시도
            url_candidates.append((label + '_https', 'https://' + u[len('http://') :]))

    _add_url('primary', primary_image_url)
    _add_url('default', cfg.img.default_thumb_url)

    for label, url in url_candidates:
        try:
            content, ctype = download_image_bytes(url, timeout_sec=25)
            ext, ctype2 = _guess_ext_and_type(ctype)
            h = hashlib.sha1((label + '|' + url).encode('utf-8')).hexdigest()[:12]
            filename = f'korean_recipe_{h}{ext}'

            if cfg.img.reuse_media_by_search:
                found = wp_find_media_by_search(cfg.wp, search=f'korean_recipe_{h}')
                if found:
                    mid, murl = found
                    return int(mid), _normalize_media_url(murl)

            mid, murl = wp_upload_media_bytes(cfg.wp, content, filename, ctype2)
            return int(mid), _normalize_media_url(murl)
        except Exception as e:
            if cfg.run.debug:
                print(f'[IMG] {label} failed:', repr(e))
            continue

    # 2) OpenAI 이미지(키/토글 있으면)
    if cfg.openai_img.use_openai_image:
        try:
            gen = openai_generate_image_bytes(cfg, keyword)
            if gen:
                content, ctype = gen
                ext, ctype2 = _guess_ext_and_type(ctype)
                h = hashlib.sha1(('openai|' + keyword).encode('utf-8')).hexdigest()[:12]
                filename = f'korean_recipe_openai_{h}{ext}'

                if cfg.img.reuse_media_by_search:
                    found = wp_find_media_by_search(cfg.wp, search=f'korean_recipe_openai_{h}')
                    if found:
                        mid, murl = found
                        return int(mid), _normalize_media_url(murl)

                mid, murl = wp_upload_media_bytes(cfg.wp, content, filename, ctype2)
                return int(mid), _normalize_media_url(murl)
        except Exception as e:
            if cfg.run.debug:
                print('[IMG] openai failed:', repr(e))

    # 3) Unsplash Source 최후 fallback
    try:
        url = unsplash_source_url(keyword)
        content, ctype = download_image_bytes(url, timeout_sec=25)
        ext, ctype2 = _guess_ext_and_type(ctype)
        h = hashlib.sha1(('unsplash|' + keyword).encode('utf-8')).hexdigest()[:12]
        filename = f'korean_recipe_unsplash_{h}{ext}'

        if cfg.img.reuse_media_by_search:
            found = wp_find_media_by_search(cfg.wp, search=f'korean_recipe_unsplash_{h}')
            if found:
                mid, murl = found
                return int(mid), _normalize_media_url(murl)

        mid, murl = wp_upload_media_bytes(cfg.wp, content, filename, ctype2)
        return int(mid), _normalize_media_url(murl)
    except Exception as e:
        if cfg.run.debug:
            print('[IMG] unsplash fallback failed:', repr(e))

    return 0, ''


# -----------------------------
# Content builder (homefeed)
# -----------------------------

def _esc(s: str) -> str:
    return html.escape(s or "")


def _no_period(s: str) -> str:
    if not s:
        return ""
    # 마침표/강한 종결부호 제거
    s = re.sub(r"[\.\!\?\u3002\u2026]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_lines(items: List[str], limit: int = 30) -> List[str]:
    out: List[str] = []
    for x in items or []:
        t = re.sub(r"\s+", " ", str(x or "")).strip()
        t = _no_period(t)
        t = t.replace("•", " ").replace("·", " ")
        t = re.sub(r"^[-\*\u2022\u25CF\u25A0]+\s*", "", t)
        t = t.strip("- ")
        if not t:
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _intro_200_300(keyword: str) -> str:
    kw = _no_period(keyword).strip()
    chunks = [
        f"요즘 {kw} 생각나면 괜히 마음이 바빠지지 않으세요",
        "저는 이런 날에 막 레시피를 뒤적이다가도 결국 실패 확률 낮은 걸 찾게 되더라고요",
        "그래서 오늘은 진짜 집에서 해먹기 좋은 흐름으로 정리해봤어요",
        "재료는 과하게 늘리지 않고 맛이 갈리는 포인트만 잡아드릴게요",
        "읽다가 바로 따라 할 수 있게 순서도 같이 적어둘게요",
    ]
    txt = " ".join(chunks)

    # 길이 맞추기(200~300자 목표)
    if len(txt) < 200:
        txt += " " + " ".join([
            "오늘 같은 날엔 따뜻한 한 그릇이 진짜 위로가 되잖아요",
            "한 번만 해보시면 다음번은 훨씬 편해지실 거예요",
        ])

    if len(txt) > 300:
        txt = txt[:300].rsplit(" ", 1)[0]

    return _no_period(txt)


def _pick_benefit_phrase(title: str) -> str:
    phrases = [
        "실패 확률 낮게 정리",
        "재료 간단 버전",
        "초보도 가능한 레시피",
        "맛 포인트만 콕",
        "집밥으로 딱 좋은 메뉴",
        "15분 안팎 완성",
    ]
    if any(k in title for k in ["국", "찌개", "탕"]):
        phrases.append("국물맛 깔끔하게")
    if any(k in title for k in ["볶음", "조림"]):
        phrases.append("간 딱 맞게")
    return random.choice(phrases)


def _title_templates(keyword: str) -> Dict[str, List[str]]:
    kw = _no_period(keyword).strip()
    return {
        "benefit": [
            f"{kw} 이번에 해먹은 사람만 '맛이 확 달라지는 포인트' 듣고 놀랐다",
            f"{kw} 한 번만 이렇게 했더니 '향이 올라오는 순간'이 딱 오더라고요",
            f"{kw} 재료 조금만 바꿨는데 '집밥 퀄리티'가 말이 안 됐다",
        ],
        "threat": [
            f"{kw} 이 순서만 놓치면 '맛이 반으로' 줄어든다 저도 처음엔 몰랐다",
            f"{kw} 간을 중간에 맞추면 '짜게 망할 확률'이 올라가요",
            f"{kw} 불 조절 놓치면 '텁텁함'이 남는다 여기만 조심해요",
        ],
        "curiosity": [
            f"{kw} 왜 밖에서 먹는 맛이 안 날까 '한 가지 차이'가 있더라고요",
            f"{kw} 실패하는 이유는 딱 하나였다 오늘은 그 부분만 잡아드릴게요",
            f"{kw} 만들면서 설레는 이유 알고 보니 '이 타이밍' 때문이었다",
        ],
        "compare": [
            f"{kw} 집에서 만든 맛 vs 밖에서 먹는 맛 결국 차이는 여기였어요",
            f"{kw} 진한 맛 vs 깔끔한 맛 누가 더 만족할까 기준을 정리했어요",
            f"{kw} 냉장 재료 vs 냉동 재료 결과 차이가 은근 크더라고요",
        ],
    }


def build_post_title(date_str: str, slot_label: str, keyword: str, title_style: str) -> str:
    tpls = _title_templates(keyword)
    style = title_style
    if style == "random":
        style = random.choice(["benefit", "threat", "curiosity", "compare"])
    base = random.choice(tpls.get(style, tpls["curiosity"]))
    return _no_period(f"{base} ({date_str} {slot_label})")


def build_hashtags(title: str, extra: List[str], max_count: int = 14) -> str:
    base = [
        "한식레시피",
        "집밥",
        "오늘뭐먹지",
        "간단요리",
        "집밥메뉴",
        "요리",
        "레시피",
        "한식",
    ]

    # 제목 토큰
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", title or "")
    toks = [x.strip() for x in t.split() if len(x.strip()) >= 2]
    for tok in toks[:6]:
        base.append(tok.replace(" ", ""))

    for x in (extra or [])[:10]:
        x = re.sub(r"\s+", "", x)
        if x:
            base.append(x)

    uniq: List[str] = []
    seen = set()
    for x in base:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    n = max(8, min(max_count, 20))
    return " ".join([f"#{x}" for x in uniq[:n]])


def _text_len_from_html(s: str) -> int:
    # 아주 러프하게 태그 제거 후 길이
    t = re.sub(r"<[^>]+>", "", s or "")
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return len(t)


def build_body_html(cfg: AppConfig, recipe: Recipe, display_img_url: str) -> str:
    title = _no_period(recipe.title)

    intro = _intro_200_300(title)

    ing = _clean_lines(recipe.ingredients, limit=22)
    stp = _clean_lines(recipe.steps, limit=12)

    # 재료/순서 줄바꿈 형태(불릿/점 제거)
    ing_lines = "<br/>".join([_esc(x) for x in ing]) if ing else _esc("재료 정보가 비어있어서 기본 재료 기준으로 진행해요")
    step_lines = "<br/>".join([_esc(f"{i+1} {x}") for i, x in enumerate(stp)]) if stp else _esc("조리 과정 정보가 비어있어서 기본 흐름으로 안내해요")

    sec1 = " ".join(
        [
            "제가 이 메뉴를 좋아하는 이유가 딱 하나가 아니거든요",
            "손이 많이 갈 것 같아 보여도 막상 해보면 흐름이 단순해서 마음이 편해져요",
            "특히 맛이 갈리는 포인트가 몇 군데만 잡히면 그 다음부터는 거의 자동으로 맛이 나요",
            "처음 하실 때는 욕심내지 말고 재료를 줄여서 시작하시는 게 오히려 성공 확률이 높더라고요",
            "그리고 간은 중간에 확 맞추기보다 마지막에 살짝 조정하는 쪽이 훨씬 안전해요",
        ]
    )

    sec1_2 = " ".join(
        [
            "저도 예전에 레시피만 믿고 중간에 간을 세게 넣었다가 짜져서 속상했던 적이 있어요",
            "그 뒤로는 불 세기랑 시간만 지키고 간은 끝에만 손대는 습관이 생겼는데 그게 진짜 도움이 됐어요",
            "오늘은 그 흐름대로만 같이 가보시면 좋겠어요",
        ]
    )

    sec2 = " ".join(
        [
            "재료는 많을수록 맛있을 것 같지만 실제로는 핵심 재료만 정확하면 충분하더라고요",
            "아래 목록은 오늘 기준으로 가장 무난하게 맛이 나는 조합이에요",
            "집에 있는 재료로 바꿔도 되는 건 많으니까 부담 없이 보셔도 돼요",
        ]
    )

    sec2_tip = " ".join(
        [
            "작은 팁 하나만 더 말씀드리면",
            "양념류는 한 번에 왕창 넣지 말고 조금씩 추가하는 게 실패를 줄여줘요",
            "그리고 국물이나 소스는 끓는 동안 농도가 바뀌니까 마지막 간 조절이 정말 중요해요",
        ]
    )

    sec3 = " ".join(
        [
            "만드는 순서는 아래처럼만 따라가시면 돼요",
            "중요한 건 속도가 아니라 타이밍이라서",
            "각 단계에서 30초만 더 기다릴지 말지 그 감각만 잡으시면 결과가 안정적으로 나와요",
        ]
    )

    sec3_tip = " ".join(
        [
            "마무리할 때 맛이 심심하면 소금보다 국간장이나 된장처럼 감칠맛 있는 쪽으로 아주 조금만 보강해보세요",
            "반대로 짜졌으면 물을 붓기보다 두부나 채소를 추가해서 밸런스를 맞추는 게 더 자연스럽더라고요",
            "그리고 다음번에 더 맛있게 하려면 오늘 드셨던 느낌을 한 줄만 메모해두시면 진짜 편해요",
        ]
    )

    # 해시태그
    hashtags = build_hashtags(title, cfg.tags.tag_names, max_count=14)

    img_html = ""
    if cfg.img.embed_in_body and display_img_url:
        img_html = f'<p><img src="{_esc(display_img_url)}" alt="{_esc(title)}"/></p>'

    body = ""
    body += img_html
    body += f"<p>{_esc(intro)}</p>"

    body += "<h3><strong>오늘 이 메뉴를 추천하는 이유</strong></h3>"
    body += f"<p>{_esc(_no_period(sec1))}</p>"
    body += f"<p>{_esc(_no_period(sec1_2))}</p>"

    body += "<h3><strong>레시피 목록  재료 준비</strong></h3>"
    body += f"<p>{_esc(_no_period(sec2))}</p>"
    body += f"<p>{ing_lines}</p>"
    body += f"<p>{_esc(_no_period(sec2_tip))}</p>"

    body += "<h3><strong>레시피 목록  만드는 순서</strong></h3>"
    body += f"<p>{_esc(_no_period(sec3))}</p>"
    body += f"<p>{step_lines}</p>"
    body += f"<p>{_esc(_no_period(sec3_tip))}</p>"

    body += f"<p>{_esc(hashtags)}</p>"

    # 글자수 보정
    cur_len = _text_len_from_html(body)
    if cur_len < cfg.content.min_total_chars:
        pads = [
            "혹시 오늘 해보시고 맛이 조금 아쉽다면 다음번에는 불을 한 단계만 낮춰서 시간을 2분 정도 더 주셔보세요",
            "그것만으로도 재료 맛이 더 부드럽게 합쳐져서 집밥 느낌이 확 살아나더라고요",
            "그리고 저는 마지막에 대파나 참기름 같은 향을 살짝 더해주면 기분까지 좋아져서 자주 그렇게 마무리해요",
            "이렇게 한 번만 성공해두면 비슷한 메뉴는 다 응용이 되니까 오늘이 제일 중요한 날이에요",
        ]
        i = 0
        while cur_len < cfg.content.min_total_chars and i < len(pads):
            body += f"<p>{_esc(_no_period(pads[i]))}</p>"
            cur_len = _text_len_from_html(body)
            i += 1

    # 너무 길면 뒤쪽만 살짝 줄이기
    if cur_len > cfg.content.max_total_chars:
        # 해시태그는 유지하고, 마지막 2~3개 패딩 문단만 제거하는 정도
        body2 = re.sub(r"(<p>[^<]{10,}</p>)\s*(<p>[^<]{10,}</p>)\s*$", "", body)
        if _text_len_from_html(body2) >= cfg.content.min_total_chars:
            body = body2

    return body


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

    # 레시피 선택
    chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(recent_pairs)

    keyword = chosen.title

    # 제목
    title = build_post_title(date_str, slot_label, keyword, cfg.content.title_style)
    slug = f"korean-recipe-{date_str}-{slot}"

    # 태그 IDs(자동 생성)
    tag_ids: List[int] = []
    if cfg.tags.auto_tags:
        # 제목에서 토큰 몇 개 + TAG_NAMES
        extra = []
        t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", keyword or "")
        toks = [x.strip() for x in t.split() if len(x.strip()) >= 2]
        extra.extend(toks[:4])
        extra.extend(cfg.tags.tag_names)
        # 중복 제거
        uniq: List[str] = []
        seen = set()
        for x in extra:
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            uniq.append(x)
        uniq = uniq[: max(1, cfg.tags.max_auto_tags)]
        for name in uniq:
            tid = wp_get_or_create_tag_id(cfg.wp, name)
            if tid:
                tag_ids.append(tid)

    # 이미지 확보
    media_id = 0
    media_url = ""
    display_img_url = ""

    if cfg.img.upload_thumb:
        try:
            # primary image url: MFDS recipe image_url
            media_id, media_url = ensure_featured_media(cfg, keyword=keyword, primary_image_url=chosen.image_url)
        except Exception as e:
            if cfg.run.debug:
                print("[IMG] ensure_featured_media failed:", repr(e))

    display_img_url = (media_url or "").strip()

    # 본문
    body_html = build_body_html(cfg, chosen, display_img_url)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략  HTML 일부")
        print(body_html[:1500])
        return

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    # 게시
    wp_post_id = 0
    wp_link = ""

    try:
        if today_meta and today_meta.get("wp_post_id") and not cfg.run.force_new:
            post_id = int(today_meta["wp_post_id"])
            wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, featured_media=featured_id, tag_ids=tag_ids)
            print("OK(updated):", wp_post_id, wp_link)
        else:
            wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, tag_ids=tag_ids)
            if featured_id:
                try:
                    wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, tag_ids=tag_ids)
                except Exception:
                    pass
            print("OK(created):", wp_post_id, wp_link)
    except Exception as e:
        # 업데이트 실패 시 새로 생성
        if cfg.run.debug:
            print("[WARN] post create/update failed, fallback to create:", repr(e))
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, tag_ids=tag_ids)
        if featured_id:
            try:
                wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, tag_ids=tag_ids)
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

    # 로그 요약
    if media_url:
        print("[IMG] featured uploaded:", media_id, media_url)
    else:
        print("[IMG] featured missing (upload failed or disabled)")


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
