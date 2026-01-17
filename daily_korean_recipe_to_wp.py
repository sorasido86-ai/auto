# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp.py (완전 통합/네이버 복붙 최적화 + 홈피드형 장문 템플릿)

- "한식 레시피만" 매일 자동 업로드 (WordPress)
- 1순위: 식품안전나라(식약처) COOKRCP01 OpenAPI 레시피 DB (MFDS_API_KEY 필요)
- 2순위(폴백): 코드 내장 한식 레시피(한국어)

★ 네이버(블로그) 복사/붙여넣기 최적화
- NAVER_STYLE=1: 스타일/스크립트 최소화, 스캔 가능한 구조 유지
- SCHEMA_MODE=comment|script|off (기본 comment)

★ 제목 템플릿(4종)
- TITLE_STYLE=benefit|threat|curiosity|compare|random (기본 random)
  * 레시피 제목(키워드)에 맞춰 후킹형 제목 생성

★ 본문 템플릿(요청 반영)
- 도입부: 홈피드형 200~300자
- 굵은 소제목 3개: 각 섹션 "1500자 이상" 자동 보강
- 총 글자수는 섹션 규칙을 만족하면 자연히 2300자 이상(대부분 5000자+)이 됩니다
- 말투: 친구에게 진심 담아 수다떠는 존댓말
- 문장 마침표(".")는 본문 텍스트에서 최대한 제거(데이터에 포함된 경우도 정리)

★ 이미지 처리(중요)
- 레시피 이미지가 비어도 DEFAULT_THUMB_URL 기본 이미지가 반드시 썸네일 후보
- 가능하면 WP media로 업로드 후 featured_media(대표이미지) 설정
- 업로드 실패 시에도 본문 상단 이미지는 URL로 삽입

필수 환경변수(Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

권장 환경변수:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS=7 (기본 7)
  - WP_TAG_IDS=1,2,3 (선택)
  - SQLITE_PATH=data/daily_korean_recipe.sqlite3

식품안전나라 OpenAPI:
  - MFDS_API_KEY=...  (없으면 내장 레시피만)

기본 이미지:
  - DEFAULT_THUMB_URL=https://.../your_default_thumb.jpg

OpenAI로 문장 다양화/장문 강화(선택):
  - USE_OPENAI=1
  - OPENAI_API_KEY=...
  - OPENAI_MODEL=... (기본 gpt-4.1-mini)

동작 옵션:
  - RUN_SLOT=day|am|pm (기본 day)
  - FORCE_NEW=0|1 (기본 0)
  - DRY_RUN=0|1 (기본 0)
  - DEBUG=0|1 (기본 0)
  - AVOID_REPEAT_DAYS=90 (기본 90)
  - MAX_TRIES=25 (기본 25)

이미지 옵션:
  - UPLOAD_THUMB=1 (기본 1)
  - SET_FEATURED=1 (기본 1)
  - EMBED_IMAGE_IN_BODY=1 (기본 1)
  - REUSE_MEDIA_BY_SEARCH=1 (기본 1)

네이버형 옵션:
  - NAVER_STYLE=1 (기본 1)
  - SCHEMA_MODE=comment|script|off (기본 comment)
  - HASHTAG_COUNT=12 (기본 12)
  - EMBED_STEP_IMAGES=1 (기본 1)
  - ADD_FAQ=1 (기본 1)
  - ADD_INTERNAL_LINKS=1 (기본 1)
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

DISCLOSURE = "※ 이 글은 레시피 데이터를 기반으로 자동 생성된 포스팅입니다"
SOURCE_NOTE = "데이터 출처 식품안전나라 식약처 OpenAPI 레시피 DB 및 내장 레시피 폴백"
SEO_NOTE = "오늘 뭐 먹지 고민될 때 재료 적고 실패 확률 낮은 레시피로 골라왔어요"


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
    run_slot: str = "day"  # day / am / pm
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 25


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""  # foodsafetykorea openapi key (optional)
    strict_korean: bool = True


@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""
    reuse_media_by_search: bool = True


@dataclass
class ContentConfig:
    naver_style: bool = True
    schema_mode: str = "comment"  # comment|script|off
    hashtag_count: int = 12
    embed_step_images: bool = True
    add_faq: bool = True
    add_internal_links: bool = True
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
    wp_status = _env("WP_STATUS", "publish") or "publish"

    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))  # 기본 7
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

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
            max_tries=_env_int("MAX_TRIES", 25),
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
        ),
        content=ContentConfig(
            naver_style=_env_bool("NAVER_STYLE", True),
            schema_mode=schema_mode,
            hashtag_count=_env_int("HASHTAG_COUNT", 12),
            embed_step_images=_env_bool("EMBED_STEP_IMAGES", True),
            add_faq=_env_bool("ADD_FAQ", True),
            add_internal_links=_env_bool("ADD_INTERNAL_LINKS", True),
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
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] UPLOAD_THUMB:", cfg.img.upload_thumb, "| SET_FEATURED:", cfg.img.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.img.embed_image_in_body)
    print("[CFG] REUSE_MEDIA_BY_SEARCH:", cfg.img.reuse_media_by_search)
    print("[CFG] NAVER_STYLE:", cfg.content.naver_style, "| SCHEMA_MODE:", cfg.content.schema_mode, "| HASHTAG_COUNT:", cfg.content.hashtag_count)
    print("[CFG] EMBED_STEP_IMAGES:", cfg.content.embed_step_images, "| ADD_FAQ:", cfg.content.add_faq, "| ADD_INTERNAL_LINKS:", cfg.content.add_internal_links)
    print("[CFG] TITLE_STYLE:", cfg.content.title_style)
    print("[CFG] USE_OPENAI:", cfg.openai.use_openai, "| OPENAI_API_KEY:", ok(cfg.openai.api_key), "| OPENAI_MODEL:", cfg.openai.model)


# -----------------------------
# SQLite (history + migrate)
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


def get_recent_wp_links(path: str, limit: int = 3) -> List[Tuple[str, str]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT wp_link, recipe_title
        FROM daily_posts
        WHERE wp_link IS NOT NULL AND wp_link != ''
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    con.close()
    out: List[Tuple[str, str]] = []
    for link, title in rows:
        if link:
            out.append((str(link), str(title or "이전 레시피")))
    return out


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/1.2"}


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
    r = requests.get(image_url, timeout=35)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code}")

    content = r.content
    ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()

    if not ctype:
        if filename.lower().endswith(".png"):
            ctype = "image/png"
        elif filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
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
    source: str  # mfds or local
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


def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int = 1, end: int = 60) -> List[Dict[str, Any]]:
    """MFDS OpenAPI 호출.

    - 외부 API(foodsafetykorea)가 종종 느리거나 끊기는 경우가 있어, 타임아웃/일시 오류는 예외로 터뜨리지 않고
      빈 결과([])로 처리해 로컬 레시피 폴백이 가능하도록 한다.
    - 환경변수로 조정 가능:
        MFDS_TIMEOUT (초, 기본 35)
        MFDS_RETRIES (재시도 횟수, 기본 2)
    """
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"

    timeout = _env_int("MFDS_TIMEOUT", 35)
    retries = _env_int("MFDS_RETRIES", 2)

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            # 429/5xx 는 잠깐 쉬고 재시도
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"MFDS transient status={r.status_code}")
                raise last_err
            if r.status_code != 200:
                return []
            try:
                data = r.json()
            except Exception:
                return []
            co = data.get("COOKRCP01") or {}
            rows = co.get("row") or []
            return rows if isinstance(rows, list) else []
        except Exception as e:
            # requests 예외/임시오류 -> 재시도 후 포기
            last_err = e
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
                continue
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
    for _ in range(cfg.run.max_tries):
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, start=1, end=60)
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
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_SEQ", recipe_id, start=1, end=5)
        for row in rows:
            rcp = mfds_row_to_recipe(row)
            if rcp.recipe_id == recipe_id:
                return rcp
        return None

    return None


# -----------------------------
# Blog rendering helpers
# -----------------------------

def _esc(s: str) -> str:
    return html.escape(s or "")


def _no_period(s: str) -> str:
    """본문 문장 마침표 제거용(가능한 범위)
    - 이미지 URL 같은 곳에는 사용하지 말 것
    """
    s = (s or "").replace(".", "")
    s = s.replace("。", "")
    return s


def _plain_len(html_s: str) -> int:
    t = re.sub(r"<[^>]+>", "", html_s or "")
    t = html.unescape(t)
    return len(t)


def _wrap_p(txt: str) -> str:
    return f"<p>{_esc(txt)}</p>"


def _wrap_p_br(lines: List[str]) -> str:
    safe = "<br/>".join([_esc(x) for x in lines if x])
    return f"<p>{safe}</p>"


def _ensure_min_chars(section_html: str, min_chars: int, fillers: List[str]) -> str:
    """섹션 글자수를 min_chars 이상으로 보강
    - fillers는 마침표 없는 문장들로 준비
    """
    out = section_html
    tries = 0
    while _plain_len(out) < min_chars and tries < 60:
        tries += 1
        out += _wrap_p(random.choice(fillers))
    return out


def _clean_title_for_tags(title: str) -> List[str]:
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
    base = [
        "한식레시피",
        "집밥",
        "오늘뭐먹지",
        "간단요리",
        "초간단레시피",
        "자취요리",
        "밥도둑",
        "국물요리",
        "반찬",
        "요리기록",
    ]
    title_tokens = _clean_title_for_tags(recipe.title)
    for tok in title_tokens[:5]:
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

    n = max(5, min(20, cfg.content.hashtag_count))
    return " ".join([f"#{x}" for x in uniq[:n]])


def choose_thumb_url(cfg: AppConfig, recipe: Recipe) -> str:
    # 레시피 이미지가 비어도 기본 이미지로 대체
    return (recipe.image_url or "").strip() or (cfg.img.default_thumb_url or "").strip()


def build_recipe_schema_json(recipe: Recipe, image_url: str, now: datetime) -> str:
    data = {
        "@context": "https://schema.org",
        "@type": "Recipe",
        "name": recipe.title.strip(),
        "image": [image_url] if image_url else [],
        "datePublished": now.astimezone(KST).isoformat(),
        "recipeCuisine": "Korean",
        "recipeCategory": "Korean Recipe",
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
    # 따옴표는 네이버에서 클릭을 유도하는 느낌으로 유지
    return {
        "benefit": [
            f"{kw} 이번에 해먹은 사람만 '맛이 확 달라지는 포인트' 듣고 놀랐다",
            f"{kw} 한 번만 이렇게 했더니 '입맛 당기는 향'이 자동으로 올라왔다",
            f"{kw} 재료 2개만 바꿨는데 '집밥 퀄리티'가 말이 안 됐다",
        ],
        "threat": [
            f"{kw} 이 순서 틀리면 '맛이 반으로' 줄어든다 저도 처음엔 몰랐다",
            f"{kw} 간을 중간에 맞추면 '짜게 망할 확률'이 확 올라가요",
            f"{kw} 불 조절 놓치면 '비린내나 텁텁함'이 남는다 여기 꼭 보세요",
        ],
        "curiosity": [
            f"{kw} 왜 밖에서 먹는 맛이 안 날까 '한 가지 차이'가 있더라고요",
            f"{kw} 실패하는 이유는 딱 하나였다 오늘은 그 부분만 잡아드릴게요",
            f"{kw} 만들면서 설레는 이유 알고 보니 '이 타이밍' 때문이었다",
        ],
        "compare": [
            f"{kw} 집에서 만든 맛 vs 밖에서 먹는 맛 '결국 차이는 여기'였어요",
            f"{kw} 국물파 vs 건더기파 '누가 더 만족할까' 기준을 정리했어요",
            f"{kw} 냉장재료로 만들기 vs 냉동재료로 만들기 '결과 차이'가 컸어요",
        ],
    }


def build_post_title(date_str: str, slot_label: str, keyword: str, title_style: str = "random") -> str:
    tpls = _title_templates(keyword)
    style = title_style
    if style == "random":
        style = random.choice(["benefit", "threat", "curiosity", "compare"])
    base = random.choice(tpls.get(style, tpls["curiosity"]))
    # 날짜는 뒤에 작게
    return _no_period(f"{base} ({date_str} {slot_label})")


# -----------------------------
# Body templates
# -----------------------------

def _build_intro_200_300(keyword: str) -> str:
    kw = _no_period(keyword).strip()
    # 홈피드형 도입부 200~300자 목표
    chunks = [
        f"요즘 {kw} 생각나면 괜히 마음이 좀 바빠지지 않으세요",
        "뭘 거창하게 하려는 게 아니라 오늘 하루를 편하게 넘기고 싶은 마음이잖아요",
        "그래서 오늘은 따라만 해도 맛이 나오는 흐름으로 정리해봤어요",
        "재료는 복잡하게 늘리지 않고 실패 포인트만 조심하는 방식이에요",
        "읽다가 그대로 장바구니에 담기 좋게 적어둘게요",
    ]
    txt = " ".join(chunks)
    # 길이 맞추기
    if len(txt) < 200:
        txt += " " + " ".join(
            [
                "특히 간 맞추는 타이밍이랑 불 조절만 잡으면 결과가 진짜 안정적이더라고요",
                "오늘은 그 부분을 제일 쉽게 풀어드릴게요",
            ]
        )
    if len(txt) > 300:
        txt = txt[:298].rstrip()  # 너무 길면 컷
    return _no_period(txt)


def _section_fillers(keyword: str) -> List[str]:
    kw = _no_period(keyword).strip()
    return [
        f"{kw}는 사실 재료보다 순서가 더 중요해요 그래서 한 번만 흐름을 잡아두면 다음부터는 눈 감고도 할 수 있어요",
        "중요한 건 한 번에 완벽하게 하려는 마음을 내려놓는 거예요 오늘은 안전하게 성공하는 쪽으로만 가요",
        "간은 중간에 맞추기보다 마지막에 조절하는 게 훨씬 안정적이에요 중간에 맞추면 졸아들면서 짜지기 쉬워요",
        "불을 세게 해서 빨리 끝내려고 하면 향이 날아가거나 재료가 굳을 때가 있어요 중불에서 천천히가 결국 이득이에요",
        "재료가 하나 빠져도 괜찮아요 핵심은 간 불 시간이라서 그 세 가지만 지키면 맛이 크게 무너지지 않아요",
        "맛이 애매하면 소금이나 간장부터 더 넣기 전에 단맛이나 향을 아주 조금만 더해보는 게 의외로 해결이 빨라요",
        "처음엔 양을 줄여서 해보는 것도 좋아요 성공 경험을 한 번 만들면 다음부터 마음이 편해져요",
        "남은 음식은 다음날이 더 맛있을 때가 있어요 대신 재가열할 때 간을 다시 보지 말고 마지막에만 살짝 잡아주세요",
        "저는 이런 레시피는 저장해두고 장볼 때마다 하나씩 돌려보는 편이에요 그러면 식단 고민이 훨씬 줄어들어요",
        f"오늘 {kw}는 과하게 꾸미지 않고도 충분히 만족스럽게 나오는 쪽으로 잡았어요",
    ]


def _render_ingredients_lines(ingredients: List[str]) -> List[str]:
    if not ingredients:
        return ["재료 정보가 비어있어요"]
    out = []
    for x in ingredients:
        x = _no_period(str(x).strip())
        if not x:
            continue
        out.append(f"- {x}")
    return out or ["재료 정보가 비어있어요"]


def _render_steps_lines(steps: List[str]) -> List[str]:
    if not steps:
        return ["과정 정보가 비어있어요"]
    out = []
    for i, s in enumerate(steps, start=1):
        ss = _no_period(str(s).strip())
        if not ss:
            continue
        out.append(f"{i} {ss}")
    return out or ["과정 정보가 비어있어요"]


def build_body_html(
    cfg: AppConfig,
    now: datetime,
    run_slot_label: str,
    recipe: Recipe,
    display_img_url: str = "",
    recent_links: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str, str]:
    """returns (body_html, excerpt)"""
    keyword = _no_period(recipe.title.strip())

    # excerpt는 도입부에서 뽑고 140자 제한
    intro_txt = _build_intro_200_300(keyword)
    excerpt = (intro_txt[:140]).strip()

    schema_block = build_schema_block(cfg, recipe, display_img_url, now)

    # 상단 이미지
    img_html = ""
    if cfg.img.embed_image_in_body and display_img_url:
        img_html = f'<p><img src="{_esc(display_img_url)}" alt="{_esc(keyword)}"/></p>'

    # 공통 안내
    head_lines = [
        DISCLOSURE,
        f"기준시각 {now.astimezone(KST).strftime('%Y-%m-%d %H:%M')}  슬롯 {run_slot_label}",
        SEO_NOTE,
        SOURCE_NOTE,
    ]
    head_block = _wrap_p_br([_no_period(x) for x in head_lines])

    # --------
    # 섹션 3개
    # --------
    fillers = _section_fillers(keyword)

    # 섹션 1: 재료/성공 포인트
    h1 = f"{keyword}  맛이 갈리는 재료 포인트  딱 여기만 보면 돼요"
    sec1 = f"<h3><strong>{_esc(_no_period(h1))}</strong></h3>"
    sec1 += _wrap_p(
        _no_period(
            f"제가 이 메뉴를 좋아하는 이유가 있어요  바쁜 날에도 마음이 편해지거든요  그런데 {keyword}는 대충 하면 대충한 티가 바로 나요  그래서 오늘은 재료를 과하게 늘리지 않으면서도 맛이 안정적으로 나오게 하는 쪽으로만 잡아볼게요"
        )
    )
    sec1 += _wrap_p(
        _no_period(
            "장보실 때 제일 먼저 볼 건 신선함보다도 밸런스예요  짠맛 신맛 단맛 향 이 네 가지가 어느 하나로 치우치지 않게만 잡으면 집밥이 갑자기 업그레이드되거든요  특히 국물류나 볶음류는 간을 빨리 맞추려고 조급해질수록 실패 확률이 올라가요"
        )
    )
    sec1 += _wrap_p_br(
        [
            _no_period("오늘 재료는 이렇게만 준비하시면 돼요  체크하면서 보시면 편해요"),
            *_render_ingredients_lines(recipe.ingredients),
            _no_period("재료가 하나 빠져도 괜찮아요  대신 핵심 재료 한두 개는 꼭 챙기는 게 좋아요"),
        ]
    )
    sec1 += _wrap_p(
        _no_period(
            f"여기서 포인트 하나만 더요  {keyword}는 처음부터 강불로 몰아가면 향이 깨지거나 식감이 애매해질 때가 있어요  중불에서 천천히 시작해서 재료가 익는 속도를 맞춰주면 결과가 훨씬 안정적이에요  그리고 간은 마지막에 한 번만 잡는다고 생각하시면 마음이 편해요"
        )
    )

    sec1 = _ensure_min_chars(sec1, 1500, fillers)

    # 섹션 2: 만드는 과정 상세
    h2 = f"{keyword}  만드는 법  그대로 따라만 해도 맛이 나오는 흐름"
    sec2 = f"<h3><strong>{_esc(_no_period(h2))}</strong></h3>"
    sec2 += _wrap_p(
        _no_period(
            "조리 과정은요  어려운 기술이 필요한 게 아니라 타이밍을 놓치지 않는 게 전부예요  그래서 오늘은 과정 자체를 길게 말하기보다  어디서 흔들리는지  그 흔들림을 어떻게 잡는지  그걸 중심으로 수다 떨듯이 풀어볼게요"
        )
    )
    sec2 += _wrap_p(
        _no_period(
            "제가 제일 자주 보는 실패가 두 가지예요  하나는 양념을 너무 빨리 넣어서 향이 날아가버리는 경우  또 하나는 간을 중간에 맞춰서 마지막에 짜지는 경우예요  이 두 가지만 피하면 웬만하면 성공해요"
        )
    )
    sec2 += _wrap_p_br(
        [
            _no_period("조리 순서는 이렇게 가요  중간중간 포인트도 같이 적어둘게요"),
            *_render_steps_lines(recipe.steps),
        ]
    )
    sec2 += _wrap_p(
        _no_period(
            f"그리고 꼭 이 느낌만 기억해두세요  {keyword}는 완성 직전에 맛이 한 번 더 올라가요  그 순간에 간을 확정하면 좋고요  혹시 맛이 애매하면 소금이나 간장을 더하기 전에  향을 살리는 재료를 아주 소량만 더해보는 게 의외로 빠르게 해결돼요"
        )
    )

    # 과정 이미지 1~3장 (추가 소제목 없이 본문에만)
    if cfg.content.embed_step_images and recipe.step_images:
        imgs = [u for u in recipe.step_images[:3] if u.startswith("http")]
        if imgs:
            sec2 += _wrap_p(_no_period("과정 사진은 참고로만 살짝 붙여둘게요  복붙해도 흐트러지지 않게요"))
            for u in imgs:
                sec2 += f'<p><img src="{_esc(u)}" alt="{_esc(keyword)} 과정사진"/></p>'

    sec2 = _ensure_min_chars(sec2, 1500, fillers)

    # 섹션 3: 응용 보관 먹는 법 + CTA
    h3 = f"{keyword}  더 맛있게 먹는 법  그리고 다음에 더 쉬워지는 팁"
    sec3 = f"<h3><strong>{_esc(_no_period(h3))}</strong></h3>"
    sec3 += _wrap_p(
        _no_period(
            "여기까지 오면 사실 절반은 끝난 거예요  나머지 절반은 내 입맛에 맞게 아주 조금만 조절하는 거예요  그게 진짜 집밥의 매력이잖아요  오늘은 과하게 바꾸지 말고  딱 한 가지 포인트만 내 취향으로 바꿔보는 걸 추천드려요"
        )
    )
    sec3 += _wrap_p(
        _no_period(
            "응용은 이렇게 해보셔도 좋아요  칼칼함이 필요하면 매운 양념을 마지막에 아주 조금만 추가  단맛이 필요하면 설탕을 확 늘리기보다 올리고당을 아주 소량  향이 필요하면 대파나 마늘을 한 번 더  이런 식으로요  한 번에 크게 바꾸면 실패 확률이 올라가니까요"
        )
    )
    sec3 += _wrap_p(
        _no_period(
            "보관은 밀폐 용기만 잘 써도 편해져요  냉장으로 하루 이틀 정도는 무난하고  다시 데울 때는 꼭 한 번 끓인 뒤에 간을 마지막에만 살짝 보세요  중간에 간을 잡으면 다음날 더 짜게 느껴질 때가 있거든요"
        )
    )

    if cfg.content.add_faq:
        faq_lines = [
            _no_period("자주 묻는 질문도 짧게 정리해볼게요"),
            _no_period("간이 심심해요  언제 보강하나요  마지막에만 국간장이나 소금으로 조절해요"),
            _no_period("재료가 하나 빠졌어요  대체 가능해요  핵심은 간 불 시간이라  한두 개 빠져도 괜찮아요"),
            _no_period("다음엔 더 맛있게 하려면  오늘 만든 뒤 간과 매운맛만 메모해두면 다음번이 훨씬 쉬워져요"),
        ]
        sec3 += _wrap_p_br(faq_lines)

    # 내부 링크(지난 글 3개) - 별도 소제목 없이
    if cfg.content.add_internal_links and recent_links:
        items = [(link, t) for link, t in recent_links if link]
        if items:
            sec3 += _wrap_p(_no_period("지난 레시피도 같이 두고 갈게요  시간 날 때 하나씩 돌려보면 식단 고민이 확 줄어요"))
            for link, t in items:
                # 링크 텍스트는 마침표 제거만, URL은 그대로
                sec3 += f"<p><a href='{_esc(link)}'>{_esc(_no_period(t))}</a></p>"

    # 저장/댓글 유도
    sec3 += _wrap_p(
        _no_period(
            "이 글은 저장해두면 진짜 편해요  다음에 오늘 뭐 먹지 할 때 고민 시간이 확 줄거든요  저는 이런 글은 장보기 전에 한 번씩만 훑어보는 편이에요"
        )
    )
    sec3 += _wrap_p(
        _no_period(
            f"댓글로 하나만 알려주세요  {keyword} 만들 때 여러분은 어떤 재료를 추가하는 편이세요  버섯 두부 대파 계란 같은 조합도 좋고  입맛 취향대로 추천해주시면 저도 다음 글에 반영해볼게요"
        )
    )

    hashtags = build_hashtags(cfg, recipe)
    sec3 += "<hr/>" + _wrap_p(_no_period(hashtags))

    sec3 = _ensure_min_chars(sec3, 1500, fillers)

    # 본문 합치기
    body_html = schema_block
    body_html += head_block
    body_html += _wrap_p(intro_txt)
    body_html += img_html
    body_html += sec1 + sec2 + sec3

    return body_html, excerpt


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
아래 레시피 제목 재료 과정은 내용이 바뀌면 안 돼 그대로 유지해
대신 말투를 친구에게 수다 떠는 존댓말로 더 자연스럽고 길게 다듬어줘
마침표는 가능한 쓰지 말고 문장 호흡은 띄어쓰기와 줄바꿈으로 만들어줘
굵은 소제목은 반드시 3개만 유지하고 각 섹션이 충분히 길게 유지되게 해줘
HTML 형태로만 출력해 코드블럭 금지
과장된 광고 문구 금지

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
    """중복 업로드 방지 가능한 경우 search로 재사용, 없으면 업로드"""
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
        chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(cfg, recent_pairs)

    assert chosen is not None

    title = build_post_title(date_str, slot_label, chosen.title, title_style=cfg.content.title_style)
    slug = f"korean-recipe-{date_str}-{slot}"

    # 레시피 이미지 없으면 기본 이미지로
    thumb_url = choose_thumb_url(cfg, chosen)
    if not chosen.image_url and not cfg.img.default_thumb_url:
        print("[WARN] recipe image empty AND DEFAULT_THUMB_URL empty  featured 이미지 없이 발행될 수 있어요")

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

    recent_links = get_recent_wp_links(cfg.sqlite_path, limit=3) if cfg.content.add_internal_links else []
    body_html, excerpt = build_body_html(cfg, now, slot_label, chosen, display_img_url=display_img_url, recent_links=recent_links)

    upgraded = generate_with_openai(cfg, chosen, body_html)
    if upgraded:
        body_html = upgraded

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략  미리보기 HTML 일부")
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
            print("[WARN] post create update failed, fallback to create:", repr(e))
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
