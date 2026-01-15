# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp.py (통합/안정화 + 네이버 복붙 친화 + 사람 말투 강화)
- TheMealDB 랜덤 레시피 수집
- OpenAI로 한국어 블로그톤(자연스러운 도입/짧은 문단/요약/포인트/실패방지/FAQ/해시태그) 생성
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정
- SQLite 발행 이력 + 스키마 자동 마이그레이션
- ✅ OpenAI 실패(크레딧/일시 오류 등) 시: 무료번역(LibreTranslate) + 템플릿으로 자연스러운 최소 글 폴백

필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY

선택 env:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS="7" (기본 7)
  - WP_TAG_IDS="1,2,3" (선택)
  - SQLITE_PATH=data/daily_recipe.sqlite3 (기본)

  - RUN_SLOT=day/am/pm (기본 day)
  - DRY_RUN=1
  - DEBUG=1

  - OPENAI_MODEL=gpt-5.2 (기본 gpt-5.2)
  - STRICT_KOREAN=1 (기본 1)
  - FORCE_NEW=0 (기본 0)
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - OPENAI_MAX_RETRIES=3

네이버 느낌 옵션(선택):
  - NAVER_STYLE=1 (기본 1)  # 짧은 문단/요약/FAQ/해시태그 강화
  - BLOG_TONE=home  # home(기본) | diary | simple
  - HASHTAG_COUNT=12
  - EXTRA_HASHTAGS="#집밥 #오늘뭐먹지"  (공백 구분)
  - PREFER_AREAS="Korean,Japanese"  # 없으면 전세계 랜덤
  - PREFER_KEYWORDS="kimchi,bulgogi" # 제목에 키워드 포함 우선(선택)

무료번역 폴백(LibreTranslate) 선택 env:
  - FREE_TRANSLATE_URL=https://libretranslate.de/translate   (기본값)
  - FREE_TRANSLATE_API_KEY=   (필요한 인스턴스면 넣기)
  - FREE_TRANSLATE_SOURCE=en  (기본 en)
  - FREE_TRANSLATE_TARGET=ko  (기본 ko)
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import openai  # 예외 타입 용도
from openai import OpenAI  # 공식 SDK

KST = timezone(timedelta(hours=9))

THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"
THEMEALDB_LOOKUP = "https://www.themealdb.com/api/json/v1/1/lookup.php?i={id}"


# -----------------------------
# ENV helpers
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
    v = _env(name, "1" if default else "0")
    return v.lower() in ("1", "true", "yes", "y", "on")


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


def _parse_csv_list(csv: str) -> List[str]:
    out: List[str] = []
    for x in (csv or "").split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def _parse_space_tags(s: str) -> List[str]:
    # "#태그 #태그2" 형태 입력 지원
    if not s:
        return []
    parts = re.split(r"\s+", s.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.startswith("#"):
            p = "#" + p
        out.append(p)
    return out


# -----------------------------
# Config models
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
    run_slot: str = "day"     # day / am / pm
    dry_run: bool = False
    debug: bool = False
    strict_korean: bool = True
    force_new: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    openai_max_retries: int = 3

    # Naver-style writing knobs
    naver_style: bool = True
    blog_tone: str = "home"  # home | diary | simple
    hashtag_count: int = 12
    extra_hashtags: List[str] = field(default_factory=list)

    prefer_areas: List[str] = field(default_factory=list)
    prefer_keywords: List[str] = field(default_factory=list)


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-5.2"


@dataclass
class FreeTranslateConfig:
    url: str = "https://libretranslate.de/translate"
    api_key: str = ""
    source: str = "en"
    target: str = "ko"


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    sqlite_path: str
    openai: OpenAIConfig
    free_tr: FreeTranslateConfig


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    sqlite_path = _env("SQLITE_PATH", "data/daily_recipe.sqlite3")

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    dry_run = _env_bool("DRY_RUN", False)
    debug = _env_bool("DEBUG", False)

    strict_korean = _env_bool("STRICT_KOREAN", True)
    force_new = _env_bool("FORCE_NEW", False)

    avoid_repeat_days = _env_int("AVOID_REPEAT_DAYS", 90)
    max_tries = _env_int("MAX_TRIES", 20)

    upload_thumb = _env_bool("UPLOAD_THUMB", True)
    set_featured = _env_bool("SET_FEATURED", True)
    embed_image_in_body = _env_bool("EMBED_IMAGE_IN_BODY", True)

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2"
    openai_max_retries = _env_int("OPENAI_MAX_RETRIES", 3)

    free_url = _env("FREE_TRANSLATE_URL", "https://libretranslate.de/translate")
    free_api_key = _env("FREE_TRANSLATE_API_KEY", "")
    free_src = _env("FREE_TRANSLATE_SOURCE", "en")
    free_tgt = _env("FREE_TRANSLATE_TARGET", "ko")

    naver_style = _env_bool("NAVER_STYLE", True)
    blog_tone = (_env("BLOG_TONE", "home") or "home").lower()
    if blog_tone not in ("home", "diary", "simple"):
        blog_tone = "home"

    hashtag_count = _env_int("HASHTAG_COUNT", 12)
    extra_hashtags = _parse_space_tags(_env("EXTRA_HASHTAGS", ""))

    prefer_areas = _parse_csv_list(_env("PREFER_AREAS", ""))
    prefer_keywords = _parse_csv_list(_env("PREFER_KEYWORDS", ""))

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
            dry_run=dry_run,
            debug=debug,
            strict_korean=strict_korean,
            force_new=force_new,
            avoid_repeat_days=avoid_repeat_days,
            max_tries=max_tries,
            upload_thumb=upload_thumb,
            set_featured=set_featured,
            embed_image_in_body=embed_image_in_body,
            openai_max_retries=openai_max_retries,
            naver_style=naver_style,
            blog_tone=blog_tone,
            hashtag_count=hashtag_count,
            extra_hashtags=extra_hashtags,
            prefer_areas=prefer_areas,
            prefer_keywords=prefer_keywords,
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
        free_tr=FreeTranslateConfig(url=free_url, api_key=free_api_key, source=free_src, target=free_tgt),
    )


def validate_cfg(cfg: AppConfig) -> None:
    missing = []
    if not cfg.wp.base_url:
        missing.append("WP_BASE_URL")
    if not cfg.wp.user:
        missing.append("WP_USER")
    if not cfg.wp.app_pass:
        missing.append("WP_APP_PASS")
    if not cfg.openai.api_key:
        missing.append("OPENAI_API_KEY")
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot)
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] STRICT_KOREAN:", cfg.run.strict_korean, "| FORCE_NEW:", cfg.run.force_new)
    print("[CFG] AVOID_REPEAT_DAYS:", cfg.run.avoid_repeat_days, "| MAX_TRIES:", cfg.run.max_tries)
    print("[CFG] UPLOAD_THUMB:", cfg.run.upload_thumb, "| SET_FEATURED:", cfg.run.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.run.embed_image_in_body)
    print("[CFG] NAVER_STYLE:", cfg.run.naver_style, "| BLOG_TONE:", cfg.run.blog_tone, "| HASHTAG_COUNT:", cfg.run.hashtag_count)
    print("[CFG] EXTRA_HASHTAGS:", " ".join(cfg.run.extra_hashtags) if cfg.run.extra_hashtags else "(none)")
    print("[CFG] PREFER_AREAS:", ",".join(cfg.run.prefer_areas) if cfg.run.prefer_areas else "(none)")
    print("[CFG] PREFER_KEYWORDS:", ",".join(cfg.run.prefer_keywords) if cfg.run.prefer_keywords else "(none)")
    print("[CFG] OPENAI_API_KEY:", ok(cfg.openai.api_key))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model)
    print("[CFG] OPENAI_MAX_RETRIES:", cfg.run.openai_max_retries)
    print("[CFG] FREE_TRANSLATE_URL:", cfg.free_tr.url)
    print("[CFG] FREE_TRANSLATE_SOURCE/TARGET:", cfg.free_tr.source, "->", cfg.free_tr.target)


# -----------------------------
# SQLite (history) + migration
# -----------------------------
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_posts (
  date_key TEXT PRIMARY KEY,
  slot TEXT,
  recipe_id TEXT,
  recipe_title TEXT,
  wp_post_id INTEGER,
  wp_link TEXT,
  media_id INTEGER,
  media_url TEXT,
  created_at TEXT
)
"""

REQUIRED_COLUMNS: Dict[str, str] = {
    "date_key": "TEXT",
    "slot": "TEXT",
    "recipe_id": "TEXT",
    "recipe_title": "TEXT",
    "wp_post_id": "INTEGER",
    "wp_link": "TEXT",
    "media_id": "INTEGER",
    "media_url": "TEXT",
    "created_at": "TEXT",
}


def init_db(path: str, debug: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(TABLE_SQL)
    con.commit()

    cur.execute("PRAGMA table_info(daily_posts)")
    cols = {row[1] for row in cur.fetchall()}

    for col, typ in REQUIRED_COLUMNS.items():
        if col not in cols:
            if debug:
                print(f"[DB] add column: {col} {typ}")
            cur.execute(f"ALTER TABLE daily_posts ADD COLUMN {col} {typ}")

    con.commit()
    con.close()


def get_today_post(path: str, date_key: str) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "SELECT date_key, slot, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at "
        "FROM daily_posts WHERE date_key = ?",
        (date_key,),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "date_key": row[0],
        "slot": row[1],
        "recipe_id": row[2],
        "recipe_title": row[3],
        "wp_post_id": row[4],
        "wp_link": row[5],
        "media_id": row[6],
        "media_url": row[7],
        "created_at": row[8],
    }


def save_post_meta(
    path: str,
    date_key: str,
    slot: str,
    recipe_id: str,
    recipe_title: str,
    wp_post_id: int,
    wp_link: str,
    media_id: Optional[int],
    media_url: str,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_key, slot, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            date_key,
            slot,
            recipe_id,
            recipe_title,
            wp_post_id,
            wp_link,
            int(media_id) if media_id is not None else None,
            media_url or "",
            datetime.utcnow().isoformat(),
        ),
    )
    con.commit()
    con.close()


def get_recent_recipe_ids(path: str, days: int) -> List[str]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT recipe_id FROM daily_posts WHERE created_at >= ? AND recipe_id IS NOT NULL", (cutoff,))
    rows = cur.fetchall()
    con.close()
    return [str(r[0]) for r in rows if r and r[0]]


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-recipe-bot/1.2"}


def wp_find_post_by_slug(cfg: WordPressConfig, slug: str) -> Optional[int]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts?slug={slug}&per_page=1&context=edit"
    r = requests.get(url, headers=wp_auth_header(cfg.user, cfg.app_pass), timeout=20)
    if r.status_code != 200:
        return None
    arr = r.json()
    if isinstance(arr, list) and arr:
        try:
            return int(arr[0].get("id"))
        except Exception:
            return None
    return None


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str, featured_media: Optional[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html, "status": cfg.status}

    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str, featured_media: Optional[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}

    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "recipe.jpg") -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=30)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code} url={image_url}")

    content = r.content
    ctype = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip().lower()

    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename_hint).strip("-") or "recipe.jpg"
    if "." not in safe_name:
        safe_name += ".jpg"

    headers["Content-Type"] = ctype
    headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    up = requests.post(media_endpoint, headers=headers, data=content, timeout=45)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {up.status_code} body={up.text[:500]}")

    data = up.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# Recipe fetch (TheMealDB)
# -----------------------------
def fetch_random_recipe() -> Dict[str, Any]:
    r = requests.get(THEMEALDB_RANDOM, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Recipe API failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("Recipe API returned empty meals")
    return _normalize_meal(meals[0])


def fetch_recipe_by_id(recipe_id: str) -> Dict[str, Any]:
    url = THEMEALDB_LOOKUP.format(id=recipe_id)
    r = requests.get(url, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Recipe lookup failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("Recipe lookup returned empty meals")
    return _normalize_meal(meals[0])


def _normalize_meal(m: Dict[str, Any]) -> Dict[str, Any]:
    recipe_id = str(m.get("idMeal") or "").strip()
    title = str(m.get("strMeal") or "").strip()
    category = str(m.get("strCategory") or "").strip()
    area = str(m.get("strArea") or "").strip()
    instructions = str(m.get("strInstructions") or "").strip()
    thumb = str(m.get("strMealThumb") or "").strip()

    ingredients: List[Dict[str, str]] = []
    for i in range(1, 21):
        ing = str(m.get(f"strIngredient{i}") or "").strip()
        mea = str(m.get(f"strMeasure{i}") or "").strip()
        if ing:
            ingredients.append({"name": ing, "measure": mea})

    return {
        "id": recipe_id,
        "title": title,
        "category": category,
        "area": area,
        "instructions": instructions,
        "ingredients": ingredients,
        "thumb": thumb,
        "source": str(m.get("strSource") or "").strip(),
        "youtube": str(m.get("strYoutube") or "").strip(),
    }


def split_steps(instructions: str) -> List[str]:
    t = (instructions or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"\r?\n+", t) if p.strip()]
    if len(parts) <= 2 and len(t) > 400:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    # 빈/너무 짧은 문장 제거
    out = [p for p in parts if len(p) >= 3]
    return out


# -----------------------------
# OpenAI helpers + quota detection
# -----------------------------
def _is_insufficient_quota_error(e: Exception) -> bool:
    s = (repr(e) or "") + " " + (str(e) or "")
    s = s.lower()
    return ("insufficient_quota" in s) or ("exceeded your current quota" in s) or ("check your plan and billing" in s)


def _openai_call_with_retry(
    client: OpenAI,
    model: str,
    instructions: str,
    input_text: str,
    max_retries: int,
    debug: bool = False,
):
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(
                model=model,
                instructions=instructions,
                input=input_text,
            )
        except openai.RateLimitError as e:
            if _is_insufficient_quota_error(e):
                raise
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] RateLimit → retry in {sleep_s:.2f}s")
            time.sleep(sleep_s)
        except openai.APIError as e:
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] APIError → retry in {sleep_s:.2f}s | {repr(e)}")
            time.sleep(sleep_s)


def _count_first_ol_li(body_html: str) -> int:
    # 첫 <ol>...</ol> 블록 안의 <li> 개수 대략 검증
    m = re.search(r"<ol[^>]*>(.*?)</ol>", body_html, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return 0
    inner = m.group(1)
    return len(re.findall(r"<li\b", inner, flags=re.IGNORECASE))


# -----------------------------
# Free translate (LibreTranslate)
# -----------------------------
def free_translate_text(cfg: FreeTranslateConfig, text: str, debug: bool = False) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    payload = {
        "q": text,
        "source": cfg.source,
        "target": cfg.target,
        "format": "text",
    }
    if cfg.api_key:
        payload["api_key"] = cfg.api_key

    try:
        r = requests.post(cfg.url, json=payload, timeout=20)
        if r.status_code != 200:
            if debug:
                print("[FREE_TR] non-200:", r.status_code, r.text[:200])
            return text
        j = r.json()
        out = (j.get("translatedText") or "").strip()
        return out or text
    except Exception as e:
        if debug:
            print("[FREE_TR] failed:", repr(e))
        return text


def build_korean_body_fallback(cfg: AppConfig, recipe: Dict[str, Any], now: datetime) -> Tuple[str, str]:
    """
    OpenAI가 안될 때도 '너무 티나는 문장' 말고, 짧고 사람 말투로 최소 구성.
    """
    title_en = recipe.get("title", "Daily Recipe")
    title_ko = free_translate_text(cfg.free_tr, title_en, debug=cfg.run.debug) or title_en

    area_en = (recipe.get("area") or "").strip()
    category_en = (recipe.get("category") or "").strip()
    area_ko = free_translate_text(cfg.free_tr, area_en, debug=cfg.run.debug) if area_en else ""
    category_ko = free_translate_text(cfg.free_tr, category_en, debug=cfg.run.debug) if category_en else ""

    # 재료
    ing_lines = []
    for it in recipe.get("ingredients", []):
        name_en = (it.get("name") or "").strip()
        mea = (it.get("measure") or "").strip()
        name_ko = free_translate_text(cfg.free_tr, name_en, debug=cfg.run.debug) if name_en else ""
        label = name_ko if name_ko else name_en
        if mea:
            ing_lines.append(f"<li>{label} <span style='opacity:.75'>({mea})</span></li>")
        else:
            ing_lines.append(f"<li>{label}</li>")

    steps_en = split_steps(recipe.get("instructions", "")) or []
    step_lines = []
    for s in steps_en:
        ko = free_translate_text(cfg.free_tr, s, debug=cfg.run.debug) or s
        step_lines.append(f"<li>{ko}</li>")

    meta = " · ".join([x for x in [area_ko, category_ko] if x]).strip()
    meta_html = f"<p style='opacity:.75;'>{meta}</p>" if meta else ""

    title_final = f"집에서 편하게 만드는 {title_ko}"
    body = f"""
<p>딱 한 번만 흐름 잡아두면, 다음엔 훨씬 편해요.</p>
{meta_html}
<h2>3줄 요약</h2>
<ul>
  <li>재료는 최대한 단순하게.</li>
  <li>과정은 순서만 지키면 무난해요.</li>
  <li>간은 마지막에 한 번 더 확인!</li>
</ul>

<h2>재료</h2>
<ul>
{''.join(ing_lines) if ing_lines else '<li>재료 정보가 비어있어요.</li>'}
</ul>

<h2>만드는 법</h2>
<ol>
{''.join(step_lines) if step_lines else '<li>과정 정보가 비어있어요.</li>'}
</ol>

<h2>마무리</h2>
<p>저장해두면 다음에 바로 꺼내 쓰기 좋아요.</p>
"""
    return title_final, body.strip()


# -----------------------------
# Hashtags (네이버 느낌: 과다 금지)
# -----------------------------
BASE_HASHTAGS = [
    "#레시피", "#집밥", "#홈쿡", "#오늘뭐먹지", "#간단요리", "#요리기록", "#한끼", "#밥상",
    "#요리", "#주말요리", "#자취요리", "#맛있는한끼", "#푸드", "#food",
]


def build_hashtags(cfg: AppConfig, recipe: Dict[str, Any], title_ko: str) -> List[str]:
    tags = []

    # 기본 태그 + 사용자 추가
    tags.extend(BASE_HASHTAGS)
    tags.extend(cfg.run.extra_hashtags or [])

    # 제목에서 한글 단어(2~6자) 몇 개만 해시태그 후보로 (과다 금지)
    words = re.findall(r"[가-힣]{2,6}", title_ko or "")
    random.shuffle(words)
    for w in words[:3]:
        tags.append("#" + w)

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for t in tags:
        t = t.strip()
        if not t:
            continue
        if not t.startswith("#"):
            t = "#" + t
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)

    # 너무 많으면 컷
    n = max(6, min(20, int(cfg.run.hashtag_count or 12)))
    return uniq[:n]


def append_hashtags_if_missing(body_html: str, hashtags: List[str]) -> str:
    if not hashtags:
        return body_html
    if re.search(r"해시태그", body_html, flags=re.IGNORECASE):
        return body_html
    line = " ".join(hashtags)
    return body_html.rstrip() + f"\n\n<p><b>해시태그</b><br/>{line}</p>\n"


# -----------------------------
# OpenAI: 네이버 복붙 친화 + 사람 말투
# -----------------------------
def generate_korean_blog_naverish(
    cfg: AppConfig,
    recipe: Dict[str, Any],
) -> Tuple[str, str]:
    client = OpenAI(api_key=cfg.openai.api_key)

    steps = split_steps(recipe.get("instructions", ""))
    payload_recipe = {
        "title_en": recipe.get("title", ""),
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients": recipe.get("ingredients", []),
        "steps_en": steps,
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
        "tone": cfg.run.blog_tone,
    }

    # 사람 말투 강화를 위해 "너무 각 잡힌 목록/AI스러운 문구" 금지
    tone_hint = {
        "home": "담백한 집밥 블로그 느낌. 짧은 문단(1~2문장) 위주. 과장 금지.",
        "diary": "요리일기 느낌. '오늘은/그래서/중간에' 같은 연결어 자연스럽게. 과장 금지.",
        "simple": "최대한 간단. 군더더기 줄이고 핵심만. 과장 금지.",
    }.get(cfg.run.blog_tone, "담백한 집밥 블로그 느낌. 과장 금지.")

    # NAVER_STYLE=0이면 구조를 더 단순하게
    if cfg.run.naver_style:
        structure = (
            "본문 구성(순서 권장):\n"
            "1) <p>도입(2~4문장). 너무 'SEO' 티 나게 쓰지 말 것.</p>\n"
            "2) <h2>3줄 요약</h2><ul><li>...</li></ul>\n"
            "3) <h2>오늘 포인트</h2><ul> (3~5개, 길이 들쭉날쭉하게)</ul>\n"
            "4) <h2>재료</h2><ul> (재료명은 한국어로, 원문은 괄호로 짧게 가능)</ul>\n"
            "5) <h2>만드는 법</h2><ol> (steps 수만큼만 li 생성. 절대 추가/삭제/합치기 금지)\n"
            "   - 각 단계 li는 1~2문장으로 자연스럽게 번역\n"
            "   - 같은 li 안에 <p style='opacity:.8;'>한 줄 팁: ...</p> 1줄만\n"
            "6) <h2>자주 하는 질문</h2><ul> 3개</ul>\n"
            "7) <h2>마무리</h2><p>저장/공유 유도는 부드럽게 1문장.</p>\n"
            "8) (해시태그는 마지막에 한 줄로만)</n"
        )
    else:
        structure = (
            "본문 구성(순서 권장):\n"
            "1) <p>도입</p>\n"
            "2) <h2>재료</h2><ul>...</ul>\n"
            "3) <h2>만드는 법</h2><ol>(steps 수만큼만)</ol>\n"
            "4) <p>짧은 마무리</p>\n"
        )

    instructions = (
        "너는 한국어로 글을 쓰는 요리 블로거다.\n"
        f"[톤]\n- {tone_hint}\n"
        "\n"
        "[절대 규칙]\n"
        "1) 제공된 ingredients/steps 범위를 벗어나서 재료/계량/단계를 추가하거나 삭제하거나 바꾸지 마.\n"
        "2) 시간/온도/비율 같은 숫자는 원문 steps에 없으면 단정하지 마. 필요하면 '상태를 보며'로 표현.\n"
        "3) 'AI', '자동생성', 'ChatGPT', 'SEO' 같은 단어는 본문에 절대 쓰지 마.\n"
        "4) 문장 패턴을 기계적으로 반복하지 말고, 길이를 조금씩 섞어 자연스럽게.\n"
        "\n"
        "[출력 형식]\n"
        "첫 줄: 제목(한국어, 24~44자, 과한 자극 금지)\n"
        "둘째 줄부터: 워드프레스용 HTML(마크다운 금지). 스타일은 최소로.\n"
        + structure
    )

    user_input = "레시피 JSON:\n" + json.dumps(payload_recipe, ensure_ascii=False, indent=2)

    resp = _openai_call_with_retry(
        client=client,
        model=cfg.openai.model,
        instructions=instructions,
        input_text=user_input,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    text = (resp.output_text or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("OpenAI 응답이 너무 짧습니다(제목/본문 분리 실패).")

    title = lines[0]
    body = "\n".join(lines[1:]).strip()

    if cfg.run.strict_korean:
        if not re.search(r"[가-힣]", title) or not re.search(r"[가-힣]", body):
            raise RuntimeError("OpenAI 응답이 한국어가 아닙니다(한글 검증 실패).")

    # 최소 HTML 방어
    if "<" not in body:
        body = "<p>" + body.replace("\n", "<br/>") + "</p>"

    # steps 개수 검증(첫 <ol> 기준)
    expected = len(split_steps(recipe.get("instructions", "")))
    if expected >= 1:
        got = _count_first_ol_li(body)
        if got != expected:
            raise RuntimeError(f"단계(li) 개수 불일치: expected={expected}, got={got}")

    return title, body


# -----------------------------
# Main flow
# -----------------------------
def pick_recipe(cfg: AppConfig, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if existing and existing.get("recipe_id") and not cfg.run.force_new:
        return fetch_recipe_by_id(str(existing["recipe_id"]))

    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))

    prefer_areas = [a.strip().lower() for a in (cfg.run.prefer_areas or []) if a.strip()]
    prefer_keywords = [k.strip().lower() for k in (cfg.run.prefer_keywords or []) if k.strip()]

    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if not rid or rid in recent_ids:
            continue

        # 지역(Area) 선호
        if prefer_areas:
            area = (cand.get("area") or "").strip().lower()
            if area and all(pa.lower() != area for pa in prefer_areas):
                continue

        # 제목 키워드 선호(선택)
        if prefer_keywords:
            t = (cand.get("title") or "").lower()
            if t and not any(k in t for k in prefer_keywords):
                continue

        return cand

    # 선호 필터 때문에 못 고르면 필터 풀고 하나라도
    for _ in range(5):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if rid and rid not in recent_ids:
            return cand

    raise RuntimeError("레시피를 가져오지 못했습니다(중복 회피/시도 횟수 초과).")


def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else ("오후" if slot == "pm" else "오늘")

    date_key = f"{date_str}_{slot}" if slot in ("am", "pm") else date_str
    slug = f"daily-recipe-{date_str}-{slot}" if slot in ("am", "pm") else f"daily-recipe-{date_str}"

    init_db(cfg.sqlite_path, debug=cfg.run.debug)
    existing = get_today_post(cfg.sqlite_path, date_key)

    wp_post_id: Optional[int] = None
    if existing and existing.get("wp_post_id"):
        wp_post_id = int(existing["wp_post_id"])
    else:
        wp_post_id = wp_find_post_by_slug(cfg.wp, slug)

    recipe = pick_recipe(cfg, existing)
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title", "") or "Daily Recipe"

    # 썸네일 업로드
    media_id: Optional[int] = None
    media_url: str = ""
    thumb_url = (recipe.get("thumb") or "").strip()

    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"recipe-{date_str}-{slot}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    # 본문 생성: OpenAI 우선, 실패 시 폴백(표시 없이 조용히)
    try:
        title_ko, body_html = generate_korean_blog_naverish(cfg, recipe)
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] OpenAI generation failed → fallback:", repr(e))
        # quota든 뭐든 조용히 폴백
        title_ko, body_html = build_korean_body_fallback(cfg, recipe, now)

    # 해시태그 붙이기(없으면)
    hashtags = build_hashtags(cfg, recipe, title_ko)
    body_html = append_hashtags_if_missing(body_html, hashtags)

    # 이미지 삽입(업로드 성공 URL 우선)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            body_html = f'<p><img src="{img}" alt="{title_ko}" style="max-width:100%;height:auto;border-radius:12px;"></p>\n' + body_html

    title = f"{date_str} {slot_label} 레시피 | {title_ko}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 ↓")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2200] + ("\n...(truncated)" if len(body_html) > 2200 else ""))
        return

    # 발행/업데이트
    if wp_post_id:
        new_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)
        print("OK(updated):", new_id, wp_link)
    else:
        new_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)
        print("OK(created):", new_id, wp_link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
