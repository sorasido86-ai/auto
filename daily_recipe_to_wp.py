# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp.py (완전 통합/안정화 + 조회수형 블로그톤 + 크레딧 소진 시 무료번역 폴백)
- TheMealDB 랜덤 레시피 수집
- OpenAI로 한국어 블로그톤(후킹/요약/포인트/실패방지/SEO) 글 생성
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정
- SQLite 발행 이력 + 스키마 자동 마이그레이션
- ✅ OpenAI 크레딧 소진(insufficient_quota) 시: 무료번역(LibreTranslate)로 간단글이라도 자동 발행

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
  - STRICT_KOREAN=1 (기본 1)  # 한글 아니면 실패. 단, '크레딧 소진'이면 무료번역으로 폴백
  - FORCE_NEW=0 (기본 0)
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - OPENAI_MAX_RETRIES=3

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

    # ✅ 기본 카테고리 7번
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-recipe-bot/1.0"}


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
    return parts


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
            # ✅ 크레딧 소진이면 재시도 의미 없음 → 즉시 상위로 올려서 폴백
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


def build_korean_simple_with_free_translate(free_cfg: FreeTranslateConfig, recipe: Dict[str, Any], now: datetime, debug: bool = False) -> Tuple[str, str]:
    """
    ✅ OpenAI 크레딧 소진 시 사용할 '간단 글' 생성
    - 제목/재료/과정만 한국어로 번역(가능한 범위)
    - 블로그톤은 템플릿 문장으로 최소 구성(과도한 생성 없음)
    """
    title_en = recipe.get("title", "Daily Recipe")
    title_ko = free_translate_text(free_cfg, title_en, debug=debug)
    category_en = recipe.get("category", "")
    area_en = recipe.get("area", "")

    category_ko = free_translate_text(free_cfg, category_en, debug=debug) if category_en else ""
    area_ko = free_translate_text(free_cfg, area_en, debug=debug) if area_en else ""

    # 재료 번역(원문 병기)
    ing_lines = []
    for it in recipe.get("ingredients", []):
        name_en = (it.get("name") or "").strip()
        mea = (it.get("measure") or "").strip()
        name_ko = free_translate_text(free_cfg, name_en, debug=debug) if name_en else ""
        label = f"{name_ko} ({name_en})" if name_ko and name_ko != name_en else name_en
        if mea:
            ing_lines.append(f"<li>{label} <span style='opacity:.75'>— {mea}</span></li>")
        else:
            ing_lines.append(f"<li>{label}</li>")

    steps_en = split_steps(recipe.get("instructions", ""))
    step_lines = []
    for s in steps_en:
        ko = free_translate_text(free_cfg, s, debug=debug)
        step_lines.append(f"<li>{ko}</li>")

    src = (recipe.get("source") or "").strip()
    yt = (recipe.get("youtube") or "").strip()

    meta_line = " / ".join([x for x in [area_ko, category_ko] if x]).strip()
    meta_html = f"<p style='opacity:.8;font-size:13px;'>분류: {meta_line}</p>" if meta_line else ""

    body = f"""
    <p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">
      ※ OpenAI 크레딧 소진으로 인해, 오늘은 무료 번역 기반의 간단 레시피로 자동 발행되었습니다.
    </p>
    <p>기준시각: <b>{now.astimezone(KST).strftime("%Y-%m-%d %H:%M")}</b></p>
    {meta_html}
    <p>오늘은 <b>{title_ko}</b>를 간단히 정리해볼게요. 재료/과정은 원문을 그대로 번역했습니다.</p>

    <h2>재료</h2>
    <ul>
      {''.join(ing_lines)}
    </ul>

    <h2>만드는 법</h2>
    <ol>
      {''.join(step_lines)}
    </ol>

    <p style="opacity:.7;font-size:13px;">
      출처: {f"<a href='{src}' target='_blank' rel='nofollow noopener'>{src}</a>" if src else "-"}
      {" | " + f"<a href='{yt}' target='_blank' rel='nofollow noopener'>YouTube</a>" if yt else ""}
    </p>
    """
    # 제목은 조회수형으로 살짝만(과하지 않게)
    title_final = f"실패 없이 따라하는 {title_ko} 레시피"
    return title_final, body


# -----------------------------
# OpenAI: 조회수형 블로그톤 (후킹/요약/포인트/실패방지/SEO)
# -----------------------------
def generate_korean_blog_highview(
    openai_cfg: OpenAIConfig,
    recipe: Dict[str, Any],
    strict_korean: bool,
    max_retries: int,
    debug: bool = False
) -> Tuple[str, str]:
    client = OpenAI(api_key=openai_cfg.api_key)

    payload_recipe = {
        "title_en": recipe.get("title", ""),
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients": recipe.get("ingredients", []),
        "steps_en": split_steps(recipe.get("instructions", "")),
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
    }

    # ✅ 조회수형 톤: 후킹 + 한줄요약 + 포인트/실패방지 + 자연스러운 CTA
    instructions = (
        "너는 한국 음식 블로거이자 SEO 글쓰기 전문가다. 반드시 한국어로만 작성해.\n"
        "\n"
        "[절대 규칙]\n"
        "1) 제공된 ingredients/steps 외의 재료, 계량, 조리과정(단계)을 절대 추가/삭제/변경하지 마.\n"
        "2) 시간/온도 숫자를 원문에 없으면 단정하지 마(예: 10분, 180도). 필요하면 '상태를 보며'처럼 표현.\n"
        "3) 부연설명은 가능: 맛 포인트, 실패 방지 체크, 식감 체크 방법 등 '일반적인 설명'을 자연스럽게 덧붙여라.\n"
        "\n"
        "[조회수 잘 나오는 말투]\n"
        "- 첫 문단은 ‘후킹’ 2문장: (누구나 실패하는 포인트) → (이 글에서 해결)\n"
        "- 문장은 짧게, 리듬감 있게. 과한 이모지/과장 광고 금지.\n"
        "- 독자가 저장/공유하고 싶게: '실패 방지 체크리스트'를 짧게.\n"
        "- 클릭 유도는 부드럽게: '저장해두면 편해요' 정도.\n"
        "\n"
        "[출력 형식]\n"
        "첫 줄: 제목(한국어, 28~48자, 너무 자극적 금지)\n"
        "둘째 줄부터: 워드프레스용 HTML(마크다운 금지)\n"
        "본문 구성(순서 고정):\n"
        "1) <p>후킹 2문장 + 한줄요약</p>\n"
        "2) <h2>오늘의 핵심 포인트 5가지</h2><ul><li>...</li></ul>\n"
        "3) <h2>재료</h2><ul> (재료명은 한국어로 풀어쓰고, 원문명은 괄호로 병기 가능)</ul>\n"
        "4) <h2>만드는 법</h2><ol>\n"
        "   - steps 수만큼만 <li>를 만들고, 각 <li> 안에:\n"
        "     (a) 원문 step을 의미 그대로 자연스러운 한국어로\n"
        "     (b) 같은 <li> 안에 <div style='font-size:13px;opacity:.8;margin-top:6px;'>실패 방지 팁: ...</div> 1줄\n"
        "5) <h2>실패 방지 체크리스트</h2><ul> 4~6개</ul>\n"
        "6) <h2>마무리</h2><p>저장/공유 유도 1문장 포함</p>\n"
        "7) <p style='opacity:.7;font-size:13px;'>출처 링크(있으면) + 자동생성 안내</p>\n"
    )

    user_input = "레시피 JSON:\n" + json.dumps(payload_recipe, ensure_ascii=False, indent=2)

    resp = _openai_call_with_retry(
        client=client,
        model=openai_cfg.model,
        instructions=instructions,
        input_text=user_input,
        max_retries=max_retries,
        debug=debug,
    )

    text = (resp.output_text or "").strip()
    if debug:
        print("[OPENAI] output_text length:", len(text))

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("OpenAI 응답이 너무 짧습니다(제목/본문 분리 실패).")

    title = lines[0]
    body = "\n".join(lines[1:]).strip()

    if strict_korean:
        if not re.search(r"[가-힣]", title) or not re.search(r"[가-힣]", body):
            raise RuntimeError("OpenAI 응답이 한국어가 아닙니다(한글 검증 실패).")

    if "<" not in body:
        body = "<p>" + body.replace("\n", "<br/>") + "</p>"

    return title, body


# -----------------------------
# Main flow
# -----------------------------
def pick_recipe(cfg: AppConfig, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if existing and existing.get("recipe_id") and not cfg.run.force_new:
        return fetch_recipe_by_id(str(existing["recipe_id"]))

    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))
    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if not rid:
            continue
        if rid in recent_ids:
            continue
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

    # ✅ 1) OpenAI로 조회수형 한국어 블로그톤 생성
    # ✅ 2) 크레딧 소진(insufficient_quota)면 무료번역 폴백으로 "간단글"이라도 발행
    try:
        title_ko, body_html = generate_korean_blog_highview(
            cfg.openai,
            recipe,
            strict_korean=cfg.run.strict_korean,
            max_retries=cfg.run.openai_max_retries,
            debug=cfg.run.debug,
        )
    except Exception as e:
        if _is_insufficient_quota_error(e):
            print("[WARN] OpenAI quota depleted → fallback to FREE translate")
            title_ko, body_html = build_korean_simple_with_free_translate(cfg.free_tr, recipe, now, debug=cfg.run.debug)
        else:
            # 크레딧 소진이 아닌데도 실패하면: 기존 정책대로
            if cfg.run.strict_korean:
                raise
            print("[WARN] OpenAI failed (non-quota) → fallback to FREE translate:", repr(e))
            title_ko, body_html = build_korean_simple_with_free_translate(cfg.free_tr, recipe, now, debug=cfg.run.debug)

    # 본문에 이미지 삽입(업로드 성공하면 WP URL 우선)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            body_html = f'<p><img src="{img}" alt="{title_ko}" style="max-width:100%;height:auto;border-radius:12px;"></p>\n' + body_html

    title = f"{date_str} {slot_label} 레시피 | {title_ko}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 ↓")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2000] + ("\n...(truncated)" if len(body_html) > 2000 else ""))
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
