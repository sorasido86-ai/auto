"""원문 레시피의 재료·수량·단계를 검증하여 한국어 HTML로 발행합니다.
검증 실패 시 게시를 중단하며 DRY_RUN=1은 HTML 미리보기만 저장합니다.
날짜·슬롯별 고정 slug와 SQLite 이력으로 중복을 막습니다."""

from __future__ import annotations

import base64
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
from content_quality import generate_recipe_article, render_recipe, save_preview, safe_url, recent_editorials, remember_editorial, recipe_response_format, split_recipe_steps
from wp_common import write_post
import html
from wp_common import find_post


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
    post = find_post(cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts", wp_auth_header(cfg.user, cfg.app_pass), slug)
    return int(post["id"]) if post else None


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

    data = write_post(url, headers, payload)
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

    data = write_post(url, headers, payload)
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
    return split_recipe_steps(instructions)


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
                text=recipe_response_format(input_text),
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


# -----------------------------
# Free translate (LibreTranslate)
# -----------------------------


# -----------------------------
# Hashtags (네이버 느낌: 과다 금지)
# -----------------------------
def build_hashtags(cfg: AppConfig, recipe: Dict[str, Any], title_ko: str) -> List[str]:
    count = max(0, min(20, int(cfg.run.hashtag_count)))
    if not count:
        return []
    # Avoid automatically appending the same broad tags to every article.
    tags = list(cfg.run.extra_hashtags or [])
    unique = []
    for tag in tags:
        tag = "#" + tag.strip().lstrip("#")
        if tag != "#" and tag not in unique:
            unique.append(tag)
    return unique[:count]


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
def generate_korean_blog_naverish(cfg: AppConfig, recipe: Dict[str, Any]) -> Tuple[str, str]:
    client = OpenAI(api_key=cfg.openai.api_key)
    ingredients = [f"{x.get('name', '')} {x.get('measure', '')}".strip() for x in recipe.get("ingredients", [])]
    steps = split_steps(recipe.get("instructions", ""))
    def call(instructions, payload):
        return _openai_call_with_retry(client, cfg.openai.model, instructions, payload, cfg.run.openai_max_retries, cfg.run.debug)
    article = generate_recipe_article(call, recipe.get("title", ""), ingredients, steps, recent=recent_editorials(cfg.sqlite_path))
    cfg.run.editorial_article = article
    source = safe_url(recipe.get("source")) or f"https://www.themealdb.com/meal/{recipe['id']}"
    return article["title"], render_recipe(article, source, "TheMealDB 레시피 원문")


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

    date_key = f"{date_str}_{slot}" if slot in ("am", "pm") else date_str
    slug = f"daily-recipe-{date_str}-{slot}" if slot in ("am", "pm") else f"daily-recipe-{date_str}"

    init_db(cfg.sqlite_path, debug=cfg.run.debug)
    existing = get_today_post(cfg.sqlite_path, date_key)

    wp_post_id: Optional[int] = None
    if existing and existing.get("wp_post_id"):
        wp_post_id = int(existing["wp_post_id"])
    elif not cfg.run.dry_run:
        wp_post_id = wp_find_post_by_slug(cfg.wp, slug)

    if wp_post_id and not cfg.run.force_new:
        print("SKIP(already posted):", wp_post_id)
        return

    recipe = pick_recipe(cfg, existing)
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title", "") or "Daily Recipe"

    # 썸네일 업로드
    media_id: Optional[int] = None
    media_url: str = ""
    thumb_url = (recipe.get("thumb") or "").strip()

    if not cfg.run.dry_run and cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"recipe-{date_str}-{slot}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    title_ko, body_html = generate_korean_blog_naverish(cfg, recipe)

    # 해시태그 붙이기(없으면)
    hashtags = build_hashtags(cfg, recipe, title_ko)
    body_html = append_hashtags_if_missing(body_html, hashtags)

    # 이미지 삽입(업로드 성공 URL 우선)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            body_html = f'<p><img src="{html.escape(safe_url(img), quote=True)}" alt="{html.escape(title_ko, quote=True)}" style="max-width:100%;height:auto;border-radius:12px;"></p>\n' + body_html

    title = title_ko

    save_preview(slug, title, body_html)

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

    article = getattr(cfg.run, "editorial_article", None)
    if article:
        remember_editorial(cfg.sqlite_path, slug, article)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()

