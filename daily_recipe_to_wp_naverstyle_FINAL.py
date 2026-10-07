"""한국어 레시피 편집기. 실제 재료·조리 순서를 입력으로 사용합니다.
원문 수량과 단계 수를 검증하고 날짜·슬롯별 한 번 발행합니다.
DRY_RUN=1은 이미지 업로드와 게시 없이 HTML을 저장합니다.
기존 환경변수 이름은 호환을 위해 유지합니다."""

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
import openai  # 예외 타입용
from openai import OpenAI
from content_quality import generate_recipe_article, render_recipe, save_preview, safe_url, recent_editorials, remember_editorial, recipe_response_format, recipe_model_options, split_recipe_steps
from wp_common import write_post, find_post, recent_recipe_posts
from content_quality import editorial_context, normalize_mealdb_source, choose_validated_recipe, recover_published_recipe, ContentQualityError


KST = timezone(timedelta(hours=9))

THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"
THEMEALDB_LOOKUP = "https://www.themealdb.com/api/json/v1/1/lookup.php?i={id}"

PEXELS_SEARCH = "https://api.pexels.com/v1/search"


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
        x = (x or "").strip()
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
        x = (x or "").strip()
        if x:
            out.append(x)
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
    run_slot: str = "day"
    dry_run: bool = False
    debug: bool = False
    always_new: bool = True  # ✅ 항상 새 글 발행
    avoid_repeat_days: int = 90
    max_tries: int = 30
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    openai_max_retries: int = 3


@dataclass
class NaverStyleConfig:
    random_level: int = 2
    experience_level: int = 2
    main_keyword: str = ""  # 비우면 자동
    recipe_keywords_csv: str = ""
    prefer_areas: List[str] = field(default_factory=list)
    block_categories: List[str] = field(default_factory=list)


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-5.4"


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    naver: NaverStyleConfig
    sqlite_path: str
    openai: OpenAIConfig
    pexels_api_key: str = ""


# -----------------------------
# Load/validate
# -----------------------------
def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    # ✅ 카테고리: env가 비어있거나 빈 문자열이면 기본 7번
    cat_raw = _env("WP_CATEGORY_IDS", "7")
    cat_ids = _parse_int_list(cat_raw)
    if not cat_ids:
        cat_ids = [7]

    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    sqlite_path = _env("SQLITE_PATH", "data/daily_recipe.sqlite3")

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    cfg_run = RunConfig(
        run_slot=run_slot,
        dry_run=_env_bool("DRY_RUN", False),
        debug=_env_bool("DEBUG", False),
        always_new=_env_bool("ALWAYS_NEW", True),
        avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
        max_tries=_env_int("MAX_TRIES", 30),
        upload_thumb=_env_bool("UPLOAD_THUMB", True),
        set_featured=_env_bool("SET_FEATURED", True),
        embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 0),
    )

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-5.4") or "gpt-5.4"

    naver_random_level = max(0, min(3, _env_int("NAVER_RANDOM_LEVEL", 2)))
    naver_exp_level = max(0, min(3, _env_int("NAVER_EXPERIENCE_LEVEL", 2)))
    main_kw = _env("NAVER_MAIN_KEYWORD", "")
    recipe_kws = _env("NAVER_RECIPE_KEYWORDS", "")  # ✅ 레시피 전용 키워드
    prefer_areas = _parse_str_list(_env("PREFER_AREAS", ""))
    block_categories = _parse_str_list(_env("BLOCK_CATEGORIES", ""))

    pexels_key = _env("PEXELS_API_KEY", "")

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=cat_ids,
            tag_ids=tag_ids,
        ),
        run=cfg_run,
        naver=NaverStyleConfig(
            random_level=naver_random_level,
            experience_level=naver_exp_level,
            main_keyword=main_kw,
            recipe_keywords_csv=recipe_kws,
            prefer_areas=prefer_areas,
            block_categories=[x.strip().lower() for x in block_categories],
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
        pexels_api_key=pexels_key,
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
    print("[CFG] ALWAYS_NEW:", int(cfg.run.always_new))
    print("[CFG] DRY_RUN:", int(cfg.run.dry_run), "| DEBUG:", int(cfg.run.debug))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model, "| OPENAI_KEY:", ok(cfg.openai.api_key))
    print("[CFG] NAVER_RANDOM_LEVEL:", cfg.naver.random_level, "| NAVER_EXPERIENCE_LEVEL:", cfg.naver.experience_level)
    print("[CFG] NAVER_MAIN_KEYWORD:", cfg.naver.main_keyword or "(auto)")
    print("[CFG] NAVER_RECIPE_KEYWORDS:", cfg.naver.recipe_keywords_csv or "(empty)")
    print("[CFG] PREFER_AREAS:", ",".join(cfg.naver.prefer_areas) if cfg.naver.prefer_areas else "(any)")
    print("[CFG] BLOCK_CATEGORIES:", ",".join(cfg.naver.block_categories) if cfg.naver.block_categories else "(none)")
    print("[CFG] PEXELS_API_KEY:", ok(cfg.pexels_api_key))


# -----------------------------
# SQLite
# -----------------------------
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_posts (
  date_key TEXT PRIMARY KEY,
  slot TEXT,
  recipe_id TEXT,
  recipe_title TEXT,
  wp_post_id INTEGER,
  wp_link TEXT,
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


def save_post_meta(path: str, date_key: str, slot: str, recipe_id: str, recipe_title: str, wp_post_id: int, wp_link: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_key, slot, recipe_id, recipe_title, wp_post_id, wp_link, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            date_key,
            slot,
            recipe_id,
            recipe_title,
            wp_post_id,
            wp_link,
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-recipe-bot/2.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str, featured_media: Optional[int], excerpt: str = "") -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt

    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    data = write_post(url, headers, payload)
    return int(data["id"]), str(data.get("link") or "")


def wp_update_editorial(cfg: WordPressConfig, post_id: int, title: str, html: str, excerpt: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{int(post_id)}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    data = write_post(url, headers, {"title": title, "content": html, "excerpt": excerpt})
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "thumb.jpg") -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=40)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code} url={image_url}")

    content = r.content
    ctype = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip().lower()

    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename_hint).strip("-") or "thumb.jpg"
    if "." not in safe_name:
        safe_name += ".jpg"

    headers["Content-Type"] = ctype
    headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    up = requests.post(media_endpoint, headers=headers, data=content, timeout=60)
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

    return normalize_mealdb_source({
        "id": recipe_id,
        "title": title,
        "category": category,
        "area": area,
        "instructions": instructions,
        "ingredients": ingredients,
        "thumb": thumb,
        "source": str(m.get("strSource") or "").strip(),
        "youtube": str(m.get("strYoutube") or "").strip(),
    })


def split_steps(instructions: str) -> List[str]:
    return split_recipe_steps(instructions)


# -----------------------------
# Pexels (thumbnail)
# -----------------------------


# -----------------------------
# OpenAI helpers
# -----------------------------
def _is_insufficient_quota_error(e: Exception) -> bool:
    s = (repr(e) or "") + " " + (str(e) or "")
    s = s.lower()
    return ("insufficient_quota" in s) or ("exceeded your current quota" in s) or ("check your plan and billing" in s)


def _openai_call_with_retry(client: OpenAI, model: str, instructions: str, input_text: str, max_retries: int, debug: bool = False):
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(model=model, instructions=instructions, input=input_text, text=recipe_response_format(input_text), **recipe_model_options(model, input_text))
        except openai.RateLimitError as e:
            if _is_insufficient_quota_error(e):
                raise
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] RateLimit retry in {sleep_s:.2f}s")
            time.sleep(sleep_s)
        except openai.APIError as e:
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] APIError retry in {sleep_s:.2f}s | {repr(e)}")
            time.sleep(sleep_s)
        except Exception as e:
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] UnknownError retry in {sleep_s:.2f}s | {repr(e)}")
            time.sleep(sleep_s)


# -----------------------------
# Editorial generation uses the source recipe and actual published history.
# -----------------------------


# -----------------------------
# Keyword safety (타미야 등 방지)
# -----------------------------
BANNED_WORDS = [
    "타미야", "tamiya", "프라모델", "도색", "건담", "rc카", "미니카", "락카", "에나멜", "에어브러시",
    "칠하는", "도료", "사포", "프라이머"
]


# -----------------------------
# Body generation (naver style)
# -----------------------------
def _style_wrap(inner_html: str) -> str:
    # 가독성: line-height/여백 강화
    return f"""
<div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Noto Sans KR,Arial; line-height:1.95; letter-spacing:-0.2px; font-size:15.5px; color:#111;">
{inner_html}
</div>
""".strip()


# -----------------------------
# Pick recipe + filters
# -----------------------------
def pick_recipe(cfg: AppConfig) -> Dict[str, Any]:
    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))
    prefer_areas = set([a.lower() for a in (cfg.naver.prefer_areas or [])])
    block_cats = set([c.lower() for c in (cfg.naver.block_categories or [])])

    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if not rid:
            continue
        if rid in recent_ids:
            continue

        area = (cand.get("area") or "").strip().lower()
        if prefer_areas and area and (area not in prefer_areas):
            continue

        cat = (cand.get("category") or "").strip().lower()
        if block_cats and cat and (cat in block_cats):
            continue

        # 재료/조리 단계가 너무 빈약하면 패스
        if len(cand.get("ingredients") or []) < 5:
            continue
        if len(split_steps(cand.get("instructions", ""))) < 5:
            continue

        return cand

    raise RuntimeError("레시피를 가져오지 못했습니다  필터 조건이 너무 빡세면 MAX_TRIES를 올려주세요")


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    slot = cfg.run.run_slot
    slug = f"naverstyle-recipe-{now.strftime('%Y-%m-%d')}-{slot}"
    existing = None
    refresh = _env("REFRESH_EDITORIAL") == "1"
    if not cfg.run.dry_run:
        endpoint = cfg.wp.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
        existing = find_post(endpoint, wp_auth_header(cfg.wp.user, cfg.wp.app_pass), slug)
        if existing and not refresh:
            print("SKIP(already posted):", existing["id"])
            return

    init_db(cfg.sqlite_path, debug=cfg.run.debug)

    client = OpenAI(api_key=cfg.openai.api_key, timeout=120.0, max_retries=0)
    def call(instructions, payload):
        return _openai_call_with_retry(client, cfg.openai.model, instructions, payload, cfg.run.openai_max_retries, cfg.run.debug)
    recent = editorial_context(recent_editorials(cfg.sqlite_path), recent_recipe_posts(cfg.wp.base_url), exclude=existing)
    def author(candidate):
        ingredients = [f"{x.get('name', '')} {x.get('measure', '')}".strip() for x in candidate.get("ingredients", [])]
        return generate_recipe_article(call, candidate.get("title", ""), ingredients,
                                       split_steps(candidate.get("instructions", "")), recent=recent)
    if existing:
        snapshot = recover_published_recipe(existing)
        if not snapshot or not snapshot["dish_name"]:
            raise ContentQualityError("기존 글의 재료·단계를 검증하지 못해 덮어쓰지 않습니다.")
        article = generate_recipe_article(call, snapshot["dish_name"], snapshot["ingredients"],
                                          snapshot["steps"], recent=recent, source_is_korean=True)
        recipe = {"id": "", "title": snapshot["dish_name"], "source": snapshot["source_url"],
                  "thumb": snapshot["image_url"]}
    else:
        recipe, article = choose_validated_recipe(lambda: pick_recipe(cfg), author)
    recipe_id = recipe.get("id", "")
    title_en = recipe.get("title", "") or "Daily Recipe"
    source = safe_url(recipe.get("source")) or (f"https://www.themealdb.com/meal/{recipe_id}" if recipe_id else "")
    body_html = render_recipe(article, source, "TheMealDB 레시피 원문")

    # 원문 레시피의 실제 요리 이미지 사용
    thumb_url = (recipe.get("thumb") or "").strip()
    final_img_url = thumb_url

    # WordPress 업로드(이미지 → 본문 상단 삽입)
    media_id = None
    media_url = ""
    if not cfg.run.dry_run and not existing and cfg.run.upload_thumb and final_img_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, final_img_url, filename_hint=f"recipe-{now.strftime('%Y%m%d-%H%M%S')}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    body_html = render_recipe(article, source, "TheMealDB 레시피 원문",
                              (media_url or final_img_url) if cfg.run.embed_image_in_body else "",
                              cfg.wp.base_url.rstrip("/") + "/" + slug + "/", recent)

    full_html = _style_wrap(body_html)

    title_final = article["title"]
    save_preview(slug, title_final, full_html)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", title_final)
        print("SLUG:", slug)
        print(full_html[:2200] + ("\n...(truncated)" if len(full_html) > 2200 else ""))
        return

    featured = int(media_id) if (cfg.run.set_featured and media_id) else None

    if existing:
        post_id, link = wp_update_editorial(cfg.wp, existing["id"], title_final, full_html, article.get("excerpt", ""))
    else:
        post_id, link = wp_create_post(cfg.wp, title_final, slug, full_html, featured_media=featured, excerpt=article.get("excerpt", ""))
        date_key = now.strftime("%Y-%m-%d") + "_" + slot
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, title_en, post_id, link)

    remember_editorial(cfg.sqlite_path, slug, article)
    print("OK(updated):" if existing else "OK(created):", post_id, link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
