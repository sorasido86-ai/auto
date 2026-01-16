# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle.py (통합 완성본)
- TheMealDB 랜덤 레시피 수집
- OpenAI로 네이버 홈피드형 '사람 글' 톤 생성 (가독성/여백/문단 호흡/존댓말 수다)
- 재료: 불릿/특수문자 없이 줄바꿈 목록
- 방법: 1단계/2단계/번호 없이 문단 나열 (순서 라벨 금지)
- 마침표 없이(후처리 포함)
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정
- SQLite 발행 이력 + 스키마 자동 마이그레이션
- OpenAI 실패/크레딧 소진 시 LibreTranslate 폴백
- (선택) PEXELS_API_KEY 있으면 더 깔끔한 고퀄 썸네일 이미지 사용

필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY

권장 env:
  - WP_STATUS=publish
  - WP_CATEGORY_IDS="7"
  - WP_TAG_IDS="1,2,3"
  - SQLITE_PATH=data/daily_recipe.sqlite3

동작:
  - RUN_SLOT=day|am|pm (기본 day)
  - FORCE_NEW=0|1 (기본 0)          # 1이면 매번 새 글(중복 의심될 때)
  - DRY_RUN=0|1 (기본 0)
  - DEBUG=0|1 (기본 0)
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - OPENAI_MODEL=gpt-4.1-mini (기본)
  - OPENAI_MAX_RETRIES=2

네이버 스타일/랜덤화:
  - NAVER_RANDOM_LEVEL=0~3 (기본 2)     # 표현 변주
  - NAVER_EXPERIENCE_LEVEL=0~3 (기본 2) # 현장감/실패포인트 강조
  - NAVER_LONGFORM=0|1 (기본 1)         # 1이면 긴 글(도입 200~300자 + 3개 소제목 + 전체 2300자 이상 목표)
  - NAVER_KEYWORDS="키워드1,키워드2,..." (선택) # 태그/본문에 자연스럽게
  - PREFER_AREAS="Korean,Japanese,Chinese" (선택)

썸네일 고퀄(선택):
  - PEXELS_API_KEY=... (있으면 Pexels에서 음식 고퀄 사진 검색해서 썸네일로 사용)

무료번역 폴백(LibreTranslate):
  - FREE_TRANSLATE_URL=https://libretranslate.de/translate
  - FREE_TRANSLATE_API_KEY=
  - FREE_TRANSLATE_SOURCE=en
  - FREE_TRANSLATE_TARGET=ko
"""

from __future__ import annotations

import base64
import html as _html
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

import openai  # 예외 타입용
from openai import OpenAI

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


def _debug(cfg_debug: bool, *args):
    if cfg_debug:
        print(*args)


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
    force_new: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    openai_max_retries: int = 2


@dataclass
class NaverStyleConfig:
    random_level: int = 2          # 0~3
    experience_level: int = 2      # 0~3
    longform: bool = True          # NAVER_LONGFORM
    keywords_csv: str = ""
    prefer_areas: List[str] = field(default_factory=list)


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4.1-mini"


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
    naver: NaverStyleConfig
    sqlite_path: str
    openai: OpenAIConfig
    free_tr: FreeTranslateConfig
    pexels_api_key: str = ""


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

    cfg_run = RunConfig(
        run_slot=run_slot,
        dry_run=_env_bool("DRY_RUN", False),
        debug=_env_bool("DEBUG", False),
        force_new=_env_bool("FORCE_NEW", False),
        avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
        max_tries=_env_int("MAX_TRIES", 20),
        upload_thumb=_env_bool("UPLOAD_THUMB", True),
        set_featured=_env_bool("SET_FEATURED", True),
        embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 2),
    )

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

    free_url = _env("FREE_TRANSLATE_URL", "https://libretranslate.de/translate")
    free_api_key = _env("FREE_TRANSLATE_API_KEY", "")
    free_src = _env("FREE_TRANSLATE_SOURCE", "en")
    free_tgt = _env("FREE_TRANSLATE_TARGET", "ko")

    naver_random_level = max(0, min(3, _env_int("NAVER_RANDOM_LEVEL", 2)))
    naver_exp_level = max(0, min(3, _env_int("NAVER_EXPERIENCE_LEVEL", 2)))
    naver_longform = _env_bool("NAVER_LONGFORM", True)
    naver_keywords = _env("NAVER_KEYWORDS", "")
    prefer_areas = _parse_str_list(_env("PREFER_AREAS", ""))

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
            longform=naver_longform,
            keywords_csv=naver_keywords,
            prefer_areas=prefer_areas,
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
        free_tr=FreeTranslateConfig(url=free_url, api_key=free_api_key, source=free_src, target=free_tgt),
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", int(cfg.run.dry_run), "| DEBUG:", int(cfg.run.debug))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model, "| OPENAI_KEY:", ok(cfg.openai.api_key))
    print("[CFG] NAVER_RANDOM_LEVEL:", cfg.naver.random_level, "| NAVER_EXPERIENCE_LEVEL:", cfg.naver.experience_level, "| LONGFORM:", int(cfg.naver.longform))
    print("[CFG] NAVER_KEYWORDS:", cfg.naver.keywords_csv or "(empty)")
    print("[CFG] PREFER_AREAS:", ",".join(cfg.naver.prefer_areas) if cfg.naver.prefer_areas else "(any)")
    print("[CFG] PEXELS_API_KEY:", "OK" if cfg.pexels_api_key else "(none)")
    print("[CFG] FREE_TRANSLATE_URL:", cfg.free_tr.url)


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
    r = requests.get(url, headers=wp_auth_header(cfg.user, cfg.app_pass), timeout=25)
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

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:700]}")
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

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:700]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "recipe.jpg") -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=45)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code} url={image_url}")

    content = r.content
    ctype = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip().lower()

    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename_hint).strip("-") or "recipe.jpg"
    if "." not in safe_name:
        safe_name += ".jpg"

    headers["Content-Type"] = ctype
    headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    up = requests.post(media_endpoint, headers=headers, data=content, timeout=75)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {up.status_code} body={up.text[:700]}")

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
# Quota detect + OpenAI retry
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
# LibreTranslate (fallback)
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
        r = requests.post(cfg.url, json=payload, timeout=25)
        if r.status_code != 200:
            if debug:
                print("[FREE_TR] non-200:", r.status_code, r.text[:250])
            return text
        j = r.json()
        out = (j.get("translatedText") or "").strip()
        return out or text
    except Exception as e:
        if debug:
            print("[FREE_TR] failed:", repr(e))
        return text


# -----------------------------
# Pexels (optional HQ thumbnail)
# -----------------------------
def pexels_pick_image(pexels_api_key: str, query: str, debug: bool = False) -> str:
    if not pexels_api_key:
        return ""
    q = (query or "").strip()
    if not q:
        return ""

    try:
        headers = {"Authorization": pexels_api_key}
        params = {"query": q, "per_page": 10, "orientation": "landscape", "size": "large"}
        r = requests.get(PEXELS_SEARCH, headers=headers, params=params, timeout=25)
        if r.status_code != 200:
            if debug:
                print("[PEXELS] non-200:", r.status_code, r.text[:200])
            return ""
        j = r.json()
        photos = j.get("photos") or []
        if not photos:
            return ""
        cand = random.choice(photos[: min(10, len(photos))])
        src = cand.get("src") or {}
        # original이 가장 고퀄, 너무 크면 large2x 사용
        return str(src.get("large2x") or src.get("original") or src.get("large") or "").strip()
    except Exception as e:
        if debug:
            print("[PEXELS] failed:", repr(e))
        return ""


# -----------------------------
# Text helpers / constraints
# -----------------------------
_ASCII_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_PUNCT_RE = re.compile(r"[.!?。…]+")


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "")


def _nl2br(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\n", "<br/>")
    return s


def _remove_punct_like_period(s: str) -> str:
    s = _PUNCT_RE.sub("", s or "")
    return s


def _too_much_english(s: str) -> bool:
    if not s:
        return False
    words = _ASCII_WORD_RE.findall(s)
    return len(words) >= 6  # 제목/재료/본문에 영어가 많이 섞이면 실패로 간주


def _html_wrap(body_inner: str) -> str:
    # 가독성: 줄간격 + 문단 여백 크게
    return (
        "<div style='line-height:1.95;font-size:16px;'>"
        + body_inner
        + "</div>"
    )


def _p(txt: str) -> str:
    # 문단 간격 넉넉하게
    return f"<p style='margin:0 0 18px 0;'>{txt}</p>"


def _h2(title: str) -> str:
    return f"<h2 style='margin:24px 0 12px 0;'><b>{_html.escape(title)}</b></h2>"


def closing_random(keyword: str) -> str:
    kw = (keyword or "").strip() or "이 메뉴"
    pool = [
        f"저는 요리가 결국  정답보다  내 생활에 붙는 기준이라고 생각하거든요  {kw}도 오늘은 완벽보다  흔들리지 않는 기준만 잡는 쪽으로 정리해봤어요  다음에 만들 땐 훨씬 편해지실 거예요",
        f"이런 글은 화려한 팁보다  한 가지 기준이 더 오래 남더라고요  {kw}는 특히 상태만 잘 보면  마음이 되게 편해져요  그 감각 한 번만 잡히면  다음부터는 속도가 빨라져요",
        f"요리는 결국  내 입맛과 내 컨디션이 같이 들어가는 거라  부담을 줄이는 게 중요하더라고요  {kw}도 오늘은 무리하지 않는 흐름으로만 잡아봤어요  편하게 해보셔도 됩니다",
        f"메뉴 고민될 때  이런 글 하나 있으면  괜히 마음이 덜 흔들리잖아요  {kw}도 다음번에는  내가 좋아하는 포인트 하나만 살짝 조절해보셔도 충분히 맛있게 나옵니다",
    ]
    return random.choice(pool)


def tags_html(keywords_csv: str, main_keyword: str) -> str:
    kws: List[str] = []
    if main_keyword:
        kws.append(main_keyword.strip())
    for x in (keywords_csv or "").split(","):
        x = x.strip()
        if x and x not in kws:
            kws.append(x)

    kws = [k for k in kws if k]
    kws = kws[:12]
    if not kws:
        return ""

    # 불릿 없이 태그 한 줄
    line = " ".join([f"#{k.replace(' ', '')}" for k in kws])
    return f"<div style='margin:18px 0 0 0;opacity:.8;font-size:14px;line-height:1.8;'>태그  {line}</div>"


# -----------------------------
# Title randomization (4유형)
# -----------------------------
def build_hook_title(keyword: str, random_level: int) -> str:
    kw = (keyword or "").strip() or "오늘 레시피"

    benefit = [
        f"{kw} 이렇게 하면  맛이 안정돼요  한 끗 차이 포인트",
        f"{kw} 집에서 만들 때  제일 편해지는 기준  딱 한 가지",
        f"{kw} 실패 확 줄이는 흐름  이 순서로만 가도 돼요",
        f"{kw} 간이 흔들릴 때  마지막에 잡아주는 방법",
    ]
    threat = [
        f"{kw} 여기서만 흔들리면  맛이 바로 깨져요",
        f"{kw} 이 타이밍 놓치면  식감이 달라져요",
        f"{kw} 이 부분 과하면  전체가 무거워져요  기준만 잡아요",
        f"{kw} 괜히 서두르면  결과가 갈려요  상태부터 봐요",
    ]
    curiosity = [
        f"{kw} 왜 밖에서 먹는 맛이 안 날까  답은 이 포인트예요",
        f"{kw} 똑같이 따라 했는데  결과가 다른 이유가 있어요",
        f"{kw} 맛이 들쭉날쭉할 때  기준이 하나 부족한 거예요",
        f"{kw} 맛이 확 살아나는 순간  보통 여기서 갈려요",
    ]
    compare = [
        f"{kw}  빠르게 하는 방법보다  맛이 안정되는 방법이 먼저예요",
        f"{kw} 센 불이냐 약불이냐보다  상태를 먼저 보면 돼요",
        f"{kw} 레시피보다 중요한 건  내가 확인할 기준이에요",
        f"{kw} 화려한 팁보다  실수 줄이는 기준이 오래 가요",
    ]

    # 가중치 (궁금형 조금 높게)
    pools = [
        ("curiosity", curiosity, 35),
        ("benefit", benefit, 30),
        ("threat", threat, 20),
        ("compare", compare, 15),
    ]
    bag: List[str] = []
    for _, arr, w in pools:
        bag += arr * max(1, w // 5)

    title = random.choice(bag) if bag else kw

    # 랜덤 레벨이 높으면 끝 표현을 가끔 바꿈
    if random_level >= 3:
        tails = ["  저장해두면 편해요", "  다음엔 훨씬 쉬워져요", "  여기만 기억해요", "  부담 없이 해요"]
        title = title + random.choice(tails)

    # 마침표류 제거
    title = _remove_punct_like_period(title)
    # 너무 길면 줄이기
    title = title[:48].strip()
    return title


# -----------------------------
# Naver-style prompt builder
# -----------------------------
def _naver_prompt(cfg: AppConfig, recipe_ko: Dict[str, Any], now: datetime) -> Tuple[str, str]:
    random_level = cfg.naver.random_level
    exp_level = cfg.naver.experience_level
    longform = cfg.naver.longform

    kw_hint = cfg.naver.keywords_csv.strip()
    season_hint = now.astimezone(KST).strftime("%m월")

    # 길이 목표
    intro_min = 200
    intro_max = 320
    total_min = 2300 if longform else 1200

    # longform이면 소제목 3개 크게
    # (너가 원했던 1500자 이상은 모델이 가끔 못 맞춰서, 코드에서 길이 검증 후 자동 재생성)
    section_target = 1500 if longform else 600

    payload = {
        "title_ko": recipe_ko.get("title_ko", ""),
        "category_ko": recipe_ko.get("category_ko", ""),
        "area_ko": recipe_ko.get("area_ko", ""),
        "ingredients_ko": recipe_ko.get("ingredients_ko", []),  # [{name_ko, measure}]
        "steps_en": recipe_ko.get("steps_en", []),              # 원문 의미 유지용
        "source_url": recipe_ko.get("source_url", ""),
        "youtube": recipe_ko.get("youtube", ""),
        "naver_style": {
            "random_level": random_level,
            "experience_level": exp_level,
            "season_hint": season_hint,
            "keywords_hint": kw_hint,
            "intro_chars": [intro_min, intro_max],
            "section_target_chars": section_target,
            "total_min_chars": total_min,
        },
    }

    # ✅ 핵심: 불릿/번호/단계 라벨 금지  마침표 금지  친구에게 수다 존댓말
    instructions = f"""
너는 한국어 블로그 운영자다  네이버 홈피드에 잘 읽히는 글을 쓴다

[중요 규칙]
너는 실제로 조리해본 적이 없다
그래서 오늘 해먹어봤는데  제가 해보니  같은 거짓 체험 단정은 금지
대신 집에서 해보면  보통 여기서  이런 식으로 현장감 있는 기준으로만 말해라

제공된 재료와 과정 의미를 절대 바꾸지 마라
재료 추가 삭제 금지
조리 과정 단계 추가 삭제 금지
시간 온도 숫자도 원문에 없으면 단정 금지
필요하면 상태를 보며  색 향 농도 식감 기준으로 표현하라

문장 끝 마침표 느낌의 기호 금지
점  느낌표  물음표  말줄임표  이런 걸 쓰지 마라
리듬은 띄어쓰기와 여백  줄바꿈으로 만들라

불릿 느낌 나는 특수문자 금지
재료 목록에 점  동그라미  대시 같은 표식 금지
방법에도 1단계 2단계  첫째 둘째  번호 라벨 금지

[톤]
친구에게 진심 담아 수다 떠는 존댓말
생각  가치관  왜 이렇게 정리했는지  마음의 흐름을 섞어라
너무 교과서처럼 정리하지 마라

[가독성]
문단을 짧게  여백을 크게
한 문단 2 4줄 정도로 끊고  문단 사이에 빈 줄을 두어라

[출력]
첫 줄  제목만 출력  한국어  26 46자  검색될 단어 1 2개 포함
둘째 줄부터 HTML만 출력
반드시 아래 HTML 구조를 지켜라

1) 도입 200 300자 정도  <p> 문단 2 3개  줄바꿈 포함
2) 굵은 소제목 3개  <h2><b>...</b></h2> 형태로
   각 소제목 아래는 길게  최소 {payload["naver_style"]["section_target_chars"]}자 이상을 목표로
3) 재료 소제목
   <h2><b>재료</b></h2>
   <div>재료명  계량</div>를 여러 줄로  표식 없이  줄마다 <br/><br/>로 여백
4) 만드는 법 소제목
   <h2><b>만드는 법</b></h2>
   과정은 steps 순서를 유지해서 문단으로 나열
   각 문단은 <p>로 감싸고  문단 아래에 작은 팁 한 줄을 붙여라
   팁은 <div style='font-size:13px;opacity:.82;margin:4px 0 14px 0;'>실패 포인트  상태 기준  </div>
   번호  단계  라벨 금지
5) 맨 아래에 출처는 짧게  링크는 있어도 되고  표시 문장은 짧게

[키워드 힌트]
가능하면 본문에 아래 키워드 느낌을 자연스럽게 섞어라  억지 나열 금지
{kw_hint if kw_hint else "없음"}

[입력]
아래 JSON 레시피를 참고해서 작성하라
""".strip()

    user_input = "레시피 JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    return instructions, user_input


def _validate_lengths(title: str, body_html: str, longform: bool) -> Tuple[bool, str]:
    txt = _strip_tags(body_html)
    txt = re.sub(r"\s+", " ", txt).strip()

    # 기본: 총 길이
    total_ok = len(txt) >= (2300 if longform else 1100)

    # 도입 길이: 첫 350자 정도 안에서 200 이상이 나오도록 (대충 체크)
    intro_slice = txt[:400]
    intro_ok = len(intro_slice) >= (200 if longform else 120)

    # 영어 섞임 체크
    eng_ok = not _too_much_english(title + " " + body_html)

    ok = total_ok and intro_ok and eng_ok
    reason = f"total_ok={total_ok} intro_ok={intro_ok} eng_ok={eng_ok} total_len={len(txt)}"
    return ok, reason


# -----------------------------
# Simple fallback builder (free translate)
# -----------------------------
def build_korean_simple_with_free_translate(
    free_cfg: FreeTranslateConfig,
    recipe: Dict[str, Any],
    now: datetime,
    keywords_csv: str,
    debug: bool = False
) -> Tuple[str, str]:
    title_ko = free_translate_text(free_cfg, recipe.get("title", "Daily Recipe"), debug=debug)
    title_ko = _remove_punct_like_period(title_ko)

    # 재료: 불릿 없이
    lines = []
    for it in recipe.get("ingredients", []):
        name = (it.get("name") or "").strip()
        mea = (it.get("measure") or "").strip()
        name_ko = free_translate_text(free_cfg, name, debug=debug) if name else ""
        name_ko = _remove_punct_like_period(name_ko)
        if mea:
            lines.append(f"{_html.escape(name_ko)}  {_html.escape(mea)}")
        else:
            lines.append(f"{_html.escape(name_ko)}")

    steps = split_steps(recipe.get("instructions", ""))
    step_ps = []
    for s in steps:
        ko = free_translate_text(free_cfg, s, debug=debug)
        ko = _remove_punct_like_period(ko)
        step_ps.append(_p(_html.escape(ko)))
        step_ps.append("<div style='font-size:13px;opacity:.82;margin:4px 0 14px 0;'>실패 포인트  상태 기준  급하게 결정하지 말고  눈으로 확인해요</div>")

    main_kw = title_ko if title_ko else "오늘 레시피"
    hook_title = build_hook_title(main_kw, random_level=2)

    body = ""
    body += _p(_html.escape("레시피는 따라 했는데  결과가 매번 다르면  마음이 좀 지치잖아요  오늘은 기준만 잡아두는 쪽으로 정리해볼게요"))
    body += _p(_html.escape("완벽하게 하려는 마음보다  실패 확률을 줄이는 흐름이 오래 가더라고요  저도 그래서 이 방식이 편했어요"))
    body += _h2("재료")
    body += "<div style='margin:0 0 14px 0;'>" + "<br/><br/>".join(lines) + "</div>"
    body += _h2("만드는 법")
    body += "".join(step_ps)

    # 마무리 + 태그
    body += _h2("마무리")
    body += _p(_html.escape(_remove_punct_like_period(closing_random(main_kw))))
    body += tags_html(keywords_csv, main_kw)

    # 출처
    src = (recipe.get("source") or "").strip()
    yt = (recipe.get("youtube") or "").strip()
    src_html = []
    if src:
        src_html.append(f"<a href='{_html.escape(src)}' target='_blank' rel='nofollow noopener'>출처</a>")
    if yt:
        src_html.append(f"<a href='{_html.escape(yt)}' target='_blank' rel='nofollow noopener'>영상</a>")
    if src_html:
        body += f"<p style='opacity:.65;font-size:13px;margin:18px 0 0 0;'>{'  '.join(src_html)}</p>"

    body = _html_wrap(body)
    return hook_title, body


# -----------------------------
# Recipe pick + translate pieces
# -----------------------------
def pick_recipe(cfg: AppConfig, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if existing and existing.get("recipe_id") and not cfg.run.force_new:
        return fetch_recipe_by_id(str(existing["recipe_id"]))

    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))
    prefer_areas = set([a.lower() for a in (cfg.naver.prefer_areas or [])])

    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if not rid:
            continue
        if rid in recent_ids:
            continue
        if prefer_areas:
            area = (cand.get("area") or "").strip().lower()
            if area and (area not in prefer_areas):
                continue
        return cand

    raise RuntimeError("레시피를 가져오지 못했습니다  중복 회피  필터  시도 횟수 초과")


def translate_recipe_bits(cfg: AppConfig, recipe: Dict[str, Any]) -> Dict[str, Any]:
    # 제목/카테고리/지역/재료명은 폴백 번역으로라도 한국어화
    title_ko = free_translate_text(cfg.free_tr, recipe.get("title", ""), debug=cfg.run.debug)
    category_ko = free_translate_text(cfg.free_tr, recipe.get("category", ""), debug=cfg.run.debug) if recipe.get("category") else ""
    area_ko = free_translate_text(cfg.free_tr, recipe.get("area", ""), debug=cfg.run.debug) if recipe.get("area") else ""

    title_ko = _remove_punct_like_period(title_ko)
    category_ko = _remove_punct_like_period(category_ko)
    area_ko = _remove_punct_like_period(area_ko)

    ings_ko: List[Dict[str, str]] = []
    for it in recipe.get("ingredients", []):
        name = (it.get("name") or "").strip()
        mea = (it.get("measure") or "").strip()
        name_ko = free_translate_text(cfg.free_tr, name, debug=cfg.run.debug) if name else ""
        name_ko = _remove_punct_like_period(name_ko)
        ings_ko.append({"name_ko": name_ko or name, "measure": mea})

    return {
        "title_ko": title_ko or (recipe.get("title") or ""),
        "category_ko": category_ko,
        "area_ko": area_ko,
        "ingredients_ko": ings_ko,
        "steps_en": split_steps(recipe.get("instructions", "")),
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
    }


def pick_thumbnail_url(cfg: AppConfig, recipe_ko_title: str, recipe: Dict[str, Any]) -> str:
    # 1) Pexels 고퀄 우선 (키 있으면)
    q = (recipe_ko_title or "").strip()
    if q:
        q2 = q + " food"
        p = pexels_pick_image(cfg.pexels_api_key, q2, debug=cfg.run.debug)
        if p:
            return p
    # 2) TheMealDB 기본
    return (recipe.get("thumb") or "").strip()


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else ("오후" if slot == "pm" else "오늘")

    # date_key/slug: 기본은 같은 날은 업데이트  FORCE_NEW=1이면 매번 새 글
    date_key = f"{date_str}_{slot}" if slot in ("am", "pm") else date_str
    base_slug = f"daily-recipe-{date_str}-{slot}" if slot in ("am", "pm") else f"daily-recipe-{date_str}"

    suffix = ""
    if cfg.run.force_new:
        suffix = "-" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    slug = base_slug + suffix

    init_db(cfg.sqlite_path, debug=cfg.run.debug)
    existing = get_today_post(cfg.sqlite_path, date_key) if not cfg.run.force_new else None

    wp_post_id: Optional[int] = None
    if existing and existing.get("wp_post_id"):
        wp_post_id = int(existing["wp_post_id"])
    else:
        wp_post_id = wp_find_post_by_slug(cfg.wp, slug)

    recipe = pick_recipe(cfg, existing)
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title", "") or "Daily Recipe"

    recipe_ko = translate_recipe_bits(cfg, recipe)
    main_keyword = (recipe_ko.get("title_ko") or "오늘 레시피").strip()
    hook_title_ko = build_hook_title(main_keyword, cfg.naver.random_level)

    # 썸네일 URL 결정
    thumb_url = pick_thumbnail_url(cfg, main_keyword, recipe)

    # 썸네일 업로드
    media_id: Optional[int] = None
    media_url: str = ""
    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"recipe-{date_str}-{slot}{suffix}.jpg")
        except Exception as e:
            _debug(cfg.run.debug, "[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    # 본문 생성
    title_ko = hook_title_ko
    body_html = ""

    # OpenAI로 본문 생성 + 길이/영어 검증 후 자동 재시도
    try:
        client = OpenAI(api_key=cfg.openai.api_key)
        inst, user_in = _naver_prompt(cfg, recipe_ko, now)

        # 1차 생성
        resp = _openai_call_with_retry(
            client=client,
            model=cfg.openai.model,
            instructions=inst,
            input_text=user_in,
            max_retries=cfg.run.openai_max_retries,
            debug=cfg.run.debug,
        )
        text = (resp.output_text or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            raise RuntimeError("OpenAI 응답이 너무 짧습니다  제목 본문 분리 실패")

        # 모델 출력 제목/본문
        out_title = _remove_punct_like_period(lines[0])
        out_body = "\n".join(lines[1:]).strip()

        # 마침표류 제거 후처리
        out_title = _remove_punct_like_period(out_title)
        out_body = _remove_punct_like_period(out_body)

        if "<" not in out_body:
            out_body = _p(_html.escape(_nl2br(out_body)))

        # 길이/영어 검증
        ok, reason = _validate_lengths(out_title, out_body, cfg.naver.longform)
        _debug(cfg.run.debug, "[CHECK]", reason)

        # 2차 재생성(영어/길이 문제면)
        if not ok:
            repair_inst = inst + "\n\n[재요청] 길이나 영어 섞임 문제가 있었다  반드시 조건을 만족하도록 더 자연스럽고 더 길게 다시 작성하라  영어 단어는 절대 섞지 마라"
            resp2 = _openai_call_with_retry(
                client=client,
                model=cfg.openai.model,
                instructions=repair_inst,
                input_text=user_in,
                max_retries=1,
                debug=cfg.run.debug,
            )
            text2 = (resp2.output_text or "").strip()
            lines2 = [ln.strip() for ln in text2.splitlines() if ln.strip()]
            if len(lines2) >= 2:
                out_title = _remove_punct_like_period(lines2[0])
                out_body = "\n".join(lines2[1:]).strip()
                out_title = _remove_punct_like_period(out_title)
                out_body = _remove_punct_like_period(out_body)
                if "<" not in out_body:
                    out_body = _p(_html.escape(_nl2br(out_body)))

        title_ko = out_title if out_title else hook_title_ko
        body_html = out_body

    except Exception as e:
        if _is_insufficient_quota_error(e):
            print("[WARN] OpenAI quota depleted  fallback to free translate")
        else:
            print("[WARN] OpenAI failed  fallback to free translate:", repr(e))

        title_ko, body_html = build_korean_simple_with_free_translate(
            cfg.free_tr,
            recipe,
            now,
            keywords_csv=cfg.naver.keywords_csv,
            debug=cfg.run.debug,
        )

    # 본문 상단에 이미지 삽입(업로드 성공하면 WP URL 우선)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            img_tag = f'<p style="margin:0 0 18px 0;"><img src="{_html.escape(img)}" alt="{_html.escape(title_ko)}" style="max-width:100%;height:auto;border-radius:14px;"></p>'
            body_html = img_tag + "\n" + body_html

    # 마무리 멘트 + 태그는 항상 맨 아래에 한 번 더 붙여서 누락 방지
    body_html = body_html.strip()
    if "</div>" not in body_html:
        body_html = _html_wrap(body_html)

    # div 끝나기 전에 붙이기
    closing = _remove_punct_like_period(closing_random(main_keyword))
    append = ""
    append += _h2("마무리")
    append += _p(_html.escape(closing))
    append += tags_html(cfg.naver.keywords_csv, main_keyword)

    body_html = body_html.replace("</div>", append + "</div>", 1)

    # 최종 제목 (날짜/슬롯 포함)
    final_title = f"{date_str} {slot_label} 레시피  {title_ko}"
    final_title = _remove_punct_like_period(final_title).strip()

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", final_title)
        print("SLUG:", slug)
        print(body_html[:1800] + ("\n...(truncated)" if len(body_html) > 1800 else ""))
        return

    # 발행/업데이트
    if wp_post_id and not cfg.run.force_new:
        new_id, wp_link = wp_update_post(cfg.wp, wp_post_id, final_title, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)
        print("OK(updated):", new_id, wp_link, "| status:", cfg.wp.status)
    else:
        new_id, wp_link = wp_create_post(cfg.wp, final_title, slug, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)
        print("OK(created):", new_id, wp_link, "| status:", cfg.wp.status)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
