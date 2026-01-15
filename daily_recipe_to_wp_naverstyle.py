# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle.py (통합 완성본: 네이버 홈피드/홈판 친화형 글 템플릿 + 제목 후킹 랜덤화)
- TheMealDB 랜덤 레시피 수집
- OpenAI로 한국어 '홈피드형' 장문 작성 (도입 200~300자 + 굵은 소제목 3개, 각 1500자 이상, 총 2300자 이상(실제로는 4500+))
- 친구에게 수다 떨듯 존댓말, 마침표 없이('.' 제거), 여백/줄바꿈으로 호흡
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정
- SQLite 발행 이력 + 스키마 자동 마이그레이션
- ✅ OpenAI 크레딧 소진(insufficient_quota) 시: 템플릿 기반(비-AI) 장문 폴백 + (선택) LibreTranslate로 재료/단계 번역

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
  - FORCE_NEW=0|1 (기본 0)          # 1이면 매번 새 slug로 "항상 새 글"
  - DRY_RUN=0|1 (기본 0)
  - DEBUG=0|1 (기본 0)
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - OPENAI_MODEL=... (기본 gpt-4.1-mini)
  - OPENAI_MAX_RETRIES=3

네이버 스타일/랜덤화:
  - NAVER_RANDOM_LEVEL=0~3 (기본 2)         # 0=거의 고정, 3=변화 큼
  - NAVER_EXPERIENCE_LEVEL=0~3 (기본 2)     # 0=정보형, 3=현장감/실패포인트 강조
  - NAVER_KEYWORDS="키워드1,키워드2,..."     # 선택: 제목/본문에 자연스럽게 사용(첫 번째가 메인)
  - NAVER_TITLE_MODE=random|benefit|threat|curiosity|compare   # 기본 random
  - NAVER_TITLE_WEIGHTS="35,30,20,15"       # curiosity,benefit,threat,compare (기본값)
  - TITLE_NUMBER_EXAMPLES="3가지,5분,10분,1번" # (선택) 제목 숫자 슬롯용

고퀄 썸네일(선택):
  - THUMB_PROVIDER=themealdb|pexels (기본 themealdb)
  - PEXELS_API_KEY=...   # THUMB_PROVIDER=pexels일 때
  - PEXELS_STYLE=clean|minimal|white|topview (기본 clean)

LibreTranslate(선택):
  - FREE_TRANSLATE_URL=https://libretranslate.de/translate
  - FREE_TRANSLATE_API_KEY=
  - FREE_TRANSLATE_SOURCE=en
  - FREE_TRANSLATE_TARGET=ko
"""

from __future__ import annotations

import base64
import html
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

import openai  # 예외 타입
from openai import OpenAI

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


def _parse_str_list(csv: str) -> List[str]:
    out: List[str] = []
    for x in (csv or "").split(","):
        x = x.strip()
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
    run_slot: str = "day"     # day / am / pm
    dry_run: bool = False
    debug: bool = False
    force_new: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    openai_max_retries: int = 3


@dataclass
class NaverStyleConfig:
    random_level: int = 2          # 0~3
    experience_level: int = 2      # 0~3
    keywords: List[str] = field(default_factory=list)
    title_mode: str = "random"     # random|benefit|threat|curiosity|compare
    title_weights: List[int] = field(default_factory=lambda: [35, 30, 20, 15])  # curiosity,benefit,threat,compare
    title_numbers: List[str] = field(default_factory=list)


@dataclass
class ThumbConfig:
    provider: str = "themealdb"    # themealdb|pexels
    pexels_api_key: str = ""
    pexels_style: str = "clean"    # clean|minimal|white|topview


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
    thumb: ThumbConfig
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
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
    )

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

    free_url = _env("FREE_TRANSLATE_URL", "https://libretranslate.de/translate")
    free_api_key = _env("FREE_TRANSLATE_API_KEY", "")
    free_src = _env("FREE_TRANSLATE_SOURCE", "en")
    free_tgt = _env("FREE_TRANSLATE_TARGET", "ko")

    naver_random_level = max(0, min(3, _env_int("NAVER_RANDOM_LEVEL", 2)))
    naver_exp_level = max(0, min(3, _env_int("NAVER_EXPERIENCE_LEVEL", 2)))
    naver_keywords = _parse_str_list(_env("NAVER_KEYWORDS", ""))
    title_mode = (_env("NAVER_TITLE_MODE", "random") or "random").strip().lower()

    w_raw = _env("NAVER_TITLE_WEIGHTS", "35,30,20,15")
    w = []
    for x in w_raw.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            w.append(int(x))
        except Exception:
            pass
    if len(w) != 4 or sum(max(0, i) for i in w) <= 0:
        w = [35, 30, 20, 15]

    title_numbers = _parse_str_list(_env("TITLE_NUMBER_EXAMPLES", "3가지,5분,10분,1번,2번"))

    thumb_provider = (_env("THUMB_PROVIDER", "themealdb") or "themealdb").strip().lower()
    pexels_key = _env("PEXELS_API_KEY", "")
    pexels_style = (_env("PEXELS_STYLE", "clean") or "clean").strip().lower()

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
            keywords=naver_keywords,
            title_mode=title_mode,
            title_weights=w,
            title_numbers=title_numbers,
        ),
        thumb=ThumbConfig(
            provider=thumb_provider,
            pexels_api_key=pexels_key,
            pexels_style=pexels_style,
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", int(cfg.run.dry_run), "| DEBUG:", int(cfg.run.debug))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model, "| OPENAI_KEY:", ok(cfg.openai.api_key))
    print("[CFG] NAVER_RANDOM_LEVEL:", cfg.naver.random_level, "| NAVER_EXPERIENCE_LEVEL:", cfg.naver.experience_level)
    print("[CFG] NAVER_KEYWORDS:", ",".join(cfg.naver.keywords) if cfg.naver.keywords else "(empty)")
    print("[CFG] NAVER_TITLE_MODE:", cfg.naver.title_mode, "| WEIGHTS:", cfg.naver.title_weights)
    print("[CFG] THUMB_PROVIDER:", cfg.thumb.provider, "| PEXELS_KEY:", ok(cfg.thumb.pexels_api_key))
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


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, featured_media: Optional[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}

    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, featured_media: Optional[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status}

    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "thumb.jpg") -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=30)
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
        except Exception as e:
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] UnknownError → retry in {sleep_s:.2f}s | {repr(e)}")
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


# -----------------------------
# Utilities
# -----------------------------
def _strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _remove_periods(s: str) -> str:
    # 사용자 요청: 마침표 제거 (영문/전각 포함)
    for ch in [".", "。", "．"]:
        s = (s or "").replace(ch, "")
    return s


def _safe_space_rhythm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _weighted_choice(items: List[str], weights: List[int]) -> str:
    w = [max(0, int(x)) for x in weights]
    if not items:
        return ""
    if len(w) != len(items) or sum(w) <= 0:
        return random.choice(items)
    total = sum(w)
    r = random.uniform(0, total)
    upto = 0.0
    for item, wi in zip(items, w):
        upto += wi
        if r <= upto:
            return item
    return items[-1]


def _looks_like_finance_keyword(k: str) -> bool:
    k = (k or "")
    finance_markers = ["계좌", "적금", "대출", "청년", "금리", "이자", "세액", "연금", "ISA", "청약", "저축", "투자"]
    return any(m in k for m in finance_markers)


# -----------------------------
# Title generator (후킹 랜덤)
# -----------------------------
def build_hook_title(keyword: str, cfg: AppConfig) -> str:
    k = (keyword or "").strip()
    if not k:
        k = "오늘 레시피"

    mode = (cfg.naver.title_mode or "random").strip().lower()
    styles = ["curiosity", "benefit", "threat", "compare"]

    if mode == "random":
        chosen = _weighted_choice(styles, cfg.naver.title_weights)
    elif mode in styles:
        chosen = mode
    else:
        chosen = _weighted_choice(styles, cfg.naver.title_weights)

    is_fin = _looks_like_finance_keyword(k)
    num_pool = cfg.naver.title_numbers or ["3가지", "5분", "10분", "1번"]

    n1 = random.choice(num_pool)
    quoted_bits = [
        "이 한 줄",
        "딱 한 가지",
        "가장 많이 갈리는 지점",
        "진짜 핵심",
        "의외로 쉬운 기준",
        "실패가 줄어드는 포인트",
    ]
    q = random.choice(quoted_bits)

    if is_fin:
        benefit = [
            f"{k} 이번달 신청한 사람만 '조건별 혜택'에서 갈립니다",
            f"{k} {n1}만 체크했는데도 체감이 달라지더라고요",
            f"월급이 평범해도 {k}로 '흐름'이 바뀌는 경우가 있어요",
        ]
        threat = [
            f"{k} 잘못 고르면 '혜택'이 줄어드는 경우가 있어요 저도 처음엔 헷갈렸어요",
            f"{k} 자동이체 놓치면 '조건'에서 손해 보는 분들이 있더라고요",
            f"친구는 웃는데 나는 후회중 {k} {q} 차이였어요",
        ]
        curiosity = [
            f"{k} 해지하면 손해일까 {q}만 알면 정리가 돼요",
            f"친구는 이득 보고 나는 그대로 {k} 차이점이 뭐길래",
            f"{k} 가입한 사람들 {n1} 지나면 제일 먼저 보는 게 있더라고요",
        ]
        compare = [
            f"{k} vs 청년내일저축계좌 진짜 '돈 되는 쪽'은 따로 있어요",
            f"{k} 선택한 사람들 {n1} 뒤 '잔액 차이'로 놀라는 경우가 있어요",
            f"직장인 친구는 적금 저는 {k} 누가 더 맞을까요",
        ]
    else:
        benefit = [
            f"{k} 오늘은 {q}만 잡으면 '실패 확률'이 확 줄어요",
            f"{k} {n1}만 챙겼더니 맛이 안정되는 느낌이었어요",
            f"{k} 재료는 평범한데 결과가 다른 이유 {q}에 있어요",
        ]
        threat = [
            f"{k} 여기서 급하면 맛이 무너져요 {q} 때문에요",
            f"{k} 순서 한 번만 틀려도 '물맛' 나는 경우가 많더라고요",
            f"친구는 맛있다는데 나는 아쉬웠던 날 {k} {q} 차이였어요",
        ]
        curiosity = [
            f"{k} 왜 자꾸 아쉬울까 {q}만 잡으면 정리가 돼요",
            f"{k} 실패는 어디서 나올까 보통 {q}에서 갈리더라고요",
            f"{k} {n1} 지나도 자꾸 떠오르는 이유가 있어요",
        ]
        compare = [
            f"{k} vs {random.choice(['에어프라이어','프라이팬','냄비'])} 진짜 쉬운 쪽은 따로 있어요",
            f"{k} 집에서 한 사람들 {n1} 뒤 '차이'에 놀라는 포인트가 있어요",
            f"친구는 간단 버전 저는 {k} 기본 버전 누가 더 잘 맞을까요",
        ]

    pick_map = {
        "benefit": benefit,
        "threat": threat,
        "curiosity": curiosity,
        "compare": compare,
    }
    title = random.choice(pick_map.get(chosen, curiosity))
    title = _remove_periods(title)
    if len(title) > 52:
        title = title[:52].strip()
    return title


# -----------------------------
# Optional: Pexels thumbnail
# -----------------------------
def pexels_search_image(api_key: str, query: str, style: str = "clean", debug: bool = False) -> Optional[str]:
    if not api_key:
        return None

    style_map = {
        "clean": "clean food minimal",
        "minimal": "minimal food",
        "white": "food white background",
        "topview": "food top view minimal",
    }
    q = (query or "food").strip()
    q2 = style_map.get(style, style_map["clean"])
    full_query = f"{q} {q2}"

    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": full_query, "per_page": 10, "orientation": "landscape"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=25)
        if r.status_code != 200:
            if debug:
                print("[PEXELS] search failed:", r.status_code, r.text[:200])
            return None
        j = r.json()
        photos = j.get("photos") or []
        candidates = []
        for p in photos:
            src = p.get("src") or {}
            u = src.get("large2x") or src.get("large") or src.get("original")
            if u:
                candidates.append(u)
        if not candidates:
            return None
        return random.choice(candidates)
    except Exception as e:
        if debug:
            print("[PEXELS] error:", repr(e))
        return None


# -----------------------------
# OpenAI prompt for homefeed-long template
# -----------------------------
def _build_homefeed_prompt(cfg: AppConfig, recipe: Dict[str, Any], keyword_ko: str, now: datetime) -> Tuple[str, str]:
    random_level = cfg.naver.random_level
    exp_level = cfg.naver.experience_level
    kws = cfg.naver.keywords[:8] if cfg.naver.keywords else []
    kw_hint = ", ".join(kws) if kws else ""

    openings = [
        "이 메뉴요  딱히 거창한 날 아니어도 생각나잖아요  그래서 오늘은 어렵게 말 안 하고  진짜 필요한 얘기만 편하게 풀어볼게요",
        "요즘처럼 기운 없을 때는  복잡한 것보다  실패 확률 낮은 쪽이 마음이 편하더라고요  그래서 이 메뉴를 기준으로 잡아봤어요",
        "레시피는 많아도  막상 집에서 하면 어딘가 한 끗이 아쉬울 때가 있잖아요  그 지점을 같이 정리해볼게요",
    ]
    rhythms = [
        "저는 요리를 볼 때  정답보다  반복 가능한 방식이 먼저라고 생각하거든요",
        "완벽하게 하려다 오히려 망하는 날이 있어서  저는 기준을 작게 잡는 편이에요",
        "맛은 결국  내 생활 리듬에 맞아야 오래 가더라고요",
    ]
    empathy = [
        "괜히 급해지면 손이 꼬이는 구간이 있잖아요  그때만 천천히 가면 생각보다 잘 풀려요",
        "이게 별거 아닌데  한 번만 데이면 다음부터는 습관이 되더라고요",
        "저도 이런 글 볼 때  너무 교과서 같으면 바로 닫아버리거든요  그래서 말하듯이 써볼게요",
    ]

    seed_open = random.choice(openings)
    seed_rhythm = random.choice(rhythms)
    seed_empathy = random.choice(empathy)

    steps_en = split_steps(recipe.get("instructions", ""))
    payload = {
        "keyword_ko": keyword_ko,
        "recipe_title_en": recipe.get("title", ""),
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients": recipe.get("ingredients", []),
        "steps_en": steps_en,
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
        "constraints": {
            "intro_chars": [200, 300],
            "section_min_chars": 1500,
            "section_count": 3,
            "total_min_chars": 2300,
            "no_period": True,
            "tone": "친구에게 수다 떠는 존댓말",
        },
        "seed": {
            "random_level": random_level,
            "experience_level": exp_level,
            "opening": seed_open,
            "rhythm": seed_rhythm,
            "empathy": seed_empathy,
            "keywords_hint": kw_hint,
            "season": now.astimezone(KST).strftime("%m월"),
        },
    }

    instructions = f"""
너는 한국어 블로그 글을 쓰는 사람이다  독자와 친구처럼 수다 떠는 존댓말로 쓴다

[절대 규칙]
- 마침표 문자 . 을 쓰지 마  문장 끝도 마침표로 끝내지 마
- 가능하면 물음표  느낌표도 최소로  대신 줄바꿈과 여백으로 호흡을 만들어
- "오늘 해먹어봤는데"  "제가 해보니" 처럼 실제 체험을 단정하지 마  대신 "집에서 해보면"  "보통은"  "저는 이런 기준을 좋아해요" 같은 말로 현장감만 만들어
- 제공된 ingredients 와 steps_en 기반으로만 요리 내용 구성  재료  계량  단계 수를 추가 삭제 변경하지 마
- 시간  온도  숫자는 원문에 없으면 단정하지 마  필요하면 "상태를 보며"로 표현
- AI  챗GPT  자동생성 같은 단어를 본문에 쓰지 마

[출력 형식  HTML만]
- 첫 줄에 제목을 쓰지 마  본문 HTML만 출력해
- 도입부는 <p> 한 블록  텍스트 길이 200~300자 사이
- 굵은 소제목 3개  각 소제목은 <h2><b>제목</b></h2> 형태
- 각 소제목 아래 본문은 길게  최소 1500자 이상  <p> 와 <br/> 로 리듬 있게
- 3번째 소제목 섹션 말미에 자연스럽게 한 번만  저장해두면 편하다는 뉘앙스 한 문장
- 마지막에 출처는 아주 짧게  <p style='opacity:.6;font-size:13px;'> 로 링크 있으면 넣기
- 전체는 자연스러운 사람 글  뻔한 교과서 톤 금지  반복 문구 금지

[내용 구성 가이드]
1) 도입  seed opening 과 rhythm 을 자연스럽게 섞기
2) 소제목 1  가치관  상황  왜 이 메뉴가 편해지는지  감정  루틴 중심
3) 소제목 2  실패 포인트  상태 기준  steps_en 을 순서대로 녹여서 흐름 설명  단계 수는 동일
4) 소제목 3  내 생활에 맞게 유지하는 방법  작은 체크리스트  마무리 수다  seed empathy 섞기  키워드 힌트 있으면 과하지 않게 한 번만

[Seed]
opening  {seed_open}
rhythm   {seed_rhythm}
empathy  {seed_empathy}
keywords_hint  {kw_hint}
""".strip()

    user_input = "JSON:\n" + json.dumps(payload, ensure_ascii=False)
    return instructions, user_input


def _extract_sections_and_validate(html_body: str) -> Tuple[bool, Dict[str, int], str]:
    body = html_body or ""
    parts = re.split(r"(<h2[^>]*>.*?</h2>)", body, flags=re.IGNORECASE | re.DOTALL)
    intro = parts[0] if parts else body
    ok = True
    metrics = {}

    intro_txt = _strip_tags(intro)
    metrics["intro"] = len(intro_txt)
    if not (200 <= len(intro_txt) <= 320):
        ok = False

    sections = []
    i = 0
    while i < len(parts):
        if parts[i].lower().startswith("<h2"):
            h = parts[i]
            sec = parts[i + 1] if i + 1 < len(parts) else ""
            sections.append(h + sec)
            i += 2
        else:
            i += 1

    if len(sections) < 3:
        ok = False

    for idx in range(3):
        txt = _strip_tags(sections[idx]) if idx < len(sections) else ""
        metrics[f"section{idx+1}"] = len(txt)
        if len(txt) < 1500:
            ok = False

    if "." in body:
        ok = False

    return ok, metrics, body


def _fallback_long_html(cfg: AppConfig, recipe: Dict[str, Any], keyword_ko: str, now: datetime) -> str:
    title_en = recipe.get("title", "") or "Recipe"
    steps_en = split_steps(recipe.get("instructions", ""))[:20]
    k = keyword_ko or title_en

    intro = (
        f"{k} 얘기요  요리 글은 많은데  막상 집에서 하면 한 끗이 아쉬울 때가 있잖아요  "
        f"그래서 오늘은 어렵게 말 안 하고  실패가 줄어드는 기준만 친구한테 수다 떨듯이 정리해볼게요  "
        f"상태를 보면서 천천히 가면  생각보다 편하게 끝나더라고요"
    )
    intro = intro[:280]
    intro = _remove_periods(intro)

    def para(lines: List[str]) -> str:
        s = "<br/><br/>".join([html.escape(_remove_periods(x)) for x in lines if x.strip()])
        return f"<p>{s}</p>"

    sec1_lines = [
        "저는 요리를 할 때  완벽한 정답보다  반복 가능한 방식이 먼저라고 생각해요",
        "특히 바쁜 날에는  레시피를 외우기보다  내 눈으로 상태를 보고 조절하는 쪽이 마음이 편하더라고요",
        "그래서 이 메뉴도  크게 두 가지로만 나눠서 봐요  처음에 흐름을 잡는 구간  그리고 끝에서 맛을 정리하는 구간",
        "처음에 급해지면 손이 꼬이고  끝에서 욕심내면 과해지는 경우가 많아서요",
        "제가 좋아하는 기준은 단순해요  지금 냄새가 올라오나  질감이 붙기 시작했나  물기가 남았나  이 정도만 체크해요",
        "이렇게 기준을 작게 잡아두면  중간에 변수가 생겨도 다시 돌아오기가 쉽더라고요",
        "오늘은 그 느낌으로  편하게 끝까지 가는 흐름을 같이 잡아볼게요",
    ]
    fillers1 = [
        "재료가 조금 달라도 괜찮아요  다만 같은 재료는 크기를 비슷하게 맞추면 흐름이 안정돼요",
        "간은 중간에 급하게 잡기보다  마지막에 정리하는 편이 실패가 줄어요  특히 짠맛은 되돌리기 어렵잖아요",
        "수분은 한 번에 늘리지 말고  조금씩 보면서 맞추면 마음이 편해요",
        "불은 숫자보다 소리로 듣는 편이에요  지글지글이 너무 거칠면 조금만 낮추고  너무 조용하면 살짝 올리고요",
        "한 번에 대단하게 하려고 하면 늘 지치더라고요  그래서 저는 부담 안 되는 루틴이 오래 간다고 믿는 편이에요",
        "지금부터는 단계 흐름을 같이 볼 건데요  단계는 그대로 두고  실패 포인트만 덧붙여볼게요",
    ]
    while len(_strip_tags("<br/>".join(sec1_lines))) < 1550:
        sec1_lines.append(random.choice(fillers1))

    sec2_lines = ["여기부터는 단계 흐름이에요  레시피 단계는 그대로 따라가되  중간중간 상태 기준만 붙여볼게요"]
    for i, st in enumerate(steps_en[:12], start=1):
        st_clean = st.strip()
        if not st_clean:
            continue
        sec2_lines.append(f"{i}단계  {st_clean}")
        sec2_lines.append("실패 포인트는 급하게 넘기는 거예요  상태 기준은 냄새와 질감이 한 번 변하는 타이밍을 느끼는 거고요")
        sec2_lines.append("만약 물기가 애매하면  바로 더 넣지 말고  잠깐 기다렸다가 결정하면 실수가 줄어요")
    fillers2 = [
        "색이 진해지는 순간이 오면  그때부터는 금방 진행되니까  그 전까지만 천천히 가면 돼요",
        "향이 올라오면  이미 반은 성공한 거예요  그때부터는 과하지 않게만 정리하면 됩니다",
        "재료가 바닥에 붙기 시작하면  불을 조금 낮추고  뒤집는 타이밍을 늘려보면 좋아요",
        "간이 불안하면  바로 더 넣지 말고  한 숟갈 떠서 식혀서 맛보는 편이 안정적이에요",
        "식감은 완성 직전이 갈림길이에요  부드럽게 갈지  살아 있게 갈지  내 취향을 하나만 정하면 됩니다",
    ]
    while len(_strip_tags("<br/>".join(sec2_lines))) < 1550:
        sec2_lines.append(random.choice(fillers2))

    sec3_lines = [
        "마지막은 제일 현실적인 얘기요  이 메뉴를 꾸준히 하려면  내 생활에 맞게 기준을 낮춰야 하더라고요",
        "그래서 저는 체크리스트를 작게 잡아요  오늘은 이것만 지키자  이렇게요",
        "제가 자주 쓰는 작은 체크리스트예요",
        "재료 크기만 비슷하게 맞추기",
        "중간에 간을 급하게 잡지 않기",
        "물기는 조금씩 조절하기",
        "향이 올라오는 타이밍 놓치지 않기",
        "완성 직전에 식감만 한 번 결정하기",
        "이렇게만 해도  이상하게 마음이 편해져요  뭔가 내 기준이 생긴 느낌이거든요",
        "다음에 메뉴 고민될 때  이 글 저장해두면  필요한 부분만 쓱 보고 다시 시작하기 편하실 거예요",
    ]
    fillers3 = [
        "완벽한 한 번보다  무난하게 여러 번이 더 힘이 됩니다  그게 진짜 내 편이 되는 느낌이거든요",
        "혹시 오늘은 기운이 없으면  단계 중에 한 번만 천천히 해도 충분해요  그 한 번이 결과를 바꿔요",
        "요리는 결국 내 삶을 덜 힘들게 하려고 하는 거잖아요  그래서 저는 부담 안 되는 쪽을 선택하는 걸 더 좋아해요",
    ]
    while len(_strip_tags("<br/>".join(sec3_lines))) < 1550:
        sec3_lines.append(random.choice(fillers3))

    src = (recipe.get("source") or "").strip()
    yt = (recipe.get("youtube") or "").strip()
    src_html = "-"
    if src:
        src_html = f"<a href='{html.escape(src)}' target='_blank' rel='nofollow noopener'>source</a>"
    elif yt:
        src_html = f"<a href='{html.escape(yt)}' target='_blank' rel='nofollow noopener'>youtube</a>"

    body = "\n".join([
        para([intro]),
        f"<h2><b>{html.escape(_remove_periods('이 메뉴를 편하게 만드는 기준'))}</b></h2>",
        para(sec1_lines),
        f"<h2><b>{html.escape(_remove_periods('실패 포인트와 상태 기준  흐름대로'))}</b></h2>",
        para(sec2_lines),
        f"<h2><b>{html.escape(_remove_periods('내 생활에 맞게 계속 가는 방법'))}</b></h2>",
        para(sec3_lines),
        f"<p style='opacity:.6;font-size:13px;'>출처  {src_html}  {now.astimezone(KST).strftime('%Y-%m-%d')}</p>",
    ])
    body = _remove_periods(body)
    body = _safe_space_rhythm(body)
    return body


# -----------------------------
# Main flow helpers
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


def pick_thumbnail_url(cfg: AppConfig, recipe: Dict[str, Any], keyword_ko: str) -> str:
    thumb = (recipe.get("thumb") or "").strip()
    if cfg.thumb.provider == "pexels" and cfg.thumb.pexels_api_key:
        q = keyword_ko or (recipe.get("title") or "food")
        u = pexels_search_image(cfg.thumb.pexels_api_key, q, style=cfg.thumb.pexels_style, debug=cfg.run.debug)
        if u:
            return u
    return thumb


def generate_homefeed_body_openai(cfg: AppConfig, recipe: Dict[str, Any], keyword_ko: str, now: datetime) -> str:
    client = OpenAI(api_key=cfg.openai.api_key)
    inst, user_in = _build_homefeed_prompt(cfg, recipe, keyword_ko, now)

    for attempt in range(4):
        resp = _openai_call_with_retry(
            client=client,
            model=cfg.openai.model,
            instructions=inst,
            input_text=user_in,
            max_retries=cfg.run.openai_max_retries,
            debug=cfg.run.debug,
        )
        text = (resp.output_text or "").strip()
        text = _remove_periods(text)
        text = _safe_space_rhythm(text)

        ok, metrics, body = _extract_sections_and_validate(text)
        if cfg.run.debug:
            print("[VALIDATE] attempt", attempt + 1, metrics, "ok=", ok)
        if ok:
            return body

        cfg.naver.random_level = max(0, min(3, cfg.naver.random_level + (1 if attempt == 1 else 0)))
        inst, user_in = _build_homefeed_prompt(cfg, recipe, keyword_ko, now)

    return text


def ko_keyword_from_recipe(cfg: AppConfig, recipe: Dict[str, Any]) -> str:
    if cfg.naver.keywords:
        return cfg.naver.keywords[0]

    title_en = (recipe.get("title") or "").strip()
    if not title_en:
        return "오늘 레시피"
    ko = free_translate_text(cfg.free_tr, title_en, debug=cfg.run.debug) if cfg.free_tr.url else title_en
    ko = re.sub(r"\s+", " ", (ko or "").strip())
    if len(ko) > 18:
        ko = ko[:18].strip()
    return _remove_periods(ko) or "오늘 레시피"


# -----------------------------
# Main run
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else ("오후" if slot == "pm" else "오늘")

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

    keyword_ko = ko_keyword_from_recipe(cfg, recipe)

    hook_title = build_hook_title(keyword_ko, cfg)

    try:
        body_html = generate_homefeed_body_openai(cfg, recipe, keyword_ko, now)
    except Exception as e:
        if _is_insufficient_quota_error(e):
            print("[WARN] OpenAI quota depleted → fallback long template")
        else:
            print("[WARN] OpenAI failed → fallback long template:", repr(e))
        body_html = _fallback_long_html(cfg, recipe, keyword_ko, now)

    body_html = _remove_periods(body_html)

    thumb_url = pick_thumbnail_url(cfg, recipe, keyword_ko)

    media_id: Optional[int] = None
    media_url: str = ""
    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"thumb-{date_str}-{slot}{suffix}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            alt = html.escape(_remove_periods(hook_title))
            body_html = f'<p><img src="{html.escape(img)}" alt="{alt}" style="max-width:100%;height:auto;border-radius:14px;"></p>\n' + body_html

    wp_title = _remove_periods(f"{date_str} {slot_label} | {hook_title}")

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략 미리보기")
        print("TITLE:", wp_title)
        print("SLUG:", slug)
        ok, metrics, _ = _extract_sections_and_validate(body_html)
        print("[BODY_VALIDATE]", metrics, "ok=", ok)
        print(body_html[:2000] + ("\n...(truncated)" if len(body_html) > 2000 else ""))
        return

    if wp_post_id and not cfg.run.force_new:
        new_id, wp_link = wp_update_post(cfg.wp, wp_post_id, wp_title, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)
        print("OK(updated):", new_id, wp_link)
    else:
        new_id, wp_link = wp_create_post(cfg.wp, wp_title, slug, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)
        print("OK(created):", new_id, wp_link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
