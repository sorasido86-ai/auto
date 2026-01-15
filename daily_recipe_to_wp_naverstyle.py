# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle.py (통합 완성본: 네이버 홈피드/홈판 친화형 톤 + 가독성 강화 + 재료 목록 자동 삽입 + 제목 후킹 랜덤)
- TheMealDB 랜덤 레시피 수집
- OpenAI로 한국어 '사람 글' 톤(친구에게 존댓말 수다 / 경험담 느낌 / 실패포인트 / 상태기준 / 여백 호흡) 생성
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정
- SQLite 발행 이력 + 스키마 자동 마이그레이션
- OpenAI 크레딧 소진(insufficient_quota) 시: 무료번역(LibreTranslate) 폴백

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
  - FORCE_NEW=0|1 (기본 0)          # 1이면 매번 새 글(중복 업로드 의심될 때)
  - DRY_RUN=0|1 (기본 0)
  - DEBUG=0|1 (기본 0)
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - OPENAI_MODEL=... (기본 gpt-4.1-mini)
  - OPENAI_MAX_RETRIES=3
  - NAVER_TEXT_TRIES=3 (기본 3)     # 글자수/형식 미달 시 재생성 횟수

네이버 스타일/랜덤화:
  - NAVER_RANDOM_LEVEL=0~3 (기본 2)       # 0=거의 고정 3=변화 큼
  - NAVER_EXPERIENCE_LEVEL=0~3 (기본 2)   # 0=정보형 3=현장감/실패포인트 강조
  - NAVER_KEYWORDS="키워드1,키워드2,..." (선택) # 제목/본문에 자연스럽게 한두 번만 녹임
  - PREFER_AREAS="Korean,Japanese,Chinese" (선택) # TheMealDB area 필터

무료번역(LibreTranslate):
  - FREE_TRANSLATE_URL=https://libretranslate.de/translate
  - FREE_TRANSLATE_API_KEY=
  - FREE_TRANSLATE_SOURCE=en
  - FREE_TRANSLATE_TARGET=ko
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
    naver_text_tries: int = 3


@dataclass
class NaverStyleConfig:
    random_level: int = 2          # 0~3
    experience_level: int = 2      # 0~3
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
        naver_text_tries=_env_int("NAVER_TEXT_TRIES", 3),
    )

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

    free_url = _env("FREE_TRANSLATE_URL", "https://libretranslate.de/translate")
    free_api_key = _env("FREE_TRANSLATE_API_KEY", "")
    free_src = _env("FREE_TRANSLATE_SOURCE", "en")
    free_tgt = _env("FREE_TRANSLATE_TARGET", "ko")

    naver_random_level = max(0, min(3, _env_int("NAVER_RANDOM_LEVEL", 2)))
    naver_exp_level = max(0, min(3, _env_int("NAVER_EXPERIENCE_LEVEL", 2)))
    naver_keywords = _env("NAVER_KEYWORDS", "")
    prefer_areas = _parse_str_list(_env("PREFER_AREAS", ""))

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
            keywords_csv=naver_keywords,
            prefer_areas=prefer_areas,
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
    print("[CFG] NAVER_KEYWORDS:", cfg.naver.keywords_csv or "(empty)")
    print("[CFG] PREFER_AREAS:", ",".join(cfg.naver.prefer_areas) if cfg.naver.prefer_areas else "(any)")
    print("[CFG] NAVER_TEXT_TRIES:", cfg.run.naver_text_tries)
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

    r = requests.post(url, headers=headers, json=payload, timeout=45)
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
        except Exception:
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] UnknownError → retry in {sleep_s:.2f}s")
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
# Title hook templates (recipe-optimized 4 types)
# -----------------------------
def _pick_weighted(items: List[Tuple[str, int]]) -> str:
    total = sum(w for _, w in items)
    r = random.randint(1, max(1, total))
    acc = 0
    for v, w in items:
        acc += w
        if r <= acc:
            return v
    return items[-1][0]


def make_hook_title(keyword: str, random_level: int = 2) -> str:
    """
    키워드를 제목 맨 앞에 두고
    이득형 / 위협형 / 궁금형 / 비교형 중 랜덤으로 제목 생성
    """
    kw = (keyword or "").strip()
    if not kw:
        kw = "오늘 레시피"

    weights = [
        ("benefit", 30),
        ("threat", 20),
        ("curiosity", 35),
        ("compare", 15),
    ]
    t = _pick_weighted(weights)

    benefits = [
        f"{kw} 이거 하나만 바꾸면 맛이 확 달라져요",
        f"{kw} 처음부터 완벽 말고 이 기준만 잡아도 돼요",
        f"{kw} 집에서 자주 하는데도 실패 줄인 포인트 1개",
    ]
    threats = [
        f"{kw} 여기서 실수하면 맛이 갑자기 무너져요",
        f"{kw} 순서만 헷갈리면 밍밍해지기 쉬워요",
        f"{kw} 간 잡는 타이밍 놓치면 한 번에 흔들려요",
    ]
    curios = [
        f"{kw} 왜 자꾸 밍밍할까 의외로 여기였어요",
        f"{kw} 유독 맛이 들쭉날쭉한 이유 딱 한 군데",
        f"{kw} 사람들이 헷갈리는 포인트 제가 한 번 정리해요",
    ]
    compares = [
        f"{kw} vs 비슷한 메뉴 집에서 더 쉬운 쪽은",
        f"{kw} 밖에서 먹는 맛이랑 다른 이유 비교해보면 보여요",
        f"{kw} 비슷한 재료인데 결과가 갈리는 지점은 여기",
    ]

    pool = benefits if t == "benefit" else threats if t == "threat" else curios if t == "curiosity" else compares
    title = random.choice(pool)

    # 랜덤 레벨이 높을수록 약간의 후킹 장치 추가
    if random_level >= 2:
        tails = [
            "저도 여기서 많이 흔들렸어요",
            "이거 알면 다음엔 훨씬 편해요",
            "저장해두면 진짜 도움돼요",
        ]
        if random.random() < 0.55:
            title = f"{title} {random.choice(tails)}"
    return title.replace(".", "").strip()


# -----------------------------
# Readability helpers (HTML spacing)
# -----------------------------
def wrap_readable(html: str) -> str:
    """
    문단 간격과 행간을 강제로 줘서
    네이버에서 복붙했을 때도 읽기 편하게
    """
    html = (html or "").strip()
    if not html:
        return html
    if "<div data-wrap=" in html:
        return html
    return (
        "<div data-wrap='1' style=\"line-height:1.9;font-size:16px;letter-spacing:-0.2px;word-break:keep-all;\">"
        + html
        + "</div>"
    )


def strip_tags(text: str) -> str:
    t = re.sub(r"<[^>]+>", " ", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def count_chars_no_space(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def extract_intro_and_sections(body_html: str) -> Tuple[str, List[str]]:
    """
    <p data-intro="1">...</p>
    <div data-sec="1">...</div> 3개
    """
    intro = ""
    m = re.search(r"<p[^>]*data-intro=['\"]1['\"][^>]*>(.*?)</p>", body_html, flags=re.S | re.I)
    if m:
        intro = strip_tags(m.group(1))

    sections: List[str] = []
    for i in (1, 2, 3):
        mm = re.search(rf"<div[^>]*data-sec=['\"]{i}['\"][^>]*>(.*?)</div>", body_html, flags=re.S | re.I)
        if mm:
            sections.append(strip_tags(mm.group(1)))
        else:
            sections.append("")
    return intro, sections


def validate_longform(body_html: str) -> Tuple[bool, str]:
    intro, secs = extract_intro_and_sections(body_html)
    intro_len = count_chars_no_space(intro)
    sec_lens = [count_chars_no_space(s) for s in secs]
    total_len = count_chars_no_space(strip_tags(body_html))

    # 사용자가 준 스펙
    if not (200 <= intro_len <= 330):
        return False, f"intro_len={intro_len} not in 200~330"
    if any(l < 1500 for l in sec_lens):
        return False, f"section_len={sec_lens} require each>=1500"
    if total_len < 2300:
        return False, f"total_len={total_len} require>=2300"
    return True, "ok"


def build_ingredients_html(cfg: AppConfig, recipe: Dict[str, Any]) -> str:
    """
    재료 목록은 모델 출력과 무관하게 항상 삽입
    이름은 가능한 경우에만 무료 번역으로 살짝 한글화
    """
    items = []
    for it in recipe.get("ingredients", []):
        name_en = (it.get("name") or "").strip()
        mea = (it.get("measure") or "").strip()
        if not name_en:
            continue
        # 번역은 best-effort
        name_ko = free_translate_text(cfg.free_tr, name_en, debug=cfg.run.debug)
        if name_ko and name_ko.lower() != name_en.lower():
            label = f"{name_ko} ({name_en})"
        else:
            label = name_en
        if mea:
            items.append(f"<li style=\"margin:0 0 8px 0;\">{label} <span style='opacity:.75'>— {mea}</span></li>")
        else:
            items.append(f"<li style=\"margin:0 0 8px 0;\">{label}</li>")

    if not items:
        return "<ul data-ingredients='1'><li>재료 정보가 비어있어요</li></ul>"

    return "<ul data-ingredients='1' style=\"margin:10px 0 18px 18px;\">" + "".join(items) + "</ul>"


# -----------------------------
# Naver-style prompt (homefeed longform + spacing)
# -----------------------------
def _naver_prompt(cfg: AppConfig, recipe: Dict[str, Any], now: datetime, title_seed: str, ingredients_html: str) -> Tuple[str, str]:
    random_level = cfg.naver.random_level
    exp_level = cfg.naver.experience_level

    hooks = [
        "레시피는 따라 했는데도 왜 결과가 매번 다르게 나오는지  은근히 신경 쓰이시죠",
        "똑같은 재료인데 맛이 갈리는 지점이 꼭 있더라고요  그 지점만 잡아도 마음이 편해요",
        "완벽하게 하려다 오히려 망하는 날이 있어서  저는 기준부터 잡는 쪽으로 바뀌었어요",
        "이 메뉴  막상 해보면 다들 비슷한 지점에서 흔들리더라고요  그래서 그 부분만 정리해요",
    ]
    one_lines = [
        "오늘은 상태 기준 하나만 잡고 가요  나머지는 크게 안 흔들려요",
        "이 글은 재료보다 순서랑 상태 기준에 집중했어요  그게 제일 안정적이더라고요",
        "실패 포인트만 미리 알고 가면  요리가 훨씬 편해져요",
    ]
    points = [
        "향이 올라오는 타이밍을 기준으로 보기",
        "수분은 한 번에 늘리지 말고 조금씩 보기",
        "간은 마지막에 정리한다는 마음으로 보기",
        "재료 크기를 맞춰서 익는 속도를 맞추기",
        "완성 직전 식감에서 한 번 더 확인하기",
    ]
    hook_seed = random.choice(hooks)
    one_seed = random.choice(one_lines)
    point_seed = random.choice(points)

    kw_list = [k.strip() for k in (cfg.naver.keywords_csv or "").split(",") if k.strip()]
    kw_hint = ", ".join(kw_list[:6]) if kw_list else ""

    steps_en = split_steps(recipe.get("instructions", ""))

    payload_recipe = {
        "title_en": recipe.get("title", ""),
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients": recipe.get("ingredients", []),
        "steps_en": steps_en,
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
        "naver_style": {
            "random_level": random_level,
            "experience_level": exp_level,
            "hook_seed": hook_seed,
            "one_seed": one_seed,
            "point_seed": point_seed,
            "title_seed": title_seed,
            "keywords_hint": kw_hint,
            "season_hint": now.astimezone(KST).strftime("%m월"),
            "ingredients_html": ingredients_html,
            "steps_count": len(steps_en),
        },
    }

    instructions = f"""
너는 한국어 블로그 글을 쓰는 사람이다  친구에게 진심 담아 수다 떠는 존댓말로 써라

[절대 규칙]
- 너는 실제로 조리해본 적이 없다  그래서 "오늘 해먹어봤는데" "제가 해보니" 같은 체험 단정은 금지
- 대신 "집에서 해보면" "보통 이 지점에서" "상태를 이렇게 보면" 같은 현장감 표현은 허용
- 제공된 ingredients 와 steps 외 재료 계량 단계 추가 삭제 변경 금지
- 원문에 없는 시간 온도 숫자 단정 금지  필요하면 "상태를 보며"로 표현
- 마침표(., 。) 사용 금지  문장 호흡은 줄바꿈과 여백으로 조절
- 과한 광고 말투 과한 이모지 AI티 문구 금지

[가독성 규칙]
- 문단은 짧게  한 문단 2~4줄 정도
- 문단 사이 여백을 확실히  <p> 를 자주 쓰고  문단 안에서는 <br/><br/> 로 호흡을 둬라
- 전체를 <div data-wrap='1' style="line-height:1.9;font-size:16px;"> 로 감싸라

[형식 요구]
- 도입부는 200~300자(공백 제외)  반드시 <p data-intro="1"> 로 감싼다
- 굵은 소제목 3개  각 섹션은 공백 제외 1500자 이상  반드시 <div data-sec="1"> 2 3 로 감싼다
- 총 글자수는 2300자 이상
- 각 섹션 맨 위에는 반드시 굵은 제목을 <p><strong>...</strong></p> 로 넣어라

[재료 목록]
- 재료 목록은 아래 HTML 을 그대로 삽입해라  수정 금지
{ingredients_html}

[조리 과정]
- steps_en 의 개수만큼만 <li> 를 만들어라  개수는 반드시 동일
- <ol data-steps="1"> 로 감싸라
- 각 <li> 안에
  1) 원문 step 의미를 자연스럽게 한국어로 풀어쓴 문장
  2) 같은 <li> 안에 <div style='font-size:13px;opacity:.82;margin-top:10px;'>실패 포인트: ...<br/><br/>상태 기준: ...</div> 한 줄을 넣어라

[제목]
- 첫 줄은 제목만 출력
- 제목은 아래 Seed 를 기반으로 하되  키워드를 맨 앞에 두고 과자극 금지  26~46자
- Seed: {title_seed}

[출력]
- 첫 줄: 제목
- 둘째 줄부터: HTML만 출력
""".strip()

    user_input = "레시피 JSON:\n" + json.dumps(payload_recipe, ensure_ascii=False, indent=2)
    return instructions, user_input


# -----------------------------
# Fallback (free translate)  가독성/재료 포함
# -----------------------------
def build_korean_simple_with_free_translate(
    cfg: AppConfig,
    recipe: Dict[str, Any],
    now: datetime,
    title_seed: str,
    ingredients_html: str,
) -> Tuple[str, str]:
    title_en = recipe.get("title", "Daily Recipe")
    title_ko = free_translate_text(cfg.free_tr, title_en, debug=cfg.run.debug) or title_en

    steps_en = split_steps(recipe.get("instructions", ""))
    step_lines = []
    for s in steps_en:
        ko = free_translate_text(cfg.free_tr, s, debug=cfg.run.debug) or s
        # 마침표 제거(문장 끝에 있는 것만 가볍게)
        ko = ko.replace(".", "").replace("。", "")
        step_lines.append(
            "<li style=\"margin:0 0 14px 0;\">"
            + f"{ko}<div style='font-size:13px;opacity:.82;margin-top:10px;'>실패 포인트: 상태를 보며 천천히<br/><br/>상태 기준: 향과 농도를 먼저 확인</div>"
            + "</li>"
        )

    src = (recipe.get("source") or "").strip()
    yt = (recipe.get("youtube") or "").strip()

    body = f"""
<div data-wrap='1' style="line-height:1.9;font-size:16px;letter-spacing:-0.2px;word-break:keep-all;">
  <p data-intro="1" style="margin:0 0 18px 0;">
    오늘은 너무 어렵게 말고  흐름만 잡고 가는 레시피로 정리해볼게요<br/><br/>
    재료는 아래 목록 기준으로 보고  과정은 상태를 보면서 따라가면 됩니다
  </p>

  <div data-sec="1">
    <p style="margin:0 0 14px 0;"><strong>재료 먼저 보고 마음을 가볍게 잡는 쪽으로요</strong></p>
    <p style="margin:0 0 18px 0;">
      저는 레시피를 볼 때  재료부터 훑고 나서 마음을 정리하는 편이에요<br/><br/>
      막상 해보면  재료가 복잡해서가 아니라  중간에 급해져서 흔들리는 경우가 많더라고요
    </p>
    {ingredients_html}
    <p style="margin:0 0 18px 0;">
      이 메뉴는 완벽하게 하려는 순간부터 손이 꼬일 수 있어서  기준만 잡고 가는 게 편합니다<br/><br/>
      오늘은 내 입맛에 맞게  마지막에만 조절한다  이 마음으로 시작해보시면 좋아요
    </p>
  </div>

  <div data-sec="2">
    <p style="margin:0 0 14px 0;"><strong>만드는 과정은 단계보다 상태가 더 중요하더라고요</strong></p>
    <p style="margin:0 0 18px 0;">
      시간 온도 숫자보다  향  농도  식감  이런 상태가 훨씬 정확할 때가 많아요<br/><br/>
      아래 과정은 순서 그대로 따라가되  중간중간 잠깐 멈추고 상태를 확인하는 느낌으로 가면 안정적입니다
    </p>
    <ol data-steps="1" style="margin:10px 0 18px 18px;">
      {''.join(step_lines)}
    </ol>
  </div>

  <div data-sec="3">
    <p style="margin:0 0 14px 0;"><strong>마무리는 내 입맛 기준 하나만 남겨두면 다음이 편해요</strong></p>
    <p style="margin:0 0 18px 0;">
      저는 요리에서 가장 좋은 변화가  다음에 다시 할 수 있을 만큼 부담이 줄어드는 거라고 생각해요<br/><br/>
      그래서 한 번 만들 때도  완벽보다  내가 다음에 또 할 수 있는 흐름을 남겨두는 편이에요
    </p>
    <p style="margin:0 0 18px 0;">
      다음에 하실 때는  간을 어디서 잡는지  딱 그 지점만 메모해두면  그 다음부터 훨씬 편해집니다<br/><br/>
      혹시 {title_ko} 하실 때  가장 헷갈리는 포인트가 뭐였는지  친구처럼 한 번 얘기해주시면 저도 참고할게요
    </p>
    <p style="opacity:.65;font-size:13px;margin:0 0 10px 0;">
      출처: {f"<a href='{src}' target='_blank' rel='nofollow noopener'>{src}</a>" if src else "-"}
      {" | " + f"<a href='{yt}' target='_blank' rel='nofollow noopener'>YouTube</a>" if yt else ""}
    </p>
  </div>
</div>
""".strip()

    # 제목은 Seed 우선
    title_final = (title_seed or f"{title_ko} 레시피").replace(".", "").replace("。", "")
    return title_final, body


# -----------------------------
# Recipe pick
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

    raise RuntimeError("레시피를 가져오지 못했습니다(중복 회피/필터/시도 횟수 초과).")


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

    # 제목 키워드(기본은 레시피명 번역 시도)
    recipe_title_ko_guess = free_translate_text(cfg.free_tr, recipe_title_en, debug=cfg.run.debug) or recipe_title_en
    kw_list = [k.strip() for k in (cfg.naver.keywords_csv or "").split(",") if k.strip()]
    title_keyword = kw_list[0] if kw_list else recipe_title_ko_guess

    title_seed = make_hook_title(title_keyword, random_level=cfg.naver.random_level)
    ingredients_html = build_ingredients_html(cfg, recipe)

    # 썸네일 업로드 (기본: TheMealDB thumb)
    media_id: Optional[int] = None
    media_url: str = ""
    thumb_url = (recipe.get("thumb") or "").strip()

    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"recipe-{date_str}-{slot}{suffix}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    # OpenAI: 네이버 홈피드형 글 생성 (글자수/형식 미달 시 재생성)
    title_ko = ""
    body_html = ""
    used_fallback = False

    for attempt in range(max(1, cfg.run.naver_text_tries)):
        try:
            client = OpenAI(api_key=cfg.openai.api_key)
            inst, user_in = _naver_prompt(cfg, recipe, now, title_seed=title_seed, ingredients_html=ingredients_html)

            # attempt 힌트(부족하면 더 늘리게)
            if attempt > 0:
                inst = inst + f"\n\n[추가 지시]\n이전 출력이 길이 또는 형식 조건을 못 맞췄다  섹션별 글자수를 더 늘리고  <p> 를 더 자주 써서 여백을 더 줘라"

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
                raise RuntimeError("OpenAI 응답이 너무 짧습니다(제목/본문 분리 실패).")

            title_ko = lines[0].replace(".", "").replace("。", "").strip()
            body_html = "\n".join(lines[1:]).strip()

            # 가독성 래퍼 강제
            body_html = wrap_readable(body_html)

            # 재료 목록이 혹시 빠졌으면 강제 삽입(섹션1 앞쪽에 끼워넣기)
            if "data-ingredients='1'" not in body_html and "data-ingredients=\"1\"" not in body_html:
                body_html = body_html.replace("</div>", ingredients_html + "</div>", 1)

            ok, reason = validate_longform(body_html)
            if cfg.run.debug:
                print(f"[TEXT] validate attempt={attempt+1} ok={ok} reason={reason}")
            if ok:
                break
        except Exception as e:
            if _is_insufficient_quota_error(e):
                if cfg.run.debug:
                    print("[WARN] OpenAI quota depleted")
                used_fallback = True
                break
            if cfg.run.debug:
                print("[WARN] OpenAI generation failed:", repr(e))
            # 다음 attempt 재시도
            if attempt == cfg.run.naver_text_tries - 1:
                used_fallback = True

    if used_fallback or not body_html:
        title_ko, body_html = build_korean_simple_with_free_translate(
            cfg,
            recipe,
            now,
            title_seed=title_seed,
            ingredients_html=ingredients_html,
        )

    # 모델 제목이 Seed랑 너무 다르면 Seed로 교체(키워드 선두 보장)
    if title_keyword and (not title_ko.startswith(title_keyword)):
        title_ko = title_seed

    # 본문에 이미지 삽입(업로드 성공하면 WP URL 우선)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            img_block = (
                f"<p style=\"margin:0 0 18px 0;\">"
                f"<img src=\"{img}\" alt=\"{title_ko}\" style=\"max-width:100%;height:auto;border-radius:12px;\"/>"
                f"</p>"
            )
            body_html = body_html.replace("<div data-wrap='1'", img_block + "<div data-wrap='1'", 1)

    title = f"{date_str} {slot_label} 레시피 | {title_ko}".replace("..", ".").strip()

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 ↓")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2500] + ("\n...(truncated)" if len(body_html) > 2500 else ""))
        return

    # 발행/업데이트
    if wp_post_id and not cfg.run.force_new:
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
