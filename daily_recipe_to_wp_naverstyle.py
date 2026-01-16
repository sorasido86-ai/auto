# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle.py (통합 완성본)
- TheMealDB 랜덤 레시피 수집
- (번역 누락 방지) 제목/재료/스텝을 먼저 한국어로 일괄 번역 (OpenAI -> 실패/영어잔존 시 LibreTranslate 폴백)
- (홈피드형) 도입 200~300자 + 굵은 소제목 3개(각 1500자 이상) + 총 2300자 이상 목표
- (사람 수다 존댓말) 경험담/생각/가치관 강조 + 마침표 제거(후처리)
- (가독성) 여백/행간 넉넉하게 HTML 구성
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정
- SQLite 발행 이력 + 스키마 자동 마이그레이션
- ✅ OpenAI 크레딧/오류 시: LibreTranslate + 템플릿 글로 폴백

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
  - RUN_SLOT=day|am|pm
  - FORCE_NEW=0|1
  - DRY_RUN=0|1
  - DEBUG=0|1
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - OPENAI_MODEL=gpt-4.1-mini (또는 사용 가능 모델)
  - OPENAI_MAX_RETRIES=3

네이버 스타일/랜덤화:
  - NAVER_RANDOM_LEVEL=0~3 (기본 2)         # 제목/표현 변주 폭
  - NAVER_EXPERIENCE_LEVEL=0~3 (기본 2)     # 경험담/실수/가치관 강도
  - NAVER_KEYWORDS="키워드1,키워드2,..."    # 제목/태그에 자연스럽게 사용
  - PREFER_AREAS="Korean,Japanese,Chinese"  # TheMealDB area 필터 (선택)

무료번역 폴백(LibreTranslate):
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
import html as _html

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

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:800]}")
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
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:800]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "recipe.jpg") -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=35)
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
        raise RuntimeError(f"WP media upload failed: {up.status_code} body={up.text[:800]}")

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
    parts = [p for p in parts if p]
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
        r = requests.post(cfg.url, json=payload, timeout=25)
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
# Text cleanup helpers
# -----------------------------
def _strip_periods(s: str) -> str:
    # 마침표 계열 제거
    s = (s or "")
    s = s.replace(".", "").replace("。", "").replace("．", "")
    return s


def _strip_bullets_and_numbers(s: str) -> str:
    s = (s or "")
    # 흔한 불릿/기호 제거
    s = re.sub(r"[•●◦∙·]", "", s)
    s = re.sub(r"^\s*[-–—*]\s*", "", s, flags=re.M)
    # 1단계 2단계 Step 1 같은 느낌 제거
    s = re.sub(r"^\s*(\d+)\s*(단계|step)\s*[:：]?\s*", "", s, flags=re.I | re.M)
    return s.strip()


def _contains_english(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s or ""))


def _safe_json_load(text: str) -> Optional[dict]:
    if not text:
        return None
    t = text.strip()
    # code fence 제거
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # 첫 { 부터 마지막 } 까지 슬라이스 시도
    if "{" in t and "}" in t:
        t2 = t[t.find("{"):t.rfind("}") + 1]
    else:
        t2 = t
    try:
        return json.loads(t2)
    except Exception:
        return None


# -----------------------------
# Batch translate (OpenAI -> Libre fallback)
# -----------------------------
def openai_translate_batch(cfg: AppConfig, texts: List[str], debug: bool = False) -> List[str]:
    # 빈 값은 그대로
    idx_map = []
    src = []
    for i, s in enumerate(texts):
        s2 = (s or "").strip()
        if not s2:
            continue
        idx_map.append(i)
        src.append(s2)

    out = list(texts)
    if not src:
        return out

    client = OpenAI(api_key=cfg.openai.api_key)
    instructions = (
        "너는 전문 번역가다\n"
        "입력은 영어 문장 배열이다\n"
        "출력은 같은 길이의 한국어 배열 JSON만 출력해라\n"
        "추가 설명 금지 코드블럭 금지\n"
        "고유명사는 적절히 음차하거나 흔한 번역이 있으면 그걸 사용\n"
    )
    payload = json.dumps({"texts": src}, ensure_ascii=False)

    try:
        resp = _openai_call_with_retry(
            client=client,
            model=cfg.openai.model,
            instructions=instructions,
            input_text=payload,
            max_retries=cfg.run.openai_max_retries,
            debug=debug,
        )
        txt = (resp.output_text or "").strip()
        # 배열만 뽑기
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        j = None
        # JSON 배열 또는 {"translations":[...]} 둘 다 허용
        try:
            j = json.loads(txt)
        except Exception:
            # 배열 브래킷만 잘라보기
            if "[" in txt and "]" in txt:
                try:
                    j = json.loads(txt[txt.find("["):txt.rfind("]") + 1])
                except Exception:
                    j = None

        if isinstance(j, dict) and isinstance(j.get("translations"), list):
            translated = j["translations"]
        elif isinstance(j, list):
            translated = j
        else:
            translated = None

        if not translated or len(translated) != len(src):
            raise RuntimeError("translate json parse failed")

        for k, original_index in enumerate(idx_map):
            out[original_index] = str(translated[k] or "").strip() or texts[original_index]

    except Exception as e:
        if debug:
            print("[WARN] openai translate batch failed -> libre fallback:", repr(e))
        # 폴백: libretranslate로 개별 번역
        for k, original_index in enumerate(idx_map):
            out[original_index] = free_translate_text(cfg.free_tr, src[k], debug=debug)

    # 영어 잔존하면 libre로 한 번 더
    for i in range(len(out)):
        if out[i] and _contains_english(out[i]):
            out[i] = free_translate_text(cfg.free_tr, out[i], debug=debug)

    return out


# -----------------------------
# Title generator (hook templates)
# -----------------------------
def _pick(pool: List[str]) -> str:
    return random.choice(pool) if pool else ""


def generate_hook_title(keyword: str, random_level: int) -> str:
    kw = (keyword or "").strip() or "오늘 메뉴"

    gain = [
        f"{kw} 이 맛이 나는 이유  한 번만 알면 계속 써먹게 돼요",
        f"{kw} 은근히 쉬운 포인트  이거 잡고 나서 마음이 편해졌어요",
        f"{kw} 딱 한 가지 기준만 잡아도 맛이 안정되는 느낌이었어요",
    ]
    threat = [
        f"{kw} 여기서 한번만 급하면  맛이 바로 흔들리더라고요",
        f"{kw} 이 부분 놓치면  재료가 아깝게 느껴질 수도 있어요",
        f"{kw} 이 타이밍만 어긋나면  왜인지 맛이 붕 뜨는 느낌이에요",
    ]
    curious = [
        f"{kw} 왜 남이 하면 맛있고  내가 하면 애매한지  그 이유가 있더라고요",
        f"{kw} 딱 여기에서 갈리는데  저는 한동안 그걸 몰랐어요",
        f"{kw} 이 한 줄 기준이  결과를 꽤 바꾸는 느낌이었어요",
    ]
    compare = [
        f"{kw} 대충 따라하기 vs 기준 잡고 하기  결과가 이렇게 달라지더라고요",
        f"{kw} 레시피대로 vs 상태 보고  어떤 쪽이 편한지 얘기해볼게요",
        f"{kw} 오래 걸리는 줄 알았는데  순서만 바꾸면 체감이 달라요",
    ]

    buckets = [gain, threat, curious, compare]
    # 랜덤 레벨이 낮으면 gain/curious 위주
    if random_level <= 1:
        buckets = [gain, curious, threat]
    else:
        random.shuffle(buckets)

    title = _pick(_pick(buckets)) if isinstance(buckets[0][0], str) else _pick(gain)
    # 안전하게 마침표 제거
    title = _strip_periods(title)
    title = _strip_bullets_and_numbers(title)
    # 길이 조정(너무 길면 컷)
    if len(title) > 60:
        title = title[:60].strip()
    return title


# -----------------------------
# Closing + tags
# -----------------------------
def closing_random(keyword: str) -> str:
    kw = (keyword or "").strip() or "이 메뉴"
    pool = [
        f"저는 요리가 완벽해야 한다는 생각을 내려놓고 나서부터  오히려 꾸준히 하게 되더라고요  {kw}도 오늘은 기준만 하나 잡아두시면  다음엔 훨씬 편해질 거예요",
        f"{kw}는 한번 감만 잡히면  그 다음부터는 부담이 확 줄어요  다음엔 내 입맛 포인트 하나만 살짝 조절해보셔도 좋겠어요",
        f"요리는 결국 내 컨디션을 배려하는 쪽이 오래 가더라고요  {kw}도 급하지 않게  편한 속도로 해보셔도 충분해요",
        f"오늘 정리한 방식은  메뉴 고민될 때 다시 꺼내 보기 좋아요  {kw}는 특히 이 지점에서 갈리니까  그 부분만 기억해두셔도 도움이 될 거예요",
    ]
    return _strip_periods(_pick(pool))


def tags_html(keywords_csv: str, keyword_main: str, extra: List[str]) -> str:
    kws: List[str] = []
    if keyword_main:
        kws.append(keyword_main.strip())
    for x in (keywords_csv or "").split(","):
        x = x.strip()
        if x and x not in kws:
            kws.append(x)
    for x in extra:
        x = (x or "").strip()
        if x and x not in kws:
            kws.append(x)

    # 너무 많으면 지저분해 보이니 10~12개 제한
    base_add = ["레시피", "집밥", "오늘의요리", "간단요리"]
    for x in base_add:
        if x not in kws:
            kws.append(x)
    kws = [k for k in kws if k][:12]

    if not kws:
        return ""

    line = " ".join([f"#{k.replace(' ', '')}" for k in kws if k])
    return (
        "<div style='margin:22px 0 0 0;opacity:.82;font-size:14px;line-height:1.9;'>"
        f"태그  {line}"
        "</div>"
    )


# -----------------------------
# Naver-style long narrative (OpenAI)
# -----------------------------
def generate_homefeed_narrative(cfg: AppConfig, recipe_ko: Dict[str, Any], keyword_main: str, debug: bool = False) -> Dict[str, Any]:
    random_level = cfg.naver.random_level
    exp_level = cfg.naver.experience_level

    # seed 문장 (너무 포맷 티 안 나게)
    hook_pool = [
        "레시피는 따라 했는데  뭔가 한 끗이 아쉬울 때가 있잖아요",
        "똑같은 재료인데도 결과가 갈리는 날이 있더라고요",
        "급하게 하면 꼭 한번 삐끗하는 구간이 있어요",
        "완벽하게 하려다 오히려 스트레스 받는 날이 있잖아요",
    ]
    value_pool = [
        "요리는 결국 내 생활 리듬에 맞아야 오래 가더라고요",
        "완벽보다 반복 가능한 방식이 제일 남는 것 같아요",
        "저는 실패를 줄이는 기준 하나만 정해두는 편이에요",
        "기준이 잡히면 그 다음부터는 마음이 확 편해져요",
    ]
    hook_seed = _pick(hook_pool)
    value_seed = _pick(value_pool)

    prompt = {
        "keyword_main": keyword_main,
        "recipe_title": recipe_ko.get("title_ko", ""),
        "area_ko": recipe_ko.get("area_ko", ""),
        "category_ko": recipe_ko.get("category_ko", ""),
        "ingredients_ko": recipe_ko.get("ingredients_ko", []),  # [{"name_ko","measure"}]
        "steps_ko": recipe_ko.get("steps_ko", []),
        "style": {
            "tone": "친구에게 진심 담아 수다떠는 존댓말",
            "no_periods": True,
            "spacing": "여백과 줄바꿈으로 호흡",
            "intro_len": "200~300자",
            "sections": 3,
            "section_min_chars": 1500,
            "total_min_chars": 2300,
            "random_level": random_level,
            "experience_level": exp_level,
            "hook_seed": hook_seed,
            "value_seed": value_seed,
        },
    }

    instructions = (
        "너는 한국어 블로그 글을 쓰는 사람이다\n"
        "출력은 반드시 JSON 하나만 출력해라  코드블럭 금지\n"
        "\n"
        "[작성 규칙]\n"
        "1) 말투는 친구에게 수다 떠는 존댓말로  과하게 교과서 말투 금지\n"
        "2) 마침표 문자 사용 금지  대신 띄어쓰기와 줄바꿈으로 호흡\n"
        "3) 도입은 200~300자\n"
        "4) 굵은 소제목 3개  각 본문은 최소 1500자 이상\n"
        "5) 경험담  생각  가치관이 보이게 쓰기  너무 완벽하게 정리된 AI 느낌 금지\n"
        "6) 재료나 조리 단계는 바꾸지 말고  글 안에서 새로운 재료나 시간을 단정하지 말기\n"
        "7) 번호 매기기 금지  1단계 2단계 같은 표현 금지\n"
        "\n"
        "[출력 JSON 형식]\n"
        "{\n"
        "  \"intro\": \"...\",\n"
        "  \"sections\": [\n"
        "    {\"h\": \"소제목1\", \"body\": \"...\"},\n"
        "    {\"h\": \"소제목2\", \"body\": \"...\"},\n"
        "    {\"h\": \"소제목3\", \"body\": \"...\"}\n"
        "  ]\n"
        "}\n"
    )

    client = OpenAI(api_key=cfg.openai.api_key)

    # 길이 조건이 빡세서 최대 2회까지 재시도
    for attempt in range(3):
        resp = _openai_call_with_retry(
            client=client,
            model=cfg.openai.model,
            instructions=instructions,
            input_text=json.dumps(prompt, ensure_ascii=False),
            max_retries=cfg.run.openai_max_retries,
            debug=debug,
        )
        data = _safe_json_load((resp.output_text or "").strip())
        if not data or not isinstance(data, dict):
            continue

        intro = _strip_bullets_and_numbers(_strip_periods(str(data.get("intro") or "").strip()))
        sections = data.get("sections") or []
        if not isinstance(sections, list) or len(sections) != 3:
            continue

        ok = True
        total_len = len(intro)
        fixed_sections = []
        for s in sections:
            h = _strip_bullets_and_numbers(_strip_periods(str((s or {}).get("h") or "").strip()))
            body = _strip_bullets_and_numbers(_strip_periods(str((s or {}).get("body") or "").strip()))
            # 길이 체크
            if len(body) < 1500:
                ok = False
            total_len += len(body)
            fixed_sections.append({"h": h, "body": body})

        # intro 길이 대략 체크
        if not (180 <= len(intro) <= 360):
            ok = False

        if total_len < 2300:
            ok = False

        if ok:
            return {"intro": intro, "sections": fixed_sections}

    raise RuntimeError("OpenAI narrative length/format constraints not met")


# -----------------------------
# HTML builder (readability)
# -----------------------------
def _wrap_div(inner: str) -> str:
    style = (
        "font-size:16px;"
        "line-height:1.95;"
        "letter-spacing:-0.2px;"
        "word-break:keep-all;"
        "max-width:760px;"
        "margin:0 auto;"
    )
    return f"<div style='{style}'>{inner}</div>"


def _p(text: str) -> str:
    # 문단 간격 크게
    safe = _html.escape(text).replace("\n", "<br/><br/>")
    return f"<p style='margin:0 0 16px 0;line-height:1.95;'>{safe}</p>"


def _h2(title: str) -> str:
    t = _html.escape(title)
    return f"<h2 style='margin:26px 0 12px 0;'><b>{t}</b></h2>"


def _ingredients_block(ingredients_ko: List[Dict[str, str]]) -> str:
    # 불릿 없이 깔끔한 라인
    lines = []
    for it in ingredients_ko:
        name = _strip_bullets_and_numbers(str(it.get("name_ko") or "").strip())
        mea = str(it.get("measure") or "").strip()
        name = _strip_periods(name)
        name = name if name else str(it.get("name_en") or "").strip()
        if mea:
            line = f"{name}   {mea}"
        else:
            line = f"{name}"
        line = line.strip()
        if not line:
            continue
        lines.append(f"<div style='margin:0 0 8px 0;opacity:.92;'>{_html.escape(line)}</div>")
    if not lines:
        return _p("재료는 본문 기준으로 준비하시면 됩니다")
    return "<div style='margin:0 0 8px 0;'>" + "".join(lines) + "</div>"


def _method_block(steps_ko: List[str]) -> str:
    # 번호/단계 없이 흐름형 문장
    out = []
    for s in steps_ko:
        s2 = _strip_bullets_and_numbers(_strip_periods(str(s or "").strip()))
        if not s2:
            continue
        # 혹시 번역 후에도 Step 같은 거 남으면 제거
        s2 = re.sub(r"^\s*step\s*\d+\s*", "", s2, flags=re.I)
        if not s2:
            continue
        out.append(f"<p style='margin:0 0 14px 0;line-height:1.95;'>{_html.escape(s2)}</p>")
    if not out:
        return _p("만드는 법은 제공된 조리 순서대로  상태를 보면서 진행하시면 됩니다")
    return "<div>" + "".join(out) + "</div>"


# -----------------------------
# Fallback narrative (no OpenAI)
# -----------------------------
def fallback_narrative(keyword: str, exp_level: int) -> Dict[str, Any]:
    kw = keyword or "오늘 메뉴"
    intro = (
        f"{kw} 같은 건요  정보만 보면 쉬워 보이는데  막상 하면 어딘가 애매해질 때가 있잖아요\n\n"
        f"저는 그런 날이 반복되면  재료보다 내 마음이 먼저 지치더라고요\n\n"
        f"그래서 오늘은 완벽하게 하려는 쪽이 아니라  실패 확률을 줄이는 기준 쪽으로 얘기해볼게요"
    )
    intro = _strip_periods(intro)

    # 1500자 채우기 위해 반복 템플릿을 길게 구성
    base = []
    for _ in range(40 if exp_level >= 2 else 28):
        base.append(
            "이상하게도 급해지는 구간이 있더라고요  그때는 손이 빨라지는데  판단이 느려져요\n\n"
            "그래서 저는 기준을 하나만 정해두는 편이에요  향이 올라오는지  수분이 너무 빨리 날아가는지  식감이 어느 정도인지\n\n"
            "완벽하게 외우는 것보다  상태를 보는 쪽이 마음이 편해요\n\n"
        )
    body_long = _strip_periods("".join(base))

    sections = [
        {"h": "오늘은 왜 이 메뉴가 끌렸는지", "body": body_long},
        {"h": "실패가 나는 지점에서 마음을 잡는 방법", "body": body_long},
        {"h": "요리를 오래 가져가기 위한 제 기준", "body": body_long},
    ]
    return {"intro": intro, "sections": sections}


# -----------------------------
# Main flow
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

    # main keyword: NAVER_KEYWORDS 첫 번째 우선, 없으면 번역된 레시피명으로
    kw_list = [k.strip() for k in (cfg.naver.keywords_csv or "").split(",") if k.strip()]
    keyword_main = kw_list[0] if kw_list else ""

    # --- 번역: 제목/카테고리/지역/재료/스텝을 먼저 한국어로 확보 ---
    steps_en = split_steps(recipe.get("instructions", ""))
    ing_en = recipe.get("ingredients", [])

    # 번역 대상 모으기
    translate_targets = []
    translate_targets.append(recipe.get("title", ""))
    translate_targets.append(recipe.get("category", ""))
    translate_targets.append(recipe.get("area", ""))
    for it in ing_en:
        translate_targets.append(str(it.get("name") or ""))
    for st in steps_en:
        translate_targets.append(st)

    translated = openai_translate_batch(cfg, translate_targets, debug=cfg.run.debug)

    title_ko = translated[0].strip()
    category_ko = translated[1].strip()
    area_ko = translated[2].strip()

    # ingredients
    ingredients_ko: List[Dict[str, str]] = []
    base_i = 3
    for idx, it in enumerate(ing_en):
        name_en = str(it.get("name") or "").strip()
        measure = str(it.get("measure") or "").strip()
        name_ko = str(translated[base_i + idx] or "").strip()
        name_ko = _strip_bullets_and_numbers(_strip_periods(name_ko))
        if not name_ko:
            name_ko = name_en
        # 영어 남으면 libre 한번 더
        if _contains_english(name_ko):
            name_ko = free_translate_text(cfg.free_tr, name_en, debug=cfg.run.debug).strip() or name_ko
            name_ko = _strip_bullets_and_numbers(_strip_periods(name_ko))
        ingredients_ko.append({"name_en": name_en, "name_ko": name_ko, "measure": measure})

    # steps
    steps_ko: List[str] = []
    base_s = base_i + len(ing_en)
    for j, st_en in enumerate(steps_en):
        st_ko = str(translated[base_s + j] or "").strip()
        st_ko = _strip_bullets_and_numbers(_strip_periods(st_ko))
        if not st_ko:
            st_ko = st_en
        if _contains_english(st_ko):
            st_ko = free_translate_text(cfg.free_tr, st_en, debug=cfg.run.debug).strip() or st_ko
            st_ko = _strip_bullets_and_numbers(_strip_periods(st_ko))
        steps_ko.append(st_ko)

    # keyword_main이 없으면 title_ko로 세팅
    if not keyword_main:
        keyword_main = _strip_bullets_and_numbers(_strip_periods(title_ko)) or "오늘 메뉴"

    recipe_ko = {
        "title_ko": _strip_bullets_and_numbers(_strip_periods(title_ko)),
        "category_ko": _strip_bullets_and_numbers(_strip_periods(category_ko)),
        "area_ko": _strip_bullets_and_numbers(_strip_periods(area_ko)),
        "ingredients_ko": ingredients_ko,
        "steps_ko": steps_ko,
    }

    # 썸네일 업로드 (TheMealDB 이미지 사용)
    media_id: Optional[int] = None
    media_url: str = ""
    thumb_url = (recipe.get("thumb") or "").strip()

    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(
                cfg.wp,
                thumb_url,
                filename_hint=f"recipe-{date_str}-{slot}{suffix}.jpg",
            )
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    # --- 본문 생성 (OpenAI) ---
    try:
        narrative = generate_homefeed_narrative(cfg, recipe_ko, keyword_main, debug=cfg.run.debug)
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] narrative openai failed -> fallback:", repr(e))
        narrative = fallback_narrative(keyword_main, cfg.naver.experience_level)

    intro = narrative.get("intro", "").strip()
    sections = narrative.get("sections", [])

    # --- HTML 조립 (순서 고정: 이미지 -> 도입 -> 소제목3개 -> 재료 -> 방법 -> 마무리 -> 태그 -> 크레딧) ---
    blocks: List[str] = []

    # 이미지(가독성 여백)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            blocks.append(
                "<p style='margin:0 0 18px 0;'>"
                f"<img src='{_html.escape(img)}' alt='{_html.escape(keyword_main)}' "
                "style='max-width:100%;height:auto;border-radius:16px;display:block;'/>"
                "</p>"
            )

    # 도입
    blocks.append(_p(_strip_bullets_and_numbers(_strip_periods(intro))))

    # 소제목 3개
    for s in sections[:3]:
        h = _strip_bullets_and_numbers(_strip_periods(str(s.get("h") or "").strip()))
        body = _strip_bullets_and_numbers(_strip_periods(str(s.get("body") or "").strip()))
        if h:
            blocks.append(_h2(h))
        if body:
            blocks.append(_p(body))

    # 재료
    blocks.append(_h2("재료"))
    blocks.append(_ingredients_block(ingredients_ko))

    # 만드는 법
    blocks.append(_h2("만드는 법"))
    blocks.append(_method_block(steps_ko))

    # 마무리
    blocks.append(_h2("마무리"))
    blocks.append(_p(closing_random(keyword_main)))

    # 태그 (extra: area/category + 재료 일부)
    extra_tags = []
    if recipe_ko.get("area_ko"):
        extra_tags.append(recipe_ko["area_ko"])
    if recipe_ko.get("category_ko"):
        extra_tags.append(recipe_ko["category_ko"])
    # 재료 2개 정도만
    for it in ingredients_ko[:2]:
        n = str(it.get("name_ko") or "").strip()
        if n:
            extra_tags.append(n)
    blocks.append(tags_html(cfg.naver.keywords_csv, keyword_main, extra_tags))

    # 크레딧 라인(티 덜나게 아주 짧게)
    src = (recipe.get("source") or "").strip()
    yt = (recipe.get("youtube") or "").strip()
    credit_parts = []
    if src:
        credit_parts.append(f"<a href='{_html.escape(src)}' target='_blank' rel='nofollow noopener'>출처</a>")
    if yt:
        credit_parts.append(f"<a href='{_html.escape(yt)}' target='_blank' rel='nofollow noopener'>영상</a>")
    credit = "  ".join(credit_parts) if credit_parts else "참고  공개 레시피 기반으로 정리"
    blocks.append(f"<div style='margin:18px 0 0 0;opacity:.65;font-size:13px;line-height:1.8;'>{credit}</div>")

    body_html = _wrap_div("".join(blocks))

    # 제목: 날짜/슬롯 + 후킹 타이틀(키워드)
    hook_title = generate_hook_title(keyword_main, cfg.naver.random_level)
    title = _strip_bullets_and_numbers(_strip_periods(f"{date_str} {slot_label}  {hook_title}"))

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2000] + ("\n...(truncated)" if len(body_html) > 2000 else ""))
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
