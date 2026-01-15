# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naver.py (완전 통합/안정화 + 네이버 복붙 친화 + 덜 AI 같은 문장)
- TheMealDB 레시피 수집 (옵션: Korean area 우선)
- OpenAI로 '사람 글 같은' 한국어 블로그 글 생성(구조는 모바일 친화, 템플릿 고정 최소화)
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정
- SQLite 발행 이력 + 스키마 자동 마이그레이션
- ✅ OpenAI 크레딧 소진(insufficient_quota) 시: 무료번역(LibreTranslate) 폴백

필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY

선택 env:
  - WP_STATUS=publish
  - WP_CATEGORY_IDS="7"
  - WP_TAG_IDS="1,2,3"
  - SQLITE_PATH=data/daily_recipe.sqlite3

  - RUN_SLOT=day/am/pm
  - DRY_RUN=1
  - DEBUG=1

  - OPENAI_MODEL=gpt-5.2
  - STRICT_KOREAN=1
  - FORCE_NEW=0
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - OPENAI_MAX_RETRIES=3

  - PREFER_KOREAN_AREA=1   # 기본 1 (가능하면 Korean area 레시피 우선)
  - DEFAULT_THUMB_URL=https://.../default.jpg  # thumb 없을 때 대체 이미지
  - NAVER_COPY_MODE=1      # 기본 1: 네이버 복붙시 보기 좋은 줄바꿈/구성

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
import openai  # 예외 타입 용도
from openai import OpenAI  # 공식 SDK

KST = timezone(timedelta(hours=9))

THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"
THEMEALDB_LOOKUP = "https://www.themealdb.com/api/json/v1/1/lookup.php?i={id}"
THEMEALDB_FILTER_AREA = "https://www.themealdb.com/api/json/v1/1/filter.php?a={area}"

# -----------------------------
# ENV helpers
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
    prefer_korean_area: bool = True
    default_thumb_url: str = ""
    naver_copy_mode: bool = True

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

    cfg = AppConfig(
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
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            force_new=_env_bool("FORCE_NEW", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 20),
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
            prefer_korean_area=_env_bool("PREFER_KOREAN_AREA", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            naver_copy_mode=_env_bool("NAVER_COPY_MODE", True),
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(
            api_key=_env("OPENAI_API_KEY", ""),
            model=_env("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2",
        ),
        free_tr=FreeTranslateConfig(
            url=_env("FREE_TRANSLATE_URL", "https://libretranslate.de/translate"),
            api_key=_env("FREE_TRANSLATE_API_KEY", ""),
            source=_env("FREE_TRANSLATE_SOURCE", "en"),
            target=_env("FREE_TRANSLATE_TARGET", "ko"),
        ),
    )
    return cfg

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
    print("[CFG] PREFER_KOREAN_AREA:", cfg.run.prefer_korean_area)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.run.default_thumb_url else "EMPTY")
    print("[CFG] NAVER_COPY_MODE:", cfg.run.naver_copy_mode)
    print("[CFG] OPENAI_API_KEY:", ok(cfg.openai.api_key))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model)
    print("[CFG] OPENAI_MAX_RETRIES:", cfg.run.openai_max_retries)
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

def fetch_korean_area_candidate_ids() -> List[str]:
    url = THEMEALDB_FILTER_AREA.format(area="Korean")
    r = requests.get(url, timeout=25)
    if r.status_code != 200:
        return []
    j = r.json()
    meals = j.get("meals") or []
    out: List[str] = []
    for it in meals:
        rid = str(it.get("idMeal") or "").strip()
        if rid:
            out.append(rid)
    return out

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
                # JSON 모드(파싱 안정화)
                response_format={"type": "json_object"},
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

def _json_load_safe(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass
    # 혹시 앞뒤가 섞였을 때 첫 JSON 블록만 추출
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# -----------------------------
# Free translate (LibreTranslate)
# -----------------------------
def free_translate_text(cfg: FreeTranslateConfig, text: str, debug: bool = False) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    payload = {"q": text, "source": cfg.source, "target": cfg.target, "format": "text"}
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
    title_en = recipe.get("title", "Daily Recipe")
    title_ko = free_translate_text(free_cfg, title_en, debug=debug)

    ing_lines = []
    for it in recipe.get("ingredients", []):
        name_en = (it.get("name") or "").strip()
        mea = (it.get("measure") or "").strip()
        name_ko = free_translate_text(free_cfg, name_en, debug=debug) if name_en else ""
        label = f"{name_ko} ({name_en})" if name_ko and name_ko != name_en else name_en
        ing_lines.append(f"<li>{label}" + (f" <span style='opacity:.75'>— {mea}</span>" if mea else "") + "</li>")

    steps_en = split_steps(recipe.get("instructions", ""))
    step_lines = []
    for s in steps_en:
        ko = free_translate_text(free_cfg, s, debug=debug)
        step_lines.append(f"<li>{ko}</li>")

    body = f"""
<p style="opacity:.85;font-size:13px;">기준시각: {now.astimezone(KST).strftime("%Y-%m-%d %H:%M")}</p>
<p>오늘은 <b>{title_ko}</b>를 간단히 정리해볼게요. (번역 기반 요약)</p>
<h2>재료</h2>
<ul>{''.join(ing_lines) if ing_lines else "<li>-</li>"}</ul>
<h2>만드는 법</h2>
<ol>{''.join(step_lines) if step_lines else "<li>-</li>"}</ol>
"""
    title_final = f"실패 없이 따라하는 {title_ko} 레시피"
    return title_final, body

# -----------------------------
# OpenAI: 덜 AI 같은 글(모바일/네이버 복붙 친화)
# -----------------------------
def generate_korean_blog_human(
    openai_cfg: OpenAIConfig,
    recipe: Dict[str, Any],
    strict_korean: bool,
    max_retries: int,
    naver_copy_mode: bool,
    debug: bool = False
) -> Tuple[str, str]:
    client = OpenAI(api_key=openai_cfg.api_key)

    steps_en = split_steps(recipe.get("instructions", ""))
    payload = {
        "title_en": recipe.get("title", ""),
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients": recipe.get("ingredients", []),
        "steps_en": steps_en,
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
        "naver_copy_mode": bool(naver_copy_mode),
    }

    # ✅ 포인트: '고정 템플릿' 느낌을 줄이기 위해, 섹션명/문장 리듬을 매번 조금씩 다르게 유도
    instructions = (
        "너는 한국어로 글을 쓰는 요리 블로거 편집자다. 반드시 한국어로만 작성한다.\n"
        "\n"
        "[절대 규칙]\n"
        "1) 제공된 ingredients/steps_en의 의미를 바꾸지 마. 재료/계량/단계를 추가하거나 삭제하지 마.\n"
        "2) 원문에 없는 시간/온도/새 도구를 단정하지 마. 필요하면 '상태를 보며'처럼 표현.\n"
        "3) '내가 해봤다/어제 만들었다' 같은 실제 경험 주장 금지. 대신 일반형(집에서 보통…)으로 자연스럽게.\n"
        "\n"
        "[홈피드/모바일에서 잘 읽히는 톤]\n"
        "- 첫 문단은 2~3문장: 실수 포인트 1개 + 이 글에서 해결 + 한 줄 요약\n"
        "- 문장 길이 섞기(짧게/길게), 너무 교과서처럼 딱딱한 말투 금지.\n"
        "- 과한 광고/과장/이모지 남발 금지. 이모지는 0~1개 정도만.\n"
        "- 섹션 이름은 매번 조금씩 바꿔도 됨(예: '맛 포인트', '준비물', '이렇게 하면 편해요' 등)\n"
        "\n"
        "[출력은 JSON 하나]\n"
        "다음 키를 정확히 포함해라:\n"
        "{\n"
        '  "title": string,  # 26~46자, 너무 자극적 금지. 메인 키워드(요리명) 포함\n'
        '  "hook": string,   # 2~3문장\n'
        '  "bullets": [string],  # 3~6개, 저장하고 싶은 팁 느낌\n'
        '  "ingredients_ko": [string],  # 재료 줄, 필요하면 (영문) 병기\n'
        '  "steps_ko": [string],        # steps_en 개수와 정확히 동일한 개수\n'
        '  "step_tips": [string],       # steps_ko와 개수 동일, 각 단계별 한 줄 팁\n'
        '  "checklist": [string],       # 4~6개\n'
        '  "closing": string,           # 2~3문장, 저장/공유 유도는 부드럽게 1문장 포함\n'
        '  "hashtags": [string]         # 6~10개, 관련 해시태그(요리명/레시피/집밥 등)\n'
        "}\n"
    )

    resp = _openai_call_with_retry(
        client=client,
        model=openai_cfg.model,
        instructions=instructions,
        input_text=json.dumps(payload, ensure_ascii=False),
        max_retries=max_retries,
        debug=debug,
    )

    raw = (resp.output_text or "").strip()
    if debug:
        print("[OPENAI] raw len:", len(raw))

    obj = _json_load_safe(raw)
    if not obj:
        raise RuntimeError("OpenAI JSON 파싱 실패")

    title = str(obj.get("title") or "").strip()
    hook = str(obj.get("hook") or "").strip()
    bullets = obj.get("bullets") or []
    ingredients_ko = obj.get("ingredients_ko") or []
    steps_ko = obj.get("steps_ko") or []
    step_tips = obj.get("step_tips") or []
    checklist = obj.get("checklist") or []
    closing = str(obj.get("closing") or "").strip()
    hashtags = obj.get("hashtags") or []

    if strict_korean:
        if not re.search(r"[가-힣]", title):
            raise RuntimeError("한글 제목 검증 실패")
        # 본문 핵심 중 하나라도 한글 없으면 실패
        if not (re.search(r"[가-힣]", hook) and (ingredients_ko or steps_ko)):
            raise RuntimeError("한글 본문 검증 실패")

    if len(steps_ko) != len(payload["steps_en"]):
        raise RuntimeError(f"steps_ko 개수 불일치: {len(steps_ko)} vs {len(payload['steps_en'])}")
    if len(step_tips) != len(steps_ko):
        # tip은 부족하면 빈값으로 채움(발행은 되게)
        step_tips = (step_tips + [""] * len(steps_ko))[:len(steps_ko)]

    # HTML 렌더링(네이버 복붙 친화: 복잡한 스타일 최소)
    def _li(items: List[str]) -> str:
        return "".join([f"<li>{_escape_html(x)}</li>" for x in items if str(x).strip()])

    def _escape_html(s: str) -> str:
        s = str(s or "")
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        s = s.replace('"', "&quot;")
        return s

    # 단계 + 단계별 한줄팁
    step_html = ""
    for s, tip in zip(steps_ko, step_tips):
        s2 = _escape_html(s)
        tip2 = _escape_html(tip)
        tip_block = f"<div style='opacity:.85;font-size:13px;margin-top:6px;'>포인트: {tip2}</div>" if tip2 else ""
        step_html += f"<li>{s2}{tip_block}</li>"

    # 해시태그는 네이버에서 잘 먹는 편이라(관련된 것만) 맨 아래에
    tags = [str(x).strip() for x in hashtags if str(x).strip()]
    tags = [t if t.startswith("#") else f"#{t.replace(' ', '')}" for t in tags]
    tag_line = " ".join(tags[:12])

    # naver_copy_mode면 줄바꿈이 더 보이도록 문단을 잘게 쪼갬
    hook_html = f"<p>{_escape_html(hook).replace('\n', '<br/>')}</p>" if hook else ""
    closing_html = f"<p>{_escape_html(closing).replace('\n', '<br/>')}</p>" if closing else ""

    body = (
        hook_html
        + f"<h2>딱 이것만 기억하면 쉬워요</h2><ul>{_li(bullets)}</ul>"
        + f"<h2>재료</h2><ul>{_li(ingredients_ko) or '<li>-</li>'}</ul>"
        + f"<h2>만드는 법</h2><ol>{step_html or '<li>-</li>'}</ol>"
        + f"<h2>실패 줄이는 체크</h2><ul>{_li(checklist)}</ul>"
        + f"<h2>마무리</h2>{closing_html}"
        + (f"<p>{_escape_html(tag_line)}</p>" if tag_line else "")
    )

    # 너무 빽빽하면 체류가 떨어지는 경우가 많아서, 문단 간 공백을 조금 추가
    if naver_copy_mode:
        body = body.replace("</h2><ul>", "</h2>\n<ul>").replace("</ul><h2>", "</ul>\n<h2>")
        body = body.replace("</h2><ol>", "</h2>\n<ol>").replace("</ol><h2>", "</ol>\n<h2>")

    return title, body

# -----------------------------
# Main flow
# -----------------------------
def pick_recipe(cfg: AppConfig, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # 이미 오늘 올린 글이 있고 force_new=0이면 같은 레시피 유지
    if existing and existing.get("recipe_id") and not cfg.run.force_new:
        return fetch_recipe_by_id(str(existing["recipe_id"]))

    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))

    # 1) 가능하면 Korean area 후보에서 먼저 뽑기
    if cfg.run.prefer_korean_area:
        try:
            ids = fetch_korean_area_candidate_ids()
            random.shuffle(ids)
            for rid in ids[:200]:
                if rid in recent_ids:
                    continue
                r = fetch_recipe_by_id(rid)
                if r.get("id") and split_steps(r.get("instructions", "")):
                    return r
        except Exception:
            pass

    # 2) 랜덤(중복 회피)
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

    # 썸네일(없으면 DEFAULT_THUMB_URL로 대체)
    thumb_url = (recipe.get("thumb") or "").strip() or (cfg.run.default_thumb_url or "").strip()

    media_id: Optional[int] = None
    media_url: str = ""
    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"recipe-{date_str}-{slot}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    # ✅ OpenAI로 '덜 AI 같은' 글 생성 + 폴백
    try:
        title_core, body_html = generate_korean_blog_human(
            cfg.openai,
            recipe,
            strict_korean=cfg.run.strict_korean,
            max_retries=cfg.run.openai_max_retries,
            naver_copy_mode=cfg.run.naver_copy_mode,
            debug=cfg.run.debug,
        )
    except Exception as e:
        if _is_insufficient_quota_error(e):
            print("[WARN] OpenAI quota depleted → fallback to FREE translate")
            title_core, body_html = build_korean_simple_with_free_translate(cfg.free_tr, recipe, now, debug=cfg.run.debug)
        else:
            if cfg.run.strict_korean:
                raise
            print("[WARN] OpenAI failed (non-quota) → fallback to FREE translate:", repr(e))
            title_core, body_html = build_korean_simple_with_free_translate(cfg.free_tr, recipe, now, debug=cfg.run.debug)

    # 본문 상단 이미지(업로드 성공하면 WP URL 우선)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            body_html = f'<p><img src="{img}" alt="{title_core}" style="max-width:100%;height:auto;border-radius:12px;"></p>\n' + body_html

    # 최종 제목: 날짜/슬롯 + 핵심제목 (너무 티나게 과장하지 않게)
    title = f"{date_str} {slot_label} | {title_core}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 ↓")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2500] + ("\n...(truncated)" if len(body_html) > 2500 else ""))
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
