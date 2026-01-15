# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle.py (통합 완성본: 홈피드 가독성 강화 + 재료목록 강제 + 제목 템플릿 랜덤 + 번역 보정)
- TheMealDB 랜덤 레시피 수집
- OpenAI로 한국어 '사람 글' 톤(홈피드형 여백/호흡/존댓말 수다) 생성
- 재료 목록/조리 단계는 파이썬이 강제로 HTML 삽입(누락 방지)
- ✅ 제목/재료에 남는 영어를 LibreTranslate로 자동 번역 보정
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정
- SQLite 발행 이력 + 스키마 자동 마이그레이션
- OpenAI 크레딧 소진/실패 시: 무료번역(LibreTranslate) 폴백

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
  - OPENAI_MODEL=... (기본 gpt-4.1-mini)
  - NAVER_TITLE_KEYWORD="김치찌개"  (없으면 NAVER_KEYWORDS 첫 키워드 사용, 그것도 없으면 레시피명 기반)
  - NAVER_TITLE_MODE=random|benefit|threat|curiosity|compare (기본 random)
  - NAVER_KEYWORDS="키워드1,키워드2" (선택)

LibreTranslate(폴백) env:
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
import html as _html
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
    keywords_csv: str = ""
    title_keyword: str = ""
    title_mode: str = "random"  # random|benefit|threat|curiosity|compare


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

    naver_keywords = _env("NAVER_KEYWORDS", "")
    title_keyword = _env("NAVER_TITLE_KEYWORD", "")
    title_mode = (_env("NAVER_TITLE_MODE", "random") or "random").strip().lower()
    if title_mode not in ("random", "benefit", "threat", "curiosity", "compare"):
        title_mode = "random"

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
            keywords_csv=naver_keywords,
            title_keyword=title_keyword,
            title_mode=title_mode,
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
    print("[CFG] NAVER_TITLE_KEYWORD:", cfg.naver.title_keyword or "(auto)")
    print("[CFG] NAVER_TITLE_MODE:", cfg.naver.title_mode)
    print("[CFG] NAVER_KEYWORDS:", cfg.naver.keywords_csv or "(empty)")
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
            return client.responses.create(model=model, instructions=instructions, input=input_text)
        except openai.RateLimitError as e:
            if _is_insufficient_quota_error(e):
                raise
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())
        except openai.APIError:
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())


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


def free_translate_bulk(cfg: FreeTranslateConfig, texts: List[str], debug: bool = False) -> List[str]:
    """
    여러 문장을 한 번에 번역  줄수 제한/속도 고려
    실패하거나 분리가 깨지면 개별 번역으로 폴백
    """
    texts = [t.strip() for t in (texts or [])]
    if not texts:
        return []
    token = "|||__SPLIT__|||"
    joined = f"\n{token}\n".join(texts)
    out = free_translate_text(cfg, joined, debug=debug)
    parts = [p.strip() for p in (out or "").split(token)]
    if len(parts) != len(texts):
        # 폴백: 개별
        if debug:
            print("[FREE_TR] bulk split mismatch → fallback individual")
        return [free_translate_text(cfg, t, debug=debug) for t in texts]
    return parts


# -----------------------------
# Translation heuristics
# -----------------------------
_RE_HAS_ASCII = re.compile(r"[A-Za-z]")
_RE_HAS_KO = re.compile(r"[가-힣]")

def looks_english(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    return bool(_RE_HAS_ASCII.search(s)) and not bool(_RE_HAS_KO.search(s))


def ensure_korean_phrase(cfg: AppConfig, s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # 영어가 섞였으면 번역 한 번 더 시도
    if looks_english(s):
        tr = free_translate_text(cfg.free_tr, s, debug=cfg.run.debug).strip()
        return tr or s
    return s


# -----------------------------
# Title templates (random)
# -----------------------------
def _first_keyword_from_csv(csv: str) -> str:
    for x in (csv or "").split(","):
        x = x.strip()
        if x:
            return x
    return ""


def make_title(keyword: str, mode: str = "random") -> str:
    keyword = (keyword or "").strip()
    if not keyword:
        keyword = "오늘 레시피"

    benefit = [
        f"{keyword} 이번엔 왜 실패가 줄었는지 한 번만 짚어볼게요",
        f"{keyword} 이 포인트만 잡으면 맛이 안정되는 이유",
        f"{keyword} 손 많이 갈 것 같아도 의외로 편해지는 순서가 있어요",
    ]
    threat = [
        f"{keyword} 여기서 급해지면 맛이 붕 뜹니다 저도 이 지점이 제일 불안해요",
        f"{keyword} 이거 한 번 놓치면 식감이 확 무너질 수 있어요",
        f"{keyword} 같은 재료인데 결과가 갈리는 건 보통 이 한 줄 차이예요",
    ]
    curiosity = [
        f"{keyword} 다들 어디서 망하는지 궁금하셨죠 그 지점만 정리해요",
        f"{keyword} 왜 어떤 날은 맛있고 어떤 날은 애매한지 그 이유만요",
        f"{keyword} 대충 해도 괜찮아지는 기준이 진짜 따로 있어요",
    ]
    compare = [
        f"{keyword} 따라하기보다 상태를 보는 쪽이 편하더라고요",
        f"{keyword} 정량파 vs 상태파 저는 이제 이쪽으로 기울었어요",
        f"{keyword} 레시피대로 vs 내 기준대로 뭐가 더 안 흔들릴까요",
    ]

    pools = {"benefit": benefit, "threat": threat, "curiosity": curiosity, "compare": compare}
    if mode == "random":
        mode = random.choices(["curiosity", "benefit", "threat", "compare"], weights=[35, 30, 20, 15], k=1)[0]
    return random.choice(pools.get(mode, curiosity))


# -----------------------------
# HTML builders (readability + forced lists + translation)
# -----------------------------
def _ingredients_html(cfg: AppConfig, recipe: Dict[str, Any]) -> str:
    items = recipe.get("ingredients", []) or []
    names_en: List[str] = []
    needs_idx: List[int] = []

    for i, it in enumerate(items):
        name = (it.get("name") or "").strip()
        if name and looks_english(name):
            needs_idx.append(i)
            names_en.append(name)

    translated_map: Dict[int, str] = {}
    if names_en:
        trans = free_translate_bulk(cfg.free_tr, names_en, debug=cfg.run.debug)
        for j, idx in enumerate(needs_idx):
            ko = (trans[j] or "").strip()
            translated_map[idx] = ko

    lis = []
    for i, it in enumerate(items):
        name = (it.get("name") or "").strip()
        mea = (it.get("measure") or "").strip()
        if not name:
            continue

        ko = translated_map.get(i, "").strip()
        if ko and ko.lower() != name.lower():
            label = f"{_html.escape(ko)} <span style='opacity:.65'>({_html.escape(name)})</span>"
        else:
            label = _html.escape(name)

        if mea:
            label += f" <span style='opacity:.78'>({_html.escape(mea)})</span>"

        lis.append(f"<li style='margin:0 0 10px 0;'>{label}</li>")

    if not lis:
        lis = ["<li style='margin:0 0 10px 0;'>재료 정보가 비어있어요</li>"]

    return "<ul style='margin:0;padding-left:18px;'>" + "".join(lis) + "</ul>"


def _steps_html(recipe: Dict[str, Any]) -> str:
    steps = split_steps(recipe.get("instructions", ""))
    lis = []
    for s in steps:
        s = (s or "").strip()
        if not s:
            continue
        lis.append(f"<li style='margin:0 0 12px 0;'>{_html.escape(s)}</li>")
    if not lis:
        lis = ["<li style='margin:0 0 12px 0;'>과정 정보가 비어있어요</li>"]
    return "<ol style='margin:0;padding-left:20px;'>" + "".join(lis) + "</ol>"


def wrap_readable(body_html: str) -> str:
    return "<div style='line-height:1.9;font-size:16px;letter-spacing:-0.2px;'>" + body_html + "</div>"


def plain_len(html_text: str) -> int:
    t = re.sub(r"<[^>]+>", "", html_text or "")
    t = re.sub(r"\s+", " ", t).strip()
    return len(t)


# -----------------------------
# OpenAI prompt for homefeed style
# -----------------------------
def build_homefeed_prompt(cfg: AppConfig, recipe: Dict[str, Any], keyword: str, now: datetime) -> Tuple[str, str]:
    ing_token = "[[INGREDIENTS]]"
    step_token = "[[STEPS]]"

    seeds = [
        "요리는 완벽보다 다음에 또 할 수 있는 방식이 더 오래 가더라고요",
        "레시피는 답이라기보다 실패를 줄이는 힌트 같아요",
        "급하게 하면 꼭 한 군데서 삐끗해서 저는 기준부터 잡는 편이에요",
        "정량보다 상태를 보면 마음이 편해지는 순간이 있어요",
    ]
    seed = random.choice(seeds)

    instructions = f"""
너는 한국어로 글을 쓰는 블로그 운영자다  아래 레시피를 바탕으로 네이버 홈피드에서 읽기 편한 글을 HTML로 작성해라

[중요 사실 규칙]
- 너는 실제로 조리해본 적이 없다  그래서 오늘 해먹어봤는데 같은 체험 단정은 쓰지 마
- 대신 집에서 해보면 보통 여기서 갈린다  상태를 이렇게 보면 된다  같은 현장감 기준으로 쓴다
- 제공된 재료 목록과 단계 외의 재료  계량  단계는 추가 삭제 변경 금지
- 시간 온도 숫자는 원문에 없으면 단정 금지  상태를 보며 같은 표현을 쓴다

[톤과 가독성]
- 친구에게 진심 담아 수다 떠는 존댓말로
- 마침표는 쓰지 않는다  대신 띄어쓰기와 여백과 줄바꿈으로 호흡을 둔다
- 문단은 짧게  대신 줄바꿈을 자주  한 문단 2~4줄 느낌

[길이]
- 전체 텍스트는 2300자 이상이 되도록
- 도입부는 200~300자 정도로
- 굵은 소제목 3개를 만들고  각 섹션은 충분히 길게 쓴다

[구성]
1) <p>도입부</p>
2) <h2><b>소제목 1</b></h2> + 내용
3) <h2><b>소제목 2</b></h2> + 내용 + 여기 어딘가에 토큰 {ing_token} 을 반드시 한 번 넣기
4) <h2><b>소제목 3</b></h2> + 내용 + 여기 어딘가에 토큰 {step_token} 을 반드시 한 번 넣기
5) 마지막에 <p style='opacity:.7;font-size:13px;'>출처 링크가 있으면 간단히</p>

[키워드]
- 글 흐름 중간중간에 '{keyword}' 단어를 자연스럽게 2~4번만 섞어라  과도하게 반복하지 마
- 씨앗 문장도 자연스럽게 녹여라  {seed}

[출력]
- HTML만 출력한다  제목은 출력하지 않는다
""".strip()

    payload = {
        "title_en": recipe.get("title", ""),
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients": recipe.get("ingredients", []),
        "steps_en": split_steps(recipe.get("instructions", "")),
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
        "now_kst": now.astimezone(KST).strftime("%Y-%m-%d %H:%M"),
        "keyword": keyword,
    }
    user_input = "레시피 JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    return instructions, user_input


def generate_homefeed_body(cfg: AppConfig, recipe: Dict[str, Any], keyword: str, now: datetime) -> str:
    ing_token = "[[INGREDIENTS]]"
    step_token = "[[STEPS]]"
    ing_html = _ingredients_html(cfg, recipe)
    steps_html = _steps_html(recipe)

    client = OpenAI(api_key=cfg.openai.api_key)
    inst, user_in = build_homefeed_prompt(cfg, recipe, keyword=keyword, now=now)
    resp = _openai_call_with_retry(
        client=client,
        model=cfg.openai.model,
        instructions=inst,
        input_text=user_in,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )
    body = (resp.output_text or "").strip()
    if "<" not in body:
        safe = _html.escape(body).replace("\n", "<br/><br/>")
        body = f"<p style='margin:0 0 16px 0;'>{safe}</p>"

    if ing_token in body:
        body = body.replace(ing_token, ing_html)
    else:
        body += "<h2><b>재료 목록</b></h2>" + ing_html

    if step_token in body:
        body = body.replace(step_token, steps_html)
    else:
        body += "<h2><b>만드는 과정</b></h2>" + steps_html

    body = wrap_readable(body)

    if plain_len(body) < 2300:
        fillers = [
            "이 메뉴는 딱 한 번만 기준을 잡아두면 다음부터는 마음이 훨씬 편해지더라고요<br/><br/>"
            "특히 간을 급하게 잡는 순간이 오면 잠깐 멈추고 향과 농도를 먼저 보는 쪽이 안정적이에요<br/><br/>",
            "저는 요리할 때 완벽하려고 하면 오히려 더 흔들리더라고요<br/><br/>"
            "그래서 오늘은 딱 한 가지 기준만 챙기자 이런 식으로 스스로를 설득하는 편이에요<br/><br/>",
            "혹시라도 중간에 꼬였다 싶으면 처음부터 다시 하기보다 상태를 먼저 확인해보세요<br/><br/>"
            "이상하게도 그 한 번의 확인이 전체 흐름을 다시 정리해주는 느낌이 있거든요<br/><br/>",
        ]
        while plain_len(body) < 2300:
            body += "<p style='margin:0 0 16px 0;'>" + random.choice(fillers) + "</p>"

    src = (recipe.get("source") or "").strip()
    yt = (recipe.get("youtube") or "").strip()
    links = []
    if src:
        links.append(f"<a href='{_html.escape(src)}' target='_blank' rel='nofollow noopener'>source</a>")
    if yt:
        links.append(f"<a href='{_html.escape(yt)}' target='_blank' rel='nofollow noopener'>youtube</a>")
    if links:
        body += "<p style='opacity:.7;font-size:13px;margin-top:18px;'>" + " | ".join(links) + "</p>"

    return body


def build_korean_simple_with_free_translate(cfg: AppConfig, recipe: Dict[str, Any], now: datetime, keyword: str) -> Tuple[str, str]:
    title_en = recipe.get("title", "Daily Recipe")
    title_ko = ensure_korean_phrase(cfg, free_translate_text(cfg.free_tr, title_en, debug=cfg.run.debug) or title_en)

    body = wrap_readable(
        f"<p style='margin:0 0 16px 0;'>오늘은 자동 생성 호출이 어려워서 공개 레시피를 한국어로 간단히 정리했어요<br/><br/>"
        f"그래도 { _html.escape(keyword) } 포인트만 놓치지 않게 흐름을 잡아둘게요</p>"
        f"<h2><b>재료 목록</b></h2>{_ingredients_html(cfg, recipe)}"
        f"<h2><b>만드는 과정</b></h2>{_steps_html(recipe)}"
    )
    return f"{title_ko} 실패 포인트만 잡아두기", body


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

    # 제목 키워드 결정 + 번역 보정
    keyword = (cfg.naver.title_keyword or "").strip()
    if not keyword:
        keyword = _first_keyword_from_csv(cfg.naver.keywords_csv)
    if not keyword:
        keyword = free_translate_text(cfg.free_tr, recipe.get("title", ""), debug=cfg.run.debug) or (recipe.get("title") or "오늘 레시피")
    keyword = ensure_korean_phrase(cfg, keyword)

    # 썸네일 업로드(기본: TheMealDB)
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

    # 본문 생성
    try:
        body_html = generate_homefeed_body(cfg, recipe, keyword=keyword, now=now)
    except Exception as e:
        if _is_insufficient_quota_error(e):
            print("[WARN] OpenAI quota depleted → fallback to free translate")
        else:
            print("[WARN] OpenAI failed → fallback to free translate:", repr(e))
        _, body_html = build_korean_simple_with_free_translate(cfg, recipe, now, keyword=keyword)

    # 본문 맨 위 이미지
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            img_tag = (
                f"<p style='margin:0 0 18px 0;'><img src=\"{_html.escape(img)}\" alt=\"{_html.escape(keyword)}\" "
                "style=\"max-width:100%;height:auto;border-radius:14px;display:block;\"/></p>"
            )
            body_html = img_tag + body_html

    # 제목 템플릿 랜덤 적용(키워드 번역 보정된 값)
    title_core = make_title(keyword, mode=cfg.naver.title_mode)
    title = f"{date_str} {slot_label} | {title_core}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 ↓")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2000] + ("\n...(truncated)" if len(body_html) > 2000 else ""))
        return

    # 발행/업데이트
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title", "") or "Daily Recipe"

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
