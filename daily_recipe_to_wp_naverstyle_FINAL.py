# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle_FINAL.py (완전 통합본)
- TheMealDB 랜덤 레시피 수집
- OpenAI로 네이버 홈피드형 '사람 글' 생성 (친구에게 수다떨듯한 존댓말 / 마침표 없이 / 여백 호흡)
- ✅ 제목: 4가지 후킹 스타일 중 랜덤 1개 선택
- ✅ 재료/만드는법: 반드시 한국어 번역 (영어 남으면 자동 재번역/수정)
- ✅ 중복 실행 시: 기존 글 수정이 아니라 "항상 새 글 발행" (slug 랜덤 suffix)
- ✅ 엉뚱한 키워드(타미야 등) 유입 방지: 키워드/본문 금지어 감지 시 재생성
- ✅ 본문 가독성: 행간/여백 강화, 리스트 점/숫자 제거(불릿/1단계/2단계 안 씀)
- ✅ 태그/마무리: 반드시 맨 끝에만 위치
- ✅ HTML 깨짐 문자 제거 (예: < =':0 0 8 0;'> 같은 조각 제거)
- SQLite 발행 이력
- 썸네일: Pexels(있으면) → 없으면 TheMealDB thumb

필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY

선택 env:
  - WP_STATUS=publish
  - WP_CATEGORY_IDS="7" (비워도 기본 7 자동)
  - WP_TAG_IDS="" (WP 태그 ID 사용 시)
  - SQLITE_PATH=data/daily_recipe.sqlite3

네이버 스타일 옵션:
  - NAVER_RANDOM_LEVEL=0~3 (기본 2)
  - NAVER_EXPERIENCE_LEVEL=0~3 (기본 2)
  - NAVER_MAIN_KEYWORD="" (비우면 레시피명에서 자동 추출)
  - NAVER_RECIPE_KEYWORDS="집밥,간단요리,..." (태그/문장에 자연스럽게 활용)
  - PREFER_AREAS="Korean,Japanese,Chinese" (선택)
  - BLOCK_CATEGORIES="Drink,Dessert" (선택)
  - PEXELS_API_KEY="" (있으면 고퀄 썸네일)

동작:
  - RUN_SLOT=day|am|pm (기본 day)
  - DRY_RUN=1 (발행 안 하고 미리보기)
  - DEBUG=1 (로그 자세히)
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
    model: str = "gpt-4.1-mini"


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
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
    )

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

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

    r = requests.post(url, headers=headers, json=payload, timeout=40)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
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
# Pexels (thumbnail)
# -----------------------------
def pexels_pick_image(api_key: str, query: str) -> Optional[str]:
    api_key = (api_key or "").strip()
    if not api_key:
        return None
    q = (query or "").strip()
    if not q:
        return None

    headers = {"Authorization": api_key}
    params = {"query": q, "per_page": 15, "orientation": "landscape"}
    r = requests.get(PEXELS_SEARCH, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        return None
    j = r.json() or {}
    photos = j.get("photos") or []
    if not photos:
        return None

    # 고해상도 우선
    cand = sorted(photos, key=lambda p: int((p.get("width") or 0) * (p.get("height") or 0)), reverse=True)[0]
    src = cand.get("src") or {}
    return (src.get("large2x") or src.get("large") or src.get("original") or "").strip() or None


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
            return client.responses.create(model=model, instructions=instructions, input=input_text)
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


def _has_english(s: str) -> bool:
    # 영어 단어(3글자 이상)가 보이면 영어 섞임으로 판단
    s = s or ""
    return bool(re.search(r"[A-Za-z]{3,}", s))


def _strip_weird_html_fragments(s: str) -> str:
    # 예: "< =':0 0 8 0;'>", "< =':...'>" 같은 깨진 조각 제거
    s = s or ""
    s = re.sub(r"<\s*=\s*['\"][^>]*?>", "", s)
    s = s.replace("<=':0 0 8 0;'>", "")
    s = re.sub(r"<\s*=\s*[^>]*>", "", s)
    return s


def _dedup_consecutive_lines(s: str) -> str:
    lines = [ln.rstrip() for ln in (s or "").split("\n")]
    out = []
    prev = None
    for ln in lines:
        key = ln.strip()
        if not key:
            out.append(ln)
            prev = None
            continue
        if prev == key:
            continue
        out.append(ln)
        prev = key
    return "\n".join(out)


def _remove_periods(text: str) -> str:
    # “마침표 없이” 요청 대응: 문장부호를 크게 줄임(링크는 최대한 보존)
    t = text or ""
    # URL 보호
    urls = {}
    def _url_repl(m):
        k = f"__URL{len(urls)}__"
        urls[k] = m.group(0)
        return k

    t = re.sub(r"https?://[^\s\"']+", _url_repl, t)

    # 문장부호 제거
    t = t.replace(".", "")
    t = t.replace("。", "")
    t = t.replace("!", "")
    t = t.replace("?", "")
    t = t.replace("…", "")
    t = t.replace("·", " ")
    t = re.sub(r"\s{2,}", " ", t).strip()

    # URL 복원
    for k, v in urls.items():
        t = t.replace(k, v)
    return t


def translate_recipe_to_korean(client: OpenAI, model: str, title_en: str, ingredients: List[Dict[str, str]], steps_en: List[str],
                              max_retries: int, debug: bool) -> Tuple[str, List[str], List[str]]:
    """
    ✅ 제목/재료/단계 전체를 한국어로 번역 (영어 남으면 자동 보정)
    """
    payload = {
        "title_en": title_en,
        "ingredients": ingredients,
        "steps_en": steps_en,
    }
    inst = """
너는 번역가다
아래 JSON을 한국어로 번역해서 JSON으로만 반환해라

[규칙]
- 반드시 한국어만 사용하고 영어 단어는 남기지 마
- ingredients는 재료명은 자연스러운 한국어로 번역하고 괄호로 영어 원문을 붙이지 마
- measure는 가능하면 한국어 단위로 자연스럽게 바꿔라 (tbsp 큰술 tsp 작은술 cup 컵 등)
- steps는 의미만 정확히 자연스럽게 번역하되 단계 수는 유지해라
- 출력은 아래 키를 정확히 지켜라

[출력 JSON]
{
  "title_ko": "...",
  "ingredients_ko": ["재료명  수량", ...],
  "steps_ko": ["...", ...]
}
""".strip()

    resp = _openai_call_with_retry(client, model, inst, json.dumps(payload, ensure_ascii=False), max_retries, debug)
    out = (resp.output_text or "").strip()
    try:
        j = json.loads(out)
        title_ko = str(j.get("title_ko") or "").strip()
        ing_ko = j.get("ingredients_ko") or []
        steps_ko = j.get("steps_ko") or []
        ing_ko = [str(x).strip() for x in ing_ko if str(x).strip()]
        steps_ko = [str(x).strip() for x in steps_ko if str(x).strip()]
    except Exception:
        # JSON 파싱 실패하면 2차로 “그냥 제목만이라도”
        title_ko = title_en
        ing_ko = [f"{it.get('name','').strip()} {it.get('measure','').strip()}".strip() for it in ingredients]
        steps_ko = steps_en[:]

    # 영어 잔존 시 1회 보정
    if _has_english(title_ko) or any(_has_english(x) for x in ing_ko) or any(_has_english(x) for x in steps_ko):
        fix_payload = {"title_ko": title_ko, "ingredients_ko": ing_ko, "steps_ko": steps_ko}
        fix_inst = """
아래 JSON에서 영어가 섞인 부분을 전부 자연스러운 한국어로 고쳐라
의미는 유지하고 출력은 JSON만
""".strip()
        resp2 = _openai_call_with_retry(client, model, fix_inst, json.dumps(fix_payload, ensure_ascii=False), max_retries, debug)
        out2 = (resp2.output_text or "").strip()
        try:
            j2 = json.loads(out2)
            title_ko = str(j2.get("title_ko") or title_ko).strip()
            ing_ko2 = j2.get("ingredients_ko") or ing_ko
            steps_ko2 = j2.get("steps_ko") or steps_ko
            ing_ko = [str(x).strip() for x in ing_ko2 if str(x).strip()]
            steps_ko = [str(x).strip() for x in steps_ko2 if str(x).strip()]
        except Exception:
            pass

    return title_ko.strip(), ing_ko, steps_ko


# -----------------------------
# Title: 4 hook styles random
# -----------------------------
def _compact_kw(kw: str) -> str:
    kw = (kw or "").strip()
    kw = re.sub(r"\s+", " ", kw)
    return kw


def make_hook_title(main_kw: str) -> str:
    kw = _compact_kw(main_kw) or "오늘의 레시피"
    # 4가지 타입
    benefit = [
        f"{kw} 해보면  맛이 달라지는 포인트가 있어요",
        f"{kw} 딱 한 기준만 잡으면  실패 확률이 확 줄어요",
        f"{kw} 처음엔 어려운데  이 한 줄만 기억하면 편해요",
    ]
    threat = [
        f"{kw} 여기 한 번 놓치면  맛이 확 무너져요",
        f"{kw} 간을 급하게 잡으면  전체가 흐려질 수 있어요",
        f"{kw} 쉬워 보여도  이 지점에서 많이 흔들려요",
    ]
    curiosity = [
        f"{kw} 왜 집에서는 맛이 다를까  답이 의외로 간단해요",
        f"{kw} 맛이 들쭉날쭉할 때  원인이 한 가지더라고요",
        f"괜히 어렵게 느껴지는 {kw}  사실 순서만 정리하면 돼요",
    ]
    compare = [
        f"{kw} 그냥 따라하기 vs 기준 잡기  결과가 다르게 나와요",
        f"{kw} 성공한 날과 아쉬운 날  차이가 여기서 갈려요",
        f"{kw} 밖에서 먹는 맛을 집에서 내는 법  포인트는 하나예요",
    ]
    buckets = [benefit, threat, curiosity, compare]
    chosen = random.choice(buckets)
    return random.choice(chosen).strip()


# -----------------------------
# Keyword safety (타미야 등 방지)
# -----------------------------
BANNED_WORDS = [
    "타미야", "tamiya", "프라모델", "도색", "건담", "rc카", "미니카", "락카", "에나멜", "에어브러시",
    "칠하는", "도료", "사포", "프라이머"
]


def contains_banned(text: str) -> bool:
    t = (text or "").lower()
    for w in BANNED_WORDS:
        if w.lower() in t:
            return True
    return False


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


def _spacer(h: int = 14) -> str:
    return f"<div style='height:{h}px;'></div>"


def _p(text: str) -> str:
    # 문장 끝 마침표 제거 + 호흡
    t = _remove_periods(text)
    t = _html.escape(t)
    # 줄바꿈은 <br/>
    t = t.replace("\n", "<br/>")
    return f"<p style='margin:0 0 16px 0;'>{t}</p>"


def _h2_bold(title: str) -> str:
    t = _remove_periods(title)
    return f"<h2 style='margin:22px 0 12px 0; font-size:18px;'><b>{_html.escape(t)}</b></h2>"


def _simple_lines_block(title: str, lines: List[str]) -> str:
    # 리스트 점/숫자 제거: div 줄로 출력
    out = [f"<p style='margin:0 0 10px 0;'><b>{_html.escape(_remove_periods(title))}</b></p>"]
    for ln in lines:
        ln2 = _remove_periods(ln)
        out.append(f"<div style='margin:0 0 10px 0; padding-left:2px;'>{_html.escape(ln2)}</div>")
    return "\n".join(out)


def _hashtags(main_kw: str, extra_csv: str) -> str:
    tags = []
    mk = re.sub(r"\s+", "", (main_kw or "").strip())
    if mk:
        tags.append("#" + mk)

    extra = [x.strip() for x in (extra_csv or "").split(",") if x.strip()]
    # 공백 제거 + # 붙이기
    extra = [("#" + re.sub(r"\s+", "", x)) if not x.startswith("#") else x for x in extra]
    # 금지어 제거
    cleaned = []
    for t in extra:
        if contains_banned(t):
            continue
        cleaned.append(t)
    # 중복 제거
    seen = set()
    for t in cleaned:
        if t in seen:
            continue
        seen.add(t)
        tags.append(t)

    tags = tags[:12]
    if not tags:
        return ""
    return f"<p style='margin:16px 0 0 0; opacity:.85; font-size:14px;'><b>태그</b><br/>{' '.join(tags)}</p>"


def generate_body_json(
    client: OpenAI,
    model: str,
    recipe_title_ko: str,
    main_kw: str,
    random_level: int,
    exp_level: int,
    max_retries: int,
    debug: bool,
) -> Dict[str, Any]:
    """
    ✅ intro(200~300자) + 소제목 3개 + 각 섹션 텍스트 생성
    - 존댓말 수다톤
    - 마침표 없이
    - 특정 문장 반복 방지
    """
    seed_moods = [
        "퇴근하고 힘 빠진 날  괜히 따뜻한 게 당기실 때",
        "주말인데도 밖에 나가기 귀찮은 날  집에서 해결하고 싶을 때",
        "뭔가 해먹고 싶은데  머리 쓰기 싫을 때",
        "오늘은 그냥 마음 편하게  기준만 잡고 싶을 때",
    ]
    mood = random.choice(seed_moods)

    payload = {
        "recipe_title_ko": recipe_title_ko,
        "main_keyword": main_kw,
        "random_level": random_level,
        "experience_level": exp_level,
        "mood_seed": mood,
        "constraints": {
            "intro_min": 200,
            "intro_max": 300,
            "sections": 3,
            "section_min_chars": 1500,
            "total_min_chars": 2300,
            "no_period": True,
            "honorific_friend_chat": True,
            "no_step_numbering": True,
            "no_bullet_marker": True,
            "no_english": True,
            "no_repeating_sentences": True
        }
    }

    inst = """
너는 친구에게 진심으로 수다 떠는 블로그 글을 쓰는 사람이다
반드시 한국어로만 작성해라
마침표는 쓰지 마라
존댓말로만 써라  반말 금지다
중간에 같은 문장을 반복하지 마라
AI 냄새 나는 말투 금지다  너무 규격적인 문장 금지다

[출력은 JSON만]
{
  "intro": "200~300자",
  "section_titles": ["굵은 소제목1","굵은 소제목2","굵은 소제목3"],
  "section1_pre": "섹션1 앞부분 1500자 이상을 채울 수 있게 충분히 길게",
  "section1_post": "섹션1 뒷부분 1500자 이상을 채울 수 있게 충분히 길게",
  "section2": "섹션2 본문 1500자 이상",
  "section3": "섹션3 본문 1500자 이상",
  "closing": "마무리 멘트 120~220자"
}

[내용 방향]
- 오늘 메뉴를 레시피처럼 ‘해보면’ 느낌으로  감정 생각 가치관을 자연스럽게 넣어라
- 직접 해봤다고 단정하지 말고  보통 집에서 해보면  이런 느낌이 든다  이런 방식으로 쓰라
- ‘기준을 잡아두면 마음이 편해진다’ 같은 가치관 문장 꼭 섞어라
- 강요하지 말고  저장해두면 좋다는 말은 딱 1번만 자연스럽게
""".strip()

    resp = _openai_call_with_retry(client, model, inst, json.dumps(payload, ensure_ascii=False), max_retries, debug)
    out = (resp.output_text or "").strip()
    try:
        j = json.loads(out)
    except Exception:
        raise RuntimeError("body_json_parse_failed")

    # 기본 검증 + 영어 섞이면 재요청 1회
    def _get(k, default=""):
        return str(j.get(k) or default).strip()

    intro = _get("intro")
    titles = j.get("section_titles") or []
    if not isinstance(titles, list) or len(titles) != 3:
        titles = ["오늘의 기준을 잡아두는 법", "실패를 줄이는 마음가짐", "다음에 더 편해지는 정리"]

    s1_pre = _get("section1_pre")
    s1_post = _get("section1_post")
    s2 = _get("section2")
    s3 = _get("section3")
    closing = _get("closing")

    whole = " ".join([intro, " ".join(titles), s1_pre, s1_post, s2, s3, closing])
    if _has_english(whole) or contains_banned(whole):
        # 재요청: 영어/금지어 제거 강화
        inst2 = inst + "\n\n추가 규칙  영어 단어를 절대 쓰지 마라  금지어가 나오면 안 된다"
        resp2 = _openai_call_with_retry(client, model, inst2, json.dumps(payload, ensure_ascii=False), max_retries, debug)
        out2 = (resp2.output_text or "").strip()
        j = json.loads(out2)

    return j


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

    init_db(cfg.sqlite_path, debug=cfg.run.debug)

    recipe = pick_recipe(cfg)
    recipe_id = recipe.get("id", "")
    title_en = recipe.get("title", "") or "Daily Recipe"
    ingredients_en = recipe.get("ingredients", []) or []
    steps_en = split_steps(recipe.get("instructions", ""))

    client = OpenAI(api_key=cfg.openai.api_key)

    # 1) 레시피 번역(필수)
    title_ko, ingredients_ko, steps_ko = translate_recipe_to_korean(
        client, cfg.openai.model, title_en, ingredients_en, steps_en, cfg.run.openai_max_retries, cfg.run.debug
    )

    # 메인 키워드 결정
    main_kw = (cfg.naver.main_keyword or "").strip()
    if not main_kw:
        main_kw = title_ko.strip()
    main_kw = re.sub(r"\s+", " ", main_kw).strip()

    if contains_banned(main_kw):
        # 혹시라도 메인키워드가 이상하면 레시피명으로 강제
        main_kw = title_ko.strip()

    # 2) 제목 후킹 4가지 랜덤
    hook_title = make_hook_title(main_kw)

    # 3) 본문(사람 글) JSON 생성
    body_json = generate_body_json(
        client=client,
        model=cfg.openai.model,
        recipe_title_ko=title_ko,
        main_kw=main_kw,
        random_level=cfg.naver.random_level,
        exp_level=cfg.naver.experience_level,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    intro = str(body_json.get("intro") or "").strip()
    sec_titles = body_json.get("section_titles") or []
    s1_pre = str(body_json.get("section1_pre") or "").strip()
    s1_post = str(body_json.get("section1_post") or "").strip()
    s2 = str(body_json.get("section2") or "").strip()
    s3 = str(body_json.get("section3") or "").strip()
    closing = str(body_json.get("closing") or "").strip()

    # 4) HTML 조립 (✅ 정확한 순서 고정)
    inner = []
    inner.append(_p(intro))
    inner.append(_spacer(8))

    # 섹션1 (중간에 재료/만드는법 블록 삽입)
    t1 = sec_titles[0] if len(sec_titles) > 0 else "오늘은 기준부터 잡아볼게요"
    inner.append(_h2_bold(t1))
    inner.append(_p(s1_pre))

    # ✅ 레시피 목록(번역 필수) - 불릿/번호 없이
    inner.append(_spacer(6))
    inner.append(_simple_lines_block("재료", ingredients_ko))
    inner.append(_spacer(10))
    inner.append(_simple_lines_block("만드는 법", steps_ko))
    inner.append(_spacer(8))

    inner.append(_p(s1_post))

    # 섹션2
    t2 = sec_titles[1] if len(sec_titles) > 1 else "실패를 줄이는 마음가짐"
    inner.append(_h2_bold(t2))
    inner.append(_p(s2))

    # 섹션3
    t3 = sec_titles[2] if len(sec_titles) > 2 else "다음에 더 편해지는 정리"
    inner.append(_h2_bold(t3))
    inner.append(_p(s3))

    # 마무리 + 태그 (✅ 반드시 맨 끝)
    inner.append(_h2_bold("마무리"))
    inner.append(_p(closing))
    inner.append(_hashtags(main_kw, cfg.naver.recipe_keywords_csv))

    body_html = "\n".join(inner)
    body_html = _strip_weird_html_fragments(body_html)
    body_html = _dedup_consecutive_lines(body_html)

    # 영어가 아직 남아 있으면 1회 자동 보정 (발행 중단 안 함)
    if _has_english(body_html):
        fix_inst = """
아래 HTML에서 영어로 남아있는 부분을 전부 자연스러운 한국어로 바꿔라
HTML 태그 구조는 유지해라
마침표는 쓰지 마라
존댓말 유지해라
출력은 HTML만
""".strip()
        resp_fix = _openai_call_with_retry(client, cfg.openai.model, fix_inst, body_html, cfg.run.openai_max_retries, cfg.run.debug)
        body_html = (resp_fix.output_text or "").strip()
        body_html = _strip_weird_html_fragments(body_html)

    # 금지어가 나오면 본문 키워드 부분을 제거하고 재시도 없이 안전 처리
    if contains_banned(body_html):
        body_html = re.sub("|".join([re.escape(w) for w in BANNED_WORDS]), "", body_html, flags=re.I)

    # 5) 썸네일 준비 (Pexels 우선)
    thumb_url = (recipe.get("thumb") or "").strip()
    best_img = None
    if cfg.pexels_api_key:
        q = f"{main_kw} food"
        best_img = pexels_pick_image(cfg.pexels_api_key, q)

    final_img_url = best_img or thumb_url

    # WordPress 업로드(이미지 → 본문 상단 삽입)
    media_id = None
    media_url = ""
    if cfg.run.upload_thumb and final_img_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, final_img_url, filename_hint=f"recipe-{now.strftime('%Y%m%d-%H%M%S')}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    if cfg.run.embed_image_in_body:
        img = media_url or final_img_url
        if img:
            img_tag = f"<p style='margin:0 0 18px 0;'><img src='{_html.escape(img)}' alt='{_html.escape(main_kw)}' style='max-width:100%;height:auto;border-radius:14px;'/></p>"
            body_html = img_tag + "\n" + body_html

    full_html = _style_wrap(body_html)

    # ✅ 제목: 메인키워드 포함 + 후킹
    title_final = f"{main_kw} {hook_title}".strip()
    title_final = _remove_periods(title_final)

    # ✅ 항상 새 글 발행: slug는 매번 랜덤 suffix
    suffix = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    slug = f"naverstyle-recipe-{now.strftime('%Y-%m-%d')}-{slot}-{suffix}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", title_final)
        print("SLUG:", slug)
        print(full_html[:2200] + ("\n...(truncated)" if len(full_html) > 2200 else ""))
        return

    featured = int(media_id) if (cfg.run.set_featured and media_id) else None

    post_id, link = wp_create_post(cfg.wp, title_final, slug, full_html, featured_media=featured)

    date_key = now.strftime("%Y-%m-%d_%H%M%S") + "_" + slot + "_" + suffix
    save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, title_en, post_id, link)

    print("OK(created):", post_id, link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
