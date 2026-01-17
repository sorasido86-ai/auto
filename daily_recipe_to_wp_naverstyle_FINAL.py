# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle_FINAL.py (완전 통합본)

- TheMealDB 랜덤 레시피 수집
- (선택) Pexels 고퀄 음식 사진 검색(썸네일) -> PEXELS_API_KEY 필요
- OpenAI로:
  1) 레시피(제목/재료/방법) 한국어 번역 JSON
  2) 네이버 홈피드형 "친구 수다 존댓말(해요체)" 본문 JSON 생성
- WordPress 발행 (항상 새 글) + 썸네일 업로드/대표이미지 설정
- SQLite: 최근 레시피 중복 회피용 기록

필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY

권장 env:
  - WP_STATUS=publish
  - WP_CATEGORY_IDS="7"
  - WP_TAG_IDS="" (선택)
  - SQLITE_PATH=data/daily_recipe.sqlite3

옵션:
  - OPENAI_MODEL=gpt-4.1-mini (기본)
  - OPENAI_MAX_RETRIES=3
  - DRY_RUN=0|1
  - DEBUG=0|1
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=30

네이버 스타일:
  - NAVER_RANDOM_LEVEL=0~3 (기본 2)
  - NAVER_EXPERIENCE_LEVEL=0~3 (기본 2)
  - NAVER_MAIN_KEYWORD="" (비우면 레시피명(한글번역)을 키워드로 사용)
  - NAVER_KEYWORDS="집밥,간단요리,..." (해시태그/자연스러운 키워드용)
  - NAVER_SECTION_MIN_CHARS=1500 (기본 1500)
  - NAVER_INTRO_MIN=200 (기본 200)
  - NAVER_INTRO_MAX=300 (기본 300)

레시피 필터:
  - PREFER_AREAS="Korean,Japanese,Chinese" (선택)
  - BLOCK_CATEGORIES="Dessert,Drink" (선택)

썸네일 고퀄(선택):
  - PEXELS_API_KEY=...
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

import openai  # exception types
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
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 30
    openai_max_retries: int = 3
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True


@dataclass
class NaverStyleConfig:
    random_level: int = 2
    experience_level: int = 2
    main_keyword: str = ""
    keywords_csv: str = ""
    prefer_areas: List[str] = field(default_factory=list)
    block_categories: List[str] = field(default_factory=list)
    intro_min: int = 200
    intro_max: int = 300
    section_min_chars: int = 1500


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4.1-mini"


@dataclass
class MediaConfig:
    pexels_api_key: str = ""


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    naver: NaverStyleConfig
    sqlite_path: str
    openai: OpenAIConfig
    media: MediaConfig


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"
    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    sqlite_path = _env("SQLITE_PATH", "data/daily_recipe.sqlite3")

    run = RunConfig(
        dry_run=_env_bool("DRY_RUN", False),
        debug=_env_bool("DEBUG", False),
        avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
        max_tries=_env_int("MAX_TRIES", 30),
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
        upload_thumb=_env_bool("UPLOAD_THUMB", True),
        set_featured=_env_bool("SET_FEATURED", True),
        embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
    )

    naver_random_level = max(0, min(3, _env_int("NAVER_RANDOM_LEVEL", 2)))
    naver_exp_level = max(0, min(3, _env_int("NAVER_EXPERIENCE_LEVEL", 2)))
    naver_main_kw = _env("NAVER_MAIN_KEYWORD", "")
    naver_kws = _env("NAVER_KEYWORDS", "")
    prefer_areas = _parse_str_list(_env("PREFER_AREAS", ""))
    block_categories = _parse_str_list(_env("BLOCK_CATEGORIES", ""))

    intro_min = max(120, _env_int("NAVER_INTRO_MIN", 200))
    intro_max = max(intro_min + 20, _env_int("NAVER_INTRO_MAX", 300))
    section_min_chars = max(700, _env_int("NAVER_SECTION_MIN_CHARS", 1500))

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

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
        run=run,
        naver=NaverStyleConfig(
            random_level=naver_random_level,
            experience_level=naver_exp_level,
            main_keyword=naver_main_kw,
            keywords_csv=naver_kws,
            prefer_areas=prefer_areas,
            block_categories=block_categories,
            intro_min=intro_min,
            intro_max=intro_max,
            section_min_chars=section_min_chars,
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
        media=MediaConfig(pexels_api_key=pexels_key),
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
    print("[CFG] DRY_RUN:", int(cfg.run.dry_run), "| DEBUG:", int(cfg.run.debug))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model, "| OPENAI_KEY:", ok(cfg.openai.api_key))
    print("[CFG] NAVER_RANDOM_LEVEL:", cfg.naver.random_level, "| NAVER_EXPERIENCE_LEVEL:", cfg.naver.experience_level)
    print("[CFG] NAVER_MAIN_KEYWORD:", cfg.naver.main_keyword or "(auto)")
    print("[CFG] NAVER_KEYWORDS:", cfg.naver.keywords_csv or "(empty)")
    print("[CFG] INTRO_MIN/MAX:", cfg.naver.intro_min, "/", cfg.naver.intro_max, "| SECTION_MIN_CHARS:", cfg.naver.section_min_chars)
    print("[CFG] PREFER_AREAS:", ",".join(cfg.naver.prefer_areas) if cfg.naver.prefer_areas else "(any)")
    print("[CFG] BLOCK_CATEGORIES:", ",".join(cfg.naver.block_categories) if cfg.naver.block_categories else "(none)")
    print("[CFG] PEXELS_API_KEY:", ok(cfg.media.pexels_api_key))


# -----------------------------
# SQLite: history for avoiding repeats
# -----------------------------
POSTS_SQL = """
CREATE TABLE IF NOT EXISTS posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  recipe_id TEXT,
  recipe_title_en TEXT,
  wp_post_id INTEGER,
  wp_link TEXT,
  created_at TEXT
)
"""


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(POSTS_SQL)
    con.commit()
    con.close()


def save_post_row(path: str, recipe_id: str, recipe_title_en: str, wp_post_id: int, wp_link: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO posts(recipe_id, recipe_title_en, wp_post_id, wp_link, created_at) VALUES (?,?,?,?,?)",
        (recipe_id, recipe_title_en, int(wp_post_id), wp_link, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


def get_recent_recipe_ids(path: str, days: int) -> List[str]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    con = sqlite3.connect(path)
    cur = con.cursor()
    try:
        cur.execute("SELECT recipe_id FROM posts WHERE created_at >= ? AND recipe_id IS NOT NULL", (cutoff,))
        rows = cur.fetchall()
        return [str(r[0]) for r in rows if r and r[0]]
    finally:
        con.close()


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
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:800]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "recipe.jpg") -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=40)
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
    if len(parts) <= 2 and len(t) > 350:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    return [p for p in parts if p]


# -----------------------------
# Pexels (optional high quality food image)
# -----------------------------
def pexels_search_food_image(api_key: str, query: str, debug: bool = False) -> str:
    if not api_key:
        return ""
    q = (query or "").strip()
    if not q:
        return ""
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": f"{q} food", "per_page": 10, "orientation": "landscape"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=25)
        if r.status_code != 200:
            if debug:
                print("[PEXELS] non-200:", r.status_code, r.text[:200])
            return ""
        j = r.json()
        photos = j.get("photos") or []
        if not photos:
            return ""
        # 최대한 고해상도
        src = photos[0].get("src") or {}
        return str(src.get("large2x") or src.get("large") or src.get("original") or "").strip()
    except Exception as e:
        if debug:
            print("[PEXELS] failed:", repr(e))
        return ""


# -----------------------------
# OpenAI helpers
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
    last = None
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(model=model, instructions=instructions, input=input_text)
        except openai.RateLimitError as e:
            last = e
            if _is_insufficient_quota_error(e):
                raise
            if attempt >= max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] RateLimit -> retry {sleep_s:.2f}s")
            time.sleep(sleep_s)
        except openai.APIError as e:
            last = e
            if attempt >= max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] APIError -> retry {sleep_s:.2f}s")
            time.sleep(sleep_s)
        except Exception as e:
            last = e
            if attempt >= max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] Unknown -> retry {sleep_s:.2f}s | {repr(e)}")
            time.sleep(sleep_s)
    raise last  # pragma: no cover


def _has_english(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s or ""))


def _strip_forbidden_punct(s: str) -> str:
    if not s:
        return ""
    # 마침표/물음표/느낌표 제거 + 중복 공백 정리
    s = s.replace(".", "").replace("!", "").replace("?", "")
    s = s.replace("。", "").replace("！", "").replace("？", "")
    s = s.replace("•", "").replace("·", " ")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _normalize_measures_ko(s: str) -> str:
    if not s:
        return ""
    rep = {
        "tsp": "작은술",
        "tbsp": "큰술",
        "tablespoon": "큰술",
        "teaspoon": "작은술",
        "cup": "컵",
        "cups": "컵",
        "oz": "온스",
        "lb": "파운드",
    }
    out = s
    for k, v in rep.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)
    return out.strip()


def _json_from_text(txt: str) -> Dict[str, Any]:
    t = (txt or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    # 앞뒤 군더더기 제거
    a = t.find("{")
    b = t.rfind("}")
    if a == -1 or b == -1:
        raise ValueError("JSON object not found")
    return json.loads(t[a : b + 1])


def translate_recipe_parts_to_ko(
    client: OpenAI,
    model: str,
    recipe_title_en: str,
    ingredients_en: List[Dict[str, str]],
    steps_en: List[str],
    max_retries: int,
    debug: bool = False,
) -> Tuple[str, List[Dict[str, str]], List[str]]:
    payload = {
        "title_en": recipe_title_en,
        "ingredients_en": [
            {"name": (it.get("name") or "").strip(), "measure": _normalize_measures_ko((it.get("measure") or "").strip())}
            for it in (ingredients_en or [])
        ],
        "steps_en": [s.strip() for s in (steps_en or []) if (s or "").strip()],
        "rules": {
            "korean_only": True,
            "no_bullets": True,
            "no_step_numbers": True,
        },
    }

    instructions = (
        "너는 번역가다\n"
        "반드시 한국어로만 번역한다\n"
        "출력은 JSON만\n"
        "영어 단어를 남기지 않는다\n"
        "불릿 특수문자(•,-,*)를 문장 앞에 붙이지 않는다\n"
        "1단계 2단계 같은 단계 번호를 만들지 않는다\n"
        "형식은 반드시 아래 키를 유지한다\n"
        '{"title_ko":"...","ingredients_ko":[{"name_ko":"...","measure_ko":"..."}],"steps_ko":["...","..."]}'
    )

    resp = _openai_call_with_retry(
        client=client,
        model=model,
        instructions=instructions,
        input_text=json.dumps(payload, ensure_ascii=False),
        max_retries=max_retries,
        debug=debug,
    )
    obj = _json_from_text(resp.output_text or "")

    title_ko = _strip_forbidden_punct(str(obj.get("title_ko") or "").strip())
    ing_ko_raw = obj.get("ingredients_ko") or []
    steps_ko_raw = obj.get("steps_ko") or []

    ingredients_ko: List[Dict[str, str]] = []
    for it in ing_ko_raw:
        name_ko = _strip_forbidden_punct(str((it or {}).get("name_ko") or "").strip())
        mea_ko = _strip_forbidden_punct(str((it or {}).get("measure_ko") or "").strip())
        if name_ko:
            ingredients_ko.append({"name_ko": name_ko, "measure_ko": mea_ko})

    steps_ko = [_strip_forbidden_punct(str(s or "").strip()) for s in steps_ko_raw if str(s or "").strip()]

    # 마지막 안전장치: 영어 남으면 재시도 1회
    if _has_english(title_ko) or any(_has_english(x["name_ko"] + x["measure_ko"]) for x in ingredients_ko) or any(_has_english(s) for s in steps_ko):
        if debug:
            print("[WARN] English remains in translation -> reprompt once")
        payload["rules"]["korean_only"] = True
        payload["rules"]["force_remove_english"] = True
        resp2 = _openai_call_with_retry(
            client=client,
            model=model,
            instructions=instructions + "\n영어가 남아 있으면 무조건 자연스러운 한국어로 바꿔서 출력해라",
            input_text=json.dumps(payload, ensure_ascii=False),
            max_retries=max_retries,
            debug=debug,
        )
        obj2 = _json_from_text(resp2.output_text or "")
        title_ko = _strip_forbidden_punct(str(obj2.get("title_ko") or "").strip())
        ing_ko_raw = obj2.get("ingredients_ko") or []
        steps_ko_raw = obj2.get("steps_ko") or []
        ingredients_ko = []
        for it in ing_ko_raw:
            name_ko = _strip_forbidden_punct(str((it or {}).get("name_ko") or "").strip())
            mea_ko = _strip_forbidden_punct(str((it or {}).get("measure_ko") or "").strip())
            if name_ko:
                ingredients_ko.append({"name_ko": name_ko, "measure_ko": mea_ko})
        steps_ko = [_strip_forbidden_punct(str(s or "").strip()) for s in steps_ko_raw if str(s or "").strip()]

    return title_ko, ingredients_ko, steps_ko


# -----------------------------
# Title hook (4 styles)
# -----------------------------
def build_hook_title(main_kw: str) -> str:
    kw = (main_kw or "").strip()
    if not kw:
        kw = "오늘 레시피"

    profit = [
        f"{kw}  한 번만 이렇게 잡으면  맛이 안정되는 이유",
        f"{kw}  이 포인트 하나로  실패 확률이 확 줄었어요",
        f"{kw}  마지막에 이거만 지키면  집밥 느낌이 딱 나요",
    ]
    threat = [
        f"{kw}  여기 놓치면  맛이 반으로 떨어져요  저도 그랬어요",
        f"{kw}  이 타이밍만 어긋나면  갑자기 망해요  처음엔 몰랐어요",
        f"{kw}  괜히 급하게 하면  맛이 흐려져요  이 부분만 조심해요",
    ]
    curious = [
        f"{kw}  왜 똑같이 해도 맛이 갈릴까요  답은 여기였어요",
        f"{kw}  제일 많이 헷갈리는 지점  딱 한 줄로 정리했어요",
        f"{kw}  이 메뉴  의외로 여기서 다 갈려요  그래서 정리했어요",
    ]
    compare = [
        f"{kw}  외식이랑 뭐가 다를까요  집에서 살리는 포인트가 따로 있어요",
        f"{kw}  간단해 보이는데 은근 어렵죠  차이를 만드는 쪽은 따로 있어요",
        f"{kw}  비슷한 메뉴 많지만  결국 돈 되는 맛은 이쪽이에요",
    ]

    pick_group = random.choice([profit, threat, curious, compare])
    title = random.choice(pick_group)
    return _strip_forbidden_punct(title)


# -----------------------------
# Naver-style content generation (JSON)
# -----------------------------
BANNED_WORDS = [
    "타미야", "프라모델", "도색", "건담", "도료", "마감재", "스프레이", "페인트",
    "주식", "코인", "청년도약계좌", "보험", "대출", "부동산"
]


def _contains_banned(s: str) -> bool:
    t = (s or "")
    return any(w in t for w in BANNED_WORDS)


def generate_naver_body_json(
    client: OpenAI,
    model: str,
    cfg: AppConfig,
    main_kw: str,
    recipe_title_ko: str,
    ingredients_ko: List[Dict[str, str]],
    steps_ko: List[str],
    max_retries: int,
    debug: bool = False,
) -> Dict[str, Any]:
    rnd = cfg.naver.random_level
    exp = cfg.naver.experience_level

    # 랜덤 씨드(문장 반복/AI스러운 패턴 줄이기)
    moods = ["퇴근하고", "주말 오후에", "비 오는 날에", "바람 부는 날에", "괜히 허전한 날에", "손이 가기 싫은 날에"]
    values = ["완벽하게 하려는 마음", "내 페이스를 지키는 느낌", "괜히 조급해지는 마음", "소소한 성취감", "오늘 하루를 다독이는 느낌"]
    angles = ["실패 포인트를 줄이는 기준", "맛이 갈리는 한 끗", "재료 상태를 보는 방법", "간을 정리하는 순서", "식감을 살리는 타이밍"]

    seed = {
        "mood": random.choice(moods),
        "value": random.choice(values),
        "angle": random.choice(angles),
        "tone": "친구에게 수다 떠는 존댓말 해요체",
        "no_periods": True,
        "no_step_numbers": True,
        "no_bullets": True,
        "intro_min": cfg.naver.intro_min,
        "intro_max": cfg.naver.intro_max,
        "section_min_chars": cfg.naver.section_min_chars,
        "random_level": rnd,
        "experience_level": exp,
    }

    kws = [k.strip() for k in (cfg.naver.keywords_csv or "").split(",") if k.strip()]
    # 키워드는 "내용 주제"로 쓰지 말고 해시태그 후보로만 쓰게 강제
    kw_hint = [k for k in kws if not _contains_banned(k)][:10]

    payload = {
        "main_keyword": main_kw,
        "recipe_title_ko": recipe_title_ko,
        "ingredients_ko": ingredients_ko,
        "steps_ko": steps_ko,
        "seed": seed,
        "keywords_hint_for_hashtags_only": kw_hint,
    }

    instructions = f"""
너는 한국어로 글을 쓰는 블로거다
말투는 친구에게 진심 담아 수다 떠는 존댓말 해요체로만 쓴다
반말 금지  명령형 금지  딱딱한 문서체 금지

절대 규칙
- 마침표 느낌표 물음표를 쓰지 않는다  대신 줄바꿈과 여백으로 호흡을 만든다
- 단계 번호 1단계 2단계 같은 표기를 만들지 않는다
- 불릿 특수문자 • - * 같은 걸 문장 앞에 붙이지 않는다
- 영어를 섞지 않는다  반드시 한국어만
- 레시피는 제공된 재료와 steps 내용에서만 말한다  새 재료 새 단계 추가 금지
- 키워드는 본문 주제로 바꾸지 않는다  해시태그 후보로만 쓰고  음식 얘기를 벗어나지 않는다

출력은 JSON만
형식은 반드시 아래 키를 가진다
{{
  "intro": "200~300자",
  "h2": [
    {{"title":"굵은 소제목1","body":"{cfg.naver.section_min_chars}자 이상"}},
    {{"title":"굵은 소제목2","body":"{cfg.naver.section_min_chars}자 이상"}},
    {{"title":"굵은 소제목3","body":"{cfg.naver.section_min_chars}자 이상"}}
  ],
  "closing":"마무리 멘트",
  "hashtags":["#해시태그", "..."]
}}

내용 구성 가이드
- intro는 200~300자  공감 후킹  오늘 기준 한 줄  저장 강요 금지
- h2 1은 감정  생각  가치관  왜 이 요리가 위로가 되는지  공감 중심
- h2 2는 재료 준비 느낌  손질하면서 생기는 생각  맛 포인트  재료 목록은 따로 표시하지 말고  아래에서 코드가 붙일 거라 자연스럽게 연결 문장만
- h2 3는 만드는 흐름  실패 포인트  상태 기준  steps를 기반으로 이야기하되  번호 없이 자연스럽게 흐름으로
- closing은 다정하게  다음에 해먹을 때 기억할 한 줄 포함
- 해시태그는 음식 관련으로만  최대 12개

""".strip()

    resp = _openai_call_with_retry(
        client=client,
        model=model,
        instructions=instructions,
        input_text=json.dumps(payload, ensure_ascii=False),
        max_retries=max_retries,
        debug=debug,
    )

    obj = _json_from_text(resp.output_text or "")

    # 간단 검증/보정
    intro = _strip_forbidden_punct(str(obj.get("intro") or "").strip())
    closing = _strip_forbidden_punct(str(obj.get("closing") or "").strip())
    h2 = obj.get("h2") or []
    hashtags = obj.get("hashtags") or []

    # 금지어/영어/반말 방지(간단)
    full_text = intro + " " + closing + " " + json.dumps(h2, ensure_ascii=False)
    if _contains_banned(full_text):
        raise RuntimeError("banned_topic_detected")
    if _has_english(full_text):
        raise RuntimeError("english_detected")

    # 길이 체크(너무 짧으면 다시 생성)
    if not (cfg.naver.intro_min <= len(intro) <= cfg.naver.intro_max):
        raise RuntimeError("intro_length_bad")

    if not isinstance(h2, list) or len(h2) != 3:
        raise RuntimeError("h2_structure_bad")

    fixed_h2 = []
    for it in h2:
        t = _strip_forbidden_punct(str((it or {}).get("title") or "").strip())
        b = _strip_forbidden_punct(str((it or {}).get("body") or "").strip())
        if len(b) < cfg.naver.section_min_chars:
            raise RuntimeError("section_too_short")
        fixed_h2.append({"title": t, "body": b})

    fixed_tags: List[str] = []
    for x in hashtags:
        s = str(x or "").strip()
        s = s.replace("##", "#")
        if not s:
            continue
        if not s.startswith("#"):
            s = "#" + s
        s = re.sub(r"\s+", "", s)
        s = _strip_forbidden_punct(s)
        if _contains_banned(s):
            continue
        if len(s) >= 2:
            fixed_tags.append(s)
    fixed_tags = fixed_tags[:12]

    return {"intro": intro, "h2": fixed_h2, "closing": closing, "hashtags": fixed_tags}


# -----------------------------
# HTML builder (safe, readable, no broken tags)
# -----------------------------
def _to_html_text(s: str) -> str:
    # escape + 줄바꿈을 <br/><br/>로
    s = (s or "").strip()
    s = _strip_forbidden_punct(s)
    s = _html.escape(s)
    # 사용자가 원하는 "호흡"을 위해 빈 줄 강화
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.replace("\r\n", "\n")
    s = s.replace("\n\n", "<br/><br/>")
    s = s.replace("\n", "<br/>")
    return s


def _p(block: str, mb: int = 18) -> str:
    return f"<p style='margin:0 0 {mb}px 0;line-height:2.05;'>{block}</p>"


def _h2(title: str) -> str:
    t = _html.escape(_strip_forbidden_punct(title))
    return f"<h2 style='margin:30px 0 12px 0;'><b>{t}</b></h2>"


def build_ingredients_block(ingredients_ko: List[Dict[str, str]]) -> str:
    lines = []
    for it in ingredients_ko:
        name = (it.get("name_ko") or "").strip()
        mea = (it.get("measure_ko") or "").strip()
        if not name:
            continue
        line = name
        if mea:
            line = f"{name}   {mea}"
        line = _html.escape(_strip_forbidden_punct(line))
        # 불릿/점 같은 느낌 제거: 그냥 줄로만
        lines.append(f"<div style='margin:0 0 10px 0;'>{line}</div>")

    if not lines:
        return _p(_html.escape("재료는 집에 있는 걸로 무리 없이 맞춰보셔도 돼요"), mb=12)

    box = (
        "<div style='padding:14px 14px 6px 14px;background:#f7f7f7;border-radius:14px;'>"
        + "".join(lines)
        + "</div>"
    )
    return box


def build_method_block(steps_ko: List[str]) -> str:
    # 번호 없이, 문단으로만
    ps = []
    for s in steps_ko:
        s2 = _strip_forbidden_punct(s)
        if not s2:
            continue
        ps.append(_p(_to_html_text(s2), mb=16))
    if not ps:
        return _p(_html.escape("만드는 흐름은 재료 상태 보면서 천천히 따라가시면 돼요"), mb=12)
    return "<div>" + "".join(ps) + "</div>"


def build_final_html(
    title_ko: str,
    main_kw: str,
    body_json: Dict[str, Any],
    ingredients_ko: List[Dict[str, str]],
    steps_ko: List[str],
    image_url: str,
) -> str:
    intro = body_json["intro"]
    h2_list = body_json["h2"]
    closing = body_json["closing"]
    hashtags = body_json.get("hashtags") or []

    wrap_open = "<div style='font-size:16px;line-height:2.05;letter-spacing:-0.1px;'>"
    wrap_close = "</div>"

    parts: List[str] = []

    if image_url:
        img = _html.escape(image_url)
        alt = _html.escape(_strip_forbidden_punct(title_ko))
        parts.append(
            f"<p style='margin:0 0 18px 0;'><img src='{img}' alt='{alt}' "
            f"style='max-width:100%;height:auto;border-radius:14px;'/></p>"
        )

    parts.append(_p(_to_html_text(intro), mb=18))

    # h2_1
    parts.append(_h2(h2_list[0]["title"]))
    parts.append(_p(_to_html_text(h2_list[0]["body"]), mb=18))

    # h2_2 + 재료 박스 삽입
    parts.append(_h2(h2_list[1]["title"]))
    parts.append(_p(_to_html_text(h2_list[1]["body"]), mb=16))
    parts.append(_h2("재료 목록"))
    parts.append(build_ingredients_block(ingredients_ko))

    # h2_3 + 만드는 흐름 삽입
    parts.append(_h2(h2_list[2]["title"]))
    parts.append(_p(_to_html_text(h2_list[2]["body"]), mb=16))
    parts.append(_h2("만드는 흐름"))
    parts.append(build_method_block(steps_ko))

    # 마무리/태그는 무조건 맨 끝
    parts.append("<div style='height:14px;'></div>")
    parts.append(_p(f"<b>마무리</b><br/><br/>{_to_html_text(closing)}", mb=14))

    if hashtags:
        tag_line = " ".join(hashtags[:12])
        parts.append(_p(_html.escape(tag_line), mb=0))

    html_out = wrap_open + "".join(parts) + wrap_close

    # 깨진 태그 유사 문자열 제거(혹시라도)
    html_out = re.sub(r"<\s*=\s*['\"][^>]*>", "", html_out)

    return html_out


# -----------------------------
# Pick recipe with filters
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

        # 최소 정보 체크
        ings = cand.get("ingredients") or []
        steps = split_steps(cand.get("instructions") or "")
        if len(ings) < 6 or len(steps) < 4:
            continue

        # area 필터
        if prefer_areas:
            area = (cand.get("area") or "").strip().lower()
            if area and (area not in prefer_areas):
                continue

        # category 차단
        cat = (cand.get("category") or "").strip().lower()
        if cat and cat in block_cats:
            continue

        return cand

    raise RuntimeError("레시피를 가져오지 못했습니다(중복/필터/시도 횟수 초과)")


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    rand6 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))

    init_db(cfg.sqlite_path)

    recipe = pick_recipe(cfg)
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title", "") or "Daily Recipe"
    steps_en = split_steps(recipe.get("instructions") or "")
    ingredients_en = recipe.get("ingredients") or []

    client = OpenAI(api_key=cfg.openai.api_key)

    # 1) 번역(제목/재료/방법) 강제
    title_ko_recipe, ingredients_ko, steps_ko = translate_recipe_parts_to_ko(
        client=client,
        model=cfg.openai.model,
        recipe_title_en=recipe_title_en,
        ingredients_en=ingredients_en,
        steps_en=steps_en,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    # 메인 키워드: 사용자가 지정하면 그걸 우선, 아니면 레시피명(한글)
    main_kw = (cfg.naver.main_keyword or "").strip() or title_ko_recipe

    # 2) 제목: 4가지 후킹 중 랜덤 + 키워드 포함
    hook_title = build_hook_title(main_kw)
    wp_title = hook_title  # WP 제목은 이걸로

    # 3) 썸네일: Pexels 우선 -> 없으면 TheMealDB
    thumb_url = (recipe.get("thumb") or "").strip()
    hiq = pexels_search_food_image(cfg.media.pexels_api_key, title_ko_recipe, debug=cfg.run.debug)
    chosen_img = hiq or thumb_url

    # 4) 본문 JSON 생성(조건 미달/금지어/영어 있으면 재생성)
    body_json = None
    for attempt in range(3):
        try:
            body_json = generate_naver_body_json(
                client=client,
                model=cfg.openai.model,
                cfg=cfg,
                main_kw=main_kw,
                recipe_title_ko=title_ko_recipe,
                ingredients_ko=ingredients_ko,
                steps_ko=steps_ko,
                max_retries=cfg.run.openai_max_retries,
                debug=cfg.run.debug,
            )
            break
        except Exception as e:
            if cfg.run.debug:
                print("[BODY] regen:", attempt + 1, "reason:", repr(e))
            if attempt >= 2:
                raise

    assert body_json is not None

    # 해시태그 보강: 사용자가 넣은 키워드가 있어도 "음식 관련만" 최대한
    base_tags = ["#집밥", "#레시피", "#간단요리", "#오늘뭐먹지", "#저녁메뉴", "#혼밥", "#자취요리"]
    extra = [k.strip() for k in (cfg.naver.keywords_csv or "").split(",") if k.strip()]
    extra = [f"#{re.sub(r'\\s+', '', x)}" if not x.startswith("#") else x for x in extra]
    extra = [x for x in extra if not _contains_banned(x)]
    body_json["hashtags"] = (body_json.get("hashtags") or []) + base_tags + extra
    # 중복 제거
    uniq = []
    seen = set()
    for t in body_json["hashtags"]:
        tt = _strip_forbidden_punct(str(t or "").strip())
        if not tt:
            continue
        if not tt.startswith("#"):
            tt = "#" + tt
        if tt in seen:
            continue
        seen.add(tt)
        uniq.append(tt)
    body_json["hashtags"] = uniq[:12]

    # 5) HTML 조립(깨진 태그 방지)
    html_body = build_final_html(
        title_ko=wp_title,
        main_kw=main_kw,
        body_json=body_json,
        ingredients_ko=ingredients_ko,
        steps_ko=steps_ko,
        image_url=(chosen_img if cfg.run.embed_image_in_body else ""),
    )

    # 6) WP 업로드(항상 새 글)
    media_id: Optional[int] = None
    media_url: str = ""

    if cfg.run.upload_thumb and chosen_img:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, chosen_img, filename_hint=f"recipe-{date_str}-{rand6}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    slug = f"naverfeed-recipe-{date_str}-{rand6}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 미리보기")
        print("TITLE:", wp_title)
        print("SLUG:", slug)
        print(html_body[:1500] + ("\n...(truncated)" if len(html_body) > 1500 else ""))
        return

    post_id, link = wp_create_post(cfg.wp, wp_title, slug, html_body, featured_media=featured)
    save_post_row(cfg.sqlite_path, recipe_id, recipe_title_en, post_id, link)

    print("OK(created):", post_id, link)
    if cfg.run.debug:
        print("[META] recipe_id:", recipe_id, "| recipe_title_en:", recipe_title_en)


def main() -> None:
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
