# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle_FINAL.py (통합 완성본)
- TheMealDB 랜덤 레시피 수집
- (중요) 제목/재료/만드는법 먼저 한국어 번역(영어 잔존 방지)
- 네이버 홈피드형 글감 스타일로 본문 생성
  - 도입부 200~300자
  - 굵은 소제목 3개 (각 1500자 이상)
  - 총 2300자 이상
  - 친구에게 수다 떠는 존댓말
  - 마침표 없이 여백으로 호흡
  - 불릿/번호(1단계 등) 금지
- WordPress: 매 실행마다 새 글 발행(중복 실행해도 수정 X)
- 썸네일: PEXELS_API_KEY 있으면 고퀄 이미지 우선, 없으면 TheMealDB 썸네일
- SQLite: 발행 이력 저장(최근 레시피 중복 회피)

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

선택 env:
  - OPENAI_MODEL=gpt-4.1-mini   (기본)
  - OPENAI_MAX_RETRIES=3
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - RUN_SLOT=day|am|pm          (제목에만 반영)
  - NAVER_KEYWORDS="키워드1,키워드2"
  - NAVER_RANDOM_LEVEL=0~3      (문장 변주)
  - PREFER_AREAS="Korean,Japanese,Chinese" (TheMealDB area 필터)
  - FORCE_NEW=1                 (기본 1로 처리됨, 유지용)

고퀄 썸네일(선택):
  - PEXELS_API_KEY=...
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

import openai  # 예외 타입용
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
    run_slot: str = "day"
    dry_run: bool = False
    debug: bool = False
    # ✅ 항상 새 글 발행 기본
    force_new: bool = True
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    openai_max_retries: int = 3


@dataclass
class NaverStyleConfig:
    random_level: int = 2
    keywords_csv: str = ""
    prefer_areas: List[str] = field(default_factory=list)


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

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

    naver_random_level = max(0, min(3, _env_int("NAVER_RANDOM_LEVEL", 2)))
    naver_keywords = _env("NAVER_KEYWORDS", "")
    prefer_areas = _parse_str_list(_env("PREFER_AREAS", ""))

    # ✅ 사용자 요청: 중복 실행 시에도 수정이 아니라 "추가 발행"
    # force_new 기본 True 유지 (ENV FORCE_NEW=0으로만 끌 수 있게)
    force_new = not (_env("FORCE_NEW", "1").strip() in ("0", "false", "False"))

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
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            force_new=force_new,
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 20),
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
        ),
        naver=NaverStyleConfig(
            random_level=naver_random_level,
            keywords_csv=naver_keywords,
            prefer_areas=prefer_areas,
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
        pexels_api_key=_env("PEXELS_API_KEY", ""),
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
    print("[CFG] NAVER_RANDOM_LEVEL:", cfg.naver.random_level)
    print("[CFG] NAVER_KEYWORDS:", cfg.naver.keywords_csv or "(empty)")
    print("[CFG] PREFER_AREAS:", ",".join(cfg.naver.prefer_areas) if cfg.naver.prefer_areas else "(any)")
    print("[CFG] PEXELS_API_KEY:", ok(cfg.pexels_api_key))


# -----------------------------
# SQLite (history)
# -----------------------------
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_key TEXT UNIQUE,
  slot TEXT,
  recipe_id TEXT,
  recipe_title_en TEXT,
  wp_post_id INTEGER,
  wp_link TEXT,
  media_id INTEGER,
  media_url TEXT,
  created_at TEXT
)
"""

def init_db(path: str, debug: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(TABLE_SQL)
    con.commit()
    con.close()


def save_post_meta(
    path: str,
    run_key: str,
    slot: str,
    recipe_id: str,
    recipe_title_en: str,
    wp_post_id: int,
    wp_link: str,
    media_id: Optional[int],
    media_url: str,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(run_key, slot, recipe_id, recipe_title_en, wp_post_id, wp_link, media_id, media_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_key,
            slot,
            recipe_id,
            recipe_title_en,
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
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:800]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "thumb.jpg") -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=45)
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
        raise RuntimeError(f"WP media upload failed: {up.status_code} body={up.text[:800]}")

    data = up.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# TheMealDB fetch
# -----------------------------
def fetch_random_recipe() -> Dict[str, Any]:
    r = requests.get(THEMEALDB_RANDOM, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Recipe API failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("Recipe API returned empty meals")
    return _normalize_meal(meals[0])


def fetch_recipe_by_id(recipe_id: str) -> Dict[str, Any]:
    url = THEMEALDB_LOOKUP.format(id=recipe_id)
    r = requests.get(url, timeout=30)
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
# Pexels thumbnail (optional)
# -----------------------------
def pexels_search_image(pexels_api_key: str, query: str) -> Optional[str]:
    if not pexels_api_key or not query:
        return None
    try:
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": pexels_api_key}
        params = {"query": query, "per_page": 10, "orientation": "landscape"}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            return None
        j = r.json() or {}
        photos = j.get("photos") or []
        if not photos:
            return None
        # 가능한 한 큰 원본 우선
        cand = random.choice(photos)
        src = (cand.get("src") or {})
        return (src.get("large2x") or src.get("large") or src.get("original") or None)
    except Exception:
        return None


# -----------------------------
# OpenAI retry + quota detect
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
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(
                model=model,
                instructions=instructions,
                input=input_text,
            )
        except openai.RateLimitError as e:
            last_err = e
            if _is_insufficient_quota_error(e):
                raise
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print("[OPENAI] RateLimit retry", attempt, "sleep", f"{sleep_s:.2f}s")
            time.sleep(sleep_s)
        except openai.APIError as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print("[OPENAI] APIError retry", attempt, "sleep", f"{sleep_s:.2f}s")
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print("[OPENAI] UnknownError retry", attempt, "sleep", f"{sleep_s:.2f}s")
            time.sleep(sleep_s)
    raise last_err  # pragma: no cover


# -----------------------------
# Text cleanup for "Korean only" + no periods
# -----------------------------
def _has_english(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s or ""))


def _strip_bullets_and_numbers(s: str) -> str:
    s = s or ""
    # 앞 불릿/특문 제거
    s = re.sub(r"(?m)^\s*[•\-\*\u2022]+\s*", "", s)
    # 1단계 2단계 3단계 제거
    s = re.sub(r"\b[0-9]+\s*단계\b", "", s)
    # 1. 2. 같은 번호 제거
    s = re.sub(r"(?m)^\s*\d+\s*[\.\)]\s*", "", s)
    return s


def _remove_periods(s: str) -> str:
    # 사용자가 원하는 스타일: 마침표 없이
    s = s or ""
    s = s.replace(".", " ")
    s = s.replace("。", " ")
    s = s.replace("!", " ")
    s = s.replace("?", " ")
    s = s.replace("…", " ")
    # 공백 정리
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _dedupe_repeated_sentences(s: str) -> str:
    """
    반복 문장 버그 방지
    - 같은 문장이 연속으로 반복되면 1개만 남김
    """
    if not s:
        return ""
    parts = [p.strip() for p in re.split(r"\n{2,}", s) if p.strip()]
    out = []
    prev = None
    for p in parts:
        key = re.sub(r"\s+", " ", p)
        if prev is not None and key == prev:
            continue
        out.append(p)
        prev = key
    return "\n\n".join(out).strip()


def _korean_only_guard(s: str) -> str:
    # 영어 단어 덩어리 제거(최후 안전장치)
    s = s or ""
    s = re.sub(r"[A-Za-z]{2,}", "", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


# -----------------------------
# Translate recipe parts to Korean (JSON output)
# -----------------------------
def translate_recipe_to_ko(
    client: OpenAI,
    model: str,
    recipe: Dict[str, Any],
    max_retries: int,
    debug: bool = False,
) -> Dict[str, Any]:
    payload = {
        "title_en": recipe.get("title", ""),
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients_en": [{"name": it.get("name", ""), "measure": it.get("measure", "")} for it in (recipe.get("ingredients") or [])],
        "steps_en": split_steps(recipe.get("instructions", "")),
    }

    instructions = (
        "너는 번역가다\n"
        "반드시 한국어로만 번역한다\n"
        "출력은 JSON만\n"
        "영어 단어를 남기지 않는다\n"
        "불릿 특수문자 1단계 2단계 번호를 만들지 않는다\n"
        "재료명은 자연스럽게 한국어로 풀고 분량은 의미만 유지한다\n"
        "형식은 아래 키를 유지한다\n"
        "{\"title_ko\":\"...\",\"category_ko\":\"...\",\"area_ko\":\"...\","
        "\"ingredients_ko\":[{\"name_ko\":\"...\",\"measure_ko\":\"...\"}],"
        "\"steps_ko\":[\"...\",...]}\n"
    )

    resp = _openai_call_with_retry(
        client=client,
        model=model,
        instructions=instructions,
        input_text=json.dumps(payload, ensure_ascii=False),
        max_retries=max_retries,
        debug=debug,
    )

    txt = (resp.output_text or "").strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)

    # JSON only parse
    left = txt.find("{")
    right = txt.rfind("}")
    if left == -1 or right == -1:
        raise RuntimeError("번역 JSON 파싱 실패")

    obj = json.loads(txt[left:right + 1])

    # cleanup
    obj["title_ko"] = _korean_only_guard(_strip_bullets_and_numbers(str(obj.get("title_ko", ""))))
    obj["category_ko"] = _korean_only_guard(_strip_bullets_and_numbers(str(obj.get("category_ko", ""))))
    obj["area_ko"] = _korean_only_guard(_strip_bullets_and_numbers(str(obj.get("area_ko", ""))))

    ing = obj.get("ingredients_ko") or []
    clean_ing = []
    for it in ing:
        n = _korean_only_guard(_strip_bullets_and_numbers(str(it.get("name_ko", ""))))
        m = _korean_only_guard(_strip_bullets_and_numbers(str(it.get("measure_ko", ""))))
        n = _remove_periods(n)
        m = _remove_periods(m)
        if n:
            clean_ing.append({"name_ko": n, "measure_ko": m})
    obj["ingredients_ko"] = clean_ing

    steps = obj.get("steps_ko") or []
    clean_steps = []
    for s in steps:
        s = _korean_only_guard(_strip_bullets_and_numbers(str(s)))
        s = _remove_periods(s)
        if s:
            clean_steps.append(s)
    obj["steps_ko"] = clean_steps

    # hard guard
    for k in ("title_ko", "category_ko", "area_ko"):
        if _has_english(obj.get(k, "")):
            obj[k] = _korean_only_guard(obj[k])

    return obj


# -----------------------------
# Title hook templates (random)
# -----------------------------
def build_hook_title(main_kw: str, random_level: int) -> str:
    kw = (main_kw or "").strip() or "레시피"
    benefit = [
        f"{kw}   오늘 이 조합만 알면 맛이 안정돼요",
        f"{kw}   이 기준 하나로 실패 확률이 확 줄어요",
        f"{kw}   힘 빼고 해도 맛이 살아나는 포인트",
    ]
    threat = [
        f"{kw}   여기서 갈리면 맛이 한 번에 무너져요",
        f"{kw}   이 지점 놓치면 간이 갑자기 튀어요",
        f"{kw}   이 순서만 헷갈리면 쉽게 흐트러져요",
    ]
    curious = [
        f"{kw}   왜 똑같이 해도 맛이 다를까요",
        f"{kw}   비슷한데 결과가 갈리는 이유가 있어요",
        f"{kw}   이 부분이 은근히 제일 중요하더라고요",
    ]
    compare = [
        f"{kw}   센불보다 중요한 건 재료 상태예요",
        f"{kw}   레시피보다 먼저 잡아야 하는 기준",
        f"{kw}   과한 욕심 빼면 더 맛있어져요",
    ]

    pools = [benefit, threat, curious, compare]
    # 레벨 높을수록 섞임 커짐
    pool = random.choice(pools) if random_level >= 2 else benefit
    t = random.choice(pool)
    # 너무 길면 살짝 줄이기
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


# -----------------------------
# Body generation (naver homefeed style, long form)
# -----------------------------
def generate_naver_homefeed_article(
    client: OpenAI,
    model: str,
    recipe_ko: Dict[str, Any],
    keywords_csv: str,
    random_level: int,
    slot_label: str,
    max_retries: int,
    debug: bool = False,
) -> Dict[str, str]:
    """
    output JSON:
      title
      intro
      s1_title, s1_body
      s2_title, s2_body
      s3_title, s3_body
      closing
    """
    kws = [k.strip() for k in (keywords_csv or "").split(",") if k.strip()]
    main_kw = recipe_ko.get("title_ko") or "레시피"

    # seeds (변주용)
    seeds_open = [
        "오늘은 막 열심히 하기보다  실패가 덜 나는 기준만 딱 잡고 가보려고 해요",
        "요리는 재능보다  흐름이더라고요  특히 이 메뉴는 기준만 알면 편해져요",
        "저는 레시피 볼 때  제일 먼저 실패 포인트부터 찾게 되더라고요  그걸 먼저 잡아볼게요",
        "완벽하게 하려는 순간 오히려 어긋날 때가 있잖아요  오늘은 힘을 조금 빼볼게요",
    ]
    seeds_value = [
        "요리는 결국 내 컨디션에 맞게 반복 가능한 방식이 제일 오래 가더라고요",
        "맛있게 먹는 것도 좋지만  마음이 덜 흔들리게 만드는 게 더 중요할 때가 있어요",
        "나는 왜 요리할 때 조급해질까  이런 생각을 하다 보면  기준이 보이더라고요",
    ]
    seed_open = random.choice(seeds_open)
    seed_value = random.choice(seeds_value)

    payload = {
        "recipe": {
            "title_ko": recipe_ko.get("title_ko", ""),
            "category_ko": recipe_ko.get("category_ko", ""),
            "area_ko": recipe_ko.get("area_ko", ""),
            "ingredients_ko": recipe_ko.get("ingredients_ko", []),
            "steps_ko": recipe_ko.get("steps_ko", []),
        },
        "style": {
            "must_polite": True,
            "chatty_friend_polite": True,
            "no_periods": True,
            "breathing_with_spaces": True,
            "no_bullets": True,
            "no_step_numbers": True,
            "intro_chars": "200~300",
            "section_min_chars_each": 1500,
            "total_min_chars": 2300,
            "slot_label": slot_label,
            "random_level": random_level,
            "keywords_hint": kws[:10],
            "seed_open": seed_open,
            "seed_value": seed_value,
        },
    }

    instructions = """
너는 네이버 홈피드에 올리는 글을 쓰는 사람이다
친구에게 진심으로 수다 떠는 느낌의 존댓말로만 작성한다

절대 반말 금지
절대 영어 금지
절대 마침표 사용 금지
문장 호흡은 띄어쓰기와 줄바꿈으로 만든다
불릿 기호 점 찍기 금지
1단계 2단계 같은 번호 표기 금지
제공된 재료와 과정 의미는 유지하고  시간을 온도처럼 숫자를 단정하지 않는다
체험담 느낌은 내되  과장 광고처럼 보이게 하지 않는다

출력은 JSON만
키는 아래 고정
{
 "title":"...",
 "intro":"...",
 "s1_title":"...",
 "s1_body":"...",
 "s2_title":"...",
 "s2_body":"...",
 "s3_title":"...",
 "s3_body":"...",
 "closing":"..."
}

형식 요구
intro는 200~300자
s1_body s2_body s3_body는 각각 1500자 이상
전체 글은 2300자 이상

s2_body 안에 반드시 재료 목록을 포함한다
재료 목록은 줄바꿈으로만 정리하고  불릿이나 번호는 쓰지 않는다

s3_body 안에 만드는 흐름을 포함한다
만드는 흐름도 줄바꿈으로만 구분하고  번호나 1단계 같은 표현은 쓰지 않는다

마지막 closing은 2~4문장 분량인데  마침표는 쓰지 않는다
""".strip()

    resp = _openai_call_with_retry(
        client=client,
        model=model,
        instructions=instructions,
        input_text=json.dumps(payload, ensure_ascii=False),
        max_retries=max_retries,
        debug=debug,
    )

    txt = (resp.output_text or "").strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)

    left = txt.find("{")
    right = txt.rfind("}")
    if left == -1 or right == -1:
        raise RuntimeError("본문 JSON 파싱 실패")

    obj = json.loads(txt[left:right + 1])

    # cleanup + guards
    for k in ["title","intro","s1_title","s1_body","s2_title","s2_body","s3_title","s3_body","closing"]:
        v = str(obj.get(k, "") or "")
        v = _strip_bullets_and_numbers(v)
        v = _dedupe_repeated_sentences(v)
        v = _remove_periods(v)
        v = _korean_only_guard(v)
        obj[k] = v

    # ✅ 최후: 영어 남으면 재시도 유도할 수 있도록 체크
    for k in ["title","intro","s1_title","s1_body","s2_title","s2_body","s3_title","s3_body","closing"]:
        if _has_english(obj.get(k, "")):
            obj[k] = _korean_only_guard(obj[k])

    # 강제 훅 제목(사용자 요구)  title이 밋밋할 때 보정
    if len(obj.get("title","")) < 10 or _has_english(obj.get("title","")):
        obj["title"] = build_hook_title(main_kw, random_level)

    return obj


def _strip_html_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def _count_chars_plain(s: str) -> int:
    s = (s or "").strip()
    return len(s)


def ensure_article_constraints(
    client: OpenAI,
    model: str,
    recipe_ko: Dict[str, Any],
    keywords_csv: str,
    random_level: int,
    slot_label: str,
    max_retries: int,
    debug: bool = False,
) -> Dict[str, str]:
    """
    조건 충족될 때까지 (최대 3회) 재생성
    """
    last = None
    for attempt in range(3):
        obj = generate_naver_homefeed_article(
            client=client,
            model=model,
            recipe_ko=recipe_ko,
            keywords_csv=keywords_csv,
            random_level=random_level,
            slot_label=slot_label,
            max_retries=max_retries,
            debug=debug,
        )
        last = obj

        intro_len = _count_chars_plain(obj.get("intro",""))
        s1_len = _count_chars_plain(obj.get("s1_body",""))
        s2_len = _count_chars_plain(obj.get("s2_body",""))
        s3_len = _count_chars_plain(obj.get("s3_body",""))
        total_len = _count_chars_plain(
            "\n".join([obj.get("intro",""), obj.get("s1_body",""), obj.get("s2_body",""), obj.get("s3_body",""), obj.get("closing","")])
        )

        ok_intro = (200 <= intro_len <= 320)
        ok_sections = (s1_len >= 1500 and s2_len >= 1500 and s3_len >= 1500)
        ok_total = (total_len >= 2300)

        no_english = not any(_has_english(obj.get(k,"")) for k in obj.keys())

        if debug:
            print("[CHECK] attempt", attempt, "intro", intro_len, "s1", s1_len, "s2", s2_len, "s3", s3_len, "total", total_len, "no_english", no_english)

        if ok_intro and ok_sections and ok_total and no_english:
            return obj

    if last is None:
        raise RuntimeError("본문 생성 실패")
    return last


# -----------------------------
# HTML rendering (readability)
# -----------------------------
def _html_escape(s: str) -> str:
    s = s or ""
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return s


def _to_paragraphs(text: str) -> str:
    """
    마침표 없이 줄바꿈으로 호흡
    - 빈 줄 단위로 문단 만들고  문단 내부는 <br/><br/>로
    """
    text = (text or "").strip()
    text = re.sub(r"\r\n", "\n", text)
    chunks = [c.strip() for c in re.split(r"\n{2,}", text) if c.strip()]
    html_parts = []
    for c in chunks:
        c = _html_escape(c)
        c = c.replace("\n", "<br/><br/>")
        html_parts.append(f"<p style='margin:0 0 18px 0;line-height:2.05;font-size:16px;'>{c}</p>")
    return "\n".join(html_parts)


def _h2_bold(title: str) -> str:
    title = _html_escape(title)
    return f"<h2 style='margin:28px 0 14px 0;line-height:1.4;font-size:20px;'><b>{title}</b></h2>"


def _tag_line(keywords_csv: str, main_kw: str) -> str:
    kws = []
    if main_kw:
        kws.append(main_kw.strip())
    for k in (keywords_csv or "").split(","):
        k = k.strip()
        if k and k not in kws:
            kws.append(k)
    kws = kws[:12]
    if not kws:
        return ""
    # 불릿 없이  #만
    line = " ".join([f"#{k.replace(' ', '')}" for k in kws if k])
    return f"<div style='margin:18px 0 0 0;opacity:.78;font-size:14px;line-height:1.9;'>태그  { _html_escape(line) }</div>"


def closing_random(main_kw: str) -> str:
    kw = (main_kw or "").strip() or "이 레시피"
    pool = [
        f"오늘은 {kw}를 완벽하게 하려는 것보다  실패 확률만 줄이는 기준으로 편하게 잡아봤어요  이렇게 해두면 다음엔 마음이 훨씬 가벼워지실 거예요",
        f"{kw}는 한 번 기준만 잡히면  그 다음부터는 손이 빨라지더라고요  다음번엔 내 입맛에 맞게 한 포인트만 살짝 조절해보셔도 좋겠어요",
        f"요리는 정답이 있는 것 같아도  결국 내 컨디션에 맞게 반복 가능한 게 제일 오래 가더라고요  {kw}도 그렇게 편하게 가져가시면 돼요",
        f"오늘처럼 정리해두면  다음에 메뉴 고민할 때도 바로 꺼내 쓰기 좋아요  {kw}는 특히 여기서 갈리니까  그 부분만 기억해두셔도 충분해요",
    ]
    return random.choice(pool)


def render_html(article: Dict[str, str], main_kw: str, keywords_csv: str, img_url: str = "") -> str:
    wrap_open = "<div style='max-width:740px;margin:0 auto;padding:18px 16px;line-height:2.05;letter-spacing:0.01em;'>"
    wrap_close = "</div>"

    parts = [wrap_open]

    if img_url:
        parts.append(
            f"<p style='margin:0 0 18px 0;'><img src='{_html_escape(img_url)}' alt='{_html_escape(main_kw)}' "
            "style='max-width:100%;height:auto;border-radius:16px;'/></p>"
        )

    parts.append(_to_paragraphs(article.get("intro","")))

    parts.append(_h2_bold(article.get("s1_title","")))
    parts.append(_to_paragraphs(article.get("s1_body","")))

    parts.append(_h2_bold(article.get("s2_title","")))
    parts.append(_to_paragraphs(article.get("s2_body","")))

    parts.append(_h2_bold(article.get("s3_title","")))
    parts.append(_to_paragraphs(article.get("s3_body","")))

    # ✅ 마무리 항상 맨 아래
    parts.append(_h2_bold("마무리"))
    closing = article.get("closing","") or closing_random(main_kw)
    parts.append(_to_paragraphs(closing))

    parts.append(_tag_line(keywords_csv, main_kw))

    parts.append("<p style='opacity:.55;font-size:13px;margin:20px 0 0 0;line-height:1.8;'>참고  공개 레시피 정보를 바탕으로 보기 좋게 정리한 글입니다</p>")
    parts.append(wrap_close)

    return "\n".join(parts)


# -----------------------------
# Pick recipe (avoid repeats)
# -----------------------------
def pick_recipe(cfg: AppConfig) -> Dict[str, Any]:
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

    # 그래도 못 찾으면 중복 허용
    return fetch_random_recipe()


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else ("오후" if slot == "pm" else "오늘")

    init_db(cfg.sqlite_path, debug=cfg.run.debug)

    recipe = pick_recipe(cfg)
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title", "") or "Daily Recipe"

    client = OpenAI(api_key=cfg.openai.api_key)

    # 1) 한국어 번역(영어 잔존 방지)
    recipe_ko = translate_recipe_to_ko(
        client=client,
        model=cfg.openai.model,
        recipe=recipe,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    main_kw = recipe_ko.get("title_ko") or "레시피"

    # 2) 고퀄 썸네일 우선(Pexels), 없으면 TheMealDB
    thumb_url = (recipe.get("thumb") or "").strip()
    pexels_img = pexels_search_image(cfg.pexels_api_key, query=main_kw) if cfg.pexels_api_key else None
    chosen_img_url = pexels_img or thumb_url

    # 3) 본문 생성(조건 검수 포함)
    article = ensure_article_constraints(
        client=client,
        model=cfg.openai.model,
        recipe_ko=recipe_ko,
        keywords_csv=cfg.naver.keywords_csv,
        random_level=cfg.naver.random_level,
        slot_label=slot_label,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    # 4) 제목 훅 템플릿 보정(항상 한글)
    hook_title = build_hook_title(main_kw, cfg.naver.random_level)
    title_core = article.get("title") or hook_title
    title_core = _korean_only_guard(_remove_periods(_strip_bullets_and_numbers(title_core)))
    if _has_english(title_core) or len(title_core) < 10:
        title_core = hook_title

    final_title = f"{date_str} {slot_label}  {title_core}"
    final_title = re.sub(r"\s{2,}", " ", final_title).strip()

    # 5) 미디어 업로드
    media_id: Optional[int] = None
    media_url: str = ""

    if cfg.run.upload_thumb and chosen_img_url:
        try:
            media_id, media_url = wp_upload_media(
                cfg.wp,
                chosen_img_url,
                filename_hint=f"naverstyle-{date_str}-{slot}-{random.randint(1000,9999)}.jpg",
            )
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    # 6) HTML 렌더링(가독성)
    img_for_body = (media_url or chosen_img_url) if cfg.run.embed_image_in_body else ""
    body_html = render_html(article, main_kw=main_kw, keywords_csv=cfg.naver.keywords_csv, img_url=img_for_body)

    # 7) 항상 새 글 발행(중복 실행해도 수정 X)
    # slug는 항상 유니크하게
    rand_suffix = now.strftime("%H%M%S") + "-" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    slug = f"daily-recipe-{date_str}-{slot}-{rand_suffix}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", final_title)
        print("SLUG:", slug)
        print(body_html[:2000] + ("\n...(truncated)" if len(body_html) > 2000 else ""))
        return

    new_id, wp_link = wp_create_post(cfg.wp, final_title, slug, body_html, featured_media=featured)

    run_key = f"{date_str}-{slot}-{rand_suffix}"
    save_post_meta(cfg.sqlite_path, run_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)

    print("OK(created):", new_id, wp_link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
