# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle_FINAL.py
- TheMealDB 랜덤 레시피 수집
- 제목: 4가지 후킹 유형(이득/위협/궁금/비교) 중 랜덤 + 키워드 포함
- 재료/만드는법/본문/제목: 한국어 번역 강제 (영어 잔존 검사 -> 재번역/정리)
- 네이버 홈피드형 글감: 도입 200~300자 + 굵은 소제목 3개(각 1500자 이상 목표) + 총 길이 충분히 길게
- 친구에게 수다떠는 존댓말 / 마침표(.) 최소화(후처리로 제거) / 행간 여백 크게
- 재료 목록: 불릿 특문 없음
- 만드는법: 단계번호(1단계, 2단계 등) 금지
- WordPress: 매번 새 글 발행(기존글 수정 X)
- 썸네일: PEXELS_API_KEY 있으면 고퀄 이미지 우선, 없으면 TheMealDB 썸네일 사용
- SQLite: 중복 회피용(선택) 캐시 저장 (actions/cache로 유지 가능)

필수 env (GitHub Secrets 권장):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY

권장 env:
  - WP_STATUS=publish
  - WP_CATEGORY_IDS="7"
  - WP_TAG_IDS="1,2,3"
  - SQLITE_PATH=data/daily_recipe.sqlite3

네이버 스타일/키워드:
  - NAVER_MAIN_KEYWORD="청년도약계좌" (선택)  # 제목 키워드 강제, 비우면 레시피명 사용
  - NAVER_KEYWORDS="키워드1,키워드2,..." (선택) # 본문 하단 해시태그 라인

기타:
  - RUN_SLOT=day|am|pm (기본 day)
  - FORCE_NEW=1 (기본 1)  # 항상 새 글
  - DRY_RUN=0|1
  - DEBUG=0|1
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - OPENAI_MODEL=gpt-4.1-mini (기본)
  - OPENAI_MAX_RETRIES=3

이미지(선택):
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
import html as _html
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import openai
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


def _now_kst() -> datetime:
    return datetime.now(tz=KST)


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
    force_new: bool = True    # 항상 새 글
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    openai_max_retries: int = 3


@dataclass
class NaverStyleConfig:
    main_keyword: str = ""      # 제목 키워드 강제 (비우면 레시피명)
    keywords_csv: str = ""      # 해시태그 힌트
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

    cfg_run = RunConfig(
        run_slot=run_slot,
        dry_run=_env_bool("DRY_RUN", False),
        debug=_env_bool("DEBUG", False),
        force_new=_env_bool("FORCE_NEW", True),  # ✅ 기본 True
        avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
        max_tries=_env_int("MAX_TRIES", 20),
        upload_thumb=_env_bool("UPLOAD_THUMB", True),
        set_featured=_env_bool("SET_FEATURED", True),
        embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
    )

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

    main_kw = _env("NAVER_MAIN_KEYWORD", "")
    keywords_csv = _env("NAVER_KEYWORDS", "")
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
            main_keyword=main_kw,
            keywords_csv=keywords_csv,
            prefer_areas=prefer_areas,
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", int(cfg.run.dry_run), "| DEBUG:", int(cfg.run.debug))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model, "| OPENAI_KEY:", ok(cfg.openai.api_key))
    print("[CFG] NAVER_MAIN_KEYWORD:", cfg.naver.main_keyword or "(auto)")
    print("[CFG] NAVER_KEYWORDS:", cfg.naver.keywords_csv or "(empty)")
    print("[CFG] PEXELS_API_KEY:", ok(cfg.pexels_api_key))


# -----------------------------
# SQLite (history)  (v2, 여러 글 허용)
# -----------------------------
DB_SQL = """
CREATE TABLE IF NOT EXISTS daily_posts_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  recipe_id TEXT,
  recipe_title_en TEXT,
  recipe_title_ko TEXT,
  wp_post_id INTEGER,
  wp_link TEXT
)
"""


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(DB_SQL)
    con.commit()
    con.close()


def save_post_meta_v2(
    path: str,
    recipe_id: str,
    recipe_title_en: str,
    recipe_title_ko: str,
    wp_post_id: int,
    wp_link: str,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO daily_posts_v2(created_at, recipe_id, recipe_title_en, recipe_title_ko, wp_post_id, wp_link) VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            recipe_id or "",
            recipe_title_en or "",
            recipe_title_ko or "",
            int(wp_post_id),
            wp_link or "",
        ),
    )
    con.commit()
    con.close()


def get_recent_recipe_ids(path: str, days: int) -> List[str]:
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute("SELECT recipe_id FROM daily_posts_v2 WHERE created_at >= ? AND recipe_id IS NOT NULL", (cutoff,))
        rows = cur.fetchall()
        con.close()
        return [str(r[0]) for r in rows if r and r[0]]
    except Exception:
        return []


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "naverstyle-recipe-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str, featured_media: Optional[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": cfg.status,
    }
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=45)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:600]}")
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
        raise RuntimeError(f"WP media upload failed: {up.status_code} body={up.text[:600]}")

    data = up.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# TheMealDB
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
        "title_en": title,
        "category_en": category,
        "area_en": area,
        "instructions_en": instructions,
        "ingredients_en": ingredients,
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
# Pexels thumbnail (고퀄)
# -----------------------------
def pexels_pick_photo_url(api_key: str, query: str, debug: bool = False) -> Optional[str]:
    if not api_key:
        return None
    q = (query or "").strip()
    if not q:
        return None
    try:
        headers = {"Authorization": api_key}
        params = {"query": q, "per_page": 10, "orientation": "landscape"}
        r = requests.get(PEXELS_SEARCH, headers=headers, params=params, timeout=25)
        if r.status_code != 200:
            if debug:
                print("[PEXELS] non-200:", r.status_code, r.text[:200])
            return None
        j = r.json()
        photos = j.get("photos") or []
        if not photos:
            return None
        # 랜덤 1개
        p = random.choice(photos)
        src = (p.get("src") or {})
        # 고해상도 우선
        for k in ("large2x", "original", "large", "medium"):
            u = (src.get(k) or "").strip()
            if u:
                return u
        return None
    except Exception as e:
        if debug:
            print("[PEXELS] failed:", repr(e))
        return None


# -----------------------------
# OpenAI helpers
# -----------------------------
def _is_insufficient_quota_error(e: Exception) -> bool:
    s = (repr(e) or "") + " " + (str(e) or "")
    s = s.lower()
    return ("insufficient_quota" in s) or ("exceeded your current quota" in s) or ("check your plan and billing" in s)


def _openai_call_with_retry(client: OpenAI, model: str, instructions: str, input_text: str, max_retries: int, debug: bool = False):
    last = None
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(model=model, instructions=instructions, input=input_text)
        except Exception as e:
            last = e
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print("[OPENAI] retry in", sleep_s, "err=", repr(e))
            time.sleep(sleep_s)
    raise last


def _has_english(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s or ""))


def _strip_periods(s: str) -> str:
    # 사용자가 말한 “마침표” 최소화: '.' 제거
    return (s or "").replace(".", "").replace("。", "").strip()


def _remove_bullets_and_stepnums(s: str) -> str:
    # 불릿/특문 제거 + "1단계"류 제거
    s = re.sub(r"[•·●◦▪▫■□▶►➤-]\s*", "", s)
    s = re.sub(r"\b\d+\s*단계\b", "", s)
    s = re.sub(r"\bstep\s*\d+\b", "", s, flags=re.I)
    s = re.sub(r"\b\d+\)\s*", "", s)
    s = re.sub(r"^\s*\d+\s*[).:-]\s*", "", s, flags=re.M)
    return s.strip()


def _dedupe_lines(s: str) -> str:
    lines = [ln.rstrip() for ln in (s or "").splitlines()]
    out = []
    seen = set()
    for ln in lines:
        key = re.sub(r"\s+", " ", ln.strip())
        if not key:
            out.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    # 연속 공백줄 3개 이상이면 2개로 줄임
    merged = "\n".join(out)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()


def _polite_enough(s: str) -> bool:
    # 아주 단순 체크: "요"가 너무 적으면 반말 가능성
    t = (s or "")
    yo = len(re.findall(r"요(\s|$)", t))
    seumnida = len(re.findall(r"습니다(\s|$)", t))
    return (yo + seumnida) >= 25


# -----------------------------
# Title hooks (4 types random)
# -----------------------------
def make_hook_title(keyword: str) -> str:
    kw = (keyword or "").strip()
    if not kw:
        kw = "오늘 메뉴"

    benefit = [
        f"{kw}  이번에 이 기준 하나 잡으니까 맛이 달라지더라고요",
        f"{kw}  괜히 어렵게 느껴졌는데  의외로 이 포인트가 전부였어요",
        f"{kw}  한 번만 이렇게 해두면  다음부터는 편하게 갑니다",
    ]
    threat = [
        f"{kw}  여기서 흔들리면  전체가 밍밍해지는 경우가 많아요",
        f"{kw}  이 타이밍 놓치면  식감이 확 달라져서 아쉬워지더라고요",
        f"{kw}  대충 넘어가면  마지막에 간이 안 잡히는 패턴이 자주 나와요",
    ]
    curious = [
        f"{kw}  왜 똑같이 했는데 결과가 갈릴까요  딱 이 지점이에요",
        f"{kw}  평소에 애매했던 포인트  오늘은 기준으로 정리해볼게요",
        f"{kw}  어렵게 느껴지는 이유가 있더라고요  그 부분만 풀어볼게요",
    ]
    compare = [
        f"{kw}  대충 따라가기 vs 기준 잡고 하기  차이가 은근 크게 납니다",
        f"{kw}  같은 재료인데도  순서만 바꾸면 결과가 달라져요",
        f"{kw}  한 번은 급하게  한 번은 기준대로  뭐가 달랐을까요",
    ]

    buckets = [benefit, threat, curious, compare]
    pick_bucket = random.choice(buckets)
    title = random.choice(pick_bucket)

    # 마침표 제거
    title = _strip_periods(title)
    title = _remove_bullets_and_stepnums(title)
    return title


# -----------------------------
# Translate recipe parts (JSON output)
# -----------------------------
def translate_recipe_to_korean(
    client: OpenAI,
    model: str,
    recipe: Dict[str, Any],
    max_retries: int,
    debug: bool = False
) -> Dict[str, Any]:
    """
    return:
      {
        "title_ko": "...",
        "category_ko": "...",
        "area_ko": "...",
        "ingredients_ko": [{"name_ko":"...","measure_ko":"..."}],
        "steps_ko": ["...","..."]
      }
    """
    steps_en = split_steps(recipe.get("instructions_en", ""))
    payload = {
        "title_en": recipe.get("title_en", ""),
        "category_en": recipe.get("category_en", ""),
        "area_en": recipe.get("area_en", ""),
        "ingredients_en": recipe.get("ingredients_en", []),
        "steps_en": steps_en,
        "rules": {
            "korean_only": True,
            "no_bullets": True,
            "no_step_numbers": True,
            "no_periods": True,
        },
    }

    instructions = (
        "너는 번역가다\n"
        "반드시 한국어로만 번역한다\n"
        "영어 단어를 남기지 않는다\n"
        "불릿 기호를 사용하지 않는다\n"
        "1단계 2단계 같은 단계 번호를 만들지 않는다\n"
        "마침표 기호를 쓰지 않는다\n"
        "출력은 JSON만\n"
        "{\"title_ko\":\"...\",\"category_ko\":\"...\",\"area_ko\":\"...\",\"ingredients_ko\":[{\"name_ko\":\"...\",\"measure_ko\":\"...\"}],\"steps_ko\":[\"...\",...]}\n"
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
    obj = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])

    # 후처리: 마침표/불릿/단계번호 제거 + 영어 잔존 시 정리
    obj["title_ko"] = _remove_bullets_and_stepnums(_strip_periods(obj.get("title_ko", "")))
    obj["category_ko"] = _remove_bullets_and_stepnums(_strip_periods(obj.get("category_ko", "")))
    obj["area_ko"] = _remove_bullets_and_stepnums(_strip_periods(obj.get("area_ko", "")))

    ings = obj.get("ingredients_ko") or []
    clean_ings = []
    for it in ings:
        n = _remove_bullets_and_stepnums(_strip_periods(it.get("name_ko", "")))
        m = _remove_bullets_and_stepnums(_strip_periods(it.get("measure_ko", "")))
        if n:
            clean_ings.append({"name_ko": n, "measure_ko": m})
    obj["ingredients_ko"] = clean_ings

    steps = obj.get("steps_ko") or []
    clean_steps = []
    for s in steps:
        s = _remove_bullets_and_stepnums(_strip_periods(str(s)))
        if s:
            clean_steps.append(s)
    obj["steps_ko"] = clean_steps

    # 영어 잔존 검사: 있으면 한 번 더 번역 시도
    joined = json.dumps(obj, ensure_ascii=False)
    if _has_english(joined):
        if debug:
            print("[KO_CHECK] english remains -> re-translate once")
        payload["retry_note"] = "영어가 남았다  영어를 절대 남기지 말고 다시 번역"
        resp2 = _openai_call_with_retry(client, model, instructions, json.dumps(payload, ensure_ascii=False), max_retries, debug)
        txt2 = (resp2.output_text or "").strip()
        txt2 = re.sub(r"^```(?:json)?\s*", "", txt2)
        txt2 = re.sub(r"\s*```$", "", txt2)
        obj2 = json.loads(txt2[txt2.find("{"):txt2.rfind("}") + 1])

        # 동일 후처리
        obj2["title_ko"] = _remove_bullets_and_stepnums(_strip_periods(obj2.get("title_ko", "")))
        obj2["category_ko"] = _remove_bullets_and_stepnums(_strip_periods(obj2.get("category_ko", "")))
        obj2["area_ko"] = _remove_bullets_and_stepnums(_strip_periods(obj2.get("area_ko", "")))

        ings = obj2.get("ingredients_ko") or []
        clean_ings = []
        for it in ings:
            n = _remove_bullets_and_stepnums(_strip_periods(it.get("name_ko", "")))
            m = _remove_bullets_and_stepnums(_strip_periods(it.get("measure_ko", "")))
            if n:
                clean_ings.append({"name_ko": n, "measure_ko": m})
        obj2["ingredients_ko"] = clean_ings

        steps = obj2.get("steps_ko") or []
        clean_steps = []
        for s in steps:
            s = _remove_bullets_and_stepnums(_strip_periods(str(s)))
            if s:
                clean_steps.append(s)
        obj2["steps_ko"] = clean_steps

        obj = obj2

    return obj


# -----------------------------
# Article generation (네이버 홈피드형)
# -----------------------------
def generate_naver_homefeed_article(
    client: OpenAI,
    model: str,
    keyword: str,
    recipe_ko: Dict[str, Any],
    max_retries: int,
    debug: bool = False
) -> str:
    """
    return HTML (본문만)
    요구:
      - 도입부 200~300자
      - 굵은 소제목 3개 (각 1500자 이상 목표)
      - 총 길이 충분히 길게
      - 친구에게 수다떠는 존댓말
      - 마침표 없이 (후처리 제거)
      - 반복문장/AI 냄새 줄이기
    """
    title_ko = recipe_ko.get("title_ko", "").strip()
    category_ko = recipe_ko.get("category_ko", "").strip()
    area_ko = recipe_ko.get("area_ko", "").strip()
    ingredients_ko = recipe_ko.get("ingredients_ko", [])
    steps_ko = recipe_ko.get("steps_ko", [])

    # 재료/방법 HTML은 “모델이 만들지 않게” 우리가 직접 붙임 (영어/번호/특문 방지)
    # 모델은 경험담/생각/가치관/포인트/리듬만 만들어주게 함
    payload = {
        "keyword": keyword,
        "dish": title_ko,
        "category": category_ko,
        "area": area_ko,
        "constraints": {
            "polite_friend_tone": True,
            "no_periods": True,
            "no_bullets": True,
            "no_step_numbers": True,
            "intro_chars": "200~300",
            "three_sections_chars_each": "1500+",
            "no_repeat_sentences": True,
        },
        "style_notes": [
            "마침표 없이 띄어쓰기와 여백으로 호흡 두기",
            "친구에게 진심 담아 수다떨듯 존댓말",
            "너무 교과서 같은 문장 피하기",
            "체험담처럼 보이되 거짓 단정은 피하기  대신 보통 겪는 포인트처럼 말하기",
        ],
    }

    instructions = (
        "너는 네이버 홈피드에 올라갈 글을 쓰는 사람이다\n"
        "출력은 HTML만\n"
        "\n"
        "필수 형식\n"
        "1) 도입부는 200~300자 정도  문장 사이 여백을 넉넉히\n"
        "2) 굵은 소제목 3개를 반드시 만든다  각 소제목 아래 본문은 1500자 이상 목표\n"
        "3) 전체 길이는 충분히 길게  가독성 좋게 줄바꿈을 자주\n"
        "4) 친구에게 수다떠는 존댓말  반말 금지\n"
        "5) 마침표 기호를 쓰지 마  대신 띄어쓰기와 줄바꿈으로 호흡\n"
        "6) 불릿 기호 금지  1단계 2단계 같은 단계번호 금지\n"
        "7) 특정 문장 반복 금지\n"
        "\n"
        "내용 방향\n"
        "- 레시피는 '기준 잡아주는 글'처럼  실패 포인트와 상태 기준을 말해준다\n"
        "- 경험담 느낌을 내되  실제로 해봤다고 단정하지 말고  보통 겪는 포인트로 말한다\n"
        "- 생각  가치관  생활 루틴 같은 이야기를 자연스럽게 섞는다\n"
        "- 너무 AI 같은 결론 문장  뻔한 문구 반복을 피한다\n"
        "\n"
        "출력 구조 예시\n"
        "<p>도입</p>\n"
        "<h2><b>소제목1</b></h2>\n"
        "<p>본문</p>\n"
        "<h2><b>소제목2</b></h2>\n"
        "<p>본문</p>\n"
        "<h2><b>소제목3</b></h2>\n"
        "<p>본문</p>\n"
    )

    def ask(add_note: str = "") -> str:
        inp = dict(payload)
        if add_note:
            inp["retry_note"] = add_note
        resp = _openai_call_with_retry(
            client=client,
            model=model,
            instructions=instructions,
            input_text=json.dumps(inp, ensure_ascii=False),
            max_retries=max_retries,
            debug=debug,
        )
        out = (resp.output_text or "").strip()
        out = _dedupe_lines(out)
        out = _remove_bullets_and_stepnums(out)
        out = _strip_periods(out)
        return out

    # 1차 생성
    html = ask()

    # 존댓말 체크 + 길이 체크 후 최대 2번 보정
    for _ in range(2):
        if not _polite_enough(html):
            html = ask("반말이 섞였다  친구에게 수다떠는 존댓말로만 다시")
            continue
        # 너무 짧으면 늘리기
        if len(re.sub(r"<[^>]+>", "", html)) < 5200:
            html = ask("분량이 부족하다  각 소제목 아래를 더 길게  여백과 호흡 유지")
            continue
        break

    # 영어 잔존 방지
    if _has_english(html):
        html = _strip_periods(re.sub(r"[A-Za-z]{2,}", "", html))

    return html.strip()


# -----------------------------
# 재료/만드는법 렌더링 (특문/번호 없음)
# -----------------------------
def render_ingredients_block(ingredients_ko: List[Dict[str, str]]) -> str:
    lines = []
    for it in ingredients_ko or []:
        n = (it.get("name_ko") or "").strip()
        m = (it.get("measure_ko") or "").strip()
        if not n:
            continue
        line = f"{n}  {m}".strip() if m else n
        line = _remove_bullets_and_stepnums(_strip_periods(line))
        lines.append(f"<p style='margin:0 0 12px 0;line-height:2.0;'>{_html.escape(line)}</p>")
    if not lines:
        return "<p style='margin:0 0 12px 0;line-height:2.0;'>재료는 집에 있는 것 기준으로 편하게 맞추셔도 됩니다</p>"
    return "<div style='margin:0 0 8px 0;'>" + "\n".join(lines) + "</div>"


def render_steps_block(steps_ko: List[str]) -> str:
    blocks = []
    for s in steps_ko or []:
        s = _remove_bullets_and_stepnums(_strip_periods(str(s))).strip()
        if not s:
            continue
        blocks.append(f"<p style='margin:0 0 16px 0;line-height:2.05;'>{_html.escape(s)}</p>")
    if not blocks:
        return "<p style='margin:0 0 16px 0;line-height:2.05;'>만드는 흐름은 제공된 순서대로  상태 보면서 진행하시면 됩니다</p>"
    return "<div>" + "\n".join(blocks) + "</div>"


def tags_html(keywords_csv: str, main_keyword: str, dish_keyword: str) -> str:
    kws = []
    for k in [main_keyword, dish_keyword]:
        k = (k or "").strip()
        if k:
            kws.append(k)
    for x in (keywords_csv or "").split(","):
        x = x.strip()
        if x and x not in kws:
            kws.append(x)
    kws = kws[:12]
    if not kws:
        return ""
    line = " ".join([f"#{k.replace(' ', '')}" for k in kws if k])
    # 불릿 특문 없이
    return f"<div style='margin:22px 0 0 0;opacity:.82;font-size:14px;line-height:1.9;'>태그  { _html.escape(line) }</div>"


def closing_random(dish: str) -> str:
    d = (dish or "").strip() or "이 메뉴"
    pool = [
        f"오늘은 {d}를 완벽하게 하려는 것보다  실패 확률만 줄이는 기준으로 정리해봤어요  이런 식으로 기준을 잡아두면  다음에는 훨씬 마음이 편해지더라고요",
        f"{d}는 한 번만 감이 잡히면  그 다음부터는 손이 빨라져요  다음번에는 내 입맛에 맞는 포인트 하나만 살짝 조절해보셔도 좋아요",
        f"요리는 정답 찾기보다  내가 반복 가능한 방식으로 가져가는 게 제일 오래 가더라고요  {d}도 그렇게 편하게 가져가시면 됩니다",
        f"메뉴 고민될 때  이런 글 하나가 있으면 진짜 편해요  {d}는 특히 이 기준만 기억해두셔도  결과가 안정됩니다",
    ]
    return _remove_bullets_and_stepnums(_strip_periods(random.choice(pool)))


def wrap_html(inner: str) -> str:
    inner = inner.strip()
    return (
        "<div style='max-width:760px;margin:0 auto;padding:6px 4px;'>\n"
        f"{inner}\n"
        "</div>"
    )


# -----------------------------
# Pick recipe (avoid repeats optional)
# -----------------------------
def pick_recipe(cfg: AppConfig) -> Dict[str, Any]:
    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))
    prefer = set([a.lower() for a in (cfg.naver.prefer_areas or [])])

    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if not rid:
            continue
        if rid in recent_ids:
            continue
        if prefer:
            area = (cand.get("area_en") or "").strip().lower()
            if area and (area not in prefer):
                continue
        return cand

    # 마지막엔 그냥 하나
    return fetch_random_recipe()


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = _now_kst()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")

    init_db(cfg.sqlite_path)

    recipe = pick_recipe(cfg)
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title_en", "") or "Daily Recipe"

    client = OpenAI(api_key=cfg.openai.api_key)

    # 1) 번역(재료/방법/제목 모두 강제)
    recipe_ko = translate_recipe_to_korean(
        client=client,
        model=cfg.openai.model,
        recipe=recipe,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    dish_ko = recipe_ko.get("title_ko", "").strip()
    if not dish_ko:
        dish_ko = "오늘 메뉴"

    # 2) 제목 키워드 결정
    main_kw = (cfg.naver.main_keyword or "").strip()
    title_kw = main_kw if main_kw else dish_ko

    # 3) 제목 후킹(4유형 랜덤)
    hook_title = make_hook_title(title_kw)

    # 4) 네이버 홈피드형 본문 생성(경험담/가치관/수다 존댓말)
    article_html = generate_naver_homefeed_article(
        client=client,
        model=cfg.openai.model,
        keyword=title_kw,
        recipe_ko=recipe_ko,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    # 5) 재료/만드는법 블록을 본문 "끝부분"에 붙여서 절대 중간 삽입 안 되게
    ingredients_block = render_ingredients_block(recipe_ko.get("ingredients_ko", []))
    steps_block = render_steps_block(recipe_ko.get("steps_ko", []))

    tail = []
    tail.append("<h2><b>재료 목록</b></h2>")
    tail.append(ingredients_block)
    tail.append("<h2><b>만드는 흐름</b></h2>")
    tail.append(steps_block)
    tail.append("<h2><b>마무리</b></h2>")
    tail.append(f"<p style='margin:0 0 14px 0;line-height:2.0;'>{_html.escape(closing_random(dish_ko))}</p>")
    tail.append(tags_html(cfg.naver.keywords_csv, main_kw, dish_ko))

    full_body = article_html.strip() + "\n\n" + "\n".join(tail)

    # 최종 후처리: 반복 라인 제거 + 마침표 제거 + 영어 토큰 제거
    full_body = _dedupe_lines(full_body)
    full_body = _strip_periods(full_body)
    full_body = _remove_bullets_and_stepnums(full_body)
    if _has_english(full_body):
        # 남은 영어는 강하게 제거(최후 안전장치)
        full_body = re.sub(r"[A-Za-z]{2,}", "", full_body)
        full_body = re.sub(r"\s{2,}", " ", full_body)
        full_body = full_body.strip()

    # 6) 썸네일 고퀄 우선 (Pexels -> TheMealDB)
    thumb_url = ""
    pex_query = f"{dish_ko} 음식"
    pex_url = pexels_pick_photo_url(cfg.pexels_api_key, pex_query, debug=cfg.run.debug)
    if pex_url:
        thumb_url = pex_url
    else:
        thumb_url = (recipe.get("thumb") or "").strip()

    media_id: Optional[int] = None
    media_url: str = ""
    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"thumb-{date_str}-{time_str}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    # 7) 본문 상단 이미지 삽입 (가독성 + 썸네일 통일)
    if cfg.run.embed_image_in_body:
        img = (media_url or thumb_url or "").strip()
        if img:
            img_tag = (
                f"<p style='margin:0 0 20px 0;'>"
                f"<img src='{_html.escape(img)}' alt='{_html.escape(hook_title)}' "
                f"style='max-width:100%;height:auto;border-radius:14px;'/>"
                f"</p>"
            )
            full_body = img_tag + "\n" + full_body

    full_body = wrap_html(full_body)

    # 8) 새 글 발행 (항상 새 글)
    # slug: 날짜 + 시간 + 랜덤 (겹칠 수 없음)
    rand6 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    slug = f"naverstyle-{date_str}-{cfg.run.run_slot}-{time_str}-{rand6}"

    # 워드프레스 최종 제목: 후킹 제목 그대로 (날짜 prefix 안 붙임)
    wp_title = hook_title

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", wp_title)
        print("SLUG:", slug)
        print(full_body[:2500] + ("\n...(truncated)" if len(full_body) > 2500 else ""))
        return

    featured = media_id if (cfg.run.set_featured and media_id) else None
    post_id, link = wp_create_post(cfg.wp, wp_title, slug, full_body, featured_media=featured)
    save_post_meta_v2(cfg.sqlite_path, recipe_id, recipe_title_en, dish_ko, post_id, link)
    print("OK(created):", post_id, link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
