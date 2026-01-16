# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle.py
- 네이버 홈피드형(가독성/수다톤/가치관 강화)
- 재료 불릿(점) 제거, 만드는 법 번호/단계 제거
- ✅ 번역 누락/영어 잔존 최소화: (1) bulk 번역에서 빈값 보존 (2) 영어 남은 필드만 재번역 후처리 (3) 제목 키워드도 한국어 보정
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

ING_TOKEN = "[[INGREDIENTS]]"
STEP_TOKEN = "[[STEPS]]"

_RE_HAS_ASCII_WORD = re.compile(r"[A-Za-z]{3,}")  # 3글자 이상 영단어
_RE_HAS_ASCII = re.compile(r"[A-Za-z]")


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


def _first_keyword(csv: str) -> str:
    for x in (csv or "").split(","):
        x = x.strip()
        if x:
            return x
    return ""


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


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4.1-mini"


@dataclass
class FreeTranslateConfig:
    urls: List[str] = field(default_factory=list)
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

    urls_csv = _env("FREE_TRANSLATE_URLS", "")
    urls = [u.strip() for u in urls_csv.split(",") if u.strip()]
    if not urls:
        urls = [
            "https://libretranslate.de/translate",
            "https://translate.astian.org/translate",
        ]

    free_api_key = _env("FREE_TRANSLATE_API_KEY", "")
    free_src = _env("FREE_TRANSLATE_SOURCE", "en")
    free_tgt = _env("FREE_TRANSLATE_TARGET", "ko")

    naver_keywords = _env("NAVER_KEYWORDS", "")
    title_keyword = _env("NAVER_TITLE_KEYWORD", "")

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
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
        free_tr=FreeTranslateConfig(urls=urls, api_key=free_api_key, source=free_src, target=free_tgt),
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
    print("[CFG] NAVER_KEYWORDS:", cfg.naver.keywords_csv or "(empty)")
    print("[CFG] FREE_TRANSLATE_URLS:", ", ".join(cfg.free_tr.urls))


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
  media_id INTEGER,
  media_url TEXT,
  created_at TEXT
)
"""


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(TABLE_SQL)
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
# WordPress
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
# Recipe fetch
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
            time.sleep((2 ** attempt) + random.random())
        except openai.APIError as e:
            if attempt == max_retries:
                raise
            if debug:
                print("[OPENAI] APIError retry:", repr(e))
            time.sleep((2 ** attempt) + random.random())
        except Exception as e:
            if attempt == max_retries:
                raise
            if debug:
                print("[OPENAI] Unknown retry:", repr(e))
            time.sleep((2 ** attempt) + random.random())


# -----------------------------
# LibreTranslate (rotate) + ✅ bulk keep-empty
# -----------------------------
def _free_translate_request(url: str, api_key: str, source: str, target: str, text: str, debug: bool = False) -> Optional[str]:
    payload = {"q": text, "source": source, "target": target, "format": "text"}
    if api_key:
        payload["api_key"] = api_key
    try:
        r = requests.post(url, json=payload, timeout=25)
        if r.status_code != 200:
            if debug:
                print("[FREE_TR] non-200:", url, r.status_code, r.text[:120])
            return None
        j = r.json()
        out = (j.get("translatedText") or "").strip()
        return out or None
    except Exception as e:
        if debug:
            print("[FREE_TR] failed:", url, repr(e))
        return None


def free_translate_text(cfg: FreeTranslateConfig, text: str, debug: bool = False) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    for u in cfg.urls:
        out = _free_translate_request(u, cfg.api_key, cfg.source, cfg.target, text, debug=debug)
        if out and out.strip():
            return out.strip()
    return text


def free_translate_bulk_keep_empty(cfg: FreeTranslateConfig, texts: List[str], debug: bool = False) -> List[str]:
    """
    ✅ 핵심 수정
    - 빈 문자열을 버리면 재료/계량이 뒤로 밀려서 번역 누락처럼 보임
    - 빈값은 그대로 유지하고, 번역이 필요한 것만 묶어서 번역한 뒤 다시 끼워 넣음
    """
    if not texts:
        return []

    idxs = [i for i, t in enumerate(texts) if (t or "").strip()]
    if not idxs:
        return list(texts)

    token = "|||__SPLIT__|||"
    non_empty = [(texts[i] or "").strip() for i in idxs]
    joined = f"\n{token}\n".join(non_empty)

    translated = None
    for _ in range(2):
        out = free_translate_text(cfg, joined, debug=debug)
        parts = [p.strip() for p in (out or "").split(token)]
        if len(parts) == len(non_empty):
            translated = parts
            break
        time.sleep(0.6 + random.random())

    if translated is None:
        if debug:
            print("[FREE_TR] bulk mismatch → individual fallback")
        translated = [free_translate_text(cfg, t, debug=debug) for t in non_empty]

    result = list(texts)
    for j, i in enumerate(idxs):
        result[i] = translated[j] if translated[j] else texts[i]
    return result


# -----------------------------
# translate recipe to korean (JSON) + ✅ fix english leftovers
# -----------------------------
def _fix_english_leftovers_with_openai(cfg: AppConfig, fields: List[str]) -> List[str]:
    client = OpenAI(api_key=cfg.openai.api_key)
    inst = """
너는 번역기다  아래 문자열 리스트를 한국어로 번역한다
- 출력은 JSON 배열만
- 영어 단어를 남기지 말라
- 측정 단위(cup, tbsp 등)는 자연스럽게 한국어로 풀어써라
""".strip()
    resp = _openai_call_with_retry(
        client=client,
        model=cfg.openai.model,
        instructions=inst,
        input_text=json.dumps(fields, ensure_ascii=False),
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )
    out = (resp.output_text or "").strip()
    data = json.loads(out)
    if isinstance(data, list) and len(data) == len(fields):
        return [str(x or "").strip() for x in data]
    return fields


def _fix_english_leftovers(cfg: AppConfig, recipe_ko: Dict[str, Any]) -> Dict[str, Any]:
    """
    제목/재료/계량/스텝에 영어가 남으면 그 부분만 재번역해서 덮어씀
    """
    to_fix: List[Tuple[str, int, str]] = []  # (path, idx, value)

    # title
    t = (recipe_ko.get("title_ko") or "").strip()
    if t and _RE_HAS_ASCII.search(t):
        to_fix.append(("title", -1, t))

    # ingredients
    ings = recipe_ko.get("ingredients") or []
    for i, it in enumerate(ings):
        nk = (it.get("name_ko") or "").strip()
        mk = (it.get("measure_ko") or "").strip()
        if nk and _RE_HAS_ASCII_WORD.search(nk):
            to_fix.append(("ing_name", i, nk))
        if mk and _RE_HAS_ASCII_WORD.search(mk):
            to_fix.append(("ing_mea", i, mk))

    # steps
    steps = recipe_ko.get("steps_ko") or []
    for i, s in enumerate(steps):
        s = (s or "").strip()
        if s and _RE_HAS_ASCII_WORD.search(s):
            to_fix.append(("step", i, s))

    if not to_fix:
        return recipe_ko

    fields = [v for (_, __, v) in to_fix]

    fixed: List[str]
    try:
        fixed = _fix_english_leftovers_with_openai(cfg, fields)
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] leftover fix via OpenAI failed → free translate:", repr(e))
        fixed = free_translate_bulk_keep_empty(cfg.free_tr, fields, debug=cfg.run.debug)

    # apply back
    ptr = 0
    for kind, idx, _v in to_fix:
        newv = (fixed[ptr] or "").strip()
        ptr += 1
        if not newv:
            continue
        if kind == "title":
            recipe_ko["title_ko"] = newv
        elif kind == "ing_name":
            recipe_ko["ingredients"][idx]["name_ko"] = newv
        elif kind == "ing_mea":
            recipe_ko["ingredients"][idx]["measure_ko"] = newv
        elif kind == "step":
            recipe_ko["steps_ko"][idx] = newv

    return recipe_ko


def translate_recipe_to_korean(cfg: AppConfig, recipe: Dict[str, Any]) -> Dict[str, Any]:
    title_en = recipe.get("title", "") or ""
    ingredients = recipe.get("ingredients", []) or []
    steps_en = split_steps(recipe.get("instructions", ""))

    # 1) OpenAI 우선
    try:
        client = OpenAI(api_key=cfg.openai.api_key)
        inst = """
너는 번역기다  아래 입력을 한국어로 정확히 번역해라
- 출력은 JSON만
- 영어 단어를 남기지 말라
- ingredients 개수 유지  steps 개수 유지
JSON 스키마:
{
 "title_ko": "string",
 "ingredients": [{"name_en":"string","name_ko":"string","measure_en":"string","measure_ko":"string"}],
 "steps_ko": ["string", ...]
}
""".strip()
        payload = {"title_en": title_en, "ingredients": ingredients, "steps_en": steps_en}
        resp = _openai_call_with_retry(
            client=client,
            model=cfg.openai.model,
            instructions=inst,
            input_text=json.dumps(payload, ensure_ascii=False),
            max_retries=cfg.run.openai_max_retries,
            debug=cfg.run.debug,
        )
        data = json.loads((resp.output_text or "").strip())
        if not isinstance(data, dict) or not data.get("title_ko"):
            raise RuntimeError("translate json invalid")
        # ✅ 잔존 영어 후처리
        return _fix_english_leftovers(cfg, data)
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] translate via OpenAI failed → fallback:", repr(e))

    # 2) LibreTranslate 폴백 (✅ 빈값 보존 bulk)
    title_ko = free_translate_text(cfg.free_tr, title_en, debug=cfg.run.debug) if title_en else "오늘 레시피"

    ing_names = [(it.get("name") or "").strip() for it in ingredients]
    ing_meas = [(it.get("measure") or "").strip() for it in ingredients]

    tr_names = free_translate_bulk_keep_empty(cfg.free_tr, ing_names, debug=cfg.run.debug) if ing_names else []
    tr_meas = free_translate_bulk_keep_empty(cfg.free_tr, ing_meas, debug=cfg.run.debug) if ing_meas else []

    tr_steps = free_translate_bulk_keep_empty(cfg.free_tr, steps_en, debug=cfg.run.debug) if steps_en else []

    out_ings = []
    for i, it in enumerate(ingredients):
        name_en = (it.get("name") or "").strip()
        mea_en = (it.get("measure") or "").strip()
        name_ko = (tr_names[i] if i < len(tr_names) else "").strip()
        mea_ko = (tr_meas[i] if i < len(tr_meas) else "").strip()

        out_ings.append(
            {
                "name_en": name_en,
                "name_ko": name_ko or name_en,
                "measure_en": mea_en,
                "measure_ko": mea_ko or mea_en,
            }
        )

    data = {
        "title_ko": title_ko or title_en or "오늘 레시피",
        "ingredients": out_ings,
        "steps_ko": tr_steps if tr_steps else steps_en,
    }
    # ✅ 잔존 영어 후처리
    return _fix_english_leftovers(cfg, data)


# -----------------------------
# Keyword/title Korean enforcement
# -----------------------------
def ensure_korean_keyword(cfg: AppConfig, s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "오늘 레시피"
    if not _RE_HAS_ASCII.search(s):
        return s
    # 영어가 섞였으면 키워드도 번역해서 제목에 영어가 남지 않게
    try:
        client = OpenAI(api_key=cfg.openai.api_key)
        inst = "다음 텍스트를 한국어로 번역해라  영어 단어를 남기지 마라  출력은 번역문만"
        resp = _openai_call_with_retry(
            client=client,
            model=cfg.openai.model,
            instructions=inst,
            input_text=s,
            max_retries=cfg.run.openai_max_retries,
            debug=cfg.run.debug,
        )
        out = (resp.output_text or "").strip()
        return out or free_translate_text(cfg.free_tr, s, debug=cfg.run.debug) or s
    except Exception:
        return free_translate_text(cfg.free_tr, s, debug=cfg.run.debug) or s


# -----------------------------
# Title generator (4 types) - Korean only
# -----------------------------
def title_random(keyword_ko: str) -> str:
    kw = (keyword_ko or "").strip() or "오늘 레시피"

    benefit = [
        f"{kw} 오늘 해볼 때  맛이 안정되는 기준  딱 한 줄만 잡아볼게요",
        f"{kw} 생각보다 쉬워요  실패 확률 줄이는 흐름만 정리해요",
        f"{kw} 손이 많이 갈 것 같아도  순서만 잡으면 편해집니다",
    ]
    threat = [
        f"{kw} 여기서 한 번만 어긋나면  맛이 갑자기 흔들리더라고요",
        f"{kw} 대충 넘어가면  꼭 같은 구간에서 망합니다  그 지점만 피하기",
        f"{kw} 이 부분을 놓치면  재료가 아깝게 느껴질 수 있어요",
    ]
    curiosity = [
        f"{kw} 왜 어떤 날은 맛있고  어떤 날은 밍밍할까요  기준이 있더라고요",
        f"{kw} 같은 재료인데  결과가 갈리는 이유가 은근 단순합니다",
        f"{kw} 처음엔 복잡해 보여도  한 번 기준을 잡으면 마음이 편해져요",
    ]
    compare = [
        f"{kw} 레시피 따라 하기 vs 상태 보면서 조절하기  저는 후자 쪽이에요",
        f"{kw} 정확한 계량보다 중요한 것  저는 이 감각을 더 믿는 편이에요",
        f"{kw} 요리는 완벽보다 리듬  이 말이 은근 맞더라고요",
    ]

    buckets = [
        (benefit, 30),
        (threat, 20),
        (curiosity, 35),
        (compare, 15),
    ]

    r = random.randint(1, 100)
    acc = 0
    chosen = curiosity
    for pool, w in buckets:
        acc += w
        if r <= acc:
            chosen = pool
            break

    return random.choice(chosen)


# -----------------------------
# HTML builders (no bullets, no numbers)
# -----------------------------
def wrap_readable(html: str) -> str:
    return "<div style='line-height:2.05;font-size:16px;letter-spacing:-0.2px;'>" + html + "</div>"


def ingredients_html(ko_pack: Dict[str, Any]) -> str:
    rows = []
    for it in ko_pack.get("ingredients", []) or []:
        name = ((it.get("name_ko") or it.get("name_en") or "")).strip()
        mea = ((it.get("measure_ko") or it.get("measure_en") or "")).strip()

        # ✅ 혹시 남은 영어 제거(안전망)
        if _RE_HAS_ASCII_WORD.search(name):
            name = re.sub(r"[A-Za-z]+", "", name).strip() or name
        if _RE_HAS_ASCII_WORD.search(mea):
            # 단위영어가 남으면 표시만 줄이되, 원문이 완전히 사라지지 않게
            mea = mea.replace("tbsp", "큰술").replace("tsp", "작은술").replace("cup", "컵")

        rows.append(
            "<div style='padding:10px 12px;border:1px solid #eee;border-radius:12px;margin:0 0 10px 0;'>"
            f"<div style='font-weight:600'>{_html.escape(name)}</div>"
            f"<div style='margin-top:2px;opacity:.9'>{_html.escape(mea) if mea else ''}</div>"
            "</div>"
        )

    if not rows:
        rows = [
            "<div style='padding:10px 12px;border:1px solid #eee;border-radius:12px;margin:0 0 10px 0;'>"
            "재료 정보가 비어있어요"
            "</div>"
        ]

    return "<div style='margin:10px 0 18px 0;'>" + "".join(rows) + "</div>"


def steps_html(ko_pack: Dict[str, Any]) -> str:
    steps = [s.strip() for s in (ko_pack.get("steps_ko") or []) if s and s.strip()]
    if not steps:
        steps = ["과정 정보가 비어있어요"]

    blocks = []
    for s in steps:
        # ✅ 번호/단계 표현 제거
        s = re.sub(r"^\s*\d+\s*[\).\:-]\s*", "", s)
        s = re.sub(r"^\s*\d+\s*단계\s*[:\-]?\s*", "", s)
        s = re.sub(r"^\s*step\s*\d+\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)
        s = s.strip()

        blocks.append(
            "<div style='padding:12px 12px;border:1px solid #f0f0f0;border-radius:12px;margin:0 0 12px 0;'>"
            f"<div>{_html.escape(s)}</div>"
            "</div>"
        )

    return "<div style='margin:10px 0 18px 0;'>" + "".join(blocks) + "</div>"


# -----------------------------
# Body generation (chatty, values) - Korean only
# -----------------------------
def build_body_instructions(keyword: str) -> str:
    return f"""
너는 한국어로 글을 쓰는 블로그 운영자다  친구에게 진심 담아 수다 떠는 존댓말로 쓴다

[톤]
- 마침표는 쓰지 않는다  대신 줄바꿈과 여백으로 호흡
- 생각  가치관  판단 기준을 자연스럽게 섞는다
- “제가 오늘 직접 해먹어봤는데” 같은 실제 체험 단정은 금지
- 대신 “집에서 해보면 보통” “이 지점에서 자주” “저는 이런 기준을 좋아하는 편” 같은 표현

[형식]
- 도입부 200~300자
- 굵은 소제목 3개  각 소제목 아래는 길게  전체 2300자 이상
- 소제목은 <h2><b>...</b></h2> 로 출력
- 본문 중간에 토큰 {ING_TOKEN} 를 한 번 넣는다
- 본문 후반에 토큰 {STEP_TOKEN} 를 한 번 넣는다

[금지]
- 영어 단어 남기지 말기
- “1단계 2단계 3단계” “Step 1” 같은 단계 번호 문구 금지
- 목록 남발 금지  과한 정리체 금지

[키워드]
- '{keyword}'는 자연스럽게 2~4번

[출력]
- HTML만 출력
""".strip()


def html_has_english_word(html: str) -> bool:
    txt = re.sub(r"<[^>]+>", " ", html or "")
    return bool(_RE_HAS_ASCII_WORD.search(txt))


def generate_body_html(cfg: AppConfig, recipe_ko: Dict[str, Any], keyword: str, now: datetime) -> str:
    client = OpenAI(api_key=cfg.openai.api_key)

    seed = {
        "keyword": keyword,
        "title_ko": recipe_ko.get("title_ko", ""),
        "month": now.astimezone(KST).strftime("%m월"),
        "hook_seed": random.choice(
            [
                "똑같은 재료인데 결과가 다르면  그건 실력이 아니라 기준이 흔들린 거더라고요",
                "요리는 완벽보다 리듬이라는 말  저는 은근 이 말에 기대게 돼요",
                "바쁘면 더 복잡한 레시피가 싫어지잖아요  그래서 기준만 남기는 쪽이 편했어요",
                "맛이 한 번 흔들리면  다음부터 손이 멀어지더라고요  그게 제일 아까워요",
            ]
        ),
        "value_seed": random.choice(
            [
                "저는 요리에서 제일 중요한 게  내 컨디션을 배신하지 않는 방식이라고 생각해요",
                "저는 정확한 계량보다  상태를 보는 감각이 쌓이는 게 더 오래 가더라고요",
                "저는 실패해도 다시 하고 싶어지는 레시피가  좋은 레시피라고 느껴요",
            ]
        ),
        "note": "재료/과정은 토큰 위치에 삽입될 거라 본문에 목록을 반복하지 말기",
    }

    resp = _openai_call_with_retry(
        client=client,
        model=cfg.openai.model,
        instructions=build_body_instructions(keyword),
        input_text=json.dumps(seed, ensure_ascii=False),
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )
    html = (resp.output_text or "").strip()

    if "<" not in html:
        safe = _html.escape(html).replace("\n", "<br/><br/>")
        html = f"<p style='margin:0 0 18px 0;'>{safe}</p>"

    if ING_TOKEN not in html:
        html += f"<h2><b>재료</b></h2>{ING_TOKEN}"
    if STEP_TOKEN not in html:
        html += f"<h2><b>만드는 법</b></h2>{STEP_TOKEN}"

    # ✅ 영어 단어가 남으면 1회 재작성
    if html_has_english_word(html):
        inst = f"""
다음 HTML을 한국어로 자연스럽게 다시 써라
- 마침표 금지  줄바꿈/여백으로 호흡
- 존댓말 수다 톤  생각/가치관/판단기준이 보이게
- 영어 단어 남기지 말기
- “1단계 2단계 Step 1” 같은 번호 문구 금지
- 토큰 {ING_TOKEN} 와 {STEP_TOKEN} 는 반드시 그대로 유지
- HTML만 출력
""".strip()
        rr = _openai_call_with_retry(
            client=client,
            model=cfg.openai.model,
            instructions=inst,
            input_text=html,
            max_retries=cfg.run.openai_max_retries,
            debug=cfg.run.debug,
        )
        html2 = (rr.output_text or "").strip()
        if html2:
            html = html2

    return html


# -----------------------------
# pick recipe
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
    raise RuntimeError("레시피를 가져오지 못했습니다(중복 회피/시도 초과).")


# -----------------------------
# main
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

    init_db(cfg.sqlite_path)
    existing = get_today_post(cfg.sqlite_path, date_key) if not cfg.run.force_new else None

    wp_post_id: Optional[int] = None
    if existing and existing.get("wp_post_id"):
        wp_post_id = int(existing["wp_post_id"])
    else:
        wp_post_id = wp_find_post_by_slug(cfg.wp, slug)

    recipe = pick_recipe(cfg, existing)

    # thumbnail
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

    # ✅ translate (title/ingredients/measures/steps) + leftover fix
    recipe_ko = translate_recipe_to_korean(cfg, recipe)

    # ✅ keyword for title (Korean enforced)
    keyword = (cfg.naver.title_keyword or "").strip()
    if not keyword:
        keyword = _first_keyword(cfg.naver.keywords_csv)
    if not keyword:
        keyword = (recipe_ko.get("title_ko") or "").strip() or "오늘 레시피"
    keyword = ensure_korean_keyword(cfg, keyword)

    # ✅ title (Korean only)
    title_core = title_random(keyword)
    title = f"{date_str} {slot_label} | {title_core}"

    # body
    body = generate_body_html(cfg, recipe_ko, keyword=keyword, now=now)

    # insert ingredients/steps
    body = body.replace(ING_TOKEN, ingredients_html(recipe_ko))
    body = body.replace(STEP_TOKEN, steps_html(recipe_ko))

    # image at top (alt도 한국어)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            body = (
                "<p style='margin:0 0 22px 0;'>"
                f"<img src=\"{_html.escape(img)}\" alt=\"{_html.escape(keyword)}\" "
                "style=\"max-width:100%;height:auto;border-radius:14px;display:block;\"/>"
                "</p>"
            ) + body

    body_html = wrap_readable(body)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2500] + ("\n...(truncated)" if len(body_html) > 2500 else ""))
        return

    # publish
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
