# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naver.py (통합 완성본 + 랜덤화 레벨 1~3 + 사람글 톤 + WP 발행 + DB이력 + 무료번역 폴백)

✅ 하는 일
- TheMealDB에서 랜덤 레시피 수집(중복 회피)
- OpenAI로 '네이버 블로그에 붙여넣기 좋은' 자연스러운 한국어 글 생성
  - 후킹/요약/포인트/재료/만드는법/체크리스트/FAQ/마무리/해시태그
  - 랜덤화 레벨(1~3)로 표현/구성/개수 변주
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 설정(옵션)
- SQLite 발행 이력 저장/마이그레이션
- OpenAI 크레딧 소진(insufficient_quota 등) 또는 실패 시:
  - LibreTranslate(무료 번역)로 "간단 글" 폴백 발행

⚠️ 참고
- 네이버 홈판(메인/홈) 알고리즘은 공개되지 않으므로 “보장”은 불가.
- 대신 네이버에서 일반적으로 잘 먹히는 형태(가독성/짧은 문단/키워드 자연삽입/해시태그/저장유도)를 강화한 생성 로직임.
- “ChatGPT 냄새” 줄이기 위해 문장 길이/표현/구성 템플릿을 랜덤화.

-----------------------------------------
필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY

권장/선택 env:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS="7" (기본 7)
  - WP_TAG_IDS="1,2,3" (선택)
  - SQLITE_PATH=data/daily_recipe.sqlite3

  - RUN_SLOT=day|am|pm (기본 day)  # 하루 1번이면 day만
  - FORCE_NEW=0|1 (기본 0)         # 오늘 이미 올렸어도 새 레시피로 교체
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - DRY_RUN=0|1
  - DEBUG=0|1

  - OPENAI_MODEL=gpt-5.2 (기본 gpt-5.2)
  - OPENAI_MAX_RETRIES=3
  - STRICT_KOREAN=1 (기본 1)  # 한글 검증

  - RANDOMIZE_LEVEL=1|2|3 (기본 2)
    1=약(거의 고정) / 2=중(추천) / 3=강(구성/문장 변주 큼)

  - UPLOAD_THUMB=1 (기본 1)
  - SET_FEATURED=1 (기본 1)
  - EMBED_IMAGE_IN_BODY=1 (기본 1)
  - DEFAULT_THUMB_URL=... (썸네일 없을 때 대체 URL)

무료번역 폴백(LibreTranslate) 선택 env:
  - FREE_TRANSLATE_URL=https://libretranslate.de/translate
  - FREE_TRANSLATE_API_KEY=
  - FREE_TRANSLATE_SOURCE=en
  - FREE_TRANSLATE_TARGET=ko
"""

from __future__ import annotations

import base64
import hashlib
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
from openai import OpenAI  # 공식 SDK

KST = timezone(timedelta(hours=9))

THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"
THEMEALDB_LOOKUP = "https://www.themealdb.com/api/json/v1/1/lookup.php?i={id}"


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


def _escape_html(s: str) -> str:
    return html.escape(s or "")


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
    run_slot: str = "day"  # day / am / pm
    dry_run: bool = False
    debug: bool = False
    strict_korean: bool = True
    force_new: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20

    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""

    openai_max_retries: int = 3
    randomize_level: int = 2  # 1~3


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
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
            randomize_level=max(1, min(3, _env_int("RANDOMIZE_LEVEL", 2))),
        ),
        sqlite_path=_env("SQLITE_PATH", "data/daily_recipe.sqlite3"),
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", int(cfg.run.dry_run), "| DEBUG:", int(cfg.run.debug))
    print("[CFG] STRICT_KOREAN:", int(cfg.run.strict_korean))
    print("[CFG] AVOID_REPEAT_DAYS:", cfg.run.avoid_repeat_days, "| MAX_TRIES:", cfg.run.max_tries)
    print("[CFG] UPLOAD_THUMB:", int(cfg.run.upload_thumb), "| SET_FEATURED:", int(cfg.run.set_featured), "| EMBED_IMAGE_IN_BODY:", int(cfg.run.embed_image_in_body))
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.run.default_thumb_url else "EMPTY")
    print("[CFG] OPENAI_API_KEY:", ok(cfg.openai.api_key), "| MODEL:", cfg.openai.model, "| RETRIES:", cfg.run.openai_max_retries)
    print("[CFG] RANDOMIZE_LEVEL:", cfg.run.randomize_level)
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
  seed TEXT,
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
    "seed": "TEXT",
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
        "SELECT date_key, slot, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, seed, created_at "
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
        "seed": row[8],
        "created_at": row[9],
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
    seed: str,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_key, slot, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, seed, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            seed or "",
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

    r = requests.post(url, headers=headers, json=payload, timeout=35)
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

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
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
    source = str(m.get("strSource") or "").strip()
    youtube = str(m.get("strYoutube") or "").strip()

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
        "source": source,
        "youtube": youtube,
    }


def split_steps(instructions: str) -> List[str]:
    t = (instructions or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"\r?\n+", t) if p.strip()]
    # 문단이 너무 적고 긴 경우 문장단위 분리 보조
    if len(parts) <= 2 and len(t) > 350:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    # 너무 짧게 쪼개진 경우는 다시 합치지 않고 그대로 둠
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


def build_simple_post_with_free_translate(free_cfg: FreeTranslateConfig, recipe: Dict[str, Any], now: datetime, randomize_level: int, debug: bool = False) -> Tuple[str, str, List[str]]:
    title_en = recipe.get("title", "Daily Recipe")
    title_ko = free_translate_text(free_cfg, title_en, debug=debug) or title_en

    steps_en = split_steps(recipe.get("instructions", ""))
    steps_ko = [free_translate_text(free_cfg, s, debug=debug) or s for s in steps_en]

    ing_lines = []
    for it in recipe.get("ingredients", []):
        name_en = (it.get("name") or "").strip()
        mea = (it.get("measure") or "").strip()
        name_ko = free_translate_text(free_cfg, name_en, debug=debug) if name_en else ""
        label = f"{name_ko} ({name_en})" if name_ko and name_ko != name_en else name_en
        if mea:
            ing_lines.append(f"<li>{_escape_html(label)} <span>— {_escape_html(mea)}</span></li>")
        else:
            ing_lines.append(f"<li>{_escape_html(label)}</li>")

    step_lines = [f"<li>{_escape_html(s)}</li>" for s in steps_ko]

    # 랜덤화(약하게)
    hooks = [
        "오늘은 실패 확률 낮은 버전으로 깔끔하게 정리했어요.",
        "복잡한 건 빼고, 흐름만 따라가면 되는 방식으로 적어둘게요.",
        "재료는 단순하게, 과정은 한 번에 보이게 정리했어요.",
    ]
    closing = [
        "저장해두면 다음에 바로 꺼내 쓰기 편해요.",
        "한 번 해보면 손에 익어서 다음엔 더 빨라요.",
        "오늘 한 끼 고민될 때, 이 방식으로 가보세요.",
    ]
    hook = random.choice(hooks)
    end = random.choice(closing)

    title_final = f"{title_ko} 레시피 (실패 줄이는 정리)"
    tags = ["#레시피", "#오늘뭐먹지", "#집밥", "#간단요리", "#한끼"] if randomize_level <= 2 else ["#레시피", "#요리기록", "#집밥", "#간단요리", "#오늘메뉴", "#초보요리"]

    body = f"""
<p><strong>{_escape_html(hook)}</strong></p>
<p>기준시각: {_escape_html(now.astimezone(KST).strftime("%Y-%m-%d %H:%M"))}</p>

<h2>재료</h2>
<ul>
{''.join(ing_lines) if ing_lines else '<li>재료 정보가 비어있어요.</li>'}
</ul>

<h2>만드는 법</h2>
<ol>
{''.join(step_lines) if step_lines else '<li>과정 정보가 비어있어요.</li>'}
</ol>

<p>{_escape_html(end)}</p>
<p>{' '.join(tags)}</p>
"""
    return title_final, body.strip(), tags


# -----------------------------
# Naver-friendly post generator (OpenAI → JSON → HTML)
# -----------------------------
def _rand_seed() -> str:
    return hashlib.sha1(f"{time.time()}-{random.random()}".encode("utf-8")).hexdigest()[:10]


def _counts_by_level(level: int) -> Tuple[int, int, int]:
    """
    point_n, faq_n, tag_n
    """
    if level <= 1:
        return random.randint(3, 5), random.randint(3, 4), random.randint(10, 12)
    if level == 2:
        return random.randint(4, 6), random.randint(3, 5), random.randint(10, 15)
    return random.randint(5, 7), random.randint(4, 6), random.randint(12, 18)


def _pick_hook_style(level: int) -> str:
    pool = [
        "실패포인트 1줄 → 해결 1줄 → 오늘 맛 포인트 1줄 → 저장 유도 1줄",
        "바쁜 날 1줄 → 오늘 방식 1줄 → 결과 1줄 → 저장 유도 1줄",
        "초보가 막히는 지점 1줄 → 이 글에서 잡는 포인트 1줄 → 맛 포인트 1줄",
    ]
    if level >= 3:
        pool += [
            "처음엔 어려워 보이지만 1줄 → 핵심만 잡으면 쉬움 1줄 → 오늘 방식 1줄 → 저장 유도 1줄",
            "오늘 한 끼 고민 1줄 → 실패 줄이는 흐름 1줄 → 맛 포인트 1줄",
        ]
    return random.choice(pool)


def _heading_variants(level: int) -> Dict[str, List[str]]:
    h2_points = ["한눈에 보는 핵심 포인트", "오늘의 포인트", "맛을 살리는 포인트", "실패 줄이는 포인트"]
    h2_ing = ["재료", "재료 준비", "재료 체크"]
    h2_steps = ["만드는 법", "조리 순서", "조리 과정"]
    h2_check = ["실패 방지 체크리스트", "이것만 체크하면 실패 줄어요", "실패 줄이는 체크"]
    h2_faq = ["자주 하는 질문", "자주 묻는 포인트", "헷갈리는 부분 Q&A"]
    h2_end = ["마무리", "마지막 정리", "마무리 한마디"]

    if level >= 3:
        h2_points += ["저장용 요약 포인트", "초보용 핵심 정리"]
        h2_check += ["초보 체크", "실수 방지 체크"]

    return {
        "points": h2_points,
        "ing": h2_ing,
        "steps": h2_steps,
        "check": h2_check,
        "faq": h2_faq,
        "end": h2_end,
    }


def _build_html_from_json(recipe: Dict[str, Any], img_url: str, out: Dict[str, Any], level: int) -> Tuple[str, str, List[str]]:
    title = str(out.get("title") or "").strip()
    hook = str(out.get("hook") or "").strip()
    one_liner = str(out.get("one_liner") or "").strip()

    points: List[str] = [str(x).strip() for x in (out.get("points") or []) if str(x).strip()]
    ingredients: List[str] = [str(x).strip() for x in (out.get("ingredients") or []) if str(x).strip()]
    steps: List[str] = [str(x).strip() for x in (out.get("steps") or []) if str(x).strip()]
    step_tips: List[str] = [str(x).strip() for x in (out.get("step_tips") or []) if str(x).strip()]
    checklist: List[str] = [str(x).strip() for x in (out.get("checklist") or []) if str(x).strip()]

    faq_raw = out.get("faq") or []
    faq: List[Tuple[str, str]] = []
    if isinstance(faq_raw, list):
        for it in faq_raw:
            if isinstance(it, dict):
                q = str(it.get("q") or "").strip()
                a = str(it.get("a") or "").strip()
                if q and a:
                    faq.append((q, a))

    hashtags: List[str] = [str(x).strip() for x in (out.get("hashtags") or []) if str(x).strip()]

    # headings
    hv = _heading_variants(level)
    h_points = random.choice(hv["points"])
    h_ing = random.choice(hv["ing"])
    h_steps = random.choice(hv["steps"])
    h_check = random.choice(hv["check"])
    h_faq = random.choice(hv["faq"])
    h_end = random.choice(hv["end"])

    # hook html (백슬래시 문제 방지: replace는 f-string 밖에서)
    hook_esc = _escape_html(hook)
    hook_esc = hook_esc.replace("\n", "<br/>")
    hook_html = f"<p><strong>{hook_esc}</strong></p>" if hook else ""

    one_esc = _escape_html(one_liner).replace("\n", "<br/>")
    one_html = f"<p>{one_esc}</p>" if one_liner else ""

    img_html = ""
    if img_url:
        img_html = f'<p><img src="{_escape_html(img_url)}" alt="{_escape_html(title)}" style="max-width:100%;height:auto;"></p>'

    points_html = "<ul>" + "".join([f"<li>{_escape_html(x)}</li>" for x in points]) + "</ul>" if points else ""
    ing_html = "<ul>" + "".join([f"<li>{_escape_html(x)}</li>" for x in ingredients]) + "</ul>" if ingredients else ""

    # steps + tips
    step_items = []
    for i, s in enumerate(steps):
        tip = step_tips[i] if i < len(step_tips) else ""
        s_html = _escape_html(s)
        if tip:
            tip_html = _escape_html(tip)
            step_items.append(f"<li>{s_html}<br/><span style='opacity:.85;font-size:13px;'>실패 방지: {tip_html}</span></li>")
        else:
            step_items.append(f"<li>{s_html}</li>")
    steps_html = "<ol>" + "".join(step_items) + "</ol>" if step_items else ""

    check_html = "<ul>" + "".join([f"<li>{_escape_html(x)}</li>" for x in checklist]) + "</ul>" if checklist else ""

    faq_html = ""
    if faq:
        qa_lines = []
        for q, a in faq:
            qa_lines.append(f"<p><strong>Q. {_escape_html(q)}</strong><br/>A. {_escape_html(a)}</p>")
        faq_html = "".join(qa_lines)

    # hashtags (네이버용: #해시태그 공백구분)
    # 이미 # 포함된 형태로 받는 걸 권장하지만, 안전 처리
    final_tags = []
    for t in hashtags:
        t = t.strip()
        if not t:
            continue
        if not t.startswith("#"):
            t = "#" + re.sub(r"\s+", "", t)
        final_tags.append(t)

    tags_line = " ".join(final_tags)
    tags_html = f"<p>{_escape_html(tags_line)}</p>" if tags_line else ""

    # assemble (네이버 붙여넣기 호환성 위해 단순 태그 위주)
    body = (
        img_html
        + hook_html
        + one_html
        + (f"<h2>{_escape_html(h_points)}</h2>{points_html}" if points_html else "")
        + (f"<h2>{_escape_html(h_ing)}</h2>{ing_html}" if ing_html else "")
        + (f"<h2>{_escape_html(h_steps)}</h2>{steps_html}" if steps_html else "")
        + (f"<h2>{_escape_html(h_check)}</h2>{check_html}" if check_html else "")
        + (f"<h2>{_escape_html(h_faq)}</h2>{faq_html}" if faq_html else "")
        + (f"<h2>{_escape_html(h_end)}</h2><p>{_escape_html(str(out.get('closing') or '').strip())}</p>" if str(out.get("closing") or "").strip() else "")
        + tags_html
    ).strip()

    return title, body, final_tags


def _derive_default_hashtags(recipe: Dict[str, Any], level: int) -> List[str]:
    base = ["#레시피", "#오늘뭐먹지", "#집밥", "#간단요리", "#홈쿡"]
    if level >= 3:
        base += ["#요리기록", "#저장각", "#초보요리"]
    title = (recipe.get("title") or "").strip()
    area = (recipe.get("area") or "").strip()
    cat = (recipe.get("category") or "").strip()
    extra = []
    for s in [title, area, cat]:
        if not s:
            continue
        k = re.sub(r"[^0-9A-Za-z가-힣]+", "", s)
        if k:
            extra.append("#" + k[:18])
    # 재료 일부도 태그 후보
    ings = recipe.get("ingredients") or []
    for it in ings[:6]:
        nm = (it.get("name") or "").strip()
        nm = re.sub(r"[^0-9A-Za-z가-힣]+", "", nm)
        if nm:
            extra.append("#" + nm[:18])
    pool = list(dict.fromkeys(base + extra))
    random.shuffle(pool)
    # 적당히 제한
    return pool[: (12 if level == 1 else 15 if level == 2 else 18)]


def generate_naver_friendly_post(
    cfg: AppConfig,
    recipe: Dict[str, Any],
    img_url: str,
    seed: str,
) -> Tuple[str, str, List[str]]:
    """
    OpenAI에게 JSON만 출력시키고 → 우리가 HTML로 조립(일관성+백슬래시 오류 방지)
    """
    client = OpenAI(api_key=cfg.openai.api_key)
    level = cfg.run.randomize_level
    point_n, faq_n, tag_n = _counts_by_level(level)
    hook_style = _pick_hook_style(level)

    # 랜덤 시드 고정(하루 글 흔들림 방지 원하면 DB seed 재사용)
    random.seed(seed)

    steps_en = split_steps(recipe.get("instructions", ""))
    payload = {
        "seed": seed,
        "randomize_level": level,
        "point_n": point_n,
        "faq_n": faq_n,
        "tag_n": tag_n,
        "hook_style": hook_style,
        "recipe": {
            "title_en": recipe.get("title", ""),
            "category_en": recipe.get("category", ""),
            "area_en": recipe.get("area", ""),
            "ingredients": recipe.get("ingredients", []),  # [{"name","measure"}]
            "steps_en": steps_en,
            "source_url": recipe.get("source", ""),
            "youtube": recipe.get("youtube", ""),
        },
        "hashtag_candidates": _derive_default_hashtags(recipe, level),
        "style_notes": [
            "네이버 블로그 붙여넣기에 자연스러운 톤",
            "짧은 문단, 군더더기 최소",
            "너무 완벽한 문장 금지(사람이 쓴 느낌)",
            "AI/챗GPT/자동생성 언급 금지",
        ],
    }

    # 사람글 톤 + '네이버스럽게' (과장 광고 금지)
    instructions = (
        "너는 한국어로 글 쓰는 개인 요리 블로거다. 결과물은 네이버 블로그에 그대로 붙여넣을 수 있게 자연스럽게 써라.\n"
        "\n"
        "[절대 규칙]\n"
        "1) 제공된 ingredients/steps 범위를 벗어나서 재료/계량/과정을 추가/삭제/변경하지 마.\n"
        "2) 시간/온도 숫자는 원문 steps에 없으면 단정 금지. 필요하면 '상태를 보며'처럼 표현.\n"
        "3) AI/챗GPT/자동생성 같은 말은 절대 쓰지 마.\n"
        "\n"
        "[톤/스타일]\n"
        "- 문장은 너무 매끈하면 안 된다. 사람이 쓴 듯한 리듬으로.\n"
        "- 과한 이모지/과장 광고 금지. (이모지는 0~1개 정도만)\n"
        "- 짧은 문단(1~2문장) 위주. 읽기 편하게.\n"
        "- hook_style에 맞춰 첫 문단을 구성.\n"
        "\n"
        "[출력 형식: JSON ONLY]\n"
        "반드시 아래 키를 가진 '유효한 JSON'만 출력해라(설명/코드블럭 금지).\n"
        "{\n"
        '  "title": "...",                        // 28~48자, 너무 자극적 금지\n'
        '  "hook": "...",                         // 줄바꿈 가능\n'
        '  "one_liner": "...",                    // 한줄요약\n'
        '  "points": ["..."],                     // point_n개\n'
        '  "ingredients": ["..."],                // 입력 재료 그대로(한국어로 풀되 원문 괄호 병기 가능)\n'
        '  "steps": ["..."],                      // steps_en과 동일 개수\n'
        '  "step_tips": ["..."],                  // steps와 동일 개수(일반 팁, 새로운 재료 금지)\n'
        '  "checklist": ["..."],                  // 4~7개(레벨에 맞게)\n'
        '  "faq": [{"q":"...","a":"..."}],        // faq_n개\n'
        '  "closing": "...",                      // 저장/공유 유도 1문장 포함\n'
        '  "hashtags": ["#..."]                   // tag_n개, 후보(candidates)에서 우선 선택\n'
        "}\n"
        "\n"
        "[추가 조건]\n"
        "- steps는 의미를 바꾸지 말고 한국어로 자연스럽게 옮겨라.\n"
        "- ingredients는 빠짐없이 포함하되, 너무 장황하지 않게.\n"
        "- 해시태그는 과도한 영어 남발 금지.\n"
    )

    input_text = "입력 JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    resp = _openai_call_with_retry(
        client=client,
        model=cfg.openai.model,
        instructions=instructions,
        input_text=input_text,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    raw = (resp.output_text or "").strip()
    if cfg.run.debug:
        print("[OPENAI] raw length:", len(raw))

    # JSON 파싱
    try:
        out = json.loads(raw)
        if not isinstance(out, dict):
            raise ValueError("not a dict")
    except Exception:
        # 모델이 앞뒤 텍스트를 붙였을 때를 대비해 JSON만 추출(최후 수단)
        m = re.search(r"\{.*\}\s*$", raw, flags=re.DOTALL)
        if not m:
            raise RuntimeError("OpenAI JSON parse failed")
        out = json.loads(m.group(0))

    # 검증(steps 개수)
    steps_en = split_steps(recipe.get("instructions", ""))
    if len(out.get("steps") or []) != len(steps_en):
        raise RuntimeError(f"OpenAI steps count mismatch: expected {len(steps_en)} got {len(out.get('steps') or [])}")

    title, body, tags = _build_html_from_json(recipe, img_url, out, cfg.run.randomize_level)

    if cfg.run.strict_korean:
        if not re.search(r"[가-힣]", title) or not re.search(r"[가-힣]", body):
            raise RuntimeError("한글 검증 실패(STRICT_KOREAN=1).")

    # 해시태그가 너무 비면 후보로 채움
    if not tags:
        tags = _derive_default_hashtags(recipe, cfg.run.randomize_level)[: _counts_by_level(cfg.run.randomize_level)[2]]

    return title.strip(), body.strip(), tags


# -----------------------------
# Main flow
# -----------------------------
def pick_recipe(cfg: AppConfig, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # 오늘 이미 올렸고 FORCE_NEW=0이면 같은 레시피 유지(조회수/체류 안정 목적)
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


def choose_thumb(recipe: Dict[str, Any], default_thumb_url: str) -> str:
    return (recipe.get("thumb") or "").strip() or (default_thumb_url or "").strip()


def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else ("오후" if slot == "pm" else "오늘")

    date_key = f"{date_str}_{slot}" if slot in ("am", "pm") else date_str
    slug = f"daily-recipe-{date_str}-{slot}" if slot in ("am", "pm") else f"daily-recipe-{date_str}"

    init_db(cfg.sqlite_path, debug=cfg.run.debug)
    existing = get_today_post(cfg.sqlite_path, date_key)

    # seed: 오늘 글은 같은 seed로 흔들림 줄이고 싶으면 DB seed 재사용
    seed = (existing.get("seed") if existing else "") or _rand_seed()

    wp_post_id: Optional[int] = None
    if existing and existing.get("wp_post_id"):
        wp_post_id = int(existing["wp_post_id"])
    else:
        wp_post_id = wp_find_post_by_slug(cfg.wp, slug)

    recipe = pick_recipe(cfg, existing)
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title", "") or "Daily Recipe"

    # 썸네일
    thumb_url = choose_thumb(recipe, cfg.run.default_thumb_url)

    # 이미지 업로드
    media_id: Optional[int] = None
    media_url: str = ""
    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"recipe-{date_str}-{slot}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None
    display_img = media_url or thumb_url

    # 글 생성 (OpenAI → 실패/쿼터 소진 → 무료번역 폴백)
    try:
        title_ko, body_html, tags = generate_naver_friendly_post(cfg, recipe, display_img if cfg.run.embed_image_in_body else "", seed)
    except Exception as e:
        if _is_insufficient_quota_error(e):
            print("[WARN] OpenAI quota depleted → fallback to FREE translate")
        else:
            print("[WARN] OpenAI failed → fallback to FREE translate:", repr(e))
        title_ko, body_html, tags = build_simple_post_with_free_translate(cfg.free_tr, recipe, now, cfg.run.randomize_level, debug=cfg.run.debug)

        # 폴백일 때도 이미지 1장 상단 삽입(옵션)
        if cfg.run.embed_image_in_body and display_img:
            body_html = f'<p><img src="{_escape_html(display_img)}" alt="{_escape_html(title_ko)}" style="max-width:100%;height:auto;"></p>\n' + body_html

    # 최종 제목(날짜 + 슬롯 + 키워드)
    # 네이버용으로 너무 ‘기사체’ 말고, 레시피 키워드+혜택
    title = f"{date_str} {slot_label} | {title_ko}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", title)
        print("SLUG:", slug)
        print("SEED:", seed)
        print(body_html[:2500] + ("\n...(truncated)" if len(body_html) > 2500 else ""))
        return

    # 발행/업데이트
    if wp_post_id:
        new_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url, seed)
        print("OK(updated):", new_id, wp_link)
    else:
        new_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url, seed)
        print("OK(created):", new_id, wp_link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
