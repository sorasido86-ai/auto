# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naver.py (통합코드 / 네이버 복붙 최적화 톤)
- 1순위: 식품안전나라(MFDS) COOKRCP01 (MFDS_API_KEY 있으면)
- 2순위: TheMealDB 랜덤 레시피
- OpenAI로 한국어 자연스러운 '사람 글' 톤 생성 (네이버 복붙 가독성 중시)
- OpenAI 실패/쿼터 이슈 시: LibreTranslate 무료번역 폴백(간단글이라도 발행)
- WordPress 발행/업데이트 + 썸네일 업로드/대표이미지 + 본문 상단 이미지
- SQLite 이력/중복 회피

필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY   (없어도 폴백 글은 가능하게 할 수 있지만, 기본은 필요로 둠)

선택 env:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS="7"
  - WP_TAG_IDS="1,2,3"
  - SQLITE_PATH=data/daily_recipe.sqlite3
  - RUN_SLOT=day|am|pm (기본 day)
  - FORCE_NEW=0|1
  - DRY_RUN=0|1
  - DEBUG=0|1
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=25
  - OPENAI_MODEL=... (기본 gpt-5.2)
  - OPENAI_MAX_RETRIES=3

이미지:
  - UPLOAD_THUMB=1
  - SET_FEATURED=1
  - EMBED_IMAGE_IN_BODY=1
  - DEFAULT_THUMB_URL=...  (레시피 원본 이미지 없을 때 대체)

MFDS:
  - MFDS_API_KEY=... (있으면 한식 레시피 우선)

무료번역 폴백:
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
from urllib.parse import quote

import requests

# OpenAI SDK (공식)
import openai
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
# Config
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
    run_slot: str = "day"   # day/am/pm
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 25

@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""

@dataclass
class OpenAIConfig:
    api_key: str = ""
    model: str = "gpt-5.2"
    max_retries: int = 3

@dataclass
class MFDSConfig:
    api_key: str = ""   # foodsafetykorea openapi key (optional)

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
    img: ImageConfig
    openai: OpenAIConfig
    mfds: MFDSConfig
    free_tr: FreeTranslateConfig
    sqlite_path: str

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

    sqlite_path = _env("SQLITE_PATH", "data/daily_recipe.sqlite3")

    cfg = AppConfig(
        wp=WordPressConfig(
            base_url=wp_base, user=wp_user, app_pass=wp_pass,
            status=wp_status, category_ids=cat_ids, tag_ids=tag_ids
        ),
        run=RunConfig(
            run_slot=run_slot,
            force_new=_env_bool("FORCE_NEW", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 25),
        ),
        img=ImageConfig(
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
        ),
        openai=OpenAIConfig(
            api_key=_env("OPENAI_API_KEY", ""),
            model=_env("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2",
            max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
        ),
        mfds=MFDSConfig(api_key=_env("MFDS_API_KEY", "")),
        free_tr=FreeTranslateConfig(
            url=_env("FREE_TRANSLATE_URL", "https://libretranslate.de/translate"),
            api_key=_env("FREE_TRANSLATE_API_KEY", ""),
            source=_env("FREE_TRANSLATE_SOURCE", "en"),
            target=_env("FREE_TRANSLATE_TARGET", "ko"),
        ),
        sqlite_path=sqlite_path,
    )
    return cfg

def validate_cfg(cfg: AppConfig) -> None:
    missing = []
    if not cfg.wp.base_url: missing.append("WP_BASE_URL")
    if not cfg.wp.user: missing.append("WP_USER")
    if not cfg.wp.app_pass: missing.append("WP_APP_PASS")
    # OPENAI는 기본 필요로 두되, 쿼터 이슈시 폴백 가능
    if not cfg.openai.api_key: missing.append("OPENAI_API_KEY")
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
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] IMG upload/set_featured/embed:", cfg.img.upload_thumb, cfg.img.set_featured, cfg.img.embed_image_in_body)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] MFDS_API_KEY:", ok(cfg.mfds.api_key))
    print("[CFG] OPENAI_API_KEY:", ok(cfg.openai.api_key), "| MODEL:", cfg.openai.model, "| retries:", cfg.openai.max_retries)
    print("[CFG] FREE_TRANSLATE_URL:", cfg.free_tr.url)

# -----------------------------
# SQLite
# -----------------------------
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_posts (
  date_slot TEXT PRIMARY KEY,
  recipe_source TEXT,
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
    "date_slot": "TEXT",
    "recipe_source": "TEXT",
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
                print("[DB] add column:", col, typ)
            cur.execute(f"ALTER TABLE daily_posts ADD COLUMN {col} {typ}")
    con.commit()
    con.close()

def get_today_post(path: str, date_slot: str) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "SELECT date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at "
        "FROM daily_posts WHERE date_slot=?",
        (date_slot,),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "date_slot": row[0] or "",
        "recipe_source": row[1] or "",
        "recipe_id": row[2] or "",
        "recipe_title": row[3] or "",
        "wp_post_id": int(row[4] or 0),
        "wp_link": row[5] or "",
        "media_id": int(row[6] or 0),
        "media_url": row[7] or "",
        "created_at": row[8] or "",
    }

def save_post_meta(path: str, meta: Dict[str, Any]) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.get("date_slot", ""),
            meta.get("recipe_source", ""),
            meta.get("recipe_id", ""),
            meta.get("recipe_title", ""),
            int(meta.get("wp_post_id", 0) or 0),
            meta.get("wp_link", ""),
            int(meta.get("media_id", 0) or 0),
            meta.get("media_url", ""),
            datetime.utcnow().isoformat(),
        ),
    )
    con.commit()
    con.close()

def get_recent_recipe_pairs(path: str, days: int) -> List[Tuple[str, str]]:
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "SELECT recipe_source, recipe_id FROM daily_posts WHERE created_at >= ?",
        (cutoff,),
    )
    rows = cur.fetchall()
    con.close()
    out: List[Tuple[str, str]] = []
    for s, rid in rows:
        if s and rid:
            out.append((str(s), str(rid)))
    return out

# -----------------------------
# WP REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-recipe-bot/1.0"}

def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, featured_media: int = 0) -> Tuple[int, str]:
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

def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, featured_media: int = 0) -> Tuple[int, str]:
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

def wp_upload_media_from_url(cfg: WordPressConfig, image_url: str, filename: str) -> Tuple[int, str]:
    # download
    rr = requests.get(image_url, timeout=35)
    if rr.status_code != 200 or not rr.content:
        raise RuntimeError(f"Image download failed: {rr.status_code}")

    content = rr.content
    ctype = (rr.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
    if not ctype:
        if filename.lower().endswith(".png"):
            ctype = "image/png"
        else:
            ctype = "image/jpeg"

    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
    }
    r = requests.post(url, headers=headers, data=content, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("source_url") or "")

# -----------------------------
# Recipe model
# -----------------------------
@dataclass
class Recipe:
    source: str              # "mfds" or "themealdb"
    recipe_id: str
    title: str
    ingredients: List[str]   # already “display lines”
    steps: List[str]
    image_url: str = ""
    source_url: str = ""
    youtube: str = ""

    def uid(self) -> str:
        s = f"{self.source}|{self.recipe_id}|{self.title}"
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

# -----------------------------
# MFDS COOKRCP01
# -----------------------------
def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int = 1, end: int = 60) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    r = requests.get(url, timeout=35)
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    co = data.get("COOKRCP01") or {}
    rows = co.get("row") or []
    return rows if isinstance(rows, list) else []

def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or ""
    title = str(row.get("RCP_NM") or "").strip() or "한식 레시피"
    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()

    ingredients: List[str] = []
    for p in re.split(r"\s*,\s*", parts):
        p = p.strip()
        if p:
            ingredients.append(p)

    steps: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        if s:
            s = re.sub(r"[a-zA-Z]\s*$", "", s).strip()
            steps.append(s)

    img_main = str(row.get("ATT_FILE_NO_MAIN") or "").strip()
    if not img_main:
        img_main = str(row.get("ATT_FILE_NO_MK") or "").strip()

    return Recipe(
        source="mfds",
        recipe_id=rid or hashlib.sha1(title.encode("utf-8")).hexdigest()[:8],
        title=title,
        ingredients=ingredients,
        steps=steps,
        image_url=img_main if img_main.startswith("http") else "",
        source_url="",
        youtube="",
    )

def pick_recipe_mfds(cfg: AppConfig, used_pairs: set) -> Optional[Recipe]:
    if not cfg.mfds.api_key:
        return None
    # 너무 한정하지 말고 “한식에서 자주 찾는 키워드”로 랜덤
    keywords = ["김치", "된장", "고추장", "찌개", "국", "볶음", "조림", "전", "비빔", "나물", "탕", "김밥", "떡"]
    for _ in range(cfg.run.max_tries):
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.mfds.api_key, "RCP_NM", kw, start=1, end=60)
        random.shuffle(rows)
        for row in rows:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if not rcp.title or not rcp.steps:
                continue
            if (rcp.source, rcp.recipe_id) in used_pairs:
                continue
            return rcp
    return None

# -----------------------------
# TheMealDB
# -----------------------------
def _normalize_meal(m: Dict[str, Any]) -> Recipe:
    recipe_id = str(m.get("idMeal") or "").strip() or ""
    title = str(m.get("strMeal") or "").strip() or "오늘의 레시피"
    instructions = str(m.get("strInstructions") or "").strip()
    thumb = str(m.get("strMealThumb") or "").strip()
    src = str(m.get("strSource") or "").strip()
    yt = str(m.get("strYoutube") or "").strip()

    ingredients: List[str] = []
    for i in range(1, 21):
        ing = str(m.get(f"strIngredient{i}") or "").strip()
        mea = str(m.get(f"strMeasure{i}") or "").strip()
        if ing:
            line = f"{ing} - {mea}".strip() if mea else ing
            ingredients.append(line)

    steps = split_steps(instructions)

    return Recipe(
        source="themealdb",
        recipe_id=recipe_id or hashlib.sha1(title.encode("utf-8")).hexdigest()[:8],
        title=title,
        ingredients=ingredients,
        steps=steps,
        image_url=thumb if thumb.startswith("http") else "",
        source_url=src,
        youtube=yt,
    )

def fetch_random_themealdb() -> Recipe:
    r = requests.get(THEMEALDB_RANDOM, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"TheMealDB failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("TheMealDB empty response")
    return _normalize_meal(meals[0])

def fetch_themealdb_by_id(recipe_id: str) -> Recipe:
    r = requests.get(THEMEALDB_LOOKUP.format(id=recipe_id), timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"TheMealDB lookup failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("TheMealDB lookup empty response")
    return _normalize_meal(meals[0])

def split_steps(instructions: str) -> List[str]:
    t = (instructions or "").strip()
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"\r?\n+", t) if p.strip()]
    if len(parts) <= 2 and len(t) > 400:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    return parts

# -----------------------------
# Rendering helpers (백슬래시 f-string 오류 방지 포함)
# -----------------------------
def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)

def nl2br_escaped(s: str) -> str:
    # f-string 내부에서 replace('\\n', ...) 하지 않기 위해 밖에서 처리
    return _esc(s).replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")

def choose_thumb_url(cfg: AppConfig, recipe: Recipe) -> str:
    return (recipe.image_url or "").strip() or (cfg.img.default_thumb_url or "").strip()

# -----------------------------
# OpenAI + fallback detection
# -----------------------------
def _is_insufficient_quota_error(e: Exception) -> bool:
    s = (repr(e) + " " + str(e)).lower()
    return ("insufficient_quota" in s) or ("exceeded your current quota" in s) or ("check your plan and billing" in s)

def openai_call_with_retry(
    client: OpenAI,
    model: str,
    instructions: str,
    input_text: str,
    max_retries: int,
    debug: bool = False,
):
    last_e: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            # 최신 SDK에서 지원: instructions + input
            return client.responses.create(model=model, instructions=instructions, input=input_text)
        except TypeError:
            # 혹시 instructions 파라미터가 막힌 환경이면, input에 합쳐서 호출
            merged = f"[지시사항]\n{instructions}\n\n[입력]\n{input_text}"
            return client.responses.create(model=model, input=merged)
        except openai.RateLimitError as e:
            last_e = e
            if _is_insufficient_quota_error(e):
                raise
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] RateLimit → retry in {sleep_s:.2f}s")
            time.sleep(sleep_s)
        except openai.APIError as e:
            last_e = e
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] APIError → retry in {sleep_s:.2f}s | {repr(e)}")
            time.sleep(sleep_s)
        except Exception as e:
            last_e = e
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] UnknownError → retry in {sleep_s:.2f}s | {repr(e)}")
            time.sleep(sleep_s)
    if last_e:
        raise last_e
    raise RuntimeError("OpenAI call failed")

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

def build_fallback_post_with_free_translate(cfg: AppConfig, now: datetime, recipe: Recipe) -> Tuple[str, str]:
    # 최소한의 자연스러운 구성(너무 AI처럼 안 보이게, 과장 X)
    title_ko = free_translate_text(cfg.free_tr, recipe.title, debug=cfg.run.debug) or recipe.title

    ing_lines = "".join([f"<li>{_esc(x)}</li>" for x in recipe.ingredients]) or "<li>-</li>"
    step_lines = ""
    for s in recipe.steps[:20]:
        s_ko = free_translate_text(cfg.free_tr, s, debug=cfg.run.debug) if recipe.source != "mfds" else s
        step_lines += f"<li>{nl2br_escaped(s_ko)}</li>"

    body = f"""
<p>오늘은 <b>{_esc(title_ko)}</b>를 간단히 정리했어요. (복잡한 설명 없이 바로 따라가기용)</p>
<h2>재료</h2>
<ul>{ing_lines}</ul>
<h2>만드는 법</h2>
<ol>{step_lines or "<li>-</li>"}</ol>
<p style="opacity:.75;font-size:13px;">※ 자동 정리 글입니다. 원문 출처는 하단 링크를 참고해주세요.</p>
"""
    # 네이버 복붙용 태그(너무 길지 않게)
    tags = f"<p>#레시피 #집밥 #오늘뭐먹지 #{_esc(title_ko).replace(' ', '')}</p>"
    body += "<h2>태그</h2>" + tags

    # 날짜/슬롯 접두는 너무 로봇 같을 수 있어 뒤로 뺌
    title = f"{title_ko} (간단 레시피)"
    return title, body.strip()

# -----------------------------
# OpenAI: “사람이 쓴 느낌” + 네이버 복붙 가독성
# -----------------------------
def generate_naver_friendly_post(cfg: AppConfig, now: datetime, recipe: Recipe) -> Tuple[str, str]:
    client = OpenAI(api_key=cfg.openai.api_key)

    # 스타일 랜덤 시드(매번 똑같은 말투 방지)
    style_seed = random.choice([
        "담백하고 현실적인 톤(과장 금지), 문장 길이 섞기",
        "친한 지인에게 알려주듯(너무 가볍지 않게), 군더더기 제거",
        "요리 초보 기준, 핵심만 콕 집는 톤(말 많지 않게)",
    ])

    # 네이버 복붙에서 “AI 티” 자주 나는 표현 금지
    banned_phrases = [
        "지금부터", "알아볼까요", "함께 알아", "완벽한", "최고의", "~해볼게요!",
        "이 글에서는", "결론적으로", "요약하자면", "본 포스팅", "독자 여러분",
        "ChatGPT", "AI", "인공지능",
    ]

    # 재료/과정은 절대 변경 금지
    payload = {
        "title": recipe.title,
        "ingredients": recipe.ingredients,
        "steps": recipe.steps,
        "source": recipe.source,
        "source_url": recipe.source_url,
        "youtube": recipe.youtube,
        "style_seed": style_seed,
        "banned_phrases": banned_phrases,
        "now_kst": now.astimezone(KST).strftime("%Y-%m-%d %H:%M"),
    }

    instructions = (
        "너는 한국어로 글을 쓰는 블로그 작성자다. 목적은 '네이버 블로그에 복사/붙여넣기 했을 때' 읽기 좋은 글.\n"
        "\n"
        "[절대 규칙]\n"
        "1) ingredients/steps는 절대 추가/삭제/변형하지 마. 순서도 바꾸지 마.\n"
        "2) 시간/온도/계량을 원문에 없으면 단정하지 마. 대신 '상태를 보며' 같은 표현으로 처리.\n"
        "3) 금지어(banned_phrases)는 절대 쓰지 마. (변형 포함)\n"
        "4) 과장 광고/과도한 감탄/뻔한 SEO 문구 금지. '사람 글'처럼 자연스럽게.\n"
        "\n"
        "[형식]\n"
        "첫 줄: 제목(한국어, 24~44자, 클릭 유도는 하되 자극적으로 금지)\n"
        "그 다음 줄부터: HTML만 출력(마크다운/코드블럭 금지)\n"
        "\n"
        "[본문 구성(네이버 복붙 가독성)]\n"
        "1) <p>첫 문단 3~4줄: 상황 1줄 + 실패 포인트 1줄 + 이 글에서 해결 1줄 + 맛 포인트 1줄</p>\n"
        "2) <h2>한눈에 보는 포인트</h2> (3~5개 bullet, 짧게)\n"
        "3) <h2>재료</h2><ul> (그대로)</ul>\n"
        "4) <h2>만드는 법</h2><ol>\n"
        "   - steps 개수만큼만 <li>를 만들고, 각 <li>에는:\n"
        "     (a) 원문 step을 자연스러운 한국어로(의미 동일)\n"
        "     (b) 바로 아래 <p style='opacity:.78;font-size:13px;'>실수 방지: ...</p> 1줄(일반 조언)\n"
        "5) <h2>자주 하는 질문</h2> Q/A 3개(짧게)\n"
        "6) <h2>마무리</h2> 저장/댓글 유도 1문장씩(강요 말고 자연스럽게)\n"
        "7) <h2>태그</h2> #태그 8~15개(너무 과하지 않게)\n"
        "8) <p style='opacity:.7;font-size:13px;'>출처/자동정리 안내</p>\n"
    )

    input_text = "레시피 데이터(JSON):\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    resp = openai_call_with_retry(
        client=client,
        model=cfg.openai.model,
        instructions=instructions,
        input_text=input_text,
        max_retries=cfg.openai.max_retries,
        debug=cfg.run.debug,
    )
    text = (getattr(resp, "output_text", None) or "").strip()
    if not text:
        raise RuntimeError("OpenAI 응답이 비었습니다.")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("OpenAI 응답 형식 오류(제목/본문 분리 실패).")

    title = lines[0]
    body = "\n".join(lines[1:]).strip()
    if "<" not in body:
        body = "<p>" + nl2br_escaped(body) + "</p>"
    return title, body

# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = {"day": "오늘", "am": "오전", "pm": "오후"}.get(slot, "오늘")
    date_slot = f"{date_str}_{slot}"

    init_db(cfg.sqlite_path, debug=cfg.run.debug)
    today_meta = get_today_post(cfg.sqlite_path, date_slot)
    used_pairs = set(get_recent_recipe_pairs(cfg.sqlite_path, cfg.run.avoid_repeat_days))

    # 레시피 선택: 이미 오늘 올린 게 있으면 재사용(강제 새글 제외)
    recipe: Optional[Recipe] = None
    if today_meta and not cfg.run.force_new and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        if today_meta["recipe_source"] == "mfds":
            # mfds는 id로 직접 조회가 애매해서, 그냥 새로 뽑되 중복만 회피
            recipe = None
        else:
            try:
                recipe = fetch_themealdb_by_id(today_meta["recipe_id"])
            except Exception:
                recipe = None

    if not recipe:
        # 1) mfds 우선
        recipe = pick_recipe_mfds(cfg, used_pairs)
    if not recipe:
        # 2) themealdb
        for _ in range(cfg.run.max_tries):
            cand = fetch_random_themealdb()
            if (cand.source, cand.recipe_id) in used_pairs:
                continue
            if not cand.steps:
                continue
            recipe = cand
            break
    if not recipe:
        raise RuntimeError("레시피 선택 실패(시도 횟수 초과).")

    # 썸네일 결정(없으면 default)
    thumb_url = choose_thumb_url(cfg, recipe)

    # 미디어 업로드(가능하면)
    media_id = 0
    media_url = ""
    if cfg.img.upload_thumb and thumb_url:
        try:
            h = hashlib.sha1(thumb_url.encode("utf-8")).hexdigest()[:12]
            ext = ".jpg"
            if thumb_url.lower().endswith(".png"):
                ext = ".png"
            filename = f"recipe_{date_str}_{slot}_{h}{ext}"
            media_id, media_url = wp_upload_media_from_url(cfg.wp, thumb_url, filename)
        except Exception as e:
            if cfg.run.debug:
                print("[IMG] upload failed:", repr(e))
            media_id, media_url = 0, ""

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    # 본문 생성(OpenAI -> 실패/쿼터면 폴백)
    try:
        title_ko, body_html = generate_naver_friendly_post(cfg, now, recipe)
    except Exception as e:
        if _is_insufficient_quota_error(e):
            print("[WARN] OpenAI quota issue → fallback translate")
            title_ko, body_html = build_fallback_post_with_free_translate(cfg, now, recipe)
        else:
            # 그 외 실패: 폴백으로라도 발행
            print("[WARN] OpenAI failed → fallback translate:", repr(e))
            title_ko, body_html = build_fallback_post_with_free_translate(cfg, now, recipe)

    # 본문 상단 이미지(업로드 성공 URL 우선)
    if cfg.img.embed_image_in_body:
        display_img = (media_url or thumb_url or "").strip()
        if display_img:
            body_html = f'<p><img src="{_esc(display_img)}" alt="{_esc(title_ko)}" style="max-width:100%;height:auto;"></p>\n' + body_html

    # 제목/슬러그(너무 규칙적이면 티가 나서, 날짜는 접두가 아니라 약하게)
    # 네이버 복붙 시에도 자연스럽게: 날짜는 제목 끝에 괄호로 처리
    final_title = f"{title_ko} ({date_str} {slot_label})"
    slug = f"daily-recipe-{date_str}-{slot}"

    # DRY_RUN
    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", final_title)
        print("SLUG:", slug)
        print(body_html[:2000] + ("\n...(truncated)" if len(body_html) > 2000 else ""))
        return

    # 발행/업데이트
    if today_meta and today_meta.get("wp_post_id"):
        post_id = int(today_meta["wp_post_id"])
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, final_title, body_html, featured_media=featured_id)
        print("OK(updated):", wp_post_id, wp_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, final_title, slug, body_html, featured_media=featured_id)
        print("OK(created):", wp_post_id, wp_link)

    # 저장
    save_post_meta(cfg.sqlite_path, {
        "date_slot": date_slot,
        "recipe_source": recipe.source,
        "recipe_id": recipe.recipe_id,
        "recipe_title": recipe.title,
        "wp_post_id": wp_post_id,
        "wp_link": wp_link,
        "media_id": media_id,
        "media_url": media_url,
        "created_at": datetime.utcnow().isoformat(),
    })

def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)

if __name__ == "__main__":
    main()
