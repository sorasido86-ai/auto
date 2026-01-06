# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp.py (완전 통합/완성본)
- "한식 레시피만" 매일 자동 업로드 (WordPress)
- 1순위: 식품안전나라(식약처) COOKRCP01 OpenAPI 레시피 DB (MFDS_API_KEY 필요)
- 2순위(폴백): 코드 내장 한식 레시피(한국어)

★ 요청 반영(중요)
- 레시피 이미지가 비어도 DEFAULT_THUMB_URL 기본 이미지가 반드시 썸네일 후보가 되도록 처리
- 가능하면 WP media로 업로드 후 featured_media(대표이미지) 설정까지 수행
- 업로드 실패 시에도 본문 상단 이미지는 URL로 삽입(가능한 선에서 "이미지 보이게")

필수 환경변수(Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

권장 환경변수:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS=7 (기본 7)
  - WP_TAG_IDS=1,2,3 (선택)
  - SQLITE_PATH=data/daily_korean_recipe.sqlite3

식품안전나라 OpenAPI:
  - MFDS_API_KEY=...  (없으면 내장 레시피만)

기본 이미지(중요):
  - DEFAULT_THUMB_URL=https://.../your_default_thumb.jpg
    -> 레시피 이미지가 없을 때 이 이미지로 "대표이미지 업로드/featured"까지 시도

OpenAI로 블로거톤 강화(선택):
  - USE_OPENAI=1
  - OPENAI_API_KEY=...
  - OPENAI_MODEL=... (기본 gpt-4.1-mini)

동작 옵션:
  - RUN_SLOT=day|am|pm (기본 day)
  - FORCE_NEW=0|1 (기본 0)  : 오늘 이미 올렸어도 새 레시피로 "교체 업데이트"
  - DRY_RUN=0|1 (기본 0)
  - DEBUG=0|1 (기본 0)
  - AVOID_REPEAT_DAYS=90 (기본 90)
  - MAX_TRIES=25 (기본 25)

이미지 옵션:
  - UPLOAD_THUMB=1 (기본 1)
  - SET_FEATURED=1 (기본 1)
  - EMBED_IMAGE_IN_BODY=1 (기본 1)
  - REUSE_MEDIA_BY_SEARCH=1 (기본 1) : 같은 파일명/검색으로 기존 미디어 재사용(중복 업로드 방지)
"""

from __future__ import annotations

import base64
import hashlib
import html
import os
import random
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

KST = timezone(timedelta(hours=9))

# -----------------------------
# 내장 한식 레시피(폴백)
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "kimchi-jjigae",
        "title": "돼지고기 김치찌개",
        "ingredients": [
            ("신김치", "2컵"),
            ("돼지고기(앞다리/삼겹)", "200g"),
            ("양파", "1/2개"),
            ("대파", "1대"),
            ("두부", "1/2모"),
            ("고춧가루", "1큰술"),
            ("다진마늘", "1큰술"),
            ("국간장", "1큰술"),
            ("멸치다시마 육수(또는 물)", "700ml"),
        ],
        "steps": [
            "냄비에 돼지고기를 넣고 중불에서 기름이 살짝 돌 때까지 볶아주세요.",
            "신김치를 넣고 2~3분 더 볶아 김치의 신맛을 한 번 눌러줍니다.",
            "고춧가루/다진마늘/국간장을 넣고 30초만 볶아 향을 내요.",
            "육수를 붓고 10~12분 끓입니다.",
            "양파를 넣고 3분, 두부를 넣고 2분 더 끓인 뒤 대파로 마무리!",
        ],
        "image_url": "",
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1.5큰술"),
            ("고추장(선택)", "1/2큰술"),
            ("애호박", "1/3개"),
            ("양파", "1/3개"),
            ("두부", "1/2모"),
            ("대파", "1/2대"),
            ("다진마늘", "1작은술"),
            ("멸치다시마 육수(또는 물)", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 5분 끓여요.",
            "양파/애호박/두부 넣고 5~6분 더 끓입니다.",
            "대파 넣고 한 번만 더 끓인 뒤 간을 보고 마무리!",
        ],
        "image_url": "",
    },
    {
        "id": "bulgogi",
        "title": "간장 불고기",
        "ingredients": [
            ("소고기 불고기용", "300g"),
            ("양파", "1/2개"),
            ("대파", "1대"),
            ("간장", "4큰술"),
            ("설탕", "1큰술"),
            ("다진마늘", "1큰술"),
            ("참기름", "1큰술"),
            ("후추", "약간"),
            ("물(또는 배즙)", "3큰술"),
        ],
        "steps": [
            "간장/설탕/다진마늘/참기름/물/후추로 양념장을 섞어요.",
            "고기에 양념장을 넣고 15분 이상 재워둡니다.",
            "팬에 고기를 볶고, 양파/대파를 넣어 숨이 죽을 때까지 볶아요.",
        ],
        "image_url": "",
    },
]

KOREAN_NEGATIVE_KEYWORDS = ["파스타", "피자", "타코", "스시", "커리", "샌드위치", "버거", "샐러드"]

DISCLOSURE = "※ 이 글은 레시피 데이터를 기반으로 자동 생성된 포스팅입니다."
SOURCE_NOTE = "데이터 출처: 식품안전나라(식약처) OpenAPI 레시피 DB 및 내장 레시피(폴백)."
SEO_NOTE = "오늘 뭐 먹지 고민될 때, 재료 적고 실패 확률 낮은 레시피로 골라왔어요 🙂"


# -----------------------------
# Env helpers
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
    run_slot: str = "day"       # day / am / pm
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 25


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""      # foodsafetykorea openapi key (optional)
    strict_korean: bool = True


@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""
    reuse_media_by_search: bool = True


@dataclass
class OpenAIConfig:
    use_openai: bool = False
    api_key: str = ""
    model: str = "gpt-4.1-mini"


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    recipe: RecipeSourceConfig
    img: ImageConfig
    openai: OpenAIConfig
    sqlite_path: str


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))  # 기본 7
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

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
            force_new=_env_bool("FORCE_NEW", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 25),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
        ),
        img=ImageConfig(
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
        ),
        openai=OpenAIConfig(
            use_openai=_env_bool("USE_OPENAI", False),
            api_key=_env("OPENAI_API_KEY", ""),
            model=_env("OPENAI_MODEL", "gpt-4.1-mini"),
        ),
        sqlite_path=_env("SQLITE_PATH", "data/daily_korean_recipe.sqlite3"),
    )


def validate_cfg(cfg: AppConfig) -> None:
    missing = []
    if not cfg.wp.base_url:
        missing.append("WP_BASE_URL")
    if not cfg.wp.user:
        missing.append("WP_USER")
    if not cfg.wp.app_pass:
        missing.append("WP_APP_PASS")
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
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| STRICT_KOREAN:", cfg.recipe.strict_korean)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] UPLOAD_THUMB:", cfg.img.upload_thumb, "| SET_FEATURED:", cfg.img.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.img.embed_image_in_body)
    print("[CFG] REUSE_MEDIA_BY_SEARCH:", cfg.img.reuse_media_by_search)
    print("[CFG] USE_OPENAI:", cfg.openai.use_openai, "| OPENAI_API_KEY:", ok(cfg.openai.api_key), "| OPENAI_MODEL:", cfg.openai.model)


# -----------------------------
# SQLite (history + migrate)
# -----------------------------
REQUIRED_COLS = {
    "date_slot": "TEXT PRIMARY KEY",
    "recipe_source": "TEXT",
    "recipe_id": "TEXT",
    "recipe_title": "TEXT",
    "wp_post_id": "INTEGER",
    "wp_link": "TEXT",
    "media_id": "INTEGER",
    "media_url": "TEXT",
    "created_at": "TEXT",
}

def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
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
    )
    con.commit()

    cur.execute("PRAGMA table_info(daily_posts)")
    existing = {row[1] for row in cur.fetchall()}
    for col, coldef in REQUIRED_COLS.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE daily_posts ADD COLUMN {col} {coldef}")
    con.commit()
    con.close()


def get_today_post(path: str, date_slot: str) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at
        FROM daily_posts WHERE date_slot = ?
        """,
        (date_slot,),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "date_slot": row[0],
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
            meta.get("created_at", datetime.utcnow().isoformat()),
        ),
    )
    con.commit()
    con.close()


def get_recent_recipe_ids(path: str, days: int) -> List[Tuple[str, str]]:
    since = datetime.utcnow() - timedelta(days=days)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT recipe_source, recipe_id
        FROM daily_posts
        WHERE created_at IS NOT NULL AND created_at != '' AND created_at >= ?
        """,
        (since.isoformat(),),
    )
    rows = cur.fetchall()
    con.close()
    out: List[Tuple[str, str]] = []
    for s, rid in rows:
        if s and rid:
            out.append((str(s), str(rid)))
    return out


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status}
    if featured_media:
        payload["featured_media"] = featured_media
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_find_media_by_search(cfg: WordPressConfig, search: str) -> Optional[Tuple[int, str]]:
    # /wp/v2/media?search=... 로 조회 (중복 업로드 방지용)
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass)
    params = {"search": search, "per_page": 10}
    r = requests.get(url, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        return None
    try:
        items = r.json()
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    # 가장 첫 항목 사용
    it = items[0]
    mid = int(it.get("id") or 0)
    src = str(it.get("source_url") or "")
    if mid and src:
        return mid, src
    return None


def wp_upload_media_from_url(cfg: WordPressConfig, image_url: str, filename: str) -> Tuple[int, str]:
    # download
    r = requests.get(image_url, timeout=35)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code}")

    content = r.content
    ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()

    # fallback mime
    if not ctype:
        if filename.lower().endswith(".png"):
            ctype = "image/png"
        elif filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            ctype = "image/jpeg"
        else:
            ctype = "application/octet-stream"

    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
    }

    rr = requests.post(url, headers=headers, data=content, timeout=60)
    if rr.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {rr.status_code} body={rr.text[:500]}")
    data = rr.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# Recipe model / MFDS provider
# -----------------------------
@dataclass
class Recipe:
    source: str          # "mfds" or "local"
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    image_url: str = ""
    step_images: List[str] = field(default_factory=list)

    def uid(self) -> str:
        s = f"{self.source}|{self.recipe_id}|{self.title}"
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _has_hangul(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))


def _is_korean_recipe_name(name: str, strict: bool = True) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if strict and not _has_hangul(n):
        return False
    for bad in KOREAN_NEGATIVE_KEYWORDS:
        if bad in n:
            return False
    return True


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
    title = str(row.get("RCP_NM") or "").strip()
    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()

    ingredients: List[str] = []
    for p in re.split(r"\s*,\s*", parts):
        p = p.strip()
        if p:
            ingredients.append(p)

    steps: List[str] = []
    step_imgs: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        img = str(row.get(f"MANUAL_IMG{str(i).zfill(2)}") or "").strip()
        if s:
            s = re.sub(r"[a-zA-Z]\s*$", "", s).strip()
            steps.append(s)
            if img.startswith("http"):
                step_imgs.append(img)

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
        step_images=step_imgs,
    )


def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡"]
    for _ in range(cfg.run.max_tries):
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, start=1, end=60)
        random.shuffle(rows)
        for row in rows:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if cfg.recipe.strict_korean and not _is_korean_recipe_name(rcp.title, strict=True):
                continue
            if (rcp.source, rcp.recipe_id) in used:
                continue
            if not rcp.title or not rcp.steps:
                continue
            return rcp
    return None


def pick_recipe_local(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Recipe:
    used = set(recent_pairs)
    pool = [x for x in LOCAL_KOREAN_RECIPES if ("local", str(x["id"])) not in used]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    pick = random.choice(pool)
    ing = [f"{a} - {b}".strip() for a, b in pick.get("ingredients", [])]
    steps = [str(s).strip() for s in pick.get("steps", []) if str(s).strip()]
    return Recipe(
        source="local",
        recipe_id=str(pick["id"]),
        title=str(pick["title"]),
        ingredients=ing,
        steps=steps,
        image_url=str(pick.get("image_url") or "").strip(),
        step_images=[],
    )


def get_recipe_by_id(cfg: AppConfig, source: str, recipe_id: str) -> Optional[Recipe]:
    if source == "local":
        for x in LOCAL_KOREAN_RECIPES:
            if str(x.get("id")) == recipe_id:
                ing = [f"{a} - {b}".strip() for a, b in x.get("ingredients", [])]
                steps = [str(s).strip() for s in x.get("steps", []) if str(s).strip()]
                return Recipe(
                    source="local",
                    recipe_id=recipe_id,
                    title=str(x.get("title") or ""),
                    ingredients=ing,
                    steps=steps,
                    image_url=str(x.get("image_url") or "").strip(),
                )
        return None

    if source == "mfds" and cfg.recipe.mfds_api_key:
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_SEQ", recipe_id, start=1, end=5)
        for row in rows:
            rcp = mfds_row_to_recipe(row)
            if rcp.recipe_id == recipe_id:
                return rcp
        return None

    return None


# -----------------------------
# Blog rendering
# -----------------------------
def _esc(s: str) -> str:
    return html.escape(s or "")


def choose_thumb_url(cfg: AppConfig, recipe: Recipe) -> str:
    # ★ 핵심: 레시피 이미지가 비어도 기본 이미지로 대체
    return (recipe.image_url or "").strip() or (cfg.img.default_thumb_url or "").strip()


def build_body_html(cfg: AppConfig, now: datetime, run_slot_label: str, recipe: Recipe, display_img_url: str = "") -> str:
    title = recipe.title.strip()

    img_html = ""
    if cfg.img.embed_image_in_body and display_img_url:
        img_html = f"""
        <p style="margin:14px 0;">
          <img src="{_esc(display_img_url)}" alt="{_esc(title)}"
               style="max-width:100%;height:auto;border-radius:10px;" />
        </p>
        """

    disclosure = f'<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{_esc(DISCLOSURE)}</p>'
    head = f"<p>기준시각: <b>{_esc(now.astimezone(KST).strftime('%Y-%m-%d %H:%M'))}</b> / 슬롯: <b>{_esc(run_slot_label)}</b></p>"
    note = f'<p style="font-size:13px;opacity:.85;">{_esc(SEO_NOTE)}<br/>{_esc(SOURCE_NOTE)}</p>'

    hook = f"""
    <h2>{_esc(title)} 레시피</h2>
    <p>
      오늘은 <b>{_esc(title)}</b>로 갑니다 🙂<br/>
      재료는 단순하게, 과정은 실패 확률 낮게 정리했어요. (바쁜 날에도 OK!)
    </p>
    """

    ing_li = "".join([f"<li>{_esc(x)}</li>" for x in recipe.ingredients]) or "<li>재료 정보가 비어있어요.</li>"
    ingredients = f"<h3>재료 준비</h3><ul>{ing_li}</ul>"

    step_ol = "".join([f"<li style='margin:6px 0;'>{_esc(s)}</li>" for s in recipe.steps]) or "<li>조리 과정 정보가 비어있어요.</li>"
    steps = f"<h3>만드는 법</h3><ol>{step_ol}</ol>"

    tips = """
    <h3>실패 줄이는 팁</h3>
    <ul>
      <li>간은 마지막에 한 번 더 잡아주면 실패 확률이 확 줄어요.</li>
      <li>시간이 없으면 재료를 크게 썰어도 괜찮아요. 대신 충분히 끓이기!</li>
      <li>매운맛은 고춧가루/고추장으로 단계적으로 조절하면 깔끔합니다.</li>
    </ul>
    """

    closing = """
    <hr/>
    <p style="opacity:.85;">
      저장해두면 다음에 바로 꺼내 쓰기 좋아요 🙂<br/>
      내일 레시피도 1개씩 업데이트됩니다.
    </p>
    """

    return disclosure + head + img_html + note + hook + ingredients + steps + tips + closing


def generate_with_openai(cfg: AppConfig, recipe: Recipe, base_html: str) -> Optional[str]:
    if not (cfg.openai.use_openai and cfg.openai.api_key):
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    try:
        client = OpenAI(api_key=cfg.openai.api_key)
        prompt = f"""
너는 한국 요리 블로그 전문 에디터야.
아래 레시피(제목/재료/과정)는 내용이 바뀌면 안 돼. 그대로 유지해.
대신 도입부/설명/팁/마무리를 더 자연스럽고 조회수 잘 나오는 블로거 말투로 다듬어줘.
HTML 형태로만 출력해. (코드블럭 금지)
너무 과장된 광고 문구는 금지. 담백하지만 먹고 싶게.

[레시피 제목]
{recipe.title}

[재료]
- """ + "\n- ".join(recipe.ingredients) + """

[과정]
""" + "\n".join([f"{i+1}) {s}" for i, s in enumerate(recipe.steps)]) + f"""

[현재 HTML 초안]
{base_html}
"""
        resp = client.responses.create(model=cfg.openai.model, input=prompt)
        out_text = getattr(resp, "output_text", None)
        if out_text and "<" in out_text:
            return out_text.strip()
    except Exception:
        return None
    return None


def ensure_media(cfg: AppConfig, image_url: str, stable_name: str) -> Tuple[int, str]:
    """
    - 중복 업로드 방지(가능하면): search로 기존 미디어 재사용
    - 없으면 업로드
    """
    if not image_url:
        return 0, ""

    # 파일명 결정(고정): 같은 URL이면 같은 이름을 쓰게 해서 search 재사용 유리
    h = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    ext = ".jpg"
    u = image_url.lower()
    if u.endswith(".png"):
        ext = ".png"
    elif u.endswith(".jpeg"):
        ext = ".jpg"
    filename = f"{stable_name}_{h}{ext}"

    if cfg.img.reuse_media_by_search:
        found = wp_find_media_by_search(cfg.wp, search=f"{stable_name}_{h}")
        if found:
            return found

    mid, murl = wp_upload_media_from_url(cfg.wp, image_url, filename)
    return mid, murl


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = {"day": "오늘", "am": "오전", "pm": "오후"}.get(slot, "오늘")
    date_slot = f"{date_str}_{slot}"

    init_db(cfg.sqlite_path)

    today_meta = get_today_post(cfg.sqlite_path, date_slot)
    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days)

    print(f"[RUN] slot={slot} force_new={int(cfg.run.force_new)} date_slot={date_slot}")

    chosen: Optional[Recipe] = None
    if today_meta and not cfg.run.force_new and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

    if not chosen:
        chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(cfg, recent_pairs)

    assert chosen is not None

    title = f"{date_str} 한식 레시피 - {chosen.title} ({slot_label})"
    slug = f"korean-recipe-{date_str}-{slot}"

    # ★ 핵심: 레시피 이미지가 없으면 기본 이미지로 대체
    thumb_url = choose_thumb_url(cfg, chosen)

    if not chosen.image_url and not cfg.img.default_thumb_url:
        # 기본 이미지가 없으면 '이미지 없는 글'이 될 수 밖에 없어서 경고
        print("[WARN] recipe image empty AND DEFAULT_THUMB_URL empty → featured 이미지 없이 발행될 수 있어요.")

    # WP 업로드 성공 시 media_url을 쓰고, 실패하면 thumb_url(직접URL)로라도 본문 삽입
    media_id = 0
    media_url = ""

    if cfg.img.upload_thumb and thumb_url:
        try:
            media_id, media_url = ensure_media(cfg, thumb_url, stable_name="korean_recipe_thumb")
            if cfg.run.debug:
                print("[IMG] media:", media_id, media_url)
        except Exception as e:
            if cfg.run.debug:
                print("[IMG] upload failed:", repr(e))
            media_id, media_url = 0, ""

    display_img_url = (media_url or thumb_url or "").strip()

    body_html = build_body_html(cfg, now, slot_label, chosen, display_img_url=display_img_url)

    upgraded = generate_with_openai(cfg, chosen, body_html)
    if upgraded:
        body_html = upgraded

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 HTML 일부 ↓")
        print(body_html[:2000])
        print("... (truncated)")
        return

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    if today_meta and today_meta.get("wp_post_id"):
        post_id = int(today_meta["wp_post_id"])
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, featured_media=featured_id)
        print("OK(updated):", wp_post_id, wp_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html)
        if featured_id:
            try:
                wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id)
            except Exception:
                pass
        print("OK(created):", wp_post_id, wp_link)

    save_post_meta(
        cfg.sqlite_path,
        {
            "date_slot": date_slot,
            "recipe_source": chosen.source,
            "recipe_id": chosen.recipe_id,
            "recipe_title": chosen.title,
            "wp_post_id": wp_post_id,
            "wp_link": wp_link,
            "media_id": media_id,
            "media_url": media_url,
            "created_at": datetime.utcnow().isoformat(),
        },
    )


def main():
    cfg = load_cfg()
    validate_cfg(cfg)
    print_safe_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
