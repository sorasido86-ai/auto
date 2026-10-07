"""식품안전나라 또는 기본 한식 레시피의 재료·조리 순서를 발행합니다.
원문 계량·문장부호를 그대로 보존하고 임의 조리 팁·경험담을 추가하지 않습니다.
DRY_RUN=1은 게시 없이 HTML을 저장합니다."""

from __future__ import annotations

import base64
import hashlib
import os
import random
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from openai import OpenAI
from content_quality import (generate_recipe_article, render_recipe, require_recipe,
                             save_preview, recent_editorials, remember_editorial, recipe_response_format, recipe_model_options,
                             split_recipe_ingredients, editorial_context, recover_published_recipe)
from wp_common import write_post, find_post, recent_recipe_posts


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
            "냄비에 돼지고기를 넣고 중불에서 기름이 살짝 돌 때까지 볶아주세요",
            "신김치를 넣고 2 3분 더 볶아 김치의 신맛을 한 번 눌러줍니다",
            "고춧가루 다진마늘 국간장을 넣고 30초만 볶아 향을 내요",
            "육수를 붓고 10 12분 끓입니다",
            "양파를 넣고 3분 두부를 넣고 2분 더 끓인 뒤 대파로 마무리해요",
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
            "끓는 육수에 된장을 풀고 5분 끓여요",
            "양파 애호박 두부 넣고 5 6분 더 끓입니다",
            "대파 넣고 한 번만 더 끓인 뒤 간을 보고 마무리해요",
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
            "간장 설탕 다진마늘 참기름 물 후추로 양념장을 섞어요",
            "고기에 양념장을 넣고 15분 이상 재워둡니다",
            "팬에 고기를 볶고 양파 대파를 넣어 숨이 죽을 때까지 볶아요",
        ],
        "image_url": "",
    },
]

KOREAN_NEGATIVE_KEYWORDS = ["파스타", "피자", "타코", "스시", "커리", "샌드위치", "버거", "샐러드"]

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


def _parse_str_list(csv: str) -> List[str]:
    out: List[str] = []
    for x in (csv or "").split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


# -----------------------------
# Hard timeout wrapper
# -----------------------------

@contextmanager
def hard_timeout(seconds: int):
    """강제 종료 타임아웃  Linux(GitHub Actions)에서만 동작"""
    if seconds <= 0:
        yield
        return

    # Windows에서는 SIGALRM 없음 / 설정 실패 시 hard-timeout 비활성화
    try:
        import signal  # type: ignore
        sig = signal.SIGALRM
        it = signal.ITIMER_REAL
    except Exception:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"hard timeout {seconds}s")

    old = signal.signal(sig, _handler)
    signal.setitimer(it, float(seconds))
    try:
        yield
    finally:
        try:
            signal.setitimer(it, 0)
        except Exception:
            pass
        try:
            signal.signal(sig, old)
        except Exception:
            pass


def safe_get(url: str, *, timeout: int = 15, hard: int = 20, **kwargs) -> requests.Response:
    with hard_timeout(hard):
        return requests.get(url, timeout=timeout, **kwargs)


def safe_post(url: str, *, timeout: int = 20, hard: int = 25, **kwargs) -> requests.Response:
    with hard_timeout(hard):
        return requests.post(url, timeout=timeout, **kwargs)


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
    run_slot: str = "day"
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 12


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True
    mfds_timeout: int = 12
    mfds_budget_sec: int = 20


@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""
    reuse_media_by_search: bool = True
    auto_image: bool = True


@dataclass
class ContentConfig:
    # 글 길이
    intro_min: int = 200
    intro_max: int = 320
    body_min: int = 1200


@dataclass
class TagConfig:
    auto_tags: bool = True
    tag_names: List[str] = field(default_factory=list)


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    recipe: RecipeSourceConfig
    img: ImageConfig
    content: ContentConfig
    tags: TagConfig
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
            max_tries=_env_int("MAX_TRIES", 12),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout=_env_int("MFDS_TIMEOUT", 12),
            mfds_budget_sec=_env_int("MFDS_BUDGET_SEC", 20),
        ),
        img=ImageConfig(
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
            auto_image=_env_bool("AUTO_IMAGE", True),
        ),
        content=ContentConfig(
            intro_min=_env_int("INTRO_MIN", 200),
            intro_max=_env_int("INTRO_MAX", 320),
            body_min=_env_int("BODY_MIN", 1200),
        ),
        tags=TagConfig(
            auto_tags=_env_bool("AUTO_TAGS", True),
            tag_names=_parse_str_list(_env("TAG_NAMES", "한식레시피,집밥,오늘뭐먹지,간단요리,초간단레시피,자취요리")),
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
    print("[CFG] MFDS_TIMEOUT:", cfg.recipe.mfds_timeout, "| MFDS_BUDGET_SEC:", cfg.recipe.mfds_budget_sec)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] AUTO_IMAGE:", cfg.img.auto_image)
    print("[CFG] UPLOAD_THUMB:", cfg.img.upload_thumb, "| SET_FEATURED:", cfg.img.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.img.embed_image_in_body)
    print("[CFG] AUTO_TAGS:", cfg.tags.auto_tags, "| TAG_NAMES:", len(cfg.tags.tag_names))
    print("[CFG] BODY_MIN:", cfg.content.body_min)


# -----------------------------
# SQLite history
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/2.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, excerpt: str, tag_ids: List[int], featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status, "excerpt": excerpt}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if tag_ids:
        payload["tags"] = tag_ids
    if featured_media:
        payload["featured_media"] = featured_media

    data = write_post(url, headers, payload)
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, excerpt: str, tag_ids: List[int], featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status, "excerpt": excerpt}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if tag_ids:
        payload["tags"] = tag_ids
    if featured_media:
        payload["featured_media"] = featured_media

    data = write_post(url, headers, payload)
    return int(data["id"]), str(data.get("link") or "")


def wp_find_media_by_search(cfg: WordPressConfig, search: str) -> Optional[Tuple[int, str]]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass)
    params = {"search": search, "per_page": 10}
    r = safe_get(url, headers=headers, params=params, timeout=20, hard=25)
    if r.status_code != 200:
        return None
    try:
        items = r.json()
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    it = items[0]
    mid = int(it.get("id") or 0)
    src = str(it.get("source_url") or "")
    if mid and src:
        return mid, src
    return None


def wp_upload_media_from_url(cfg: WordPressConfig, image_url: str, filename: str) -> Tuple[int, str]:
    # 다운로드
    r = safe_get(image_url, timeout=20, hard=25, allow_redirects=True)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code}")

    content = r.content
    ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
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

    rr = safe_post(url, headers=headers, data=content, timeout=35, hard=45)
    if rr.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {rr.status_code} body={rr.text[:500]}")
    data = rr.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# Tags (optional)
# -----------------------------

def wp_find_tag_id(cfg: WordPressConfig, name: str) -> int:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/tags"
    headers = wp_auth_header(cfg.user, cfg.app_pass)
    params = {"search": name, "per_page": 20}
    r = safe_get(url, headers=headers, params=params, timeout=20, hard=25)
    if r.status_code != 200:
        return 0
    try:
        items = r.json()
    except Exception:
        return 0
    if not isinstance(items, list):
        return 0
    for it in items:
        if str(it.get("name") or "").strip() == name:
            return int(it.get("id") or 0)
    return 0


def wp_create_tag(cfg: WordPressConfig, name: str) -> int:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/tags"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload = {"name": name}
    r = safe_post(url, headers=headers, json=payload, timeout=25, hard=35)
    if r.status_code not in (200, 201):
        return 0
    try:
        data = r.json()
    except Exception:
        return 0
    return int(data.get("id") or 0)


def build_tag_names(recipe_title: str, base: List[str]) -> List[str]:
    # 제목 토큰 일부 + 기본 태그
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", recipe_title or "")
    toks = [x.strip() for x in t.split() if x.strip()]
    out: List[str] = []
    seen = set()

    for x in base + toks[:6]:
        x = re.sub(r"\s+", "", x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        if len(x) <= 1:
            continue
        out.append(x)
    return out[:15]


def ensure_tag_ids(cfg: AppConfig, names: List[str]) -> List[int]:
    if not cfg.tags.auto_tags:
        return cfg.wp.tag_ids[:]  # env의 ID 태그만

    out = cfg.wp.tag_ids[:]
    for nm in names:
        nm = nm.strip()
        if not nm:
            continue
        tid = wp_find_tag_id(cfg.wp, nm)
        if not tid:
            tid = wp_create_tag(cfg.wp, nm)
        if tid and tid not in out:
            out.append(tid)
        # 속도/부하 방지
        if len(out) >= 20:
            break
    return out


# -----------------------------
# Recipe model / MFDS provider
# -----------------------------

@dataclass
class Recipe:
    source: str  # mfds|local
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    image_url: str = ""

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


def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int, end: int, timeout: int) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    try:
        r = safe_get(url, timeout=timeout, hard=timeout + 6)
        if r.status_code != 200:
            return []
        data = r.json()
        co = data.get("COOKRCP01") or {}
        rows = co.get("row") or []
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or ""
    title = str(row.get("RCP_NM") or "").strip()
    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()

    ingredients = split_recipe_ingredients(parts)

    steps: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        if s:
            # MANUAL fields carry display labels such as "1.손질한다".
            # Do not mistake labels for quantities or strip decimal values/units.
            s = re.sub(r"^\d+[.)](?!\d)\s*", "", s).strip()
            if s:
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
    )


def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡"]

    t0 = time.time()
    tries = 0

    while tries < cfg.run.max_tries:
        tries += 1
        if (time.time() - t0) > float(cfg.recipe.mfds_budget_sec):
            return None

        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, 1, 60, cfg.recipe.mfds_timeout)
        if not rows:
            continue

        random.shuffle(rows)
        for row in rows[:20]:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if cfg.recipe.strict_korean and not _is_korean_recipe_name(rcp.title, True):
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
    ing = [f"{a} {b}".strip() for a, b in pick.get("ingredients", [])]
    steps = [str(s).strip() for s in pick.get("steps", []) if str(s).strip()]

    return Recipe(
        source="local",
        recipe_id=str(pick["id"]),
        title=str(pick["title"]),
        ingredients=ing,
        steps=steps,
        image_url=str(pick.get("image_url") or "").strip(),
    )


def get_recipe_by_id(cfg: AppConfig, source: str, recipe_id: str) -> Optional[Recipe]:
    if source == "local":
        for x in LOCAL_KOREAN_RECIPES:
            if str(x.get("id")) == recipe_id:
                ing = [f"{a} {b}".strip() for a, b in x.get("ingredients", [])]
                steps = [str(s).strip() for s in x.get("steps", []) if str(s).strip()]
                return Recipe("local", recipe_id, str(x.get("title") or ""), ing, steps, str(x.get("image_url") or "").strip())
        return None

    if source == "mfds" and cfg.recipe.mfds_api_key:
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_SEQ", recipe_id, 1, 5, cfg.recipe.mfds_timeout)
        for row in rows:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if rcp.recipe_id == recipe_id:
                return rcp

    return None


# -----------------------------
# Stable selection seed
# -----------------------------


def _seed_rng(seed: str) -> random.Random:
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return random.Random(int(h, 16))


# -----------------------------
# Title
# -----------------------------

def build_post_title(date_str: str, slot_label: str, recipe_title: str, rng: random.Random) -> str:
    return recipe_title


# -----------------------------
# Images
# -----------------------------









def choose_thumb_url(cfg: AppConfig, recipe: Recipe) -> str:
    # A generic stock-photo search must not pretend to show the chosen dish.
    return (recipe.image_url or cfg.img.default_thumb_url or "").strip()



def ensure_media(cfg: AppConfig, image_url: str, stable_name: str) -> Tuple[int, str]:
    if not image_url:
        return 0, ""

    h = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    ext = ".jpg"
    u = image_url.lower()
    if u.endswith(".png"):
        ext = ".png"
    elif u.endswith(".jpeg"):
        ext = ".jpg"

    filename = f"{stable_name}_{h}{ext}"

    if cfg.img.reuse_media_by_search:
        try:
            found = wp_find_media_by_search(cfg.wp, search=f"{stable_name}_{h}")
            if found:
                return found
        except Exception:
            pass

    mid, murl = wp_upload_media_from_url(cfg.wp, image_url, filename)
    return mid, murl


# -----------------------------
# Body builder (homefeed)
# -----------------------------


def build_body_html(cfg: AppConfig, recipe: Recipe, display_img_url: str, rng: random.Random) -> Tuple[str, str]:
    require_recipe(recipe.ingredients, recipe.steps)
    key = _env("OPENAI_API_KEY", "")
    if key:
        client = OpenAI(api_key=key, timeout=120.0, max_retries=1)
        model = _env("OPENAI_MODEL", "gpt-5.4") or "gpt-5.4"
        def call(instructions, payload):
            return client.responses.create(model=model, instructions=instructions, input=payload, text=recipe_response_format(payload), **recipe_model_options(model))
        cfg.run.recent_recipes = editorial_context(recent_editorials(cfg.sqlite_path), recent_recipe_posts(cfg.wp.base_url))
        article = generate_recipe_article(call, recipe.title, recipe.ingredients, recipe.steps,
                                          recent=cfg.run.recent_recipes, source_is_korean=True)
    else:
        # Without an API key, present the recipe itself without a canned introduction.
        article = {"title": recipe.title, "intro": "", "ingredients": recipe.ingredients, "steps": recipe.steps}
    cfg.run.editorial_article = article
    source = "https://www.foodsafetykorea.go.kr/" if recipe.source == "mfds" else ""
    label = f"식품안전나라 공개 레시피 (레시피 번호 {recipe.recipe_id})" if recipe.source == "mfds" else "저장소 기본 레시피"
    now = datetime.now(KST).strftime("%Y-%m-%d")
    page_url = cfg.wp.base_url.rstrip("/") + f"/korean-recipe-{now}-{cfg.run.run_slot}/"
    return render_recipe(article, source, label, display_img_url if cfg.img.embed_image_in_body else "",
                         page_url, getattr(cfg.run, "recent_recipes", [])), article.get("excerpt", article["intro"])


# -----------------------------
# Run
# -----------------------------

def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = {"day": "오늘", "am": "오전", "pm": "오후"}.get(slot, "오늘")
    date_slot = f"{date_str}_{slot}"

    init_db(cfg.sqlite_path)

    today_meta = get_today_post(cfg.sqlite_path, date_slot)
    existing = None
    refresh_editorial = _env("REFRESH_EDITORIAL", "0") == "1"
    can_refresh = refresh_editorial and today_meta and today_meta.get("recipe_source") and today_meta.get("recipe_id")
    if not cfg.run.dry_run and not cfg.run.force_new:
        endpoint = cfg.wp.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
        existing = find_post(endpoint, wp_auth_header(cfg.wp.user, cfg.wp.app_pass), f"korean-recipe-{date_str}-{slot}")
        if existing and not can_refresh:
            print("SKIP(already posted):", existing["id"])
            return
    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days)

    rng = _seed_rng(date_slot)

    print(f"[RUN] slot={slot} force_new={int(cfg.run.force_new)} date_slot={date_slot}")

    chosen: Optional[Recipe] = None
    if today_meta and not cfg.run.force_new and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

    if not chosen:
        if existing and can_refresh:
            facts = recover_published_recipe(existing)
            if not facts:
                raise RuntimeError("기존 글의 원문 레시피를 다시 읽지 못해 다른 요리로 덮어쓰지 않았습니다.")
            chosen = Recipe(today_meta["recipe_source"], today_meta["recipe_id"], today_meta["recipe_title"],
                            facts["ingredients"], facts["steps"], facts["image_url"])
            print("[EDITORIAL] 기존 글에서 본문과 일치하는 레시피 사실을 복구했습니다.")
    if not chosen:
        print("[RECIPE] ... choosing (mfds -> local)")
        chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(cfg, recent_pairs)

    assert chosen is not None

    print(f"[RECIPE] source={chosen.source} id={chosen.recipe_id} title={chosen.title}")

    title = build_post_title(date_str, slot_label, chosen.title, rng)
    slug = f"korean-recipe-{date_str}-{slot}"

    # 이미지 URL 결정
    thumb_url = choose_thumb_url(cfg, chosen)
    if thumb_url:
        print("[IMG] thumb_url:", (thumb_url[:120] + "...") if len(thumb_url) > 120 else thumb_url)
    else:
        print("[IMG] thumb_url: EMPTY")

    # WP 태그(이름 -> id) 준비
    tag_names = build_tag_names(chosen.title, cfg.tags.tag_names)
    if tag_names:
        print("[TAGS] names:", ", ".join(tag_names[:12]))

    tag_ids: List[int] = []
    if not cfg.run.dry_run:
        tag_ids = ensure_tag_ids(cfg, tag_names)
        if tag_ids:
            print("[TAGS] ids:", tag_ids)

    # media 업로드 시도(옵션)
    media_id = 0
    media_url = ""
    if (not cfg.run.dry_run) and cfg.img.upload_thumb and thumb_url:
        try:
            print("[IMG] uploading/ensuring media...")
            media_id, media_url = ensure_media(cfg, thumb_url, stable_name="korean_recipe_thumb")
            if media_id:
                print("[IMG] media OK:", media_id)
        except Exception as e:
            if cfg.run.debug:
                print("[IMG] upload failed:", repr(e))
            media_id, media_url = 0, ""

    display_img_url = (media_url or thumb_url or "").strip()

    body_html, excerpt = build_body_html(cfg, chosen, display_img_url, rng)
    title = cfg.run.editorial_article["title"]

    save_preview(slug, title, body_html)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략  HTML 일부")
        print(body_html[:2000])
        print("... (truncated)")
        return

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    # WP 발행
    wp_post_id = 0
    wp_link = ""

    print("[WP] publishing...")
    if today_meta and today_meta.get("wp_post_id"):
        wp_post_id, wp_link = wp_update_post(cfg.wp, int(today_meta["wp_post_id"]), title, body_html, excerpt, tag_ids, featured_media=featured_id)
        print("OK(updated):", wp_post_id, wp_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt, tag_ids, featured_media=featured_id)
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

    if cfg.run.editorial_article["intro"]:
        remember_editorial(cfg.sqlite_path, slug, cfg.run.editorial_article)


def main() -> None:
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
