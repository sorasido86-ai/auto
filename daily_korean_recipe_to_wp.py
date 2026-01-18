# -*- coding: utf-8 -*-
"""daily_korean_recipe_to_wp.py (완성 교체본)

한식 레시피를 WordPress에 자동 발행합니다

요구사항 반영
- 홈피드형
- 도입부 200~300자(텍스트 기준)
- 굵은 소제목 3개
- 각 섹션 최소 글자수: 기본 1200자(SECTION_MIN_CHARS)
- 친구에게 진심 담아 수다 떠는 존댓말
- 마침표 없이 여백과 줄바꿈으로 호흡
- 레시피 항목에서 불릿/번호/체크 등 점 찍힌 특수문자 제거
- 자동생성 문구 / 기준시각 슬롯 / 출처 / 지난 레시피 링크 기본 미노출

이미지
- MFDS(식품안전나라) 이미지가 없거나 API가 느려도 사진이 최소 1장 나오도록 처리
  우선순위: MFDS 이미지 > DEFAULT_THUMB_URL > Unsplash Source(제목 기반)
- 가능하면 WP Media로 업로드 후 featured_media 설정
- 업로드 실패해도 본문 상단에 이미지 URL 삽입

필수 환경변수(Secrets)
- WP_BASE_URL
- WP_USER
- WP_APP_PASS

선택
- MFDS_API_KEY (없거나 느리면 로컬 레시피로 폴백)
- MFDS_ONLY=1 (MFDS 실패 시 발행 건너뜀)
- DEFAULT_THUMB_URL (없으면 Unsplash 자동)

권장
- SQLITE_PATH=data/daily_korean_recipe.sqlite3

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
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

KST = timezone(timedelta(hours=9))

# -----------------------------
# Local fallback recipes (Korean)
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "kimchi-jjigae",
        "title": "돼지고기 김치찌개",
        "ingredients": [
            ("신김치", "2컵"),
            ("돼지고기(앞다리 또는 삼겹)", "200g"),
            ("양파", "1/2개"),
            ("대파", "1대"),
            ("두부", "1/2모"),
            ("고춧가루", "1큰술"),
            ("다진마늘", "1큰술"),
            ("국간장", "1큰술"),
            ("멸치다시마 육수 또는 물", "700ml"),
        ],
        "steps": [
            "냄비에 돼지고기를 넣고 중불에서 기름이 살짝 돌 때까지 볶아주세요",
            "신김치를 넣고 2에서 3분 더 볶아 김치의 신맛을 한 번 눌러줍니다",
            "고춧가루 다진마늘 국간장을 넣고 30초만 볶아 향을 내요",
            "육수를 붓고 10에서 12분 끓입니다",
            "양파를 넣고 3분 두부를 넣고 2분 더 끓인 뒤 대파로 마무리해요",
        ],
        "image_url": "",
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1.5큰술"),
            ("고추장 선택", "1/2큰술"),
            ("애호박", "1/3개"),
            ("양파", "1/3개"),
            ("두부", "1/2모"),
            ("대파", "1/2대"),
            ("다진마늘", "1작은술"),
            ("멸치다시마 육수 또는 물", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 5분 끓여요",
            "양파 애호박 두부를 넣고 5에서 6분 더 끓입니다",
            "대파를 넣고 한 번만 더 끓인 뒤 간을 보고 마무리해요",
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
            ("물 또는 배즙", "3큰술"),
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
    run_slot: str = "day"  # day|am|pm
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 10


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True
    mfds_timeout: int = 10
    mfds_budget_seconds: int = 20
    mfds_only: bool = False


@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    default_thumb_url: str = ""
    reuse_media_by_search: bool = True


@dataclass
class ContentConfig:
    intro_min_chars: int = 200
    intro_max_chars: int = 300
    section_min_chars: int = 1200
    hashtag_count: int = 0


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    recipe: RecipeSourceConfig
    img: ImageConfig
    content: ContentConfig
    sqlite_path: str


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=_env("WP_STATUS", "publish") or "publish",
            category_ids=_parse_int_list(_env("WP_CATEGORY_IDS", "7")),
            tag_ids=_parse_int_list(_env("WP_TAG_IDS", "")),
        ),
        run=RunConfig(
            run_slot=run_slot,
            force_new=_env_bool("FORCE_NEW", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 10),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout=_env_int("MFDS_TIMEOUT", 10),
            mfds_budget_seconds=_env_int("MFDS_BUDGET_SECONDS", 20),
            mfds_only=_env_bool("MFDS_ONLY", False),
        ),
        img=ImageConfig(
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
        ),
        content=ContentConfig(
            intro_min_chars=_env_int("INTRO_MIN_CHARS", 200),
            intro_max_chars=_env_int("INTRO_MAX_CHARS", 300),
            section_min_chars=_env_int("SECTION_MIN_CHARS", 1200),
            hashtag_count=_env_int("HASHTAG_COUNT", 0),
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
        raise RuntimeError("필수 설정 누락: " + ", ".join(missing))


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
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| MFDS_ONLY:", cfg.recipe.mfds_only)
    print("[CFG] MFDS_TIMEOUT:", cfg.recipe.mfds_timeout, "| MFDS_BUDGET_SECONDS:", cfg.recipe.mfds_budget_seconds)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] UPLOAD_THUMB:", cfg.img.upload_thumb, "| SET_FEATURED:", cfg.img.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.img.embed_image_in_body)
    print(
        "[CFG] INTRO_MIN_CHARS:",
        cfg.content.intro_min_chars,
        "| INTRO_MAX_CHARS:",
        cfg.content.intro_max_chars,
        "| SECTION_MIN_CHARS:",
        cfg.content.section_min_chars,
    )
    print("[CFG] HASHTAG_COUNT:", cfg.content.hashtag_count)


# -----------------------------
# SQLite history
# -----------------------------

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
            meta.get("created_at", ""),
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
    return [(str(s), str(rid)) for s, rid in rows if s and rid]


# -----------------------------
# WordPress REST
# -----------------------------

def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/3.0"}


def wp_create_post(
    cfg: WordPressConfig,
    title: str,
    slug: str,
    html_body: str,
    excerpt: str = "",
    featured_media: int = 0,
) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if featured_media:
        payload["featured_media"] = featured_media
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    d = r.json()
    return int(d["id"]), str(d.get("link") or "")


def wp_update_post(
    cfg: WordPressConfig,
    post_id: int,
    title: str,
    html_body: str,
    excerpt: str = "",
    featured_media: int = 0,
) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if featured_media:
        payload["featured_media"] = featured_media
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    d = r.json()
    return int(d["id"]), str(d.get("link") or "")


def wp_find_media_by_search(cfg: WordPressConfig, search: str) -> Optional[Tuple[int, str]]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass)
    r = requests.get(url, headers=headers, params={"search": search, "per_page": 10}, timeout=25)
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


def wp_upload_media_from_url(cfg: WordPressConfig, image_url: str, filename: str, timeout: int = 25) -> Tuple[int, str]:
    r = requests.get(image_url, timeout=timeout, allow_redirects=True)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code}")
    content = r.content
    ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower() or "image/jpeg"

    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
    }
    rr = requests.post(url, headers=headers, data=content, timeout=60)
    if rr.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {rr.status_code} body={rr.text[:500]}")
    d = rr.json()
    return int(d["id"]), str(d.get("source_url") or "")


# -----------------------------
# Recipe model + MFDS
# -----------------------------

@dataclass
class Recipe:
    source: str
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


def mfds_fetch_by_param(
    api_key: str,
    param: str,
    value: str,
    start: int = 1,
    end: int = 60,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    try:
        r = requests.get(url, timeout=timeout)
    except Exception:
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    co = data.get("COOKRCP01") or {}
    rows = co.get("row") or []
    return rows if isinstance(rows, list) else []


def _strip_trailing_ascii(s: str) -> str:
    return re.sub(r"[a-zA-Z]\s*$", "", s or "").strip()


def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or ""
    title = str(row.get("RCP_NM") or "").strip()
    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()
    ings = [p.strip() for p in re.split(r"\s*,\s*", parts) if p.strip()]

    steps: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        if s:
            steps.append(_strip_trailing_ascii(s))

    img_main = str(row.get("ATT_FILE_NO_MAIN") or "").strip() or str(row.get("ATT_FILE_NO_MK") or "").strip()

    return Recipe(
        source="mfds",
        recipe_id=rid or hashlib.sha1(title.encode("utf-8")).hexdigest()[:8],
        title=title,
        ingredients=ings,
        steps=steps,
        image_url=img_main if img_main.startswith("http") else "",
    )


def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡"]

    start_ts = time.time()
    tries = 0

    while tries < cfg.run.max_tries and (time.time() - start_ts) < cfg.recipe.mfds_budget_seconds:
        tries += 1
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, 1, 60, cfg.recipe.mfds_timeout)
        if not rows:
            continue
        random.shuffle(rows)
        for row in rows:
            if (time.time() - start_ts) >= cfg.recipe.mfds_budget_seconds:
                break
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
# Text helpers (no bullets, no periods)
# -----------------------------

def _esc(s: str) -> str:
    return html.escape(s or "")


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def _text_len_html(s: str) -> int:
    return len((_strip_tags(s) or "").replace("\n", "").replace("\r", "").replace("\t", ""))


def _no_ai_symbols(s: str) -> str:
    s = s or ""
    # 대표적인 불릿/체크/점 문자 제거
    s = s.replace("•", " ").replace("●", " ").replace("○", " ").replace("✅", " ").replace("✔", " ").replace("★", " ").replace("※", " ")
    s = re.sub(r"[\u2022\u25cf\u25cb\u2713\u2714]+", " ", s)
    # 줄 시작 하이픈/번호 제거
    s = re.sub(r"^\s*[-–—]\s*", "", s)
    s = re.sub(r"^\s*\d+\s*[\)\.\-]\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _remove_period_like(s: str) -> str:
    s = s or ""
    # 마침표 느낌 문장부호 제거
    for ch in [".", "!", "?", ";", ":", "…", "·"]:
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_sentence(s: str) -> str:
    return _remove_period_like(_no_ai_symbols(s))


def _p(lines: List[str]) -> str:
    out: List[str] = []
    for ln in lines:
        ln = _sanitize_sentence(ln)
        if ln == "":
            out.append("<p>&nbsp;</p>")
            continue
        if not ln:
            continue
        out.append(f"<p>{_esc(ln)}</p>")
    return "\n".join(out)


def _seed_rng(seed: str) -> random.Random:
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]
    return random.Random(int(h, 16))


# -----------------------------
# Title
# -----------------------------

def build_post_title(date_str: str, slot_label: str, recipe_title: str, rng: random.Random) -> str:
    hooks = [
        "집에서 진짜 잘되는 포인트만 정리",
        "실패 확률 줄이는 흐름 그대로",
        "오늘 저녁 고민 끝내는 메뉴",
        "재료 적어도 맛이 나는 방식",
        "한 번 해두면 계속 쓰는 레시피",
        "이 단계만 놓치면 맛이 확 달라져요",
    ]
    pick = rng.choice(hooks)
    return _sanitize_sentence(f"{recipe_title} 레시피 {pick} {date_str} {slot_label}")


# -----------------------------
# Images
# -----------------------------

def _korean_title_to_image_query(title: str) -> str:
    t = title or ""
    mapping = [
        (["김치찌개"], "kimchi jjigae korean stew"),
        (["된장찌개"], "doenjang jjigae korean stew"),
        (["불고기"], "bulgogi korean beef"),
        (["비빔밥"], "bibimbap korean food"),
        (["김밥"], "kimbap korean roll"),
        (["떡볶이"], "tteokbokki korean food"),
        (["잡채"], "japchae korean noodles"),
        (["제육", "제육볶음"], "jeyuk bokkeum spicy pork"),
        (["국", "탕"], "korean soup"),
        (["찌개"], "korean stew"),
        (["볶음"], "korean stir fry"),
        (["전"], "korean pancake"),
    ]
    for keys, q in mapping:
        if any(k in t for k in keys):
            return q
    return (re.sub(r"\s+", " ", t).strip() + " korean food").strip()


def _unsplash_source_url(query: str) -> str:
    # Unsplash Source는 안정적으로 랜덤 대표 이미지를 돌려줍니다
    return f"https://source.unsplash.com/1600x900/?{quote(query)}"


def choose_thumb_url(cfg: AppConfig, recipe: Recipe) -> str:
    if recipe.image_url and recipe.image_url.startswith("http"):
        return recipe.image_url.strip()
    if cfg.img.default_thumb_url and cfg.img.default_thumb_url.startswith("http"):
        return cfg.img.default_thumb_url.strip()
    return _unsplash_source_url(_korean_title_to_image_query(recipe.title))


def ensure_media(cfg: AppConfig, image_url: str, stable_name: str, timeout: int = 25) -> Tuple[int, str]:
    if not image_url:
        return 0, ""
    h = hashlib.sha1(image_url.encode("utf-8")).hexdigest()[:12]
    filename = f"{stable_name}_{h}.jpg"
    if cfg.img.reuse_media_by_search:
        found = wp_find_media_by_search(cfg.wp, f"{stable_name}_{h}")
        if found:
            return found
    return wp_upload_media_from_url(cfg.wp, image_url, filename, timeout=timeout)


def wp_image_block(image_url: str, alt: str, media_id: int = 0) -> str:
    if media_id:
        return (
            f'<!-- wp:image {{"id":{media_id},"sizeSlug":"large","linkDestination":"none"}} -->\n'
            f'<figure class="wp-block-image size-large"><img src="{_esc(image_url)}" alt="{_esc(alt)}"/></figure>\n'
            f'<!-- /wp:image -->'
        )
    return (
        "<!-- wp:image -->\n"
        f'<figure class="wp-block-image"><img src="{_esc(image_url)}" alt="{_esc(alt)}"/></figure>\n'
        "<!-- /wp:image -->"
    )


# -----------------------------
# Homefeed body builder
# -----------------------------

def _build_intro(cfg: AppConfig, title: str, rng: random.Random) -> str:
    chunks = [
        f"오늘은 {title} 얘기 좀 해도 될까요",
        "요즘은 뭐 하나 해먹는 것도 마음 먹어야 하잖아요",
        "근데 이 메뉴는 이상하게요",
        "딱 한 번만 흐름대로 해두면 다음부터는 몸이 먼저 기억하더라고요",
        "괜히 거창하게 안 하고 집에 있는 재료로도 기분이 확 올라오는 쪽으로 정리해봤어요",
        "같이 수다 떨듯이 천천히 따라오시면 좋겠어요",
    ]
    rng.shuffle(chunks)
    lines = [chunks[0], chunks[1], "", chunks[2], chunks[3], "", chunks[4], chunks[5]]

    def render(ls: List[str]) -> str:
        return _p(ls)

    intro_html = render(lines)
    minc, maxc = cfg.content.intro_min_chars, cfg.content.intro_max_chars

    txt = _strip_tags(intro_html).replace("\xa0", " ").replace("\n", " ")
    if len(txt) < minc:
        lines += ["", "저는 이런 날에 특히 이 메뉴가 위로가 되더라고요"]
        intro_html = render(lines)

    txt = _strip_tags(intro_html).replace("\xa0", " ").replace("\n", " ")
    if len(txt) > maxc:
        # 길면 뒤에서부터 줄임
        while len(txt) > maxc and lines:
            lines.pop()
            intro_html = render(lines)
            txt = _strip_tags(intro_html).replace("\xa0", " ").replace("\n", " ")

    return intro_html


def _ingredients_prose(ingredients: List[str], rng: random.Random) -> str:
    ings = [_sanitize_sentence(x) for x in (ingredients or []) if _sanitize_sentence(x)]
    ings = ings[:10]
    if not ings:
        return _sanitize_sentence("재료는 집에 있는 걸로 편하게 맞추셔도 괜찮아요")
    head = rng.choice(["준비하실 건", "손에 잡히는 재료는", "오늘은 이 정도만 있으면 돼요", "재료는 생각보다 단순해요"])
    return _sanitize_sentence(head + " " + " ".join(ings))


def _steps_as_lines(steps: List[str]) -> List[str]:
    st = [_sanitize_sentence(x) for x in (steps or []) if _sanitize_sentence(x)]
    if not st:
        return [_sanitize_sentence("과정은 크게 어렵지 않아요 불 조절만 천천히 따라가시면 돼요")]
    connectors = ["먼저", "그 다음", "이어서", "그때", "그리고", "마지막으로"]
    lines: List[str] = []
    for i, s in enumerate(st[:12]):
        c = connectors[min(i, len(connectors) - 1)]
        lines.append(_sanitize_sentence(f"{c} {s}"))
    return lines


def _unique_add(out: List[str], s: str) -> None:
    s2 = _sanitize_sentence(s)
    if not s2:
        return
    if any(_sanitize_sentence(x) == s2 for x in out):
        return
    out.append(s2)


def _expand_section(recipe_title: str, base_lines: List[str], min_chars: int, rng: random.Random) -> List[str]:
    out: List[str] = []
    for ln in base_lines:
        if ln == "":
            out.append("")
        else:
            _unique_add(out, ln)

    pool = [
        f"저는 {recipe_title} 할 때 제일 마음이 놓이는 순간이 있어요 불이 너무 세지 않게 잡아두고 냄비 안에서 향이 올라올 때요",
        "처음엔 저도 꼭 뭔가를 더 넣어야 맛이 날 것 같아서 욕심을 냈는데요 오히려 기본을 지키는 쪽이 결과가 안정적이더라고요",
        "간은 중간에 확정하지 않는 게 좋아요 끓고 나면 농도가 바뀌니까 마지막에 한 번만 살짝 조절하는 게 제일 덜 흔들려요",
        "만약 재료가 하나 빠져도 너무 불안해하지 않으셔도 돼요 이 레시피는 흐름만 지키면 맛이 크게 무너지지 않아요",
        "같은 재료인데도 불 조절이 조금만 달라도 맛이 달라져요 그래서 저는 중불로 시작해서 천천히 정리하는 편이에요",
        "이 메뉴는 이상하게요 먹기 전보다 만들고 난 뒤에 기분이 더 좋아지는 날이 있어요 내가 나를 챙겼다는 느낌이 들어서요",
        "저는 요리할 때 머리가 복잡하면 자꾸 손이 빨라지더라고요 그럴수록 속도를 줄이는 게 오히려 빨리 끝나요",
        "혹시 오늘 하루가 조금 지쳤다면요 이 메뉴는 따뜻하게 한 숟갈 뜨는 순간 마음이 풀리는 편이에요",
        "저는 여기서 한 번 숨을 고르는 걸 추천해요 잠깐만 멈추고 냄새를 확인하면 다음 단계가 더 쉬워져요",
        "기본 재료로 시작해서 마지막에 취향을 살짝 얹는 방식이 제일 오래 가더라고요",
        "저도 처음엔 레시피를 그대로 따라가도 맛이 들쭉날쭉해서 속상했는데요 어느 순간부터는 흐름이 보이더라고요",
        "특히 국물요리는요 한 번만 끓이는 게 아니라 맛이 정리되는 시간을 주는 게 진짜 중요하더라고요",
    ]
    rng.shuffle(pool)

    i = 0
    while _text_len_html(_p(out)) < min_chars and i < 200:
        phrase = pool[i % len(pool)]
        if rng.random() < 0.25:
            phrase += " " + rng.choice(
                [
                    "이건 해보시면 바로 감이 오실 거예요",
                    "괜히 조급해지면 맛이 흔들리니까요",
                    "저도 여러 번 해보면서 이게 제일 편하더라고요",
                    "그냥 오늘은 이 정도만 해도 충분해요",
                ]
            )
        _unique_add(out, phrase)
        if rng.random() < 0.22:
            out.append("")
        i += 1

    safety = 0
    while _text_len_html(_p(out)) < min_chars and safety < 80:
        dyn = rng.choice(
            [
                f"{recipe_title} 만들 때는 한 번에 완벽하려고 하기보다 한 번에 한 단계만 생각하시면 편해요",
                f"{recipe_title}는 따뜻할 때도 좋은데요 살짝 식어서 맛이 정리됐을 때도 의외로 좋더라고요",
                "저는 간을 보면서도 마음이 급해지면 자꾸 뭔가를 더 넣고 싶어지는데요 그럴 때 한 번만 참으면 더 맛있어져요",
                "혹시 중간에 애매하다 싶으면요 불을 낮추고 잠깐만 기다려보세요 그게 제일 큰 치트키예요",
            ]
        )
        _unique_add(out, dyn)
        if rng.random() < 0.18:
            out.append("")
        safety += 1

    return out


def _clean_title_for_tags(title: str) -> List[str]:
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", title or "")
    toks = [x.strip() for x in t.split() if x.strip()]
    out: List[str] = []
    seen = set()
    for x in toks:
        if len(x) <= 1:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def build_hashtags(cfg: AppConfig, recipe: Recipe) -> str:
    base = ["한식레시피", "집밥", "오늘뭐먹지", "간단요리", "레시피"]
    for tok in _clean_title_for_tags(recipe.title)[:4]:
        base.append(tok.replace(" ", ""))

    uniq: List[str] = []
    seen = set()
    for x in base:
        x = re.sub(r"\s+", "", x)
        k = x.lower()
        if not x or k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    n = max(3, min(15, cfg.content.hashtag_count))
    return " ".join([f"#{x}" for x in uniq[:n]])


def build_homefeed_html(cfg: AppConfig, recipe: Recipe, img_url: str, media_id: int, rng: random.Random) -> Tuple[str, str]:
    excerpt = _sanitize_sentence(f"{recipe.title} 레시피 집에서 편하게 따라가는 버전")[:140]

    parts: List[str] = []

    if cfg.img.embed_image_in_body and img_url:
        parts.append(wp_image_block(img_url, recipe.title, media_id))

    parts.append(_build_intro(cfg, recipe.title, rng))

    smin = max(600, int(cfg.content.section_min_chars))
    ing_txt = _ingredients_prose(recipe.ingredients, rng)
    steps_lines = _steps_as_lines(recipe.steps)

    sec1_title = "오늘 이 메뉴를 고른 이유랑 재료 이야기"
    sec1_base = [
        f"{recipe.title}는요 저는 이상하게 마음이 복잡할 때 더 찾게 되더라고요",
        "누구에게 보여주려고 하는 요리가 아니라요",
        "그냥 제 컨디션을 조금이라도 올리는 데 도움이 되는 쪽이라서요",
        "",
        ing_txt,
        "",
        "재료를 다 갖추지 않아도 괜찮아요",
        "대신 맛의 중심이 되는 것만 놓치지 않으면 결과가 꽤 안정적으로 나와요",
    ]
    sec1 = _expand_section(recipe.title, sec1_base, smin, rng)
    parts.append(f"<h3><strong>{_esc(sec1_title)}</strong></h3>\n{_p(sec1)}")

    sec2_title = "만들면서 느꼈던 포인트랑 과정 흐름"
    sec2_base = [
        "과정은요 제가 말로 옆에서 계속 얘기해드리는 느낌으로 적어볼게요",
        "번호로 딱딱 끊으면 괜히 부담되잖아요",
        "",
    ] + steps_lines + [
        "",
        "이 흐름대로만 가도 맛이 크게 벗어나지 않아요",
        "중간에 흔들릴 때는 불을 조금만 낮추고 한 번 숨을 고르시면 좋아요",
    ]
    sec2 = _expand_section(recipe.title, sec2_base, smin, rng)
    parts.append(f"<h3><strong>{_esc(sec2_title)}</strong></h3>\n{_p(sec2)}")

    sec3_title = "실패 줄이는 팁이랑 먹는 방법"
    sec3_base = [
        "제가 제일 많이 했던 실수는요 간을 너무 빨리 확정해버리는 거였어요",
        "끓고 나면 맛이 정리되니까 마지막에만 살짝 조절해도 충분하더라고요",
        "",
        "그리고 불 조절이요",
        "센 불로 확 밀어붙이면 처음엔 빨라 보이는데요 끝맛이 좀 거칠어질 때가 있어요",
        "중불로 천천히 정리하면 훨씬 편해요",
        "",
        "남으면 보관도 어렵지 않아요",
        "다만 다시 데울 때는 한 번 끓이고 나서 간을 마지막에만 보세요",
        "그게 진짜 안정적이에요",
        "",
        "오늘 이 레시피 해보시면요 아마 다음엔 더 쉽게 하실 거예요",
        "한 번만 흐름을 몸에 넣어두면 그 다음부터는 고민이 줄거든요",
    ]
    sec3 = _expand_section(recipe.title, sec3_base, smin, rng)
    sec3.extend(
        [
            "",
            _sanitize_sentence(
                f"{recipe.title} 하실 때 혹시 좋아하는 추가 재료가 있으세요 댓글로 하나만 살짝 알려주시면 저도 다음에 참고해볼게요"
            ),
        ]
    )
    parts.append(f"<h3><strong>{_esc(sec3_title)}</strong></h3>\n{_p(sec3)}")

    if cfg.content.hashtag_count and cfg.content.hashtag_count > 0:
        tags = build_hashtags(cfg, recipe)
        if tags:
            parts.append(_p([tags]))

    return "\n\n".join([p for p in parts if p]), excerpt


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
    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days)

    print(f"[RUN] slot={slot} force_new={int(cfg.run.force_new)} date_slot={date_slot}")

    chosen: Optional[Recipe] = None
    if today_meta and not cfg.run.force_new and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

    if not chosen:
        chosen = pick_recipe_mfds(cfg, recent_pairs)
        if chosen is None and cfg.recipe.mfds_only:
            print("[SKIP] MFDS_ONLY=1 인데 MFDS에서 레시피를 못 가져와서 오늘 발행을 건너뜹니다")
            return
        chosen = chosen or pick_recipe_local(cfg, recent_pairs)

    assert chosen is not None
    print(f"[RECIPE] source={chosen.source} id={chosen.recipe_id} title={chosen.title}")

    rng = _seed_rng(chosen.uid() + "|" + date_slot)
    title = build_post_title(date_str, slot_label, chosen.title, rng)
    slug = f"korean-recipe-{date_str}-{slot}"

    thumb_url = choose_thumb_url(cfg, chosen)
    media_id, media_url = 0, ""

    if cfg.img.upload_thumb and thumb_url:
        try:
            media_id, media_url = ensure_media(cfg, thumb_url, "korean_recipe_thumb", timeout=25)
        except Exception as e:
            if cfg.run.debug:
                print("[IMG] upload failed:", repr(e))
            media_id, media_url = 0, ""

    display_url = (media_url or thumb_url or "").strip()
    body_html, excerpt = build_homefeed_html(cfg, chosen, display_url, media_id, rng)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print(body_html[:2500])
        return

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    if today_meta and today_meta.get("wp_post_id"):
        wp_post_id, wp_link = wp_update_post(cfg.wp, int(today_meta["wp_post_id"]), title, body_html, excerpt=excerpt, featured_media=featured_id)
        print("OK(updated):", wp_post_id, wp_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt, featured_media=featured_id)
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
