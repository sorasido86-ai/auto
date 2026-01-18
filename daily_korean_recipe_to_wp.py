# -*- coding: utf-8 -*-
"""daily_korean_recipe_to_wp.py (홈피드형 1200자+ / 네이버 복붙 친화 / 타임아웃 강제)

요약
- 1순위: 식품안전나라(식약처) COOKRCP01 OpenAPI (MFDS_API_KEY 있으면)
- 2순위: 내장 한식 레시피(폴백)
- 글 구성: 도입부(200~300자 내외) + 굵은 소제목 3개 + 전체 1200자 이상
- 말투: 친구에게 수다떠는 존댓말  마침표 없이  띄어쓰기와 여백으로 호흡
- 특수문자/불릿(• ✅ 등) 제거
- 이미지: MFDS 메인 이미지가 없으면
  1) DEFAULT_THUMB_URL 있으면 그걸 사용
  2) 없으면 AUTO_IMAGE=1일 때 Unsplash Source(원격 링크)로 임시 대체
- "자동 생성"/"기준시각 슬롯"/"출처" 문구는 기본적으로 넣지 않음
- 실행이 1시간씩 늘어지는 문제 방지:
  - 네트워크 호출에 "하드 타임아웃"(signal) + requests timeout 동시 적용
  - MFDS 호출 전체 예산(MFDS_BUDGET_SEC) 초과 시 즉시 폴백

필수 환경변수
- WP_BASE_URL, WP_USER, WP_APP_PASS

권장 환경변수
- MFDS_API_KEY (없으면 내장 레시피만)
- DEFAULT_THUMB_URL (대표이미지 확실히 보이게 하려면 강력 추천)

옵션
- RUN_SLOT=day|am|pm (기본 day)
- FORCE_NEW=0|1 (기본 0)
- DRY_RUN=0|1 (기본 0)
- DEBUG=0|1 (기본 0)
- AVOID_REPEAT_DAYS=90 (기본 90)

이미지
- UPLOAD_THUMB=1 (기본 1)  : 원격 이미지를 WP 미디어로 업로드 시도
- SET_FEATURED=1 (기본 1)  : 업로드 성공 시 featured_media 설정
- EMBED_IMAGE_IN_BODY=1 (기본 1) : 본문 상단 이미지 1장 삽입
- AUTO_IMAGE=1 (기본 1)     : DEFAULT_THUMB_URL 없고 MFDS 이미지도 없을 때 Unsplash Source 링크 생성

태그
- AUTO_TAGS=1 (기본 1) : 제목 토큰+기본 태그를 WP 태그로 자동 생성/연결(권한/속도 이슈 시 0)
- TAG_NAMES="한식레시피,집밥,오늘뭐먹지" 같은 형태로 추가 가능

SQLite
- SQLITE_PATH=data/daily_korean_recipe.sqlite3

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
import sys
import time
from contextlib import contextmanager
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

    r = safe_post(url, headers=headers, json=payload, timeout=25, hard=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
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

    r = safe_post(url, headers=headers, json=payload, timeout=25, hard=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
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
# Text helpers (no bullets, no periods)
# -----------------------------

def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "")


def _text_len_html(s: str) -> int:
    return len((_strip_tags(s) or "").replace("\n", "").replace("\r", "").replace("\t", ""))


def _no_ai_symbols(s: str) -> str:
    s = s or ""
    s = s.replace("•", " ").replace("●", " ").replace("○", " ").replace("✅", " ").replace("✔", " ").replace("★", " ")
    s = s.replace("※", " ").replace("-", " ")
    s = re.sub(r"[\u2022\u25cf\u25cb\u2713\u2714]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _remove_period_like(s: str) -> str:
    s = s or ""
    for ch in [".", "!", "?", ";", ":", "…", "·"]:
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_sentence(s: str) -> str:
    return _remove_period_like(_no_ai_symbols(s))


def _josa_eul_reul(word: str) -> str:
    """목적격 조사(을/를) 선택"""
    word = (word or "").strip()
    if not word:
        return "을"
    last = word[-1]
    if "가" <= last <= "힣":
        code = ord(last) - 0xAC00
        jong = code % 28
        return "를" if jong == 0 else "을"
    return "을"


def _p(lines: List[str]) -> str:
    out: List[str] = []
    for ln in lines:
        ln = _sanitize_sentence(ln)
        if ln == "":
            out.append("<p>&nbsp;</p>")
            continue
        if not ln:
            continue
        out.append(f"<p>{html.escape(ln)}</p>")
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
        (["비빔"], "bibimbap korean"),
        (["김밥"], "kimbap"),
        (["떡"], "tteokbokki"),
        (["국"], "korean soup"),
        (["찌개"], "korean stew"),
        (["전"], "korean pancake jeon"),
    ]
    for keys, q in mapping:
        if any(k in t for k in keys):
            return q
    # fallback
    t2 = re.sub(r"\s+", " ", t).strip()
    if t2:
        return f"{t2} korean food"
    return "korean food"


def _unsplash_source_url(query: str) -> str:
    # Unsplash Source는 API키 없이도 랜덤 이미지를 줌 (리다이렉트)
    q = quote(query)
    return f"https://source.unsplash.com/1600x900/?{q}"


def choose_thumb_url(cfg: AppConfig, recipe: Recipe) -> str:
    # 1) MFDS 제공 이미지
    if (recipe.image_url or "").strip().startswith("http"):
        return recipe.image_url.strip()
    # 2) 기본 썸네일
    if (cfg.img.default_thumb_url or "").strip():
        return cfg.img.default_thumb_url.strip()
    # 3) 자동 이미지
    if cfg.img.auto_image:
        return _unsplash_source_url(_korean_title_to_image_query(recipe.title))
    return ""


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

def _compose_intro(recipe: Recipe, rng: random.Random, min_len: int, max_len: int) -> str:
    title = recipe.title
    base = [
        f"오늘은 {title} 얘기 좀 해볼게요",
        "요리 글 보면 다들 말은 쉬운데 막상 해보면 애매한 포인트가 있잖아요",
        "저도 그게 싫어서 진짜 집에서 잘 되는 흐름으로만 정리했어요",
        "시간 없을 때도 그대로 따라가면 맛이 나게끔 순서랑 간 타이밍을 맞춰놨어요",
        "혹시 오늘 뭐 먹지 고민 중이면 이거 한 번만 따라가봐요",
    ]
    # 길이 맞추기
    txt = " ".join([_sanitize_sentence(x) for x in base])
    if len(txt) < min_len:
        extra = [
            "대충 만들어도 되는 메뉴 같지만 은근히 불 조절 하나로 맛이 갈려요",
            "그래서 제가 자주 하는 실수도 같이 적어둘게요",
        ]
        txt = txt + " " + _sanitize_sentence(rng.choice(extra))

    txt = txt.strip()
    if len(txt) > max_len:
        txt = txt[:max_len].rstrip()
    return txt


def _recipe_list_block(recipe: Recipe) -> str:
    # 사용자가 말한 "레시피 목록" -> 재료 목록 + 만드는 순서 목록
    lines: List[str] = []
    lines.append("재료 목록")
    if recipe.ingredients:
        for x in recipe.ingredients[:30]:
            lines.append(f"{x}")
    else:
        lines.append("집에 있는 재료로도 가능해요")

    lines.append("")
    lines.append("만드는 순서")
    if recipe.steps:
        ords = [
            "첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덟째", "아홉째", "열째",
            "열한째", "열둘째", "열셋째", "열넷째", "열다섯째", "열여섯째", "열일곱째", "열여덟째", "열아홉째", "스무째",
        ]
        for i, s in enumerate(recipe.steps[:25], start=1):
            prefix = ords[i-1] if (i-1) < len(ords) else f"{i}번째"
            lines.append(f"{prefix} {s}")
    else:
        lines.append("과정은 간단하게 끓이고 간 맞추면 끝이에요")

    # 헤더는 굵게 처리
    out: List[str] = []
    for ln in lines:
        if ln in ("재료 목록", "만드는 순서"):
            out.append(f"<p><b>{html.escape(ln)}</b></p>")
        elif ln == "":
            out.append("<p>&nbsp;</p>")
        else:
            out.append(f"<p>{html.escape(_sanitize_sentence(ln))}</p>")
    return "\n".join(out)


def _section1(recipe: Recipe, rng: random.Random) -> List[str]:
    t = recipe.title
    return [
        f"{t} 할 때 제가 제일 신경 쓰는 건 시작 5분이에요",
        "처음에 불을 너무 세게 하면 향이 먼저 날아가고 나중에 간을 잡아도 밍밍해지더라고요",
        "그래서 저는 중불로 시작해서 재료가 숨이 죽을 때까지 기다리는 편이에요",
        "그리고 간은 중간에 확 넣기보단 마지막에 한 번만 조절하는 게 결과가 안정적이었어요",
        "재료가 조금 부족해도 괜찮아요",
        "대신 지금 있는 재료로 뭘 살릴지 한 가지 포인트만 잡아보면 맛이 확 달라져요",
        "예를 들면 국물은 육수 쪽  볶음은 불향 쪽  조림은 농도 쪽 이런 느낌이요",
        "저는 오늘은 실패 확률 줄이는 흐름으로만 적어둘게요",
    ]


def _section2(recipe: Recipe, rng: random.Random) -> List[str]:
    return [
        "재료는 목록 그대로 준비하면 제일 편해요",
        "근데 솔직히 집밥은 늘 딱 맞게 준비하기 어렵잖아요",
        "그래서 저는 대체 기준을 이렇게 잡아요",
        "양파나 대파 같은 향채는 있으면 넣고 없으면 마늘 타이밍을 조금 늦춰요",
        "두부나 버섯 같은 부재료는 식감용이라서 취향대로 빼도 괜찮아요",
        "대신 간장 소금 된장 고추장 같은 핵심 간 재료는 마지막에 조금씩 추가하는 방식으로 가요",
        "그리고 손질은 너무 완벽하게 하려고 하면 지쳐요",
        "대충 같은 크기로만 맞춰도 익는 속도가 비슷해져서 결과가 좋아요",
        "이렇게만 해도 집에서 먹는 맛이 훨씬 안정적이에요",
    ]


def _section3(recipe: Recipe, rng: random.Random) -> List[str]:
    return [
        "이제 만드는 순서를 그대로 따라가면 되는데요",
        "저는 한 가지를 꼭 해요",
        "중간에 맛을 볼 때 간을 확 올리지 않고  물이나 육수 양을 먼저 조절해요",
        "그래야 짜지지 않고  마지막에 딱 맞추기 쉬워요",
        "그리고 불 조절은 중불에서 시작해서  끓거나 볶이는 느낌이 잡히면 약불로 내려요",
        "이게 진짜 체감이 큰데  같은 재료로도 맛이 훨씬 부드럽게 나와요",
        "응용은 간단해요",
        "칼칼하게 가고 싶으면 고춧가루를 마지막에 아주 조금만 추가하고  단맛은 올리고당을 소량만 써요",
        "마지막으로 남은 건 냉장 보관했다가 데울 때  한 번 끓이고 간을 다시 마지막에만 보시면 돼요",
        "이렇게 하면 다음날 먹어도 맛이 크게 안 무너져요",
    ]


def build_body_html(cfg: AppConfig, recipe: Recipe, display_img_url: str, rng: random.Random) -> Tuple[str, str]:
    # 도입부
    intro = _compose_intro(recipe, rng, cfg.content.intro_min, cfg.content.intro_max)

    # 3섹션 + 레시피 목록
    s1 = _section1(recipe, rng)
    s2 = _section2(recipe, rng)
    s3 = _section3(recipe, rng)

    # 레시피 목록(재료/순서)
    recipe_list = _recipe_list_block(recipe)

    # 태그(본문 하단에 해시태그 형태)
    tag_names = build_tag_names(recipe.title, cfg.tags.tag_names)
    def _to_hash(t: str) -> str:
        return "#" + re.sub(r"\s+", "", t or "").strip()
    hashtags = " ".join([_to_hash(t) for t in tag_names[:12] if t])

    # 이미지(상단 1장)
    img_html = ""
    if cfg.img.embed_image_in_body and display_img_url:
        img_html = f"<p><img src=\"{html.escape(display_img_url)}\" alt=\"{html.escape(recipe.title)}\"/></p>"

    # 굵은 소제목 3개
    obj = _josa_eul_reul(recipe.title)
    h1 = f"<p><b>{html.escape(_sanitize_sentence('오늘 ' + recipe.title + obj + ' 추천하는 이유'))}</b></p>"
    h2 = f"<p><b>{html.escape(_sanitize_sentence('재료 준비하면서 제가 꼭 지키는 기준'))}</b></p>"
    h3 = f"<p><b>{html.escape(_sanitize_sentence('만드는 흐름과 실패 줄이는 포인트'))}</b></p>"

    # 본문 조립
    blocks: List[str] = []
    blocks.append(_p([intro]))
    blocks.append("<p>&nbsp;</p>")
    if img_html:
        blocks.append(img_html)
        blocks.append("<p>&nbsp;</p>")

    blocks.append(h1)
    blocks.append(_p(s1))
    blocks.append("<p>&nbsp;</p>")

    blocks.append(h2)
    blocks.append(_p(s2))
    blocks.append("<p>&nbsp;</p>")

    blocks.append(h3)
    blocks.append(_p(s3))
    blocks.append("<p>&nbsp;</p>")

    # 레시피 목록
    blocks.append(f"<p><b>{html.escape(_sanitize_sentence('레시피 목록'))}</b></p>")
    blocks.append(recipe_list)

    blocks.append("<p>&nbsp;</p>")
    blocks.append(_p(["저는 이렇게 해먹는 편인데요  혹시 {0}는 어떤 재료를 추가하는 편이세요  댓글로 추천해주시면 다음 글에도 반영해볼게요".format(recipe.title)]))

    if hashtags:
        blocks.append("<p>&nbsp;</p>")
        blocks.append(_p([hashtags]))

    body = "\n".join(blocks)

    # 길이 부족하면 섹션3 보강
    if _text_len_html(body) < cfg.content.body_min:
        extra = [
            "그리고 남은 재료가 애매하면 반찬처럼 곁들이는 쪽으로 돌리는 것도 좋아요",
            "계란이나 김 같은 건 진짜 실패를 줄여줘요",
            "맛이 조금 약하면 간을 올리기 전에 먼저 한 번 더 끓여보는 게 도움이 되더라고요",
            "저는 이 방식이 제일 마음이 편했어요",
        ]
        while _text_len_html(body) < cfg.content.body_min and extra:
            body += "\n" + _p([extra.pop(0)])

    excerpt = f"{recipe.title} 레시피  집에서 잘되는 흐름으로만 정리했어요"[:140]
    return body, excerpt


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

    rng = _seed_rng(date_slot)

    print(f"[RUN] slot={slot} force_new={int(cfg.run.force_new)} date_slot={date_slot}")

    chosen: Optional[Recipe] = None
    if today_meta and not cfg.run.force_new and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

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
    try:
        if today_meta and today_meta.get("wp_post_id"):
            post_id = int(today_meta["wp_post_id"])
            wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, excerpt, tag_ids, featured_media=featured_id)
            print("OK(updated):", wp_post_id, wp_link)
        else:
            wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt, tag_ids, featured_media=featured_id)
            print("OK(created):", wp_post_id, wp_link)
    except Exception as e:
        # 업데이트 실패 등 -> 새로 생성
        if cfg.run.debug:
            print("[WARN] post create/update failed, fallback to create:", repr(e))
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt, tag_ids, featured_media=featured_id)
        print("OK(created-fallback):", wp_post_id, wp_link)

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
