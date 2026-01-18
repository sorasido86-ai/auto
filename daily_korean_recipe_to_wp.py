# -*- coding: utf-8 -*-
"""daily_korean_recipe_to_wp.py (Homefeed v3 + OpenAI Image fallback)

요구사항(핵심)
- 홈피드형: 도입부 200~300자 + 굵은 소제목 3개 + 레시피 목록(재료/순서) + 해시태그
- 친구에게 수다 떠는 존댓말 톤
- 마침표( . ) 없이 줄바꿈/여백으로 호흡
- 레시피 목록 앞에 불릿/체크/점 같은 특수문자 없이 출력
- '자동 생성' '기준시각/슬롯' '출처' '지난 레시피 링크' 같은 문구 기본 미노출
- 전체 글 최소 글자수: MIN_TOTAL_CHARS(기본 1200)

이미지 우선순위
1) MFDS(식품안전나라) 이미지
2) DEFAULT_THUMB_URL
3) USE_OPENAI_IMAGE=1 이면 OpenAI 이미지 생성해서 WP에 업로드
4) AUTO_IMAGE=1 이면 Unsplash Source 자동 이미지

필수 환경변수
- WP_BASE_URL
- WP_USER
- WP_APP_PASS

권장 환경변수
- WP_STATUS=publish
- WP_CATEGORY_IDS=7
- SQLITE_PATH=data/daily_korean_recipe.sqlite3

선택 환경변수
- MFDS_API_KEY
- RUN_SLOT=day|am|pm
- FORCE_NEW=0|1
- DRY_RUN=0|1
- DEBUG=0|1

이미지 관련
- DEFAULT_THUMB_URL=https://...
- USE_OPENAI_IMAGE=0|1
- OPENAI_API_KEY=...
- OPENAI_IMAGE_MODEL=gpt-image-1-mini
- OPENAI_IMAGE_SIZE=1024x1024
- OPENAI_IMAGE_QUALITY=auto
- AUTO_IMAGE=1

주의
- MFDS 서버가 느릴 수 있어 타임아웃/예산 시간을 짧게 잡고 실패 시 로컬 레시피로 폴백합니다
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

KST = timezone(timedelta(hours=9))

# -----------------------------
# 로컬 폴백 레시피
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "kimchi-jjigae",
        "title": "돼지고기 김치찌개",
        "ingredients": [
            ("신김치", "2컵"),
            ("돼지고기", "200g"),
            ("양파", "반 개"),
            ("대파", "한 대"),
            ("두부", "반 모"),
            ("고춧가루", "1큰술"),
            ("다진마늘", "1큰술"),
            ("국간장", "1큰술"),
            ("멸치다시마 육수", "700ml"),
        ],
        "steps": [
            "냄비에 돼지고기를 먼저 볶아 기름을 살짝 내주세요",
            "신김치를 넣고 같이 볶아서 신맛을 눌러주세요",
            "고춧가루 다진마늘 국간장을 넣고 향만 올려주세요",
            "육수를 붓고 끓인 뒤 약불로 줄여서 더 끓여주세요",
            "양파를 넣고 두부를 넣고 마지막에 대파로 마무리해요",
        ],
        "image_url": "",
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1큰술 반"),
            ("애호박", "1/3개"),
            ("양파", "1/3개"),
            ("두부", "반 모"),
            ("대파", "반 대"),
            ("다진마늘", "1작은술"),
            ("멸치다시마 육수", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 한 번 끓여주세요",
            "양파 애호박을 넣고 부드럽게 익혀주세요",
            "두부를 넣고 한 번 더 끓인 뒤 대파로 향만 더해요",
        ],
        "image_url": "",
    },
    {
        "id": "bulgogi",
        "title": "간장 불고기",
        "ingredients": [
            ("소고기 불고기용", "300g"),
            ("양파", "반 개"),
            ("대파", "한 대"),
            ("간장", "4큰술"),
            ("설탕", "1큰술"),
            ("다진마늘", "1큰술"),
            ("참기름", "1큰술"),
            ("후추", "약간"),
            ("물 또는 배즙", "3큰술"),
        ],
        "steps": [
            "간장 설탕 다진마늘 참기름 물 후추를 섞어 양념을 만들어주세요",
            "고기에 양념을 넣고 잠깐 재워두면 맛이 안정적으로 들어가요",
            "팬에 고기를 볶다가 양파 대파를 넣고 숨이 죽을 때까지 볶아주세요",
        ],
        "image_url": "",
    },
]

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
        if not x:
            continue
        out.append(x)
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
    run_slot: str = "day"  # day/am/pm
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True
    mfds_timeout_sec: int = 10
    mfds_budget_sec: int = 18
    mfds_max_fails: int = 2


@dataclass
class ImageConfig:
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    reuse_media_by_search: bool = True

    default_thumb_url: str = ""
    auto_image: bool = True

    use_openai_image: bool = False
    openai_api_key: str = ""
    openai_model: str = "gpt-image-1-mini"
    openai_size: str = "1024x1024"
    openai_quality: str = "auto"
    openai_timeout_sec: int = 60


@dataclass
class ContentConfig:
    intro_min: int = 200
    intro_max: int = 300
    min_total_chars: int = 1200
    hashtag_count: int = 18


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
    wp_status = _env("WP_STATUS", "publish") or "publish"

    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    sqlite_path = _env("SQLITE_PATH", "data/daily_korean_recipe.sqlite3")

    use_openai_image = _env_bool("USE_OPENAI_IMAGE", False)
    openai_key = _env("OPENAI_API_KEY", "")

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
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout_sec=_env_int("MFDS_TIMEOUT_SEC", 10),
            mfds_budget_sec=_env_int("MFDS_BUDGET_SEC", 18),
            mfds_max_fails=_env_int("MFDS_MAX_FAILS", 2),
        ),
        img=ImageConfig(
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            auto_image=_env_bool("AUTO_IMAGE", True),
            use_openai_image=use_openai_image,
            openai_api_key=openai_key,
            openai_model=_env("OPENAI_IMAGE_MODEL", "gpt-image-1-mini"),
            openai_size=_env("OPENAI_IMAGE_SIZE", "1024x1024"),
            openai_quality=_env("OPENAI_IMAGE_QUALITY", "auto"),
            openai_timeout_sec=_env_int("OPENAI_TIMEOUT_SEC", 60),
        ),
        content=ContentConfig(
            intro_min=_env_int("INTRO_MIN", 200),
            intro_max=_env_int("INTRO_MAX", 300),
            min_total_chars=_env_int("MIN_TOTAL_CHARS", 1200),
            hashtag_count=_env_int("HASHTAG_COUNT", 18),
        ),
        sqlite_path=sqlite_path,
    )


def print_cfg(cfg: AppConfig) -> None:
    def ok(v: str) -> str:
        return "OK" if v else "EMPTY"

    print("[CFG] WP_BASE_URL:", ok(cfg.wp.base_url))
    print("[CFG] WP_USER:", ok(cfg.wp.user), f"(len={len(cfg.wp.user)})")
    print("[CFG] WP_APP_PASS:", ok(cfg.wp.app_pass), f"(len={len(cfg.wp.app_pass)})")
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CFG] WP_TAG_IDS:", cfg.wp.tag_ids)
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| STRICT_KOREAN:", cfg.recipe.strict_korean)
    print("[CFG] MFDS_TIMEOUT_SEC:", cfg.recipe.mfds_timeout_sec, "| MFDS_BUDGET_SEC:", cfg.recipe.mfds_budget_sec)
    print("[CFG] DEFAULT_THUMB_URL:", ok(cfg.img.default_thumb_url))
    print("[CFG] USE_OPENAI_IMAGE:", cfg.img.use_openai_image, "| OPENAI_API_KEY:", ok(cfg.img.openai_api_key))
    print("[CFG] AUTO_IMAGE:", cfg.img.auto_image)
    print("[CFG] MIN_TOTAL_CHARS:", cfg.content.min_total_chars)


# -----------------------------
# SQLite history
# -----------------------------


def init_db(sqlite_path: str) -> None:
    Path(os.path.dirname(sqlite_path) or ".").mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(sqlite_path)
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
            created_at TEXT
        )
        """
    )
    con.commit()
    con.close()


def get_today_post(sqlite_path: str, date_slot: str) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link
        FROM daily_posts
        WHERE date_slot=?
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
        "wp_post_id": row[4] or 0,
        "wp_link": row[5] or "",
    }


def upsert_today_post(sqlite_path: str, date_slot: str, recipe_source: str, recipe_id: str, recipe_title: str, wp_post_id: int, wp_link: str) -> None:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO daily_posts(date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link, created_at)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(date_slot) DO UPDATE SET
          recipe_source=excluded.recipe_source,
          recipe_id=excluded.recipe_id,
          recipe_title=excluded.recipe_title,
          wp_post_id=excluded.wp_post_id,
          wp_link=excluded.wp_link
        """,
        (date_slot, recipe_source, recipe_id, recipe_title, int(wp_post_id), wp_link, datetime.now(tz=KST).isoformat()),
    )
    con.commit()
    con.close()


def get_recent_recipe_ids(sqlite_path: str, avoid_days: int) -> List[Tuple[str, str]]:
    cutoff = datetime.now(tz=KST) - timedelta(days=max(1, avoid_days))
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT recipe_source, recipe_id
        FROM daily_posts
        WHERE created_at >= ?
        ORDER BY created_at DESC
        LIMIT 500
        """,
        (cutoff.isoformat(),),
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/3.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, excerpt: str = "") -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data.get("id") or 0), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, featured_media: int = 0, excerpt: str = "") -> Tuple[int, str]:
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
    data = r.json()
    return int(data.get("id") or 0), str(data.get("link") or "")


def wp_find_media_by_search(cfg: WordPressConfig, search: str) -> Optional[Tuple[int, str]]:
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
    it = items[0]
    mid = int(it.get("id") or 0)
    src = str(it.get("source_url") or "")
    if mid and src:
        return mid, src
    return None


def wp_upload_media_bytes(cfg: WordPressConfig, content: bytes, filename: str, content_type: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": content_type,
    }
    rr = requests.post(url, headers=headers, data=content, timeout=90)
    if rr.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {rr.status_code} body={rr.text[:500]}")
    data = rr.json()
    return int(data.get("id") or 0), str(data.get("source_url") or "")


def wp_upload_media_from_url(cfg: WordPressConfig, image_url: str, filename: str) -> Tuple[int, str]:
    r = requests.get(image_url, timeout=35)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code}")

    ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
    if not ctype:
        if filename.lower().endswith(".png"):
            ctype = "image/png"
        elif filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            ctype = "image/jpeg"
        else:
            ctype = "application/octet-stream"

    return wp_upload_media_bytes(cfg, r.content, filename, ctype)


# -----------------------------
# Recipe model / MFDS
# -----------------------------


@dataclass
class Recipe:
    source: str  # mfds|local
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    image_url: str = ""


def _has_hangul(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s or ""))


def _is_korean_recipe_name(name: str, strict: bool = True) -> bool:
    name = (name or "").strip()
    if not name:
        return False
    if strict and not _has_hangul(name):
        return False
    bad = ["pasta", "pizza", "taco", "sushi", "curry", "sandwich", "burger", "salad"]
    low = name.lower()
    if any(x in low for x in bad):
        return False
    return True


def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int = 1, end: int = 60, timeout_sec: int = 10) -> List[Dict[str, Any]]:
    if not api_key:
        return []
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(str(value))}"
    try:
        r = requests.get(url, timeout=(5, max(5, int(timeout_sec))))
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


def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or ""
    title = str(row.get("RCP_NM") or "").strip()
    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()

    # 재료는 MFDS가 긴 문장으로 오는 경우가 많아서 ',' 기준으로만 가볍게 나눔
    ingredients: List[str] = []
    for p in re.split(r"\s*,\s*", parts):
        p = p.strip()
        if p:
            ingredients.append(p)

    steps: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        if not s:
            continue
        s = re.sub(r"\s+", " ", s).strip()
        s = s.replace(".", "")
        steps.append(s)

    img_main = str(row.get("ATT_FILE_NO_MAIN") or "").strip()
    if not img_main:
        img_main = str(row.get("ATT_FILE_NO_MK") or "").strip()

    return Recipe(
        source="mfds",
        recipe_id=rid or hashlib.sha1(title.encode("utf-8")).hexdigest()[:10],
        title=title,
        ingredients=ingredients,
        steps=steps,
        image_url=img_main if img_main.startswith("http") else "",
    )


def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡", "불고기", "잡채", "비빔밥"]

    t0 = time.time()
    fails = 0

    for _ in range(30):
        if time.time() - t0 > max(8, cfg.recipe.mfds_budget_sec):
            break

        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, start=1, end=60, timeout_sec=cfg.recipe.mfds_timeout_sec)
        if not rows:
            fails += 1
            if fails >= max(1, cfg.recipe.mfds_max_fails):
                break
            continue

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


def pick_recipe_local(recent_pairs: List[Tuple[str, str]]) -> Recipe:
    used = set(recent_pairs)
    pool = [x for x in LOCAL_KOREAN_RECIPES if ("local", str(x.get("id"))) not in used]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    pick = random.choice(pool)
    ing = [f"{a} {b}".strip() for a, b in pick.get("ingredients", [])]
    steps = [str(s).strip().replace(".", "") for s in pick.get("steps", []) if str(s).strip()]
    return Recipe(
        source="local",
        recipe_id=str(pick.get("id")),
        title=str(pick.get("title")),
        ingredients=ing,
        steps=steps,
        image_url=str(pick.get("image_url") or "").strip(),
    )


def get_recipe_by_id(cfg: AppConfig, source: str, recipe_id: str) -> Optional[Recipe]:
    if source == "local":
        for x in LOCAL_KOREAN_RECIPES:
            if str(x.get("id")) == recipe_id:
                ing = [f"{a} {b}".strip() for a, b in x.get("ingredients", [])]
                steps = [str(s).strip().replace(".", "") for s in x.get("steps", []) if str(s).strip()]
                return Recipe(source="local", recipe_id=recipe_id, title=str(x.get("title") or ""), ingredients=ing, steps=steps, image_url=str(x.get("image_url") or "").strip())
        return None

    if source == "mfds" and cfg.recipe.mfds_api_key:
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_SEQ", recipe_id, start=1, end=5, timeout_sec=cfg.recipe.mfds_timeout_sec)
        for row in rows:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if rcp.recipe_id == recipe_id:
                return rcp
    return None


# -----------------------------
# OpenAI Image generator
# -----------------------------


def openai_generate_recipe_image_bytes(cfg: AppConfig, recipe: Recipe) -> Optional[bytes]:
    if not cfg.img.use_openai_image:
        return None
    if not cfg.img.openai_api_key:
        return None

    # 너무 많은 정보를 넣으면 음식이 흐려져서 핵심만
    key_ing = []
    for x in recipe.ingredients[:8]:
        x = re.sub(r"\(.*?\)", " ", x)
        x = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", x)
        w = [t for t in x.split() if t]
        if w:
            key_ing.append(w[0])
    key_ing = list(dict.fromkeys(key_ing))[:5]

    prompt = (
        f"A high-quality appetizing food photo of Korean dish '{recipe.title.strip()}', "
        f"served nicely on a simple Korean table setting, natural light, realistic, "
        f"ingredients hint: {', '.join(key_ing) if key_ing else 'Korean home-cooking'}, "
        f"no text, no watermark, no logo, no people"
    )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=cfg.img.openai_api_key, timeout=cfg.img.openai_timeout_sec)
        res = client.images.generate(
            model=cfg.img.openai_model,
            prompt=prompt,
            size=cfg.img.openai_size,
            quality=cfg.img.openai_quality,
            n=1,
        )
        b64 = res.data[0].b64_json
        if not b64:
            return None
        return base64.b64decode(b64)
    except Exception:
        return None


# -----------------------------
# Image selection / upload
# -----------------------------


def choose_thumb_url(cfg: AppConfig, recipe: Recipe) -> str:
    if (recipe.image_url or "").strip():
        return recipe.image_url.strip()
    if (cfg.img.default_thumb_url or "").strip():
        return cfg.img.default_thumb_url.strip()
    if cfg.img.auto_image:
        q = recipe.title.strip()
        return f"https://source.unsplash.com/1200x800/?{quote(q)},korean,food"
    return ""


def ensure_media_from_url(cfg: AppConfig, image_url: str, stable_name: str) -> Tuple[int, str]:
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

    return wp_upload_media_from_url(cfg.wp, image_url, filename)


def ensure_media_from_bytes(cfg: AppConfig, content: bytes, stable_name: str) -> Tuple[int, str]:
    h = hashlib.sha1(content).hexdigest()[:12]
    filename = f"{stable_name}_{h}.png"

    if cfg.img.reuse_media_by_search:
        found = wp_find_media_by_search(cfg.wp, search=f"{stable_name}_{h}")
        if found:
            return found

    return wp_upload_media_bytes(cfg.wp, content, filename, "image/png")


# -----------------------------
# Content (homefeed)
# -----------------------------


def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def _strip_period(s: str) -> str:
    return (s or "").replace(".", "").replace("·", " ")


def _plain_text_from_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _fit_intro(text: str, min_len: int, max_len: int) -> str:
    t = _strip_period(re.sub(r"\s+", " ", text or "")).strip()
    if len(t) > max_len:
        t = t[:max_len]
        t = re.sub(r"\s+\S*$", "", t).strip()
    if len(t) < min_len:
        # 짧으면 자연스럽게 한 줄 덧붙이기
        extra = "  괜히 어렵게 생각하지 말고 오늘은 가볍게 한 번만 해보셔요"
        t = (t + extra).strip()
        if len(t) > max_len:
            t = t[:max_len]
            t = re.sub(r"\s+\S*$", "", t).strip()
    return t


INTRO_TEMPLATES = [
    "오늘은 {t} 얘기 좀 해볼게요  이상하게 바쁜 날일수록 이런 메뉴가 더 당기더라고요  손이 많이 갈 것 같아 보여도 흐름만 잡으면 금방 끝나요",
    "{t} 는 한 번만 성공하면 그 다음부터는 마음이 엄청 편해져요  실패가 줄어드는 포인트를 제가 겪은 그대로 풀어볼게요",
    "요즘 같은 날씨에는 {t} 가 진짜 든든하잖아요  집에서 해먹으면 재료도 조절되고 간도 내 입맛대로 잡히니까 더 좋더라고요",
]


def _pick_key_ingredients(recipe: Recipe, n: int = 4) -> List[str]:
    out: List[str] = []
    for x in recipe.ingredients[:12]:
        x = re.sub(r"\(.*?\)", " ", x)
        x = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", x)
        w = [t for t in x.split() if t]
        if w:
            out.append(w[0])
    # de-dup
    uniq: List[str] = []
    seen = set()
    for x in out:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq[:n]


def _summarize_steps(recipe: Recipe) -> Tuple[str, str]:
    # 첫 2개는 흐름, 마지막 1개는 마무리
    steps = [re.sub(r"\s+", " ", s).strip() for s in recipe.steps if s.strip()]
    if not steps:
        return "재료를 준비하고 불을 너무 세게만 안 쓰면 대부분 편하게 끝나요", "마지막에 간만 살짝 보고 바로 드시면 제일 좋아요"
    a = steps[0]
    b = steps[1] if len(steps) > 1 else steps[0]
    c = steps[-1]
    return _strip_period(a), _strip_period(c if c else b)


def _make_paragraph(lines: List[str]) -> str:
    lines = [re.sub(r"\s+", " ", _strip_period(x)).strip() for x in lines if str(x).strip()]
    if not lines:
        return ""
    # 홈피드 호흡
    return "<p>" + "<br/><br/>".join(_esc(x) for x in lines) + "</p>"


def build_hashtags(cfg: AppConfig, recipe: Recipe) -> str:
    base = []
    title = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", recipe.title)
    for w in title.split():
        w = re.sub(r"\s+", "", w)
        if w:
            base.append(w)

    for w in _pick_key_ingredients(recipe, n=6):
        base.append(w)

    base += [
        "한식레시피",
        "집밥",
        "오늘뭐먹지",
        "간단요리",
        "자취요리",
        "요리팁",
        "집밥메뉴",
        "한식",
        "맛있게",
        "저녁메뉴",
        "점심메뉴",
    ]

    uniq: List[str] = []
    seen = set()
    for x in base:
        x = re.sub(r"\s+", "", x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    n = max(10, min(30, cfg.content.hashtag_count))
    return " ".join([f"#{x}" for x in uniq[:n]])


def render_recipe_list(recipe: Recipe) -> str:
    ing_lines = [re.sub(r"\s+", " ", _strip_period(x)).strip() for x in recipe.ingredients if str(x).strip()]
    step_lines = [re.sub(r"\s+", " ", _strip_period(x)).strip() for x in recipe.steps if str(x).strip()]

    ing_html = "<br/>".join(_esc(x) for x in ing_lines) if ing_lines else _esc("재료는 집에 있는 걸로 유연하게 맞추셔도 돼요")

    # 번호는 특수문자 불릿이 아니라 숫자라서 허용
    steps_out: List[str] = []
    for i, s in enumerate(step_lines[:20]):
        steps_out.append(f"{i+1}  {s}")
    step_html = "<br/>".join(_esc(x) for x in steps_out) if steps_out else _esc("순서는 어렵지 않게 한 번에 이어가시면 돼요")

    out = ""
    out += "<p><strong>재료 목록</strong><br/>" + ing_html + "</p>"
    out += "<p><strong>만드는 순서</strong><br/>" + step_html + "</p>"
    return out


def build_body_html(cfg: AppConfig, recipe: Recipe, display_img_url: str) -> Tuple[str, str]:
    intro = random.choice(INTRO_TEMPLATES).format(t=recipe.title.strip())
    intro = _fit_intro(intro, cfg.content.intro_min, cfg.content.intro_max)

    key_ing = _pick_key_ingredients(recipe, n=4)
    step_first, step_last = _summarize_steps(recipe)

    h1 = f"{recipe.title.strip()}  오늘 이 메뉴를 추천하는 이유"
    h2 = "실패가 줄어드는 흐름  제가 해보면서 편했던 순서"
    h3 = "맛이 흔들릴 때 잡는 방법  다음번이 더 쉬워져요"

    s1 = [
        intro,
        f"저는 {recipe.title.strip()} 가 딱 그 느낌이에요  힘이 빠질 때도 한 숟갈 뜨면 마음이 좀 풀리는 메뉴요",
        f"재료도 {', '.join(key_ing) if key_ing else '집에 있는 기본 재료'} 정도만 잡아두면 생각보다 간단해져요",
        "처음부터 완벽하게 하려는 마음만 내려두면 맛이 훨씬 안정적으로 나오더라고요",
    ]

    s2 = [
        f"흐름은 이렇게만 기억해두셔도 돼요  {step_first}",
        "불 조절은 센 불로 끝까지 밀지 말고 중불로 천천히 가는 쪽이 실패가 훨씬 적어요",
        "간은 중간에 한 번만 보고 마지막에 아주 살짝만 조정하면 과해질 일이 거의 없어요",
        "혹시 오늘 시간이 없으면 재료 손질만 먼저 해두고 조리만 이어가셔도 편해요",
    ]

    s3 = [
        f"먹어보면 한 번에 확 오는 포인트가 있어요  마지막에 {step_last} 이 부분이 저는 제일 중요하더라고요",
        "그리고 남았을 때도 너무 걱정하지 마세요  다음날 데울 때는 물 조금만 보태고 천천히 데우면 괜찮아요",
        "혹시 입맛이 심심하면 대파나 고춧가루 같은 걸 마지막에 아주 조금만 더해보시면 느낌이 살아나요",
        "다음번에는 오늘 한 번 해본 흐름이 몸에 남아서 훨씬 빠르게 끝나실 거예요",
    ]

    # 글자수 최소 보정
    def text_len(body_html: str) -> int:
        return len(_plain_text_from_html(body_html))

    def pack() -> str:
        img_html = f'<p><img src="{_esc(display_img_url)}" alt="{_esc(recipe.title.strip())}"/></p>' if (cfg.img.embed_image_in_body and display_img_url) else ""
        body = ""
        body += img_html
        body += f"<h3><strong>{_esc(_strip_period(h1))}</strong></h3>" + _make_paragraph(s1)
        body += f"<h3><strong>{_esc(_strip_period(h2))}</strong></h3>" + _make_paragraph(s2)
        body += render_recipe_list(recipe)
        body += f"<h3><strong>{_esc(_strip_period(h3))}</strong></h3>" + _make_paragraph(s3)
        body += "<p><br/><br/>" + _esc(build_hashtags(cfg, recipe)) + "</p>"
        return body

    body_html = pack()

    fillers = [
        "혹시 오늘 기분이 축 처져 있으면요  이런 따뜻한 메뉴 하나만 있어도 하루가 좀 덜 날카로워지더라고요",
        "저는 요리할 때 음악 한 곡 틀어두고 천천히 움직이면 괜히 마음이 정리되는 느낌이 있어요",
        "처음엔 양을 작게 해보셔도 좋아요  작은 성공 한 번이 다음번을 엄청 편하게 만들어줘요",
        "간이 애매하면 소금보다 간장이나 된장처럼 익숙한 쪽으로 아주 조금만 움직이는 게 안전해요",
        "오늘은 완벽이 아니라 편안함이 목표라고 생각하면 훨씬 잘 되실 거예요",
    ]

    tries = 0
    while text_len(body_html) < max(600, cfg.content.min_total_chars) and tries < 6:
        s3.append(random.choice(fillers))
        body_html = pack()
        tries += 1

    excerpt = _strip_period(f"{recipe.title.strip()} 레시피  집에서 편하게 해먹는 흐름으로 정리했어요")
    excerpt = re.sub(r"\s+", " ", excerpt).strip()[:140]

    return body_html, excerpt


# -----------------------------
# Title
# -----------------------------

TITLE_BENEFIT = [
    "{t} 집에서 해먹으니까 마음이 편해지더라고요",
    "{t} 이렇게 하면 실패가 확 줄어요",
    "{t} 오늘 저녁 메뉴로 딱이에요",
]
TITLE_CURIOSITY = [
    "{t} 왜 자꾸 생각나는지 오늘 알았어요",
    "{t} 딱 한 가지 포인트가 있어요",
    "{t} 한 번만 해보시면 감이 와요",
]
TITLE_COMPARE = [
    "{t} 사먹는 맛이랑 비교해보니 결론이 났어요",
    "{t} 배달 대신 해먹어보면 차이가 커요",
    "{t} 집밥으로 돌리면 진짜 이득이에요",
]


def build_post_title(date_str: str, slot: str, recipe_title: str) -> str:
    style = (_env("TITLE_STYLE", "random") or "random").lower()
    if style not in ("benefit", "curiosity", "compare", "random"):
        style = "random"

    if style == "random":
        style = random.choice(["benefit", "curiosity", "compare"])

    if style == "benefit":
        fmt = random.choice(TITLE_BENEFIT)
    elif style == "curiosity":
        fmt = random.choice(TITLE_CURIOSITY)
    else:
        fmt = random.choice(TITLE_COMPARE)

    core = _strip_period(fmt.format(t=recipe_title.strip())).strip()

    # 제목에 기준시각/슬롯 노출 싫으면 DATE_PREFIX=0
    if _env_bool("DATE_PREFIX", True):
        return f"{date_str} {core}"
    return core


# -----------------------------
# Main
# -----------------------------


def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    date_slot = f"{date_str}_{slot}"

    init_db(cfg.sqlite_path)
    today_meta = get_today_post(cfg.sqlite_path, date_slot)
    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days)

    chosen: Optional[Recipe] = None
    if today_meta and (not cfg.run.force_new) and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

    if not chosen:
        chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(recent_pairs)

    assert chosen is not None

    title = build_post_title(date_str, slot, chosen.title)
    slug = f"korean-recipe-{date_str}-{slot}"

    # 1) 기본 이미지 URL
    thumb_url = choose_thumb_url(cfg, chosen)

    media_id = 0
    media_url = ""

    # 2) URL 이미지 업로드 시도
    if cfg.img.upload_thumb and thumb_url:
        try:
            media_id, media_url = ensure_media_from_url(cfg, thumb_url, stable_name="korean_recipe")
        except Exception:
            media_id, media_url = 0, ""

    # 3) 이미지가 여전히 없으면 OpenAI 생성 폴백
    if cfg.img.upload_thumb and not media_url and cfg.img.use_openai_image:
        try:
            img_bytes = openai_generate_recipe_image_bytes(cfg, chosen)
            if img_bytes:
                media_id, media_url = ensure_media_from_bytes(cfg, img_bytes, stable_name="korean_recipe_ai")
        except Exception:
            media_id, media_url = 0, ""

    display_img_url = (media_url or thumb_url or "").strip()

    body_html, excerpt = build_body_html(cfg, chosen, display_img_url)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print(body_html[:2000])
        return

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    if today_meta and int(today_meta.get("wp_post_id") or 0):
        post_id = int(today_meta["wp_post_id"])
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, featured_media=featured_id, excerpt=excerpt)
        print("OK(updated):", wp_post_id, wp_link, "| source=", chosen.source)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt)
        # featured는 생성 후 업데이트하는 테마가 있어서 한 번 더
        if featured_id:
            try:
                wp_post_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured_id, excerpt=excerpt)
            except Exception:
                pass
        print("OK(created):", wp_post_id, wp_link, "| source=", chosen.source)

    upsert_today_post(cfg.sqlite_path, date_slot, chosen.source, chosen.recipe_id, chosen.title, wp_post_id, wp_link)


def main() -> None:
    cfg = load_cfg()
    print_cfg(cfg)

    if not cfg.wp.base_url or not cfg.wp.user or not cfg.wp.app_pass:
        raise SystemExit("WP_BASE_URL / WP_USER / WP_APP_PASS 가 필요합니다")

    print(f"[RUN] slot={cfg.run.run_slot} force_new={int(cfg.run.force_new)}")
    run(cfg)


if __name__ == "__main__":
    main()
