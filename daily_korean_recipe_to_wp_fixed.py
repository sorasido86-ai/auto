# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp.py (홈피드형 / 1200자 보장 / 이미지 안정화 / 중복업로드 허용 옵션)

요구사항 반영
- “레시피(재료/순서)”는 항상 포함
- 특수문자(불릿, 체크표시, 점찍힌 문자) 제거
- 해시태그 12~20개 자동 생성
- 중복 업로드 허용(ALLOW_DUPLICATE_POSTS=1)
- 글자수 총 MIN_TOTAL_CHARS(기본 1200) 이상만 보장 + 과도하게 길어지지 않게 MAX_TOTAL_CHARS(기본 1800)

이미지 전략(깨진 이미지 방지)
- 1) MFDS 레시피 이미지 다운로드 → WP 미디어 업로드
- 2) DEFAULT_THUMB_URL 다운로드 → WP 미디어 업로드
- 3) USE_OPENAI_IMAGE=1 이면 OpenAI 이미지 생성 → WP 미디어 업로드
- 4) AUTO_STOCK_IMAGE=1(기본 1) 이면 picsum 이미지 다운로드 → WP 미디어 업로드
- 업로드 실패 시에는 (가능한 경우) 원격 URL을 본문에 직접 삽입(최후 수단)

MFDS 타임아웃/장시간 실행 방지
- MFDS_TIMEOUT_SEC(기본 10), MFDS_BUDGET_SEC(기본 18) 내에서만 시도
- 예외/타임아웃 시 MFDS는 즉시 포기하고 로컬 레시피로 폴백

필수 환경변수(Secrets)
- WP_BASE_URL, WP_USER, WP_APP_PASS

선택 환경변수
- MFDS_API_KEY
- DEFAULT_THUMB_URL
- USE_OPENAI_IMAGE=1 + OPENAI_API_KEY (+ OPENAI_IMAGE_MODEL=gpt-image-1)
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
# 폴백 로컬 레시피 (필요시 추가 가능)
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1.5큰술"),
            ("고추장(선택)", "반 큰술"),
            ("애호박", "삼분의 일개"),
            ("양파", "삼분의 일개"),
            ("두부", "반 모"),
            ("대파", "반 대"),
            ("다진마늘", "작은술 하나"),
            ("멸치다시마 육수(또는 물)", "칠백 밀리리터"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 약불로 은근히 끓여요",
            "양파와 애호박을 넣고 부드럽게 익혀요",
            "두부를 넣고 한 번 더 끓여요",
            "대파와 다진마늘을 넣고 불을 끄기 직전에 향만 살려요",
            "간은 마지막에만 보고 싱거우면 된장을 아주 조금만 더 풀어줘요",
        ],
        "image_url": "",
    },
    {
        "id": "kimchi-jjigae",
        "title": "돼지고기 김치찌개",
        "ingredients": [
            ("신김치", "두 컵"),
            ("돼지고기", "이백 그램"),
            ("양파", "반 개"),
            ("대파", "한 대"),
            ("두부", "반 모"),
            ("고춧가루", "큰술 하나"),
            ("다진마늘", "큰술 하나"),
            ("국간장", "큰술 하나"),
            ("물 또는 육수", "칠백 밀리리터"),
        ],
        "steps": [
            "냄비에 돼지고기를 먼저 볶아서 고기향을 내요",
            "김치를 넣고 같이 볶아 신맛을 한 번 눌러줘요",
            "고춧가루와 다진마늘을 넣고 향만 살려요",
            "육수를 붓고 중불로 끓인 다음 약불로 줄여요",
            "두부와 대파를 마지막에 넣고 한 번만 더 끓여요",
        ],
        "image_url": "",
    },
    {
        "id": "bulgogi",
        "title": "간장 불고기",
        "ingredients": [
            ("소고기 불고기용", "삼백 그램"),
            ("양파", "반 개"),
            ("대파", "한 대"),
            ("간장", "큰술 네 개"),
            ("설탕", "큰술 하나"),
            ("다진마늘", "큰술 하나"),
            ("참기름", "큰술 하나"),
            ("후추", "약간"),
            ("물 또는 배즙", "큰술 세 개"),
        ],
        "steps": [
            "양념을 먼저 섞어서 고기에 골고루 버무려요",
            "최소 십오 분은 재워두면 훨씬 부드러워요",
            "팬을 달군 뒤 고기를 먼저 볶아 육즙을 잡아요",
            "양파와 대파를 넣고 숨이 죽을 때까지만 볶아요",
            "불을 끄기 직전에 참기름 향이 과하지 않게 한 번만 섞어요",
        ],
        "image_url": "",
    },
]

# 불릿류/체크표시/점찍힌 문자 제거 대상
BAD_SYMBOLS = [
    "•", "·", "●", "○", "▪", "▫", "■", "□", "◆", "◇", "▶", "▷", "★", "☆",
    "✅", "✔", "☑", "✓", "✳", "✴", "❗", "❌", "‼", "…",
]

KOREAN_NEGATIVE_KEYWORDS = ["파스타", "피자", "타코", "스시", "커리", "샌드위치", "버거", "샐러드"]


# -----------------------------
# env helpers
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
    auto_tags: bool = True
    auto_tags_max: int = 10


@dataclass
class RunConfig:
    run_slot: str = "day"  # day/am/pm
    allow_duplicate_posts: bool = False
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 12
    min_total_chars: int = 1200
    max_total_chars: int = 1800


@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True
    mfds_timeout_sec: int = 10
    mfds_budget_sec: int = 18


@dataclass
class ImageConfig:
    default_thumb_url: str = ""
    upload_media: bool = True
    set_featured: bool = True
    embed_in_body: bool = True
    use_openai_image: bool = False
    openai_api_key: str = ""
    openai_image_model: str = "gpt-image-1"
    openai_image_size: str = "1024x1024"
    auto_stock_image: bool = True
    stock_width: int = 1200
    stock_height: int = 800


@dataclass
class ContentConfig:
    hashtag_min: int = 12
    hashtag_max: int = 20


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

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=cat_ids,
            tag_ids=tag_ids,
            auto_tags=_env_bool("WP_AUTO_TAGS", True),
            auto_tags_max=_env_int("WP_AUTO_TAGS_MAX", 10),
        ),
        run=RunConfig(
            run_slot=run_slot,
            allow_duplicate_posts=_env_bool("ALLOW_DUPLICATE_POSTS", False),
            force_new=_env_bool("FORCE_NEW", False),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 12),
            min_total_chars=_env_int("MIN_TOTAL_CHARS", 1200),
            max_total_chars=_env_int("MAX_TOTAL_CHARS", 1800),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout_sec=_env_int("MFDS_TIMEOUT_SEC", 10),
            mfds_budget_sec=_env_int("MFDS_BUDGET_SEC", 18),
        ),
        img=ImageConfig(
            default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
            upload_media=_env_bool("UPLOAD_MEDIA", True),
            set_featured=_env_bool("SET_FEATURED", True),
            embed_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
            use_openai_image=_env_bool("USE_OPENAI_IMAGE", False),
            openai_api_key=_env("OPENAI_API_KEY", ""),
            openai_image_model=_env("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            openai_image_size=_env("OPENAI_IMAGE_SIZE", "1024x1024"),
            auto_stock_image=_env_bool("AUTO_STOCK_IMAGE", True),
            stock_width=_env_int("STOCK_W", 1200),
            stock_height=_env_int("STOCK_H", 800),
        ),
        content=ContentConfig(
            hashtag_min=_env_int("HASHTAG_MIN", 12),
            hashtag_max=_env_int("HASHTAG_MAX", 20),
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

    print("[CFG] WP_BASE_URL:", "OK" if cfg.wp.base_url else "MISSING")
    print("[CFG] WP_USER:", ok(cfg.wp.user))
    print("[CFG] WP_APP_PASS:", ok(cfg.wp.app_pass))
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CFG] WP_TAG_IDS:", cfg.wp.tag_ids)
    print("[CFG] WP_AUTO_TAGS:", cfg.wp.auto_tags, "MAX:", cfg.wp.auto_tags_max)
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| ALLOW_DUPLICATE_POSTS:", int(cfg.run.allow_duplicate_posts), "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| STRICT_KOREAN:", cfg.recipe.strict_korean)
    print("[CFG] MFDS_TIMEOUT_SEC:", cfg.recipe.mfds_timeout_sec, "| MFDS_BUDGET_SEC:", cfg.recipe.mfds_budget_sec)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] USE_OPENAI_IMAGE:", cfg.img.use_openai_image, "| OPENAI_API_KEY:", ok(cfg.img.openai_api_key))
    print("[CFG] AUTO_STOCK_IMAGE:", cfg.img.auto_stock_image)
    print("[CFG] MIN_TOTAL_CHARS:", cfg.run.min_total_chars, "| MAX_TOTAL_CHARS:", cfg.run.max_total_chars)


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
          media_id INTEGER,
          media_url TEXT,
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


def save_post_meta(sqlite_path: str, meta: Dict[str, Any]) -> None:
    con = sqlite3.connect(sqlite_path)
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


def get_recent_recipe_ids(sqlite_path: str, days: int) -> List[Tuple[str, str]]:
    since = datetime.utcnow() - timedelta(days=days)
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT recipe_source, recipe_id, created_at
        FROM daily_posts
        WHERE created_at IS NOT NULL AND created_at != ''
        """
    )
    rows = cur.fetchall()
    con.close()
    out: List[Tuple[str, str]] = []
    for s, rid, created_at in rows:
        try:
            if created_at and datetime.fromisoformat(created_at) < since:
                continue
        except Exception:
            pass
        if s and rid:
            out.append((str(s), str(rid)))
    return out


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/2.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, excerpt: str = "", tag_ids: Optional[List[int]] = None, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if tag_ids:
        payload["tags"] = tag_ids
    elif cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = featured_media

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, excerpt: str = "", tag_ids: Optional[List[int]] = None, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status}
    if excerpt:
        payload["excerpt"] = excerpt
    if featured_media:
        payload["featured_media"] = featured_media
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if tag_ids:
        payload["tags"] = tag_ids
    elif cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media_bytes(cfg: WordPressConfig, content: bytes, filename: str, content_type: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": content_type,
    }
    rr = requests.post(url, headers=headers, data=content, timeout=60)
    if rr.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {rr.status_code} body={rr.text[:500]}")
    data = rr.json()
    return int(data["id"]), str(data.get("source_url") or "")


def wp_find_or_create_tag(cfg: WordPressConfig, tag_name: str) -> Optional[int]:
    try:
        url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/tags"
        headers = wp_auth_header(cfg.user, cfg.app_pass)
        r = requests.get(url, headers=headers, params={"search": tag_name, "per_page": 20}, timeout=20)
        if r.status_code == 200:
            items = r.json()
            if isinstance(items, list):
                for it in items:
                    if str(it.get("name") or "").strip() == tag_name:
                        tid = int(it.get("id") or 0)
                        if tid:
                            return tid
                if items:
                    tid = int(items[0].get("id") or 0)
                    if tid:
                        return tid
        headers2 = {**headers, "Content-Type": "application/json"}
        rr = requests.post(url, headers=headers2, json={"name": tag_name}, timeout=25)
        if rr.status_code in (200, 201):
            data = rr.json()
            return int(data.get("id") or 0) or None
    except Exception:
        return None
    return None


# -----------------------------
# Recipe model / MFDS provider
# -----------------------------
@dataclass
class Recipe:
    source: str  # "mfds" or "local"
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]
    image_url: str = ""


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


def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int = 1, end: int = 60, timeout_sec: int = 10) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    try:
        r = requests.get(url, timeout=timeout_sec)
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
    if parts:
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

    img_main = str(row.get("ATT_FILE_NO_MAIN") or "").strip() or str(row.get("ATT_FILE_NO_MK") or "").strip()
    return Recipe(
        source="mfds",
        recipe_id=rid or hashlib.sha1(title.encode("utf-8")).hexdigest()[:8],
        title=title,
        ingredients=ingredients,
        steps=steps,
        image_url=img_main if (img_main or "").startswith("http") else "",
    )


def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡"]

    t0 = time.time()
    budget = max(3, int(cfg.recipe.mfds_budget_sec))

    while time.time() - t0 < budget:
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(
            cfg.recipe.mfds_api_key,
            "RCP_NM",
            kw,
            start=1,
            end=60,
            timeout_sec=max(3, int(cfg.recipe.mfds_timeout_sec)),
        )
        if not rows:
            continue
        random.shuffle(rows)
        for row in rows:
            if time.time() - t0 >= budget:
                break
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


# -----------------------------
# Content rendering (홈피드형)
# -----------------------------
def _esc(s: str) -> str:
    return html.escape(s or "")


def _strip_bad_symbols(s: str) -> str:
    out = s or ""
    for sym in BAD_SYMBOLS:
        out = out.replace(sym, " ")
    out = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219]", " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _remove_period_like(s: str) -> str:
    s = (s or "").replace(".", " ").replace("!", " ").replace("?", " ").replace("。", " ").replace("…", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_for_title(s: str) -> str:
    return _remove_period_like(_strip_bad_symbols(s))


def _clean_tokens(s: str) -> List[str]:
    s = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", s or "")
    toks = [x.strip() for x in s.split() if x.strip()]
    out: List[str] = []
    seen = set()
    for t in toks:
        if len(t) <= 1:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def build_hashtags(cfg: AppConfig, recipe: Recipe) -> str:
    base = [
        "한식", "집밥", "오늘뭐먹지", "간단요리", "초간단레시피",
        "자취요리", "집밥레시피", "요리", "한식레시피", "레시피",
        "저녁메뉴", "점심메뉴", "국물요리", "밥반찬", "집밥일기",
    ]
    for tok in _clean_tokens(recipe.title)[:6]:
        base.append(tok)
    for ing in (recipe.ingredients or [])[:6]:
        for tok in _clean_tokens(ing)[:2]:
            base.append(tok)

    uniq: List[str] = []
    seen = set()
    for x in base:
        x = re.sub(r"\s+", "", x)
        x = _strip_bad_symbols(x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    pool = uniq[:]
    random.shuffle(pool)
    picked = pool[: min(20, max(12, len(pool)))] if pool else uniq

    if len(picked) < 12:
        picked = (uniq + picked)[:12]
    if len(picked) > 20:
        picked = picked[:20]
    return " ".join([f"#{x}" for x in picked])


def _text_len_from_html(html_text: str) -> int:
    t = re.sub(r"<[^>]+>", " ", html_text or "")
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return len(t)


def _paragraph(*lines: str) -> str:
    cleaned = [_remove_period_like(_strip_bad_symbols(x)) for x in lines if (x or "").strip()]
    cleaned = [x for x in cleaned if x]
    if not cleaned:
        return ""
    return "<p>" + "<br/>".join(_esc(x) for x in cleaned) + "</p>"


def _strong(title: str) -> str:
    return f"<p><strong>{_esc(_clean_for_title(title))}</strong></p>"


def build_intro_200_300(title: str) -> str:
    base = [
        f"오늘은 {title}을 같이 해보려고요",
        "요즘은 레시피가 너무 화려해서 따라가다 지치는 날이 있잖아요",
        "그래서 저는 딱 실패 확률 낮게만 정리해두고 마음 편하게 끓이거나 볶는 쪽을 선택하더라고요",
        "괜히 거창하게 준비하지 말고 지금 집에 있는 재료로 충분히 맛 내는 방향으로 가져가볼게요",
        "한 번만 해두면 다음에는 눈 감고도 할 수 있게 순서와 간 포인트만 남겨둘게요",
    ]
    txt = " ".join(random.sample(base, k=4))
    txt = _remove_period_like(_strip_bad_symbols(txt))
    if len(txt) < 200:
        txt += " " + _remove_period_like("저도 이렇게 해두면 마음이 정말 편해지더라고요")
    if len(txt) > 320:
        txt = txt[:300].rstrip()
    return _paragraph(txt)


def build_recipe_block(recipe: Recipe) -> str:
    ing_lines: List[str] = []
    for x in (recipe.ingredients or []):
        x = _remove_period_like(_strip_bad_symbols(x))
        if x:
            ing_lines.append(x)

    step_lines: List[str] = []
    for i, s in enumerate(recipe.steps or [], start=1):
        s = _remove_period_like(_strip_bad_symbols(s))
        if s:
            step_lines.append(f"{i}단계 {s}")

    if not ing_lines:
        ing_lines = ["재료 정보가 비어있어서 오늘은 집에 있는 재료로 응용해도 괜찮아요"]
    if not step_lines:
        step_lines = ["1단계 오늘은 순서 데이터가 비어있어서 기본 조리 흐름대로만 진행해도 좋아요"]

    out = ""
    out += _strong("레시피 재료")
    out += _paragraph(*ing_lines[:20])
    out += _strong("레시피 순서")
    out += _paragraph(*step_lines[:20])
    return out


def build_homefeed_body(cfg: AppConfig, recipe: Recipe, image_url: str) -> Tuple[str, str, List[str]]:
    title = _clean_for_title(recipe.title)
    intro = build_intro_200_300(title)

    sec1_title = f"{title}에서 제일 중요한 준비"
    sec2_title = f"{title} 순서가 헷갈릴 때 제가 잡는 기준"
    sec3_title = f"{title} 맛이 달라지는 마무리 팁"

    sec1_bank = [
        "저는 요리할 때 재료를 완벽하게 맞추려고 하면 오히려 시작이 늦어지더라고요",
        "그래서 오늘은 꼭 필요한 것만 먼저 잡고 나머지는 집에 있는 걸로 자연스럽게 이어가볼게요",
        "특히 간은 중간에 욕심내면 마지막에 애매해지기 쉬워서 저는 끝에서 한 번만 조절하는 편이에요",
        "그리고 불 조절은 강불로 밀어붙이기보다 중불에서 안정적으로 가는 쪽이 실패가 적었어요",
        "딱 한 번만 해보시면 아 이 정도면 되는구나 하고 감이 잡히실 거예요",
    ]
    sec2_bank = [
        "순서는 사실 복잡해 보이지만 흐름만 기억하면 마음이 편해져요",
        "저는 시작에서 향을 만들고 중간에는 재료를 익히고 끝에서 간과 향을 마무리하는 순서로 정리해요",
        "이렇게만 잡아두면 레시피가 길어도 안 헤매고 내 속도대로 따라갈 수 있어요",
        "혹시 중간에 텀이 생기면 그냥 불을 약하게 줄이고 한 호흡 쉬었다가 다시 이어가면 돼요",
        "급하게 마무리하려고 하면 재료가 덜 익거나 간이 튀는 일이 생기더라고요",
    ]
    sec3_bank = [
        "맛이 애매할 때 저는 소금이나 간장을 바로 넣기보다 먼저 한 번 더 끓이거나 한 번 더 볶아봐요",
        "생각보다 물기나 농도가 정리되면서 맛이 따라오는 경우가 많아서요",
        "그리고 마지막 향은 과하게 넣지 말고 살짝만 얹는 느낌으로 가면 훨씬 깔끔해져요",
        "오늘 한 번 해보고 내 입맛 기준으로 짠맛과 매운맛만 메모해두면 다음이 진짜 쉬워져요",
        "저도 그렇게 하면서 집밥 스트레스가 확 줄었어요",
    ]

    recipe_block = build_recipe_block(recipe)

    img_html = ""
    if cfg.img.embed_in_body and image_url:
        img_html = f'<figure class="wp-block-image size-large"><img src="{_esc(image_url)}" alt="{_esc(title)}"/></figure>'

    body_parts: List[str] = [intro]
    if img_html:
        body_parts.append(img_html)

    body_parts.append(_strong(sec1_title))
    body_parts.append(_paragraph(*random.sample(sec1_bank, k=4)))

    body_parts.append(_strong(sec2_title))
    body_parts.append(_paragraph(*random.sample(sec2_bank, k=4)))
    body_parts.append(recipe_block)

    body_parts.append(_strong(sec3_title))
    body_parts.append(_paragraph(*random.sample(sec3_bank, k=4)))

    hashtags = build_hashtags(cfg, recipe)
    body_parts.append(_paragraph(hashtags))

    body_html = "\n".join([p for p in body_parts if p])

    min_chars = max(600, int(cfg.run.min_total_chars))
    max_chars = max(min_chars + 100, int(cfg.run.max_total_chars))

    extra_bank = [
        "요리는 잘하려고 하기보다 오늘 한 끼를 편하게 해결한다고 생각하면 훨씬 마음이 가벼워져요",
        "저도 예전에는 레시피대로 안 되면 괜히 기분이 상했는데 지금은 내 방식대로 잡아두는 게 더 좋더라고요",
        "혹시 해보시고 간이 애매하면 그 느낌 그대로 댓글로 남겨주시면 다음에 같이 조정해볼게요",
        "저는 이렇게 적어두면 다음에 다시 할 때 진짜 고민이 줄어서 계속 쓰게 되더라고요",
    ]
    used_extra: set[str] = set()
    while _text_len_from_html(body_html) < min_chars:
        cand = random.choice(extra_bank)
        if cand in used_extra:
            continue
        used_extra.add(cand)
        body_html = body_html + "\n" + _paragraph(cand)
        if len(used_extra) >= len(extra_bank):
            break

    if _text_len_from_html(body_html) > max_chars:
        def shrink_paragraph(html_block: str) -> str:
            m = re.search(r"<p>(.*?)</p>", html_block, re.DOTALL)
            if not m:
                return html_block
            inner = m.group(1)
            lines = inner.split("<br/>")
            inner2 = "<br/>".join(lines[:3])
            return "<p>" + inner2 + "</p>"

        parts = body_html.split("\n")
        new_parts = []
        for p in parts:
            if "<strong>" in p:
                new_parts.append(p)
            elif "<p>" in p and "레시피" not in p and "#" not in p:
                new_parts.append(shrink_paragraph(p))
            else:
                new_parts.append(p)
        body_html = "\n".join(new_parts)

    body_html = re.sub(r">\s+<", "><", body_html)
    excerpt = _remove_period_like(_strip_bad_symbols(f"{title} 레시피 재료와 순서를 깔끔하게 정리했어요"))[:140]
    tag_names = [x.strip("#") for x in hashtags.split() if x.startswith("#")]
    return body_html, excerpt, tag_names


# -----------------------------
# Image helpers
# -----------------------------
def _download_image(url: str, timeout: int = 20) -> Optional[Tuple[bytes, str]]:
    if not url:
        return None
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200 or not r.content:
            return None
        ctype = (r.headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
        if ctype and not ctype.startswith("image/"):
            return None
        b = r.content[:12]
        if not (b.startswith(b"\xFF\xD8\xFF") or b.startswith(b"\x89PNG") or b.startswith(b"RIFF") or b.startswith(b"GIF8")):
            if len(r.content) < 1024:
                return None
        return r.content, (ctype or "image/jpeg")
    except Exception:
        return None


def _picsum_url(seed: str, w: int, h: int) -> str:
    seed = re.sub(r"[^0-9a-zA-Z]", "", seed or "koreanrecipe")[:32] or "koreanrecipe"
    return f"https://picsum.photos/seed/{seed}/{w}/{h}"


def _openai_generate_image_bytes(api_key: str, model: str, prompt: str, size: str) -> Optional[bytes]:
    try:
        from openai import OpenAI
    except Exception:
        return None
    try:
        client = OpenAI(api_key=api_key)
        result = client.images.generate(model=model, prompt=prompt, size=size)
        b64 = result.data[0].b64_json
        return base64.b64decode(b64)
    except Exception:
        return None


def build_image_prompt(title: str) -> str:
    title = _clean_for_title(title)
    return f"Korean food photo of {title} in a clean bowl on a wooden table, warm lighting, realistic, high detail, no text"


def ensure_image_media(cfg: AppConfig, recipe: Recipe, run_seed: str) -> Tuple[int, str]:
    candidates: List[str] = []
    if recipe.image_url:
        candidates.append(recipe.image_url)
    if cfg.img.default_thumb_url:
        candidates.append(cfg.img.default_thumb_url)

    image_bytes: Optional[bytes] = None
    content_type = "image/jpeg"
    chosen_remote_url = ""

    for u in candidates:
        got = _download_image(u, timeout=25)
        if got:
            image_bytes, content_type = got
            chosen_remote_url = u
            break

    if image_bytes is None and cfg.img.use_openai_image and cfg.img.openai_api_key:
        prompt = build_image_prompt(recipe.title)
        b = _openai_generate_image_bytes(cfg.img.openai_api_key, cfg.img.openai_image_model, prompt, cfg.img.openai_image_size)
        if b:
            image_bytes = b
            content_type = "image/png"

    if image_bytes is None and cfg.img.auto_stock_image:
        stock_url = _picsum_url(seed=run_seed, w=cfg.img.stock_width, h=cfg.img.stock_height)
        got = _download_image(stock_url, timeout=25)
        if got:
            image_bytes, content_type = got
            chosen_remote_url = stock_url

    if not image_bytes:
        return 0, ""

    if not cfg.img.upload_media:
        return 0, chosen_remote_url

    ext = ".jpg"
    if "png" in content_type:
        ext = ".png"
    elif "webp" in content_type:
        ext = ".webp"
    fname = f"korean_recipe_{run_seed}{ext}"

    mid, murl = wp_upload_media_bytes(cfg.wp, image_bytes, fname, content_type)
    return mid, murl


# -----------------------------
# Title / Slug
# -----------------------------
def build_post_title(recipe_title: str) -> str:
    t = _clean_for_title(recipe_title)
    bank = [
        f"{t} 오늘은 이렇게 해요",
        f"{t} 집에서 편하게 만드는 방법",
        f"{t} 실패 확률 낮게 정리해봤어요",
        f"{t} 이렇게 해보시면 마음이 편해져요",
    ]
    return random.choice(bank)


def build_slug(date_str: str, slot: str, suffix: str = "") -> str:
    base = f"korean-recipe-{date_str}-{slot}"
    if suffix:
        return f"{base}-{suffix}"
    return base


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot

    suffix = ""
    if cfg.run.allow_duplicate_posts or cfg.run.force_new:
        suffix = "".join(random.choice("0123456789abcdefghijklmnopqrstuvwxyz") for _ in range(4))

    run_seed = hashlib.sha1(f"{date_str}|{slot}|{suffix or 'main'}".encode("utf-8")).hexdigest()[:10]

    date_slot = f"{date_str}_{slot}"
    if cfg.run.allow_duplicate_posts:
        date_slot = f"{date_slot}_{suffix or run_seed}"

    init_db(cfg.sqlite_path)

    today_meta = None
    if not cfg.run.allow_duplicate_posts:
        today_meta = get_today_post(cfg.sqlite_path, date_slot)

    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days)

    chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(cfg, recent_pairs)

    media_id = 0
    media_url = ""
    try:
        media_id, media_url = ensure_image_media(cfg, chosen, run_seed=run_seed)
    except Exception as e:
        if cfg.run.debug:
            print("[IMG] ensure_image_media failed:", repr(e))
        media_id, media_url = 0, ""

    display_img_url = (media_url or cfg.img.default_thumb_url or chosen.image_url or "").strip()
    body_html, excerpt, tag_names = build_homefeed_body(cfg, chosen, image_url=display_img_url)

    wp_tag_ids: List[int] = []
    if cfg.wp.auto_tags and tag_names:
        for nm in tag_names[: cfg.wp.auto_tags_max]:
            nm2 = _strip_bad_symbols(nm)
            nm2 = re.sub(r"[^0-9가-힣a-zA-Z\s]", "", nm2).strip()
            if not nm2:
                continue
            tid = wp_find_or_create_tag(cfg.wp, nm2)
            if tid:
                wp_tag_ids.append(tid)
    wp_tag_ids = list(dict.fromkeys([int(x) for x in wp_tag_ids if int(x) > 0]))

    title = build_post_title(chosen.title)
    slug = build_slug(date_str, slot, suffix if cfg.run.allow_duplicate_posts else "")

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", title)
        print("SLUG:", slug)
        print("CHARS:", _text_len_from_html(body_html))
        print(body_html[:1200])
        return

    wp_post_id = 0
    wp_link = ""

    if today_meta and today_meta.get("wp_post_id") and not cfg.run.force_new:
        post_id = int(today_meta["wp_post_id"])
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, excerpt=excerpt, tag_ids=wp_tag_ids, featured_media=featured_id)
        print("OK(updated):", wp_post_id, wp_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt, tag_ids=wp_tag_ids, featured_media=featured_id)
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
