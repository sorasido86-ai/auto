# -*- coding: utf-8 -*-
"""daily_korean_recipe_to_wp.py

한식 레시피 자동 포스팅 (WordPress)

문제 해결 포인트
- MFDS(식품안전나라) API가 느리거나 타임아웃이어도 1시간씩 안 붙고 빠르게 폴백
- 이미지가 없는 레시피도 사진이 반드시 1장 나오도록 처리
  1) MFDS 메인/스텝 이미지 다운로드 후 WP Media 업로드
  2) DEFAULT_THUMB_URL 다운로드 후 WP Media 업로드
  3) (선택) OpenAI 이미지 생성 -> WP Media 업로드
- 본문은 홈피드형 + 굵은 소제목 3개 + 레시피(재료/순서) 목록 포함
- 마침표 없이 줄바꿈으로 호흡
- 자동생성 고지 / 슬롯 문구 / 지난 레시피 링크 기본 미노출

필수 환경변수
- WP_BASE_URL
- WP_USER
- WP_APP_PASS

선택 환경변수
- WP_STATUS (기본 publish)
- WP_CATEGORY_IDS (예: 7 또는 7,19)
- SQLITE_PATH (기본 data/daily_korean_recipe.sqlite3)
- RUN_SLOT (기본 day)
- FORCE_NEW (1이면 같은 슬롯이어도 새 글)
- MIN_TOTAL_CHARS (기본 1200)

MFDS
- MFDS_API_KEY
- MFDS_TIMEOUT_SEC (기본 10)
- MFDS_BUDGET_SEC (기본 18)
- MFDS_ONLY (1이면 MFDS 실패 시 발행 안 함)

이미지
- DEFAULT_THUMB_URL
- UPLOAD_THUMB (기본 1)
- SET_FEATURED (기본 1)
- EMBED_IMAGE_IN_BODY (기본 1)

OpenAI 이미지 (선택)
- USE_OPENAI_IMAGE (1이면 활성)
- OPENAI_API_KEY
- OPENAI_IMAGE_MODEL (기본 gpt-image-1.5)
- OPENAI_IMAGE_SIZE (기본 1024x1024)
- AUTO_IMAGE (기본 1)  # 이미지가 없을 때 OpenAI로 생성

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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

KST = timezone(timedelta(hours=9))

# -----------------------------
# Local fallback recipes
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
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
        "id": "kimchi-jjigae",
        "title": "돼지고기 김치찌개",
        "ingredients": [
            ("신김치", "2컵"),
            ("돼지고기", "200g"),
            ("양파", "1/2개"),
            ("대파", "1대"),
            ("두부", "1/2모"),
            ("고춧가루", "1큰술"),
            ("다진마늘", "1큰술"),
            ("국간장", "1큰술"),
            ("육수 또는 물", "700ml"),
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
            "고기를 양념에 15분 정도 재워요",
            "팬을 달군 뒤 고기를 볶다가 양파 대파를 넣고 윤기나게 마무리해요",
        ],
        "image_url": "",
    },
]

MFDS_KEYWORDS = [
    "찌개",
    "볶음",
    "국",
    "무침",
    "덮밥",
    "김밥",
    "잡채",
    "전",
    "비빔",
    "나물",
]

# -----------------------------
# Config
# -----------------------------

def _env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v is not None else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: List[int] = []
    for p in re.split(r"[\s,]+", s):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out


@dataclass
class WPConfig:
    base_url: str
    user: str
    app_pass: str
    status: str = "publish"
    category_ids: List[int] = None
    tag_ids: List[int] = None


@dataclass
class MFDSConfig:
    api_key: str = ""
    timeout_sec: int = 10
    budget_sec: int = 18
    only: bool = False


@dataclass
class ImageConfig:
    default_thumb_url: str = ""
    upload_thumb: bool = True
    set_featured: bool = True
    embed_in_body: bool = True

    use_openai_image: bool = False
    openai_api_key: str = ""
    openai_image_model: str = "gpt-image-1.5"
    openai_image_size: str = "1024x1024"
    auto_image: bool = True


@dataclass
class RunConfig:
    sqlite_path: str = "data/daily_korean_recipe.sqlite3"
    run_slot: str = "day"
    force_new: bool = False
    dry_run: bool = False
    debug: bool = False
    min_total_chars: int = 1200


@dataclass
class Config:
    wp: WPConfig
    mfds: MFDSConfig
    img: ImageConfig
    run: RunConfig


def load_config() -> Config:
    wp = WPConfig(
        base_url=_env("WP_BASE_URL", "").rstrip("/"),
        user=_env("WP_USER", ""),
        app_pass=_env("WP_APP_PASS", ""),
        status=_env("WP_STATUS", "publish"),
        category_ids=_parse_int_list(_env("WP_CATEGORY_IDS", "")),
        tag_ids=_parse_int_list(_env("WP_TAG_IDS", "")),
    )

    mfds = MFDSConfig(
        api_key=_env("MFDS_API_KEY", ""),
        timeout_sec=_env_int("MFDS_TIMEOUT_SEC", 10),
        budget_sec=_env_int("MFDS_BUDGET_SEC", 18),
        only=_env_bool("MFDS_ONLY", False),
    )

    img = ImageConfig(
        default_thumb_url=_env("DEFAULT_THUMB_URL", ""),
        upload_thumb=_env_bool("UPLOAD_THUMB", True),
        set_featured=_env_bool("SET_FEATURED", True),
        embed_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
        use_openai_image=_env_bool("USE_OPENAI_IMAGE", False),
        openai_api_key=_env("OPENAI_API_KEY", ""),
        openai_image_model=_env("OPENAI_IMAGE_MODEL", "gpt-image-1.5"),
        openai_image_size=_env("OPENAI_IMAGE_SIZE", "1024x1024"),
        auto_image=_env_bool("AUTO_IMAGE", True),
    )

    run = RunConfig(
        sqlite_path=_env("SQLITE_PATH", "data/daily_korean_recipe.sqlite3"),
        run_slot=_env("RUN_SLOT", "day"),
        force_new=_env_bool("FORCE_NEW", False),
        dry_run=_env_bool("DRY_RUN", False),
        debug=_env_bool("DEBUG", False),
        min_total_chars=_env_int("MIN_TOTAL_CHARS", 1200),
    )

    return Config(wp=wp, mfds=mfds, img=img, run=run)


def _ok(s: str) -> str:
    return "OK" if s else "EMPTY"


def print_cfg(cfg: Config) -> None:
    print("[CFG] WP_BASE_URL:", _ok(cfg.wp.base_url))
    print("[CFG] WP_USER:", _ok(cfg.wp.user), f"(len={len(cfg.wp.user)})")
    print("[CFG] WP_APP_PASS:", _ok(cfg.wp.app_pass), f"(len={len(cfg.wp.app_pass)})")
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids or [])
    print("[CFG] WP_TAG_IDS:", cfg.wp.tag_ids or [])
    print("[CFG] SQLITE_PATH:", cfg.run.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", _ok(cfg.mfds.api_key), "| MFDS_TIMEOUT_SEC:", cfg.mfds.timeout_sec, "| MFDS_BUDGET_SEC:", cfg.mfds.budget_sec)
    print("[CFG] DEFAULT_THUMB_URL:", _ok(cfg.img.default_thumb_url))
    print("[CFG] USE_OPENAI_IMAGE:", cfg.img.use_openai_image, "| AUTO_IMAGE:", cfg.img.auto_image, "| OPENAI_API_KEY:", _ok(cfg.img.openai_api_key))
    print("[CFG] MIN_TOTAL_CHARS:", cfg.run.min_total_chars)


# -----------------------------
# DB
# -----------------------------


def init_db(sqlite_path: str) -> None:
    Path(os.path.dirname(sqlite_path) or ".").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                date_slot TEXT PRIMARY KEY,
                recipe_id TEXT,
                recipe_title TEXT,
                slug TEXT,
                post_id INTEGER,
                media_id INTEGER,
                created_at TEXT
            )
            """
        )
        conn.commit()


def db_get(conn: sqlite3.Connection, date_slot: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT date_slot, recipe_id, recipe_title, slug, post_id, media_id, created_at FROM posts WHERE date_slot=?", (date_slot,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "date_slot": row[0],
        "recipe_id": row[1],
        "recipe_title": row[2],
        "slug": row[3],
        "post_id": row[4],
        "media_id": row[5],
        "created_at": row[6],
    }


def db_recent_recipe_ids(conn: sqlite3.Connection, days: int = 14) -> List[str]:
    cur = conn.execute(
        "SELECT recipe_id FROM posts WHERE created_at >= datetime('now', ?) AND recipe_id IS NOT NULL",
        (f"-{days} day",),
    )
    return [r[0] for r in cur.fetchall() if r and r[0]]


def db_upsert(conn: sqlite3.Connection, data: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO posts(date_slot, recipe_id, recipe_title, slug, post_id, media_id, created_at)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(date_slot) DO UPDATE SET
            recipe_id=excluded.recipe_id,
            recipe_title=excluded.recipe_title,
            slug=excluded.slug,
            post_id=excluded.post_id,
            media_id=excluded.media_id,
            created_at=excluded.created_at
        """,
        (
            data.get("date_slot"),
            data.get("recipe_id"),
            data.get("recipe_title"),
            data.get("slug"),
            data.get("post_id"),
            data.get("media_id"),
            data.get("created_at"),
        ),
    )
    conn.commit()


# -----------------------------
# WP helpers
# -----------------------------


def _wp_auth_header(cfg: WPConfig) -> Dict[str, str]:
    token = base64.b64encode(f"{cfg.user}:{cfg.app_pass}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def wp_request(cfg: WPConfig, method: str, path: str, *, json_body: Optional[dict] = None, headers: Optional[dict] = None, data: Optional[bytes] = None, timeout: int = 25) -> requests.Response:
    url = cfg.base_url.rstrip("/") + path
    h = {}
    h.update(_wp_auth_header(cfg))
    if headers:
        h.update(headers)
    return requests.request(method, url, headers=h, json=json_body, data=data, timeout=timeout)


def wp_upload_media_bytes(cfg: WPConfig, image_bytes: bytes, filename: str, mime: str) -> Tuple[Optional[int], Optional[str]]:
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": mime,
    }
    r = wp_request(cfg, "POST", "/wp-json/wp/v2/media", headers=headers, data=image_bytes, timeout=60)
    if r.status_code not in (200, 201):
        return None, None
    j = r.json()
    return j.get("id"), j.get("source_url")


def wp_get_or_create_tag(cfg: WPConfig, name: str) -> Optional[int]:
    # search
    try:
        r = wp_request(cfg, "GET", f"/wp-json/wp/v2/tags?search={quote(name)}&per_page=100", timeout=25)
        if r.status_code == 200:
            arr = r.json()
            for t in arr:
                if (t.get("name") or "").strip() == name:
                    return t.get("id")
        # create
        r2 = wp_request(cfg, "POST", "/wp-json/wp/v2/tags", json_body={"name": name}, timeout=25)
        if r2.status_code in (200, 201):
            return r2.json().get("id")
    except Exception:
        return None
    return None


def wp_create_or_update_post(
    cfg: WPConfig,
    *,
    title: str,
    slug: str,
    content_html: str,
    status: str,
    category_ids: List[int],
    tag_ids: List[int],
    featured_media_id: Optional[int],
    existing_post_id: Optional[int] = None,
) -> Tuple[Optional[int], Optional[str]]:
    payload: Dict[str, Any] = {
        "title": title,
        "content": content_html,
        "status": status,
        "slug": slug,
    }
    if category_ids:
        payload["categories"] = category_ids
    if tag_ids:
        payload["tags"] = tag_ids
    if featured_media_id:
        payload["featured_media"] = featured_media_id

    if existing_post_id:
        r = wp_request(cfg, "POST", f"/wp-json/wp/v2/posts/{existing_post_id}", json_body=payload, timeout=60)
        if r.status_code not in (200, 201):
            return None, None
        j = r.json()
        return j.get("id"), j.get("link")

    r = wp_request(cfg, "POST", "/wp-json/wp/v2/posts", json_body=payload, timeout=60)
    if r.status_code not in (200, 201):
        return None, None
    j = r.json()
    return j.get("id"), j.get("link")


# -----------------------------
# MFDS
# -----------------------------


def mfds_fetch_by_param(api_key: str, param: str, value: str, *, start: int, end: int, timeout: int) -> List[Dict[str, Any]]:
    value_enc = quote(value)
    url = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}/{param}/{value_enc}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return []
        j = r.json()
        block = j.get("COOKRCP01") or {}
        rows = block.get("row") or []
        if isinstance(rows, list):
            return rows
        return []
    except Exception:
        return []


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_bullets(s: str) -> str:
    # 텍스트 앞의 특수 기호 제거
    s = (s or "").strip()
    s = re.sub(r"^[\-\*\u2022\u00B7\u25AA\u25AB\u25CF\u25CB\u2713\u2705\u2611\u2733\u2734\u2757\u27A1\u2192\u25B6\u25C0\u2665\u2661\u25AA\u25A0]+\s*", "", s)
    return s


def parse_mfds_row(row: Dict[str, Any]) -> Dict[str, Any]:
    rid = str(row.get("RCP_SEQ") or "").strip() or hashlib.md5((row.get("RCP_NM") or "").encode("utf-8", "ignore")).hexdigest()[:10]
    title = _clean_text(str(row.get("RCP_NM") or ""))

    # ingredients
    parts_raw = str(row.get("RCP_PARTS_DTLS") or "")
    parts_lines = [p.strip() for p in re.split(r"[\r\n]+", parts_raw) if p and p.strip()]
    if not parts_lines:
        parts_lines = [p.strip() for p in re.split(r"[,\u3001]", parts_raw) if p and p.strip()]
    ingredients: List[str] = []
    for p in parts_lines:
        p = _strip_bullets(_clean_text(p))
        if p:
            ingredients.append(p)

    # steps + step images
    steps: List[str] = []
    step_imgs: List[str] = []
    for i in range(1, 21):
        key = f"MANUAL{i:02d}"
        txt = _clean_text(str(row.get(key) or ""))
        txt = _strip_bullets(txt)
        if txt:
            steps.append(txt)
        ikey = f"MANUAL_IMG{i:02d}"
        img = _clean_text(str(row.get(ikey) or ""))
        if img:
            step_imgs.append(img)

    main_img = _clean_text(str(row.get("ATT_FILE_NO_MAIN") or ""))
    if not main_img and step_imgs:
        main_img = step_imgs[0]

    # http -> https 시도(혼합콘텐츠 방지)
    if main_img.startswith("http://"):
        main_img = "https://" + main_img[len("http://") :]

    step_imgs2: List[str] = []
    for u in step_imgs[:4]:
        if u.startswith("http://"):
            u = "https://" + u[len("http://") :]
        step_imgs2.append(u)

    return {
        "id": rid,
        "title": title,
        "ingredients": ingredients,
        "steps": steps,
        "image_url": main_img,
        "step_images": step_imgs2,
    }


def pick_recipe_mfds(cfg: Config, recent_ids: List[str]) -> Optional[Dict[str, Any]]:
    if not cfg.mfds.api_key:
        return None

    deadline = time.monotonic() + max(5, cfg.mfds.budget_sec)
    keywords = MFDS_KEYWORDS[:]
    random.shuffle(keywords)

    for kw in keywords:
        if time.monotonic() > deadline:
            break
        rows = mfds_fetch_by_param(cfg.mfds.api_key, "RCP_NM", kw, start=1, end=60, timeout=cfg.mfds.timeout_sec)
        if not rows:
            continue
        random.shuffle(rows)
        for row in rows[:20]:
            rec = parse_mfds_row(row)
            if not rec.get("title"):
                continue
            if rec.get("id") in recent_ids:
                continue
            # 최소한의 구조
            if len(rec.get("steps") or []) < 2:
                continue
            return rec

    return None


def pick_recipe_local(recent_ids: List[str]) -> Dict[str, Any]:
    pool = [r for r in LOCAL_KOREAN_RECIPES if r.get("id") not in recent_ids]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    r = random.choice(pool)
    return {
        "id": r["id"],
        "title": r["title"],
        "ingredients": [f"{a} {b}".strip() for a, b in (r.get("ingredients") or [])],
        "steps": [s for s in (r.get("steps") or [])],
        "image_url": r.get("image_url") or "",
        "step_images": [],
    }


# -----------------------------
# Image
# -----------------------------


def _guess_mime_from_url(url: str) -> Tuple[str, str]:
    u = url.lower()
    if u.endswith(".png"):
        return "image/png", "png"
    if u.endswith(".webp"):
        return "image/webp", "webp"
    return "image/jpeg", "jpg"


def download_bytes(url: str, *, timeout: int = 20) -> Optional[bytes]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.content
    except Exception:
        return None


def openai_generate_image_bytes(cfg: Config, prompt: str) -> Optional[bytes]:
    if not (cfg.img.use_openai_image and cfg.img.openai_api_key):
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=cfg.img.openai_api_key)
        result = client.images.generate(
            model=cfg.img.openai_image_model,
            prompt=prompt,
            n=1,
            size=cfg.img.openai_image_size,
        )
        b64 = result.data[0].b64_json
        return base64.b64decode(b64)
    except Exception:
        return None


def make_image_prompt(recipe_title: str) -> str:
    # 텍스트 렌더링(글자) 지시를 피해서 실패율을 낮춤
    return (
        "Korean home cooking food photography, warm natural light, shallow depth of field, "
        "a bowl of {t}, steam rising, realistic, high detail, clean background, "
        "no text, no watermark"
    ).format(t=recipe_title)


def ensure_post_image(cfg: Config, recipe: Dict[str, Any], slug: str) -> Tuple[Optional[int], Optional[str]]:
    # returns (media_id, media_url)
    if not cfg.img.upload_thumb:
        return None, None

    # 1) MFDS image
    candidates: List[str] = []
    if recipe.get("image_url"):
        candidates.append(str(recipe["image_url"]))
    for u in (recipe.get("step_images") or [])[:2]:
        if u:
            candidates.append(str(u))
    if cfg.img.default_thumb_url:
        candidates.append(cfg.img.default_thumb_url)

    for url in candidates:
        url = url.strip()
        if not url:
            continue
        b = download_bytes(url, timeout=20)
        if not b:
            continue
        mime, ext = _guess_mime_from_url(url)
        mid, murl = wp_upload_media_bytes(cfg.wp, b, f"{slug}.{ext}", mime)
        if mid and murl:
            return mid, murl

    # 2) OpenAI image
    if cfg.img.use_openai_image and cfg.img.auto_image:
        prompt = make_image_prompt(str(recipe.get("title") or "Korean food"))
        b = openai_generate_image_bytes(cfg, prompt)
        if b:
            mid, murl = wp_upload_media_bytes(cfg.wp, b, f"{slug}.png", "image/png")
            if mid and murl:
                return mid, murl

    return None, None


# -----------------------------
# Content
# -----------------------------


def _now_kst() -> datetime:
    return datetime.now(tz=KST)


def build_slug(date_slot: str) -> str:
    # 예: korean-recipe-2026-01-17-day
    return f"korean-recipe-{date_slot}"


def no_period(s: str) -> str:
    # 마침표/느낌표/물음표를 제거
    return s.replace(".", " ").replace("!", " ").replace("?", " ")


def html_p(lines: List[str]) -> str:
    safe = [html.escape(no_period(x)).strip() for x in lines if x and x.strip()]
    safe = [re.sub(r"\s+", " ", x) for x in safe]
    return "<p>" + "<br/>".join(safe) + "</p>"


def bold_title(s: str) -> str:
    return f"<p><b>{html.escape(no_period(s))}</b></p>"


def make_intro(recipe_title: str) -> str:
    # 200~300자 정도
    chunks = [
        f"요즘 집에서 밥 챙겨 드시기 더 힘드실 때 있지요",
        f"오늘은 {recipe_title}로 마음을 조금 덜 바쁘게 만들어 보려고 해요",
        "재료는 익숙한데 맛은 확 좋아지는 포인트가 있어서요",
        "처음부터 거창하게 하실 필요 없고요",
        "냄비 하나로 천천히 끓이듯이 하루도 그렇게 풀어가면 괜찮더라고요",
    ]
    txt = " ".join(chunks)
    # 길이 맞추기
    if len(txt) < 200:
        txt += "  " + "따뜻한 한 그릇이 생각보다 큰 위로가 되니까요"
    return txt[:320]


def make_sections(recipe_title: str) -> List[Tuple[str, List[str]]]:
    s1_title = "오늘 이 레시피가 딱인 이유"
    s1 = [
        f"{recipe_title}는 조리 과정이 복잡하지 않아서 시작이 편해요",
        "불 앞에서 오래 서 있어야 하는 요리만 떠올리면 부담이 확 올라가는데요",
        "이건 순서를 크게 세 덩어리로만 생각하시면 됩니다",
        "재료 손질  간 맞추기  마지막에 향 올리기",
        "이렇게만 잡아두면 중간에 마음이 덜 흔들려요",
        "그리고 한 번 끓여두면 다음 끼니에 더 맛있어지는 타입이라서요",
        "오늘 한 번만 해두면 내일은 덜 애쓰게 되지요",
    ]

    s2_title = "재료 준비와 맛이 살아나는 포인트"
    s2 = [
        "재료는 다 꺼내놓고 시작하시면 속도가 진짜 빨라져요",
        "대파  양파  두부 같은 것들은 손질만 해두면 절반은 끝난 느낌이 들고요",
        "양념은 한 번에 넣기보다 두 번에 나눠서 조절해 보세요",
        "처음에는 기본 간을 잡고",
        "마지막에는 향이 살아나는 재료를 넣어서 마무리해요",
        "이때 간을 세게 하는 것보다 향을 올리는 쪽으로 가면 실패가 훨씬 줄어요",
        "혹시 싱거우면 국간장이나 소금으로 아주 조금씩만",
        "반대로 짜면 물을 더하기보다 두부나 채소를 조금 더 넣는 방식이 편하더라고요",
    ]

    s3_title = "순서대로 만들기  실패 줄이는 작은 팁"
    s3 = [
        "순서는 아주 단순하게만 적어둘게요",
        "끓이기 전에 팬이나 냄비를 미리 데워두면 재료에서 물이 덜 생겨요",
        "중불에서 시작해서 끓기 시작하면 약불로 내려서 시간을 주면 맛이 더 깊어지고요",
        "중간에 뚜껑을 덮었다가 살짝 열어두면 향이 답답하지 않게 올라와요",
        "마지막 1분이 중요하니까요",
        "대파나 참기름 같은 향 재료는 진짜 끝에서 넣으시면 좋아요",
        "그리고 한 숟갈 떠서 식혀서 맛을 보면 간이 더 정확하게 느껴져요",
        "뜨거울 때는 짠맛이 덜 느껴져서 과하게 넣기 쉽거든요",
    ]

    return [
        (s1_title, s1),
        (s2_title, s2),
        (s3_title, s3),
    ]


def make_recipe_block(recipe: Dict[str, Any]) -> str:
    title = str(recipe.get("title") or "")
    ingredients = [
        _strip_bullets(_clean_text(x)) for x in (recipe.get("ingredients") or []) if x and str(x).strip()
    ]
    steps = [_strip_bullets(_clean_text(x)) for x in (recipe.get("steps") or []) if x and str(x).strip()]

    html_parts: List[str] = []
    html_parts.append(bold_title("레시피 목록"))
    html_parts.append(html_p([f"오늘 메뉴  {title}"]))

    if ingredients:
        html_parts.append(bold_title("재료"))
        # 불릿 대신 줄바꿈
        ing_lines = [f"{x}" for x in ingredients[:35]]
        html_parts.append(html_p(ing_lines))

    if steps:
        html_parts.append(bold_title("만드는 법"))
        step_lines = [f"{i+1}  {steps[i]}" for i in range(min(len(steps), 20))]
        html_parts.append(html_p(step_lines))

    return "\n".join(html_parts)


def make_hashtags(recipe_title: str, n: int = 12) -> str:
    base = [
        "한식",
        "집밥",
        "오늘의요리",
        "레시피",
        "간단요리",
        "저녁메뉴",
        "혼밥",
        "따뜻한한끼",
        "주방루틴",
    ]
    # 제목에서 추출
    words = re.findall(r"[가-힣]{2,}", recipe_title)
    for w in words[:5]:
        base.append(w)
    # 중복 제거
    seen = set()
    tags: List[str] = []
    for t in base:
        t = re.sub(r"\s+", "", t)
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        tags.append(t)
    random.shuffle(tags)
    tags = tags[: max(6, min(n, len(tags)))]
    return " ".join([f"#{t}" for t in tags])


def build_post_html(cfg: Config, recipe: Dict[str, Any], *, image_url: Optional[str]) -> str:
    title = str(recipe.get("title") or "오늘의 집밥")

    parts: List[str] = []

    if cfg.img.embed_in_body and image_url:
        parts.append(
            f"<p><img src=\"{html.escape(image_url)}\" alt=\"{html.escape(title)}\" style=\"max-width:100%;height:auto;\" loading=\"lazy\"/></p>"
        )

    # intro
    intro = make_intro(title)
    parts.append(html_p([intro]))

    # sections
    for sec_title, sec_lines in make_sections(title):
        parts.append(bold_title(sec_title))
        parts.append(html_p(sec_lines))

    # recipe list block
    parts.append(make_recipe_block(recipe))

    # hashtags
    parts.append(html_p([make_hashtags(title, n=12)]))

    body = "\n".join(parts)

    # 최소 글자수 확보(HTML 태그 제외 대략값)
    plain_len = len(re.sub(r"<[^>]+>", "", body))
    if plain_len < cfg.run.min_total_chars:
        # 부족하면 마지막에 부드러운 수다 문단을 추가
        extra_lines = [
            "혹시 오늘 컨디션이 애매하시면요",
            "레시피를 완벽하게 따라야 한다는 마음을 먼저 내려놓으셔도 괜찮아요",
            "한 숟갈씩 맛 보면서 조절하는 게 집밥의 매력이라서요",
            "저는 이런 요리가 사람을 조금 덜 외롭게 만든다고 생각하거든요",
            "따뜻하게 드시고 물 한 잔까지 챙기시면 오늘은 그걸로 충분해요",
        ]
        parts.insert(-2, html_p(extra_lines))
        body = "\n".join(parts)

    return body


def build_title(recipe_title: str) -> str:
    # 제목도 과한 특수문자 없이
    hooks = [
        "오늘 저녁은 이거로 가요",
        "재료 적고 맛은 확 좋아져요",
        "한 그릇으로 마음이 풀려요",
        "실패 확률 낮추는 순서로 정리했어요",
    ]
    h = random.choice(hooks)
    return no_period(f"{recipe_title} {h}").strip()


def build_tag_names(recipe_title: str) -> List[str]:
    tags = []
    for w in re.findall(r"[가-힣]{2,}", recipe_title):
        if w not in tags:
            tags.append(w)
    tags += ["한식", "집밥", "레시피", "오늘의요리", "저녁메뉴", "간단요리"]
    # 정리
    out: List[str] = []
    for t in tags:
        t = re.sub(r"\s+", "", t)
        if not t:
            continue
        if t in out:
            continue
        out.append(t)
    return out[:10]


# -----------------------------
# Run
# -----------------------------


def run(cfg: Config) -> None:
    if not (cfg.wp.base_url and cfg.wp.user and cfg.wp.app_pass):
        raise RuntimeError("WP_BASE_URL / WP_USER / WP_APP_PASS 는 필수입니다")

    init_db(cfg.run.sqlite_path)

    now = _now_kst()
    # 기존 퍼머링크 스타일과 맞추기 위해 구분자는 하이픈 사용
    # 예: 2026-01-17-day
    date_slot = f"{now:%Y-%m-%d}-{cfg.run.run_slot}"
    slug = build_slug(date_slot)

    print("[RUN] date_slot=", date_slot)

    with sqlite3.connect(cfg.run.sqlite_path) as conn:
        existing = None if cfg.run.force_new else db_get(conn, date_slot)
        recent_ids = db_recent_recipe_ids(conn, days=14)

        recipe = pick_recipe_mfds(cfg, recent_ids)
        if not recipe:
            if cfg.mfds.only:
                print("[MFDS] MFDS_ONLY=1 이라서 MFDS 실패 시 발행을 건너뜁니다")
                return
            recipe = pick_recipe_local(recent_ids)

        recipe_title = str(recipe.get("title") or "오늘의 집밥")
        title = build_title(recipe_title)

        # tags
        tag_ids: List[int] = []
        if cfg.wp.tag_ids:
            tag_ids.extend(cfg.wp.tag_ids)
        # auto create tags (best-effort)
        for name in build_tag_names(recipe_title):
            tid = wp_get_or_create_tag(cfg.wp, name)
            if tid and tid not in tag_ids:
                tag_ids.append(tid)

        # image
        media_id = None
        media_url = None
        if cfg.img.upload_thumb:
            media_id, media_url = ensure_post_image(cfg, recipe, slug)

        # content
        content_html = build_post_html(cfg, recipe, image_url=media_url)

        # wp create/update
        if cfg.run.dry_run:
            print("[DRY_RUN] title:", title)
            print("[DRY_RUN] slug:", slug)
            print("[DRY_RUN] featured_media_id:", media_id)
            print("[DRY_RUN] tags:", tag_ids)
            print("[DRY_RUN] content chars:", len(re.sub(r"<[^>]+>", "", content_html)))
            return

        post_id, link = wp_create_or_update_post(
            cfg.wp,
            title=title,
            slug=slug,
            content_html=content_html,
            status=cfg.wp.status,
            category_ids=cfg.wp.category_ids or [],
            tag_ids=tag_ids,
            featured_media_id=media_id if cfg.img.set_featured else None,
            existing_post_id=existing.get("post_id") if existing else None,
        )

        if not post_id:
            raise RuntimeError("WordPress 포스팅 실패")

        print("[OK] post_id=", post_id)
        if link:
            print("[OK] link=", link)

        db_upsert(
            conn,
            {
                "date_slot": date_slot,
                "recipe_id": recipe.get("id"),
                "recipe_title": recipe_title,
                "slug": slug,
                "post_id": post_id,
                "media_id": media_id,
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )


def main() -> None:
    cfg = load_config()
    print_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR]", str(e))
        if _env_bool("DEBUG", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)
