# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp.py (완전 통합)
- "한식 레시피만" 매일 자동 업로드 (WordPress)
- 1순위: 식품안전나라(식약처) COOKRCP01 OpenAPI 레시피 DB (이미지/과정 포함)
  - 요청주소 형식: http://openapi.foodsafetykorea.go.kr/api/인증키/COOKRCP01/json/1/5/... :contentReference[oaicite:1]{index=1}
  - 데이터 설명/속성 및 이용허락(출처표시, 변형 가능 등) :contentReference[oaicite:2]{index=2}
- 2순위(폴백): 코드 내장 "한식 레시피 묶음" (한국어)
- RUN_SLOT: day / am / pm 지원 (기본 day)
- 슬롯별로 "하루 1개(or 2개)" 운영 가능: slug/date_slot 고정 → 같은 슬롯은 업데이트
- 대표이미지(썸네일) 자동 업로드 + featured 설정 지원

필수 환경변수(Secrets):
  - WP_BASE_URL   예) https://example.com
  - WP_USER       (워드프레스 계정)
  - WP_APP_PASS   (Application Password)

권장 환경변수:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS=7 (기본 7)
  - WP_TAG_IDS=1,2,3 (선택)
  - SQLITE_PATH=data/daily_korean_recipe.sqlite3

식품안전나라 OpenAPI 사용(진짜 레시피 DB):
  - MFDS_API_KEY=발급받은키   (없으면 내장 레시피로만 운영)

OpenAI로 블로거톤 강화(선택):
  - USE_OPENAI=1
  - OPENAI_API_KEY=...
  - OPENAI_MODEL=... (기본 gpt-4.1-mini 로 설정)

동작 옵션:
  - RUN_SLOT=day|am|pm (기본 day)
  - FORCE_NEW=0|1 (기본 0)  : 오늘 이미 올렸어도 새 레시피로 교체 발행(업데이트)
  - DRY_RUN=0|1 (기본 0)    : 워드프레스 발행 안하고 미리보기 출력
  - DEBUG=0|1 (기본 0)
  - AVOID_REPEAT_DAYS=90 (기본 90) : 최근 N일 내 레시피 재사용 회피
  - MAX_TRIES=25 (기본 25)  : 조건(한식/중복회피) 맞는 레시피 찾는 시도 횟수

이미지 옵션:
  - UPLOAD_THUMB=1 (기본 1)      : 이미지 업로드 시도
  - SET_FEATURED=1 (기본 1)      : featured_media 설정
  - EMBED_IMAGE_IN_BODY=1 (기본 1): 본문 상단에 대표이미지 삽입
  - DEFAULT_THUMB_URL=... (선택) : 레시피 이미지가 없을 때 대신 사용할 이미지 URL(직접 준비)
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

KST = timezone(timedelta(hours=9))

# -----------------------------
# 내장 한식 레시피(폴백용, 한국어)
# - "진짜 DB"는 MFDS_API_KEY 설정 시 OpenAPI에서 가져옴
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "kimchi-jjigae",
        "title": "돼지고기 김치찌개",
        "summary": "집밥의 정석! 김치와 돼지고기만 있으면 실패 확률 0%.",
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
            ("설탕(선택)", "1/2작은술"),
        ],
        "steps": [
            "냄비에 돼지고기를 넣고 중불에서 기름이 살짝 돌 때까지 볶아주세요.",
            "신김치를 넣고 2~3분 더 볶아 김치의 신맛을 한 번 눌러줍니다.",
            "고춧가루/다진마늘/국간장을 넣고 30초만 볶아 향을 내요.",
            "육수를 붓고 10~12분 끓입니다.",
            "양파를 넣고 3분, 두부를 넣고 2분 더 끓인 뒤 대파로 마무리!",
        ],
        "image_url": "",
        "tags": ["한식", "찌개", "집밥", "김치"],
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "summary": "냉장고 털이도 가능한 만능 찌개. 된장만 좋으면 반은 성공.",
        "ingredients": [
            ("된장", "1.5큰술"),
            ("고추장(선택)", "1/2큰술"),
            ("애호박", "1/3개"),
            ("양파", "1/3개"),
            ("두부", "1/2모"),
            ("감자(선택)", "1/2개"),
            ("대파", "1/2대"),
            ("다진마늘", "1작은술"),
            ("멸치다시마 육수(또는 물)", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고(체에 걸러주면 더 깔끔) 5분 끓여요.",
            "감자/양파를 먼저 넣고 6~7분 익힙니다.",
            "애호박/두부/다진마늘 넣고 3~4분 더 끓입니다.",
            "대파 넣고 한 번만 더 끓인 뒤 간을 보고 마무리!",
        ],
        "image_url": "",
        "tags": ["한식", "찌개", "된장", "집밥"],
    },
    {
        "id": "bulgogi",
        "title": "간장 불고기",
        "summary": "달짝지근한 간장 양념으로 밥도둑 확정.",
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
            "간장/설탕/다진마늘/참기름/물/후추로 양념장을 먼저 섞어요.",
            "고기에 양념장을 넣고 15분 이상 재워둡니다.",
            "팬을 달군 뒤 고기를 먼저 볶고, 양파/대파를 넣어 숨이 죽을 때까지 볶아요.",
            "불을 끄기 직전에 간을 보고 부족하면 간장 1작은술로 조정!",
        ],
        "image_url": "",
        "tags": ["한식", "볶음", "불고기", "메인"],
    },
    {
        "id": "bibimbap",
        "title": "비빔밥 (초간단 집비빔)",
        "summary": "나물 2~3개만 있어도 충분히 맛있게!",
        "ingredients": [
            ("밥", "1공기"),
            ("계란", "1개"),
            ("고추장", "1큰술"),
            ("참기름", "1큰술"),
            ("김가루", "한 줌"),
            ("나물/반찬(시금치/콩나물 등)", "2~3가지"),
        ],
        "steps": [
            "그릇에 밥을 담고 나물을 보기 좋게 올립니다.",
            "계란후라이를 반숙으로 올려요.",
            "고추장/참기름/김가루 넣고, 먹기 직전에 쓱쓱 비벼 마무리!",
        ],
        "image_url": "",
        "tags": ["한식", "밥", "비빔밥", "초간단"],
    },
    {
        "id": "tteokbokki",
        "title": "국물 떡볶이",
        "summary": "달달·매콤 밸런스만 맞추면 밖에서 사먹는 맛 나와요.",
        "ingredients": [
            ("떡볶이떡", "300g"),
            ("어묵", "2장"),
            ("대파", "1대"),
            ("고추장", "2큰술"),
            ("고춧가루", "1큰술"),
            ("설탕", "1큰술"),
            ("간장", "1큰술"),
            ("물", "500ml"),
        ],
        "steps": [
            "냄비에 물/고추장/고춧가루/설탕/간장을 넣고 먼저 풀어 끓입니다.",
            "떡과 어묵을 넣고 7~10분, 중불에서 끓여요.",
            "대파 넣고 1분 더 끓인 뒤 농도 맞추고 마무리!",
        ],
        "image_url": "",
        "tags": ["한식", "분식", "떡볶이", "간식"],
    },
]

# "한식만" 필터(기본값)
KOREAN_POSITIVE_KEYWORDS = [
    "김치", "된장", "고추장", "비빔", "찌개", "국", "탕", "전", "조림", "볶음",
    "나물", "무침", "김밥", "떡", "갈비", "불고기", "제육", "순두부", "냉면", "잡채", "밥", "죽"
]
KOREAN_NEGATIVE_KEYWORDS = [
    "파스타", "피자", "타코", "부리또", "스시", "리조또", "스테이크", "커리", "샌드위치", "버거", "샐러드"
]

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
        ),
        openai=OpenAIConfig(
            use_openai=_env_bool("USE_OPENAI", False),
            api_key=_env("OPENAI_API_KEY", ""),
            model=_env("OPENAI_MODEL", "gpt-4.1-mini"),
        ),
        sqlite_path=_env("SQLITE_PATH", "data/daily_korean_recipe.sqlite3"),
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
    print("[CFG] STRICT_KOREAN:", cfg.recipe.strict_korean, "| MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key))
    print("[CFG] UPLOAD_THUMB:", cfg.img.upload_thumb, "| SET_FEATURED:", cfg.img.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.img.embed_image_in_body)
    print("[CFG] DEFAULT_THUMB_URL:", "SET" if cfg.img.default_thumb_url else "EMPTY")
    print("[CFG] USE_OPENAI:", cfg.openai.use_openai, "| OPENAI_API_KEY:", ok(cfg.openai.api_key), "| OPENAI_MODEL:", cfg.openai.model)


# -----------------------------
# SQLite (history + post meta)
# - "media_id 컬럼 없음" 같은 구버전 DB도 자동으로 ALTER TABLE 처리
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

    # migrate missing columns (구버전 DB 대응)
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
    """최근 N일 내 (recipe_source, recipe_id) 목록"""
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


def wp_upload_media_from_url(cfg: WordPressConfig, image_url: str, filename: str) -> Tuple[int, str]:
    # download
    r = requests.get(image_url, timeout=35)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code}")

    content = r.content
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if not ctype:
        # fallback
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
# Recipe model / utils
# -----------------------------
@dataclass
class Recipe:
    source: str          # "mfds" or "local"
    recipe_id: str
    title: str
    ingredients: List[str]   # already formatted strings
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
    # negative keyword filter
    for bad in KOREAN_NEGATIVE_KEYWORDS:
        if bad in n:
            return False
    # positive heuristic: if strict, require at least one positive keyword OR hangul+length
    if strict:
        if any(k in n for k in KOREAN_POSITIVE_KEYWORDS):
            return True
        # 그래도 한글+짧지 않으면 통과(예: "미역국", "호박죽" 등)
        return _has_hangul(n) and len(n) >= 2
    return True


# -----------------------------
# MFDS(OpenAPI) provider: COOKRCP01
# - 공식 요청 URL 형식 기반 :contentReference[oaicite:3]{index=3}
# -----------------------------
def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int = 1, end: int = 50) -> List[Dict[str, Any]]:
    # 추가요청인자 형태: .../startIdx/endIdx/변수명=값
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
    if not isinstance(rows, list):
        return []
    return rows


def mfds_fetch_random_batch(api_key: str, start: int = 1, end: int = 50) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    r = requests.get(base, timeout=35)
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    co = data.get("COOKRCP01") or {}
    rows = co.get("row") or []
    if not isinstance(rows, list):
        return []
    return rows


def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or str(row.get("RCP_SEQ", ""))
    title = str(row.get("RCP_NM") or "").strip()
    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()

    # ingredients: comma-separated string → list
    ingredients = []
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
            # 원문에 a,b,c 같은 꼬리문자가 붙는 경우가 있어 정리
            s = re.sub(r"[a-zA-Z]\s*$", "", s).strip()
            steps.append(s)
            step_imgs.append(img if img.startswith("http") else "")

    img_main = str(row.get("ATT_FILE_NO_MAIN") or "").strip()
    if not img_main:
        # 일부는 다른 필드에 있을 수 있어 안전하게
        img_main = str(row.get("ATT_FILE_NO_MK") or "").strip()

    return Recipe(
        source="mfds",
        recipe_id=rid or hashlib.sha1(title.encode("utf-8")).hexdigest()[:8],
        title=title,
        ingredients=ingredients,
        steps=steps,
        image_url=img_main if img_main.startswith("http") else "",
        step_images=[x for x in step_imgs if x.startswith("http")],
    )


def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    api_key = cfg.recipe.mfds_api_key
    if not api_key:
        return None

    used = set(recent_pairs)
    # 검색 키워드를 섞어서 "한식 느낌" 확률을 올림
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡"]
    random.shuffle(keywords)

    for _ in range(cfg.run.max_tries):
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(api_key, "RCP_NM", kw, start=1, end=60)
        if not rows:
            rows = mfds_fetch_random_batch(api_key, start=1, end=60)
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
    pool = []
    for x in LOCAL_KOREAN_RECIPES:
        rid = str(x["id"])
        if (("local", rid) in used):
            continue
        pool.append(x)
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]  # 어쩔 수 없으면 재사용

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
        # MFDS는 RCP_SEQ로 조회 가능하다고 알려진 케이스가 많아 시도
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_SEQ", recipe_id, start=1, end=5)
        for row in rows:
            rcp = mfds_row_to_recipe(row)
            if rcp.recipe_id == recipe_id:
                return rcp
        return None

    return None


# -----------------------------
# Blog rendering (Korean blogger tone)
# -----------------------------
def _esc(s: str) -> str:
    return html.escape(s or "")


def build_body_html(cfg: AppConfig, now: datetime, run_slot_label: str, recipe: Recipe, featured_img_url: str = "") -> str:
    title = recipe.title.strip()

    # 대표이미지(본문삽입)
    img_html = ""
    if cfg.img.embed_image_in_body and featured_img_url:
        img_html = f"""
        <p style="margin:14px 0;">
          <img src="{_esc(featured_img_url)}" alt="{_esc(title)}" style="max-width:100%;height:auto;border-radius:10px;" />
        </p>
        """

    disclosure = f'<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{_esc(DISCLOSURE)}</p>'
    head = f"<p>기준시각: <b>{_esc(now.astimezone(KST).strftime('%Y-%m-%d %H:%M'))}</b> / 슬롯: <b>{_esc(run_slot_label)}</b></p>"
    note = f'<p style="font-size:13px;opacity:.85;">{_esc(SEO_NOTE)}<br/>{_esc(SOURCE_NOTE)}</p>'

    # 훅(블로그톤)
    hook = f"""
    <h2>{_esc(title)} 레시피</h2>
    <p>
      오늘은 <b>{_esc(title)}</b>로 갑니다. <br/>
      재료는 단순하게, 과정은 실패 확률 낮게 정리했어요. (바쁜 날에도 OK!)
    </p>
    """

    # 재료
    ing_li = "".join([f"<li>{_esc(x)}</li>" for x in recipe.ingredients]) or "<li>재료 정보가 비어있어요.</li>"
    ingredients = f"""
    <h3>재료 준비</h3>
    <ul>{ing_li}</ul>
    """

    # 과정
    step_ol = ""
    for s in recipe.steps:
        step_ol += f"<li style='margin:6px 0;'>{_esc(s)}</li>"
    if not step_ol:
        step_ol = "<li>조리 과정 정보가 비어있어요.</li>"

    steps = f"""
    <h3>만드는 법</h3>
    <ol>{step_ol}</ol>
    """

    # 팁
    tips = """
    <h3>실패 줄이는 팁</h3>
    <ul>
      <li>간은 한 번에 세게 하지 말고, 마지막에 한 번 더 잡아주세요.</li>
      <li>시간이 없으면 재료를 “크게” 썰어도 괜찮아요. 대신 충분히 끓이기!</li>
      <li>매운맛은 고춧가루/고추장으로 조절하면 깔끔합니다.</li>
    </ul>
    """

    closing = """
    <hr/>
    <p style="opacity:.85;">
      도움이 됐다면 즐겨찾기 해두고, 내일 레시피도 받아가세요 🙂<br/>
      (매일 1개씩 업데이트됩니다.)
    </p>
    """

    return disclosure + head + img_html + note + hook + ingredients + steps + tips + closing


def generate_with_openai(cfg: AppConfig, recipe: Recipe, base_html: str) -> Optional[str]:
    """
    OpenAI 사용 시: '블로거톤' 강화 + 부연설명 자연스럽게.
    - 실패(쿼터/모듈없음/에러)하면 None 반환 → 템플릿 그대로 업로드
    """
    if not (cfg.openai.use_openai and cfg.openai.api_key):
        return None

    try:
        from openai import OpenAI  # 지연 import (미설치 시에도 전체 실패 방지)
    except Exception:
        return None

    try:
        client = OpenAI(api_key=cfg.openai.api_key)
        prompt = f"""
너는 한국 요리 블로그 전문 에디터야.
아래 레시피(제목/재료/과정)는 "내용을 바꾸지 말고" 그대로 유지해.
대신 도입부/설명/팁/마무리를 더 자연스럽고 조회수 잘 나오는 블로거 말투로 다듬어줘.
HTML 형태로만 출력해. (코드블럭 금지)
너무 과장된 광고 문구는 금지. 담백하지만 먹고 싶게.

[레시피 제목]
{recipe.title}

[재료]
- """ + "\n- ".join(recipe.ingredients) + """

[과정]
1) """ + "\n".join([f"{i+1}) {s}" for i, s in enumerate(recipe.steps)]) + """

[현재 HTML 초안]
{base_html}
"""
        resp = client.responses.create(
            model=cfg.openai.model,
            input=prompt,
        )
        # responses API의 안전한 텍스트 추출
        out_text = getattr(resp, "output_text", None)
        if not out_text:
            # 일부 SDK 버전 호환
            try:
                out_text = resp.output[0].content[0].text  # type: ignore
            except Exception:
                out_text = None
        if out_text and "<" in out_text:
            return out_text.strip()
    except Exception:
        return None

    return None


# -----------------------------
# Main run
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

    # 이미 오늘 올린 글이 있고, FORCE_NEW=0이면 "같은 레시피로 업데이트" 시도
    chosen: Optional[Recipe] = None
    if today_meta and not cfg.run.force_new and today_meta.get("recipe_source") and today_meta.get("recipe_id"):
        chosen = get_recipe_by_id(cfg, today_meta["recipe_source"], today_meta["recipe_id"])

    # 없다면 새 레시피 선택
    if not chosen:
        # 1) MFDS(OpenAPI) → 2) Local fallback
        chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(cfg, recent_pairs)

    assert chosen is not None
    title = f"{date_str} 한식 레시피 - {chosen.title} ({slot_label})"
    slug = f"korean-recipe-{date_str}-{slot}"

    # 대표 이미지 URL 결정 (레시피 이미지 > default)
    chosen_img_url = chosen.image_url or cfg.img.default_thumb_url

    # 미리 HTML 생성
    body_html = build_body_html(cfg, now, slot_label, chosen, featured_img_url=chosen_img_url if cfg.img.embed_image_in_body else "")

    # OpenAI로 톤 강화(실패하면 그대로)
    upgraded = generate_with_openai(cfg, chosen, body_html)
    if upgraded:
        body_html = upgraded

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 HTML 일부 ↓")
        print(body_html[:2000])
        print("... (truncated)")
        return

    # 이미지 업로드(선택)
    media_id = 0
    media_url = ""
    if cfg.img.upload_thumb and chosen_img_url:
        try:
            ext = ".jpg"
            if chosen_img_url.lower().endswith(".png"):
                ext = ".png"
            filename = f"korean_recipe_{chosen.uid()}{ext}"
            media_id, media_url = wp_upload_media_from_url(cfg.wp, chosen_img_url, filename)
            if cfg.run.debug:
                print("[IMG] uploaded:", media_id, media_url)
        except Exception as e:
            if cfg.run.debug:
                print("[IMG] upload failed:", repr(e))
            media_id, media_url = 0, ""

    featured_id = media_id if (cfg.img.set_featured and media_id) else 0

    # 글 생성/업데이트
    if today_meta and today_meta.get("wp_post_id"):
        post_id = int(today_meta["wp_post_id"])
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, featured_media=featured_id)
        print("OK(updated):", wp_post_id, wp_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html)
        # 생성 후 featured 설정이 필요하면 업데이트로 한 번 더
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
