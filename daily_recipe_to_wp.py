# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp.py (완전 통합/안정화)
- 랜덤 레시피 수집(TheMealDB) → 중복 회피 → 한글 블로거톤 변환(OpenAI) → WP 발행/업데이트
- 썸네일 자동 업로드 + 대표이미지(Featured) 설정 + 본문 내 이미지 삽입 옵션
- SQLite 발행 이력 저장 + 스키마 자동 마이그레이션(컬럼 누락 자동 추가)

필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

선택 env:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS="7" (기본 7)
  - WP_TAG_IDS="1,2,3" (선택)
  - SQLITE_PATH=data/daily_recipe.sqlite3 (기본)
  - AVOID_REPEAT_DAYS=90 (기본 90)
  - MAX_TRIES=20 (기본 20)
  - RUN_SLOT=am/pm/day (기본 day)
  - DRY_RUN=1 (발행 안함, 출력만)
  - DEBUG=1 (상세로그)

한글/블로거톤(추천):
  - KOREANIZE=1 (기본 1)
  - BLOG_TONE=1 (기본 1)
  - OPENAI_API_KEY=... (GitHub Secret로 추가)
  - OPENAI_MODEL=gpt-5-mini (기본 gpt-5-mini)
"""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

KST = timezone(timedelta(hours=9))


# -----------------------------
# ENV helpers
# -----------------------------
def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name, str(default))
    try:
        return int(v)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = _env(name, "1" if default else "0")
    return v.lower() in ("1", "true", "yes", "y", "on")


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
    run_slot: str = "day"     # day / am / pm
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    koreanize: bool = True
    blog_tone: bool = True
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True


@dataclass
class OpenAIConfig:
    api_key: str = ""
    model: str = "gpt-5-mini"   # 필요 시 env로 변경


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    sqlite_path: str
    openai: OpenAIConfig


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    # ✅ 기본 카테고리 7번
    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    sqlite_path = _env("SQLITE_PATH", "data/daily_recipe.sqlite3")

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    dry_run = _env_bool("DRY_RUN", False)
    debug = _env_bool("DEBUG", False)

    avoid_repeat_days = _env_int("AVOID_REPEAT_DAYS", 90)
    max_tries = _env_int("MAX_TRIES", 20)

    koreanize = _env_bool("KOREANIZE", True)
    blog_tone = _env_bool("BLOG_TONE", True)

    upload_thumb = _env_bool("UPLOAD_THUMB", True)
    set_featured = _env_bool("SET_FEATURED", True)
    embed_image_in_body = _env_bool("EMBED_IMAGE_IN_BODY", True)

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-5-mini") or "gpt-5-mini"

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
            dry_run=dry_run,
            debug=debug,
            avoid_repeat_days=avoid_repeat_days,
            max_tries=max_tries,
            koreanize=koreanize,
            blog_tone=blog_tone,
            upload_thumb=upload_thumb,
            set_featured=set_featured,
            embed_image_in_body=embed_image_in_body,
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot)
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] AVOID_REPEAT_DAYS:", cfg.run.avoid_repeat_days, "| MAX_TRIES:", cfg.run.max_tries)
    print("[CFG] KOREANIZE:", cfg.run.koreanize, "| BLOG_TONE:", cfg.run.blog_tone)
    print("[CFG] UPLOAD_THUMB:", cfg.run.upload_thumb, "| SET_FEATURED:", cfg.run.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.run.embed_image_in_body)
    print("[CFG] OPENAI_API_KEY:", "OK" if cfg.openai.api_key else "MISSING", "| OPENAI_MODEL:", cfg.openai.model)


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
  created_at TEXT
)
"""

# 예전 DB에 컬럼이 없을 때(너가 겪은 media_id 오류) 자동으로 추가
REQUIRED_COLUMNS: Dict[str, str] = {
    "date_key": "TEXT",
    "slot": "TEXT",
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

    # migration: missing columns → ALTER TABLE ADD COLUMN
    cur.execute("PRAGMA table_info(daily_posts)")
    cols = {row[1] for row in cur.fetchall()}  # name at index 1

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
        "SELECT date_key, slot, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at "
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
        "created_at": row[8],
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
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_key, slot, recipe_id, recipe_title, wp_post_id, wp_link, media_id, media_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-recipe-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str, featured_media: Optional[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html, "status": cfg.status}

    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str, featured_media: Optional[int]) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}

    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "recipe.jpg") -> Tuple[int, str]:
    """
    WP 미디어 업로드:
    - WP가 보안 플러그인/설정에 따라 REST 미디어 업로드를 막을 수 있음.
    """
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    # 이미지 다운로드
    r = requests.get(image_url, timeout=30)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code} url={image_url}")

    content = r.content
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower() or "image/jpeg"
    # 파일명 정리
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename_hint).strip("-") or "recipe.jpg"
    if "." not in safe_name:
        safe_name += ".jpg"

    headers["Content-Type"] = ctype
    headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    up = requests.post(media_endpoint, headers=headers, data=content, timeout=45)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {up.status_code} body={up.text[:500]}")

    data = up.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# Recipe fetch (TheMealDB)
# -----------------------------
THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"


def fetch_random_recipe() -> Dict[str, Any]:
    r = requests.get(THEMEALDB_RANDOM, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Recipe API failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("Recipe API returned empty meals")
    m = meals[0]

    recipe_id = str(m.get("idMeal") or "").strip()
    title = str(m.get("strMeal") or "").strip()
    category = str(m.get("strCategory") or "").strip()
    area = str(m.get("strArea") or "").strip()
    instructions = str(m.get("strInstructions") or "").strip()
    thumb = str(m.get("strMealThumb") or "").strip()

    # ingredients: strIngredient1..20 + strMeasure1..20
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
        "source": str(m.get("strSource") or "").strip(),
        "youtube": str(m.get("strYoutube") or "").strip(),
    }


def split_steps(instructions: str) -> List[str]:
    t = (instructions or "").strip()
    if not t:
        return []
    # 문단/줄 기준 분리
    parts = [p.strip() for p in re.split(r"\r?\n+", t) if p.strip()]
    # 너무 길면 문장 분리 보조
    if len(parts) <= 2 and len(t) > 400:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    return parts


# -----------------------------
# OpenAI: Korean blogger tone generation
# -----------------------------
def openai_generate_korean_blog(cfg: OpenAIConfig, recipe: Dict[str, Any], debug: bool = False) -> Tuple[str, str]:
    """
    레시피 원문(영문)을 '한글 + 블로거톤'으로 변환.
    - Responses API 사용: POST https://api.openai.com/v1/responses :contentReference[oaicite:1]{index=1}
    """
    if not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY가 없어 한글 변환을 할 수 없습니다. (KOREANIZE=0으로 끄거나 Key를 추가하세요)")

    title_en = recipe.get("title", "")
    ingredients = recipe.get("ingredients", [])
    steps = split_steps(recipe.get("instructions", ""))

    # 입력 데이터(환각 방지: 제공된 재료/단계만 쓰라고 강하게 지시)
    payload_recipe = {
        "title_en": title_en,
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients": ingredients,
        "steps_en": steps,
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
    }

    instructions = (
        "You are a Korean food blogger. "
        "Rewrite the given recipe into natural Korean blog tone. "
        "Do NOT invent new ingredients or steps. Use ONLY the provided ingredients and steps. "
        "Return ONLY in this exact format:\n"
        "[TITLE]\n<one line Korean title>\n[/TITLE]\n"
        "[BODY_HTML]\n<valid HTML body in Korean>\n[/BODY_HTML]\n"
        "In BODY_HTML, include sections: 소개, 재료, 만드는 법(번호), 팁(선택), 마무리 한 줄.\n"
        "Optionally add the English title in parentheses after the Korean title."
    )

    user_input = (
        "Here is the recipe JSON.\n"
        + json.dumps(payload_recipe, ensure_ascii=False, indent=2)
    )

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }
    body = {
        "model": cfg.model,
        "instructions": instructions,
        "input": user_input,
        "store": False,
    }

    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"OpenAI API failed: {r.status_code} body={r.text[:500]}")

    data = r.json()

    # Responses JSON에서 텍스트 추출(필드가 다를 수 있어 방어적으로)
    text = ""
    if isinstance(data, dict):
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            text = data["output_text"]
        else:
            out = data.get("output") or []
            # message content를 훑어서 합치기
            chunks: List[str] = []
            for item in out:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "message":
                    content = item.get("content") or []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            chunks.append(str(c.get("text") or ""))
            text = "\n".join(chunks)

    if debug:
        print("[OPENAI] raw length:", len(text))

    m_title = re.search(r"\[TITLE\]\s*(.*?)\s*\[/TITLE\]", text, re.DOTALL | re.IGNORECASE)
    m_body = re.search(r"\[BODY_HTML\]\s*(.*?)\s*\[/BODY_HTML\]", text, re.DOTALL | re.IGNORECASE)

    if not m_title or not m_body:
        raise RuntimeError("OpenAI 출력 파싱 실패(포맷 불일치).")

    title_ko = m_title.group(1).strip()
    body_html = m_body.group(1).strip()
    return title_ko, body_html


# -----------------------------
# HTML rendering fallback(영문)
# -----------------------------
DISCLOSURE = "※ 본 포스팅은 레시피 정보를 바탕으로 자동 생성된 글이며, 일부 번역/표현은 자연스럽게 다듬어질 수 있습니다."


def build_fallback_html(recipe: Dict[str, Any], now: datetime) -> str:
    title = recipe.get("title", "")
    ingredients = recipe.get("ingredients", [])
    steps = split_steps(recipe.get("instructions", ""))

    ing_html = "<ul>" + "".join(
        f"<li>{(i.get('name') or '').strip()} {('('+i.get('measure','').strip()+')' if (i.get('measure') or '').strip() else '')}</li>"
        for i in ingredients
    ) + "</ul>"

    step_html = "<ol>" + "".join(f"<li>{s}</li>" for s in steps) + "</ol>"

    thumb = recipe.get("thumb", "")
    thumb_html = f'<p><img src="{thumb}" alt="{title}" style="max-width:100%;height:auto;"></p>' if thumb else ""

    return f"""
    <p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{DISCLOSURE}</p>
    <p>기준시각: <b>{now.astimezone(KST).strftime("%Y-%m-%d %H:%M")}</b></p>
    {thumb_html}
    <h2>{title}</h2>
    <h3>Ingredients</h3>
    {ing_html}
    <h3>Steps</h3>
    {step_html}
    """


# -----------------------------
# Main flow
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot  # day/am/pm
    slot_label = "오전" if slot == "am" else ("오후" if slot == "pm" else "오늘")

    # 슬롯별로 글을 분리하고 싶으면 RUN_SLOT=am/pm 사용
    date_key = f"{date_str}_{slot}" if slot in ("am", "pm") else date_str
    slug = f"daily-recipe-{date_str}-{slot}" if slot in ("am", "pm") else f"daily-recipe-{date_str}"

    init_db(cfg.sqlite_path, debug=cfg.run.debug)

    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))
    recipe: Optional[Dict[str, Any]] = None

    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if not rid:
            continue
        if rid in recent_ids:
            continue
        recipe = cand
        break

    if recipe is None:
        raise RuntimeError("레시피를 가져오지 못했습니다(중복 회피/시도 횟수 초과). AVOID_REPEAT_DAYS 또는 MAX_TRIES 조정 필요.")

    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title", "") or "Daily Recipe"

    # 썸네일 업로드
    media_id: Optional[int] = None
    media_url: str = ""
    thumb_url = (recipe.get("thumb") or "").strip()
    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"recipe-{date_str}-{slot}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    # 본문/제목 생성 (한글 블로거톤)
    title = ""
    html = ""
    if cfg.run.koreanize and cfg.run.blog_tone:
        try:
            title_ko, body_ko = openai_generate_korean_blog(cfg.openai, recipe, debug=cfg.run.debug)
            title = title_ko
            html = body_ko
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] OpenAI koreanize failed. fallback to EN:", repr(e))
            title = f"{date_str} 오늘의 레시피 ({slot_label}) - {recipe_title_en}"
            html = build_fallback_html(recipe, now)
    else:
        title = f"{date_str} 오늘의 레시피 ({slot_label}) - {recipe_title_en}"
        html = build_fallback_html(recipe, now)

    # 본문에 업로드한 이미지 삽입(가능하면 WP에 올린 URL 사용)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            html = f'<p><img src="{img}" alt="{title}" style="max-width:100%;height:auto;"></p>\n' + html

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 ↓\n")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(html[:2000] + ("\n...(truncated)" if len(html) > 2000 else ""))
        return

    existing = get_today_post(cfg.sqlite_path, date_key)
    if existing and existing.get("wp_post_id"):
        post_id = int(existing["wp_post_id"])
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, wp_post_id, wp_link, media_id, media_url)
        print("OK(updated):", wp_post_id, wp_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, wp_post_id, wp_link, media_id, media_url)
        print("OK(created):", wp_post_id, wp_link)


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
        raise
