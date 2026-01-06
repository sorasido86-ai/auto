# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp.py (완전 통합/안정화)
- 랜덤 레시피(TheMealDB) 수집 → 한글 블로거톤 변환(OpenAI) → WordPress 발행/업데이트
- 썸네일(대표이미지) 자동 업로드 + featured_media 지정
- SQLite 발행 이력 저장 + 스키마 자동 마이그레이션(컬럼 누락 자동 추가)
- DB가 날아가도: WP slug로 기존 글을 찾아서 "업데이트" (중복 생성 방지)

필수 env (GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - OPENAI_API_KEY  (한글 블로거톤용)

선택 env:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS="7" (기본 7)
  - WP_TAG_IDS="1,2,3" (선택)
  - SQLITE_PATH=data/daily_recipe.sqlite3 (기본)

  - RUN_SLOT=day/am/pm (기본 day)
  - DRY_RUN=1 (발행 안함)
  - DEBUG=1 (상세 로그)

  - OPENAI_MODEL=gpt-5.2 (기본 gpt-5.2)
  - STRICT_KOREAN=1 (기본 1)  # 한글 출력이 아니면 실패 처리
  - FORCE_NEW=0 (기본 0)      # 이미 오늘 글이 있으면 같은 recipe_id 유지(있으면) / 1이면 새 레시피로 교체
  - AVOID_REPEAT_DAYS=90 (기본 90)
  - MAX_TRIES=20 (기본 20)

참고: OpenAI SDK 사용( pip install openai ) :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI  # 공식 SDK :contentReference[oaicite:2]{index=2}

KST = timezone(timedelta(hours=9))

THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"
THEMEALDB_LOOKUP = "https://www.themealdb.com/api/json/v1/1/lookup.php?i={id}"


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
    strict_korean: bool = True
    force_new: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-5.2"  # 문서 예시 기준 :contentReference[oaicite:3]{index=3}


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

    strict_korean = _env_bool("STRICT_KOREAN", True)
    force_new = _env_bool("FORCE_NEW", False)

    avoid_repeat_days = _env_int("AVOID_REPEAT_DAYS", 90)
    max_tries = _env_int("MAX_TRIES", 20)

    upload_thumb = _env_bool("UPLOAD_THUMB", True)
    set_featured = _env_bool("SET_FEATURED", True)
    embed_image_in_body = _env_bool("EMBED_IMAGE_IN_BODY", True)

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-5.2") or "gpt-5.2"

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
            strict_korean=strict_korean,
            force_new=force_new,
            avoid_repeat_days=avoid_repeat_days,
            max_tries=max_tries,
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
    if not cfg.openai.api_key:
        missing.append("OPENAI_API_KEY")
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
    print("[CFG] STRICT_KOREAN:", cfg.run.strict_korean, "| FORCE_NEW:", cfg.run.force_new)
    print("[CFG] AVOID_REPEAT_DAYS:", cfg.run.avoid_repeat_days, "| MAX_TRIES:", cfg.run.max_tries)
    print("[CFG] UPLOAD_THUMB:", cfg.run.upload_thumb, "| SET_FEATURED:", cfg.run.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.run.embed_image_in_body)
    print("[CFG] OPENAI_API_KEY:", ok(cfg.openai.api_key))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model)


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


def wp_find_post_by_slug(cfg: WordPressConfig, slug: str) -> Optional[int]:
    # slug로 기존 글을 찾아서 "중복 생성"을 막고 업데이트하도록
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts?slug={slug}&per_page=1&context=edit"
    r = requests.get(url, headers=wp_auth_header(cfg.user, cfg.app_pass), timeout=20)
    if r.status_code != 200:
        return None
    arr = r.json()
    if isinstance(arr, list) and arr:
        try:
            return int(arr[0].get("id"))
        except Exception:
            return None
    return None


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
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=30)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code} url={image_url}")

    content = r.content
    ctype = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip().lower()

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
def fetch_random_recipe() -> Dict[str, Any]:
    r = requests.get(THEMEALDB_RANDOM, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Recipe API failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("Recipe API returned empty meals")
    return _normalize_meal(meals[0])


def fetch_recipe_by_id(recipe_id: str) -> Dict[str, Any]:
    url = THEMEALDB_LOOKUP.format(id=recipe_id)
    r = requests.get(url, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Recipe lookup failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("Recipe lookup returned empty meals")
    return _normalize_meal(meals[0])


def _normalize_meal(m: Dict[str, Any]) -> Dict[str, Any]:
    recipe_id = str(m.get("idMeal") or "").strip()
    title = str(m.get("strMeal") or "").strip()
    category = str(m.get("strCategory") or "").strip()
    area = str(m.get("strArea") or "").strip()
    instructions = str(m.get("strInstructions") or "").strip()
    thumb = str(m.get("strMealThumb") or "").strip()

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
    parts = [p.strip() for p in re.split(r"\r?\n+", t) if p.strip()]
    if len(parts) <= 2 and len(t) > 400:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    return parts


# -----------------------------
# OpenAI: Korean blogger tone generation
# -----------------------------
def generate_korean_blog(openai_cfg: OpenAIConfig, recipe: Dict[str, Any], debug: bool = False) -> Tuple[str, str]:
    """
    공식 SDK + Responses API: response.output_text 사용 :contentReference[oaicite:4]{index=4}
    """
    client = OpenAI(api_key=openai_cfg.api_key)

    payload_recipe = {
        "title_en": recipe.get("title", ""),
        "category_en": recipe.get("category", ""),
        "area_en": recipe.get("area", ""),
        "ingredients": recipe.get("ingredients", []),
        "steps_en": split_steps(recipe.get("instructions", "")),
        "source_url": recipe.get("source", ""),
        "youtube": recipe.get("youtube", ""),
    }

    instructions = (
        "너는 한국의 음식 블로거다. 반드시 한국어로만 작성해.\n"
        "중요: 제공된 ingredients/steps 외의 재료/양/과정을 절대 추가하거나 바꾸지 마.\n"
        "출력은 '제목 1줄' + 'HTML 본문'으로 구성.\n"
        "본문 섹션은 아래 순서로:\n"
        "1) 소개(짧게)\n"
        "2) 재료(리스트)\n"
        "3) 만드는 법(번호)\n"
        "4) 팁(선택)\n"
        "5) 마무리 한 줄\n"
        "HTML은 워드프레스에 바로 붙여넣을 수 있게 깨끗하게."
    )

    user_input = "레시피 JSON:\n" + json.dumps(payload_recipe, ensure_ascii=False, indent=2)

    resp = client.responses.create(
        model=openai_cfg.model,
        instructions=instructions,
        input=user_input,
    )

    text = (resp.output_text or "").strip()
    if debug:
        print("[OPENAI] output_text length:", len(text))

    # 첫 줄 = 제목, 나머지 = HTML 본문(간단 규칙)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError("OpenAI 응답이 너무 짧습니다(제목/본문 분리 실패).")

    title = lines[0]
    body = "\n".join(lines[1:]).strip()

    # 한국어 검증(STRICT_KOREAN일 때)
    if not re.search(r"[가-힣]", title) or not re.search(r"[가-힣]", body):
        raise RuntimeError("OpenAI 응답이 한국어가 아닙니다(한글 검증 실패). OPENAI_MODEL/프롬프트/키 권한 확인.")

    # body가 HTML이 아니면 최소 래핑
    if "<" not in body:
        body = "<p>" + body.replace("\n", "<br/>") + "</p>"

    return title, body


# -----------------------------
# HTML rendering fallback (절대권장 X)
# -----------------------------
DISCLOSURE = "※ 본 포스팅은 레시피 정보를 바탕으로 자동 생성된 글입니다."


def build_basic_html(recipe: Dict[str, Any], now: datetime) -> str:
    title = recipe.get("title", "")
    ingredients = recipe.get("ingredients", [])
    steps = split_steps(recipe.get("instructions", ""))

    ing_html = "<ul>" + "".join(
        f"<li>{(i.get('name') or '').strip()} {('('+i.get('measure','').strip()+')' if (i.get('measure') or '').strip() else '')}</li>"
        for i in ingredients
    ) + "</ul>"

    step_html = "<ol>" + "".join(f"<li>{s}</li>" for s in steps) + "</ol>"

    return f"""
    <p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{DISCLOSURE}</p>
    <p>기준시각: <b>{now.astimezone(KST).strftime("%Y-%m-%d %H:%M")}</b></p>
    <h2>{title}</h2>
    <h3>재료</h3>
    {ing_html}
    <h3>만드는 법</h3>
    {step_html}
    """


# -----------------------------
# Main flow
# -----------------------------
def pick_recipe(cfg: AppConfig, existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # 오늘 글이 있고, recipe_id도 저장돼있고, FORCE_NEW=0이면 같은 레시피를 유지
    if existing and existing.get("recipe_id") and not cfg.run.force_new:
        return fetch_recipe_by_id(str(existing["recipe_id"]))

    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))
    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if not rid:
            continue
        if rid in recent_ids:
            continue
        return cand

    raise RuntimeError("레시피를 가져오지 못했습니다(중복 회피/시도 횟수 초과). AVOID_REPEAT_DAYS 또는 MAX_TRIES 조정 필요.")


def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else ("오후" if slot == "pm" else "오늘")

    date_key = f"{date_str}_{slot}" if slot in ("am", "pm") else date_str
    slug = f"daily-recipe-{date_str}-{slot}" if slot in ("am", "pm") else f"daily-recipe-{date_str}"

    init_db(cfg.sqlite_path, debug=cfg.run.debug)

    existing = get_today_post(cfg.sqlite_path, date_key)

    # DB에 wp_post_id가 없더라도 slug로 WP에서 찾아서 업데이트 가능하게
    wp_post_id: Optional[int] = None
    if existing and existing.get("wp_post_id"):
        wp_post_id = int(existing["wp_post_id"])
    else:
        wp_post_id = wp_find_post_by_slug(cfg.wp, slug)

    recipe = pick_recipe(cfg, existing)
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

    # ✅ 한글 블로거톤 생성 (실패하면 STRICT_KOREAN=1일 때 바로 실패)
    title_ko, body_html = generate_korean_blog(cfg.openai, recipe, debug=cfg.run.debug)

    # 본문에 이미지 삽입(업로드 성공하면 WP URL 우선)
    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            body_html = f'<p><img src="{img}" alt="{title_ko}" style="max-width:100%;height:auto;"></p>\n' + body_html

    title = f"{date_str} {slot_label} 레시피 | {title_ko}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 ↓")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2000] + ("\n...(truncated)" if len(body_html) > 2000 else ""))
        return

    # 발행/업데이트
    if wp_post_id:
        new_id, wp_link = wp_update_post(cfg.wp, wp_post_id, title, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)
        print("OK(updated):", new_id, wp_link)
    else:
        new_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, featured_media=featured)
        save_post_meta(cfg.sqlite_path, date_key, slot, recipe_id, recipe_title_en, new_id, wp_link, media_id, media_url)
        print("OK(created):", new_id, wp_link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
