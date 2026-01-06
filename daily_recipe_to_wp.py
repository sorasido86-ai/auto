# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp.py (완전 통합 / 매일 1개 레시피 자동 발행 + 대표이미지 업로드)

✅ 기능
- TheMealDB 공개 레시피 API에서 매일 다른 레시피 1개 랜덤 수집
- SQLite로 오늘 발행 여부/레시피 중복(최근 N일) 방지
- WordPress에 글 생성/업데이트
- ✅ 썸네일 이미지 자동 다운로드 → WP Media 업로드 → featured_media(대표이미지) 지정
- ✅ 제목/본문을 "블로거 톤"으로 자동 구성(소개/재료/레시피/팁/보관/출처)

필수 환경변수(GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

옵션 환경변수:
  - WP_STATUS: publish (기본 publish)
  - WP_CATEGORY_IDS: "7" (기본 7)
  - WP_TAG_IDS: "1,2,3" (선택)
  - SQLITE_PATH: data/daily_recipe.sqlite3
  - DRY_RUN: 1이면 WP 발행 안하고 HTML 미리보기 출력
  - DEBUG: 1이면 로그 상세
  - AVOID_REPEAT_DAYS: 90 (최근 N일 내 동일 레시피 id 재사용 방지)
  - MAX_TRIES: 20 (중복 피하려고 랜덤 재시도 횟수)
  - UPLOAD_THUMB: 1/0 (기본 1)  썸네일 WP 업로드
  - SET_FEATURED: 1/0 (기본 1)  대표이미지 지정
  - EMBED_IMAGE_IN_BODY: 1/0 (기본 1) 본문 상단에 이미지 삽입(테마에 따라 대표이미지만으로 충분하면 0 추천)
"""

from __future__ import annotations

import base64
import html as htmlmod
import mimetypes
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

KST = timezone(timedelta(hours=9))
THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"


# -----------------------------
# Config helpers
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
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    sqlite_path: str


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    # ✅ 기본 카테고리 7번
    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    sqlite_path = _env("SQLITE_PATH", "data/daily_recipe.sqlite3")
    dry_run = _env_bool("DRY_RUN", False)
    debug = _env_bool("DEBUG", False)

    avoid_repeat_days = _env_int("AVOID_REPEAT_DAYS", 90)
    max_tries = _env_int("MAX_TRIES", 20)

    upload_thumb = _env_bool("UPLOAD_THUMB", True)
    set_featured = _env_bool("SET_FEATURED", True)
    embed_image_in_body = _env_bool("EMBED_IMAGE_IN_BODY", True)

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
            dry_run=dry_run,
            debug=debug,
            avoid_repeat_days=avoid_repeat_days,
            max_tries=max_tries,
            upload_thumb=upload_thumb,
            set_featured=set_featured,
            embed_image_in_body=embed_image_in_body,
        ),
        sqlite_path=sqlite_path,
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
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] AVOID_REPEAT_DAYS:", cfg.run.avoid_repeat_days, "| MAX_TRIES:", cfg.run.max_tries)
    print("[CFG] UPLOAD_THUMB:", cfg.run.upload_thumb, "| SET_FEATURED:", cfg.run.set_featured, "| EMBED_IMAGE_IN_BODY:", cfg.run.embed_image_in_body)


# -----------------------------
# SQLite
# -----------------------------
def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()

    # 오늘 발행 이력(재실행 시 update)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_posts (
          date_key TEXT PRIMARY KEY,
          recipe_id TEXT,
          wp_post_id INTEGER,
          wp_link TEXT,
          media_id INTEGER,
          media_url TEXT,
          created_at TEXT
        )
        """
    )

    # 레시피 사용 이력(중복 방지)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS used_recipes (
          recipe_id TEXT PRIMARY KEY,
          used_at TEXT
        )
        """
    )

    con.commit()
    con.close()


def get_today_post(path: str, date_key: str) -> Optional[Tuple[str, int, str, int, str]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT recipe_id, wp_post_id, wp_link, media_id, media_url FROM daily_posts WHERE date_key = ?", (date_key,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    recipe_id = str(row[0] or "")
    wp_post_id = int(row[1] or 0)
    wp_link = str(row[2] or "")
    media_id = int(row[3] or 0)
    media_url = str(row[4] or "")
    return recipe_id, wp_post_id, wp_link, media_id, media_url


def save_today_post(path: str, date_key: str, recipe_id: str, post_id: int, link: str, media_id: int, media_url: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_key, recipe_id, wp_post_id, wp_link, media_id, media_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (date_key, recipe_id, post_id, link, media_id, media_url, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


def mark_used_recipe(path: str, recipe_id: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO used_recipes(recipe_id, used_at)
        VALUES (?, ?)
        """,
        (recipe_id, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


def was_used_recently(path: str, recipe_id: str, days: int) -> bool:
    if days <= 0:
        return False
    cutoff = datetime.utcnow() - timedelta(days=days)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT used_at FROM used_recipes WHERE recipe_id = ?", (recipe_id,))
    row = cur.fetchone()
    con.close()
    if not row or not row[0]:
        return False
    try:
        used_at = datetime.fromisoformat(row[0])
        return used_at >= cutoff
    except Exception:
        return False


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-recipe-bot/2.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_bytes: bytes, filename: str, mime: str, title: str, alt_text: str) -> Tuple[int, str]:
    """
    업로드 성공 시: (media_id, source_url) 반환
    """
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass)

    # 1) multipart 업로드 시도
    files = {
        "file": (filename, image_bytes, mime),
    }
    data = {"title": title}
    r = requests.post(url, headers=headers, files=files, data=data, timeout=40)

    # 일부 환경에서 multipart가 막히면 raw 업로드가 더 잘 먹는 경우가 있어 fallback
    if r.status_code not in (200, 201):
        headers2 = {
            **headers,
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": mime,
        }
        r = requests.post(url, headers=headers2, data=image_bytes, timeout=40)

    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {r.status_code} body={r.text[:500]}")

    j = r.json()
    media_id = int(j["id"])
    source_url = str(j.get("source_url") or "")

    # alt_text 업데이트(가능하면)
    try:
        url2 = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/media/{media_id}"
        headers_json = {**headers, "Content-Type": "application/json"}
        requests.post(url2, headers=headers_json, json={"alt_text": alt_text}, timeout=25)
    except Exception:
        pass

    return media_id, source_url


# -----------------------------
# Recipe fetching (TheMealDB)
# -----------------------------
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; daily-recipe-bot/2.0)"})
    return s


def clean_text(s: str) -> str:
    s = htmlmod.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_ingredients(meal: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for i in range(1, 21):
        ing = clean_text(str(meal.get(f"strIngredient{i}", "") or ""))
        meas = clean_text(str(meal.get(f"strMeasure{i}", "") or ""))
        if not ing:
            continue
        out.append(f"{ing} — {meas}" if meas else ing)
    return out


def pick_instructions_steps(instr: str) -> List[str]:
    t = clean_text(instr or "")
    if not t:
        return []

    lines = [x.strip() for x in re.split(r"[\r\n]+", t) if x.strip()]
    if len(lines) >= 3:
        return lines[:25]

    parts = [x.strip() for x in re.split(r"\.\s+", t) if x.strip()]
    if len(parts) >= 3:
        out = []
        for p in parts[:25]:
            out.append(p if p.endswith(".") else p + ".")
        return out

    return [t]


def fetch_random_recipe() -> Dict[str, Any]:
    with _session() as s:
        r = s.get(THEMEALDB_RANDOM, timeout=20)
        r.raise_for_status()
        data = r.json()
    meals = data.get("meals") or []
    if not meals:
        raise RuntimeError("레시피 API 응답에 meals가 없습니다.")
    return meals[0]


def fetch_unique_recipe(cfg: AppConfig) -> Dict[str, Any]:
    last = None
    for _ in range(max(1, cfg.run.max_tries)):
        meal = fetch_random_recipe()
        rid = str(meal.get("idMeal") or "")
        if not rid:
            last = meal
            continue
        if not was_used_recently(cfg.sqlite_path, rid, cfg.run.avoid_repeat_days):
            return meal
        last = meal
        if cfg.run.debug:
            print("[DEBUG] repeat avoided:", rid)
    if last:
        return last
    raise RuntimeError("레시피를 가져오지 못했습니다.")


def download_image(url: str) -> Tuple[bytes, str]:
    """
    returns: (bytes, mime)
    """
    if not url:
        raise RuntimeError("썸네일 URL이 없습니다.")
    with _session() as s:
        r = s.get(url, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"썸네일 다운로드 실패: {r.status_code}")
        mime = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        b = r.content
    if not mime:
        mime = "image/jpeg"
    return b, mime


def safe_filename(base: str, mime: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", base).strip("_") or "thumb"
    ext = mimetypes.guess_extension(mime) or ".jpg"
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        ext = ".jpg"
    return f"{base}{ext}"


# -----------------------------
# Blogger-tone Rendering
# -----------------------------
DISCLOSURE = "※ 본 글은 공개 레시피 데이터(TheMealDB) 기반으로 자동 생성되었습니다. 원문/출처는 하단 링크를 참고하세요."


def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")


def blogger_intro(name: str, area: str, cat: str) -> str:
    # 블로거톤 인트로(과하지 않게)
    bits = []
    if cat:
        bits.append(cat)
    if area:
        bits.append(area)
    vibe = " · ".join(bits)
    vibe = f" ({vibe})" if vibe else ""
    return (
        f"<p>오늘은 <b>{htmlmod.escape(name)}</b>{htmlmod.escape(vibe)} 레시피를 가져왔어요. "
        f"바쁜 날에도 부담 없이 따라 할 수 있게 핵심만 정리해둘게요 🙂</p>"
    )


def build_recipe_html(cfg: AppConfig, now: datetime, meal: Dict[str, Any], media_url: str = "") -> Tuple[str, str, str, str]:
    """
    returns: (recipe_id, title, slug, html)
    """
    date_str = now.strftime("%Y-%m-%d")
    rid = str(meal.get("idMeal") or "")
    name = clean_text(str(meal.get("strMeal") or "오늘의 레시피"))
    area = clean_text(str(meal.get("strArea") or ""))
    cat = clean_text(str(meal.get("strCategory") or ""))

    thumb_src = clean_text(str(meal.get("strMealThumb") or ""))
    source_url = clean_text(str(meal.get("strSource") or ""))
    yt = clean_text(str(meal.get("strYoutube") or ""))

    ingredients = extract_ingredients(meal)
    steps = pick_instructions_steps(str(meal.get("strInstructions") or ""))

    # 제목도 블로거톤: 너무 길면 자연스럽게
    title = f"{date_str} 오늘의 레시피 | {name}"
    if area:
        title += f" ({area})"

    slug = f"daily-recipe-{date_str}"  # 하루 1개 고정(오늘 재실행하면 update)

    mealdb_link = f"https://www.themealdb.com/meal/{rid}" if rid else "https://www.themealdb.com/"
    ref = source_url or mealdb_link

    disclosure = f'<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{htmlmod.escape(DISCLOSURE)}</p>'
    head = f"<p style='opacity:.85;'>기준시각: <b>{htmlmod.escape(fmt_dt(now))}</b></p>"

    meta_bits = []
    if cat:
        meta_bits.append(f"카테고리: <b>{htmlmod.escape(cat)}</b>")
    if area:
        meta_bits.append(f"스타일: <b>{htmlmod.escape(area)}</b>")
    meta = f"<p>{' · '.join(meta_bits)}</p>" if meta_bits else ""

    # 이미지: 업로드된 media_url 우선, 없으면 원본 thumb 사용
    img_url = media_url or thumb_src
    img_block = ""
    if cfg.run.embed_image_in_body and img_url:
        img_block = (
            f"<figure style='margin:14px 0;'>"
            f"<img src='{htmlmod.escape(img_url)}' alt='{htmlmod.escape(name)}' "
            f"style='max-width:100%;height:auto;border-radius:12px;'/>"
            f"<figcaption style='font-size:12px;opacity:.7;margin-top:6px;'>오늘의 레시피: {htmlmod.escape(name)}</figcaption>"
            f"</figure>"
        )

    intro = blogger_intro(name, area, cat)

    ing_html = (
        "<ul>"
        + "".join(f"<li>{htmlmod.escape(x)}</li>" for x in ingredients)
        + "</ul>"
        if ingredients
        else "<p>-</p>"
    )

    # 블로거톤 스텝(원문을 그대로 복붙 느낌 줄이려고 '요약' 문장 + 원문 스텝 제공)
    # 번역은 하지 않음(영문일 수 있음) — 대신 읽기 편하게 정돈
    step_items = []
    for s in steps:
        s2 = clean_text(s)
        if not s2:
            continue
        step_items.append(f"<li>{htmlmod.escape(s2)}</li>")
    step_html = "<ol>" + "".join(step_items) + "</ol>" if step_items else "<p>-</p>"

    tips = (
        "<ul>"
        "<li>재료 계량은 집마다 컵/스푼이 달라서, 처음엔 조금씩 넣어가며 맛을 맞추는 게 좋아요.</li>"
        "<li>불 조절이 맛을 좌우해요. 센 불로 시작했다면 중약불로 마무리해 주세요.</li>"
        "<li>남은 음식은 완전히 식힌 뒤 밀폐 보관하면 다음 날 더 맛있어지는 경우가 많아요.</li>"
        "</ul>"
    )

    refs = (
        f"<hr/>"
        f"<p style='font-size:13px;opacity:.85;'>"
        f"출처/원문 링크: <a href='{htmlmod.escape(ref)}' target='_blank' rel='nofollow noopener'>{htmlmod.escape(ref)}</a><br/>"
        f"데이터 제공: <a href='{htmlmod.escape(mealdb_link)}' target='_blank' rel='nofollow noopener'>TheMealDB</a>"
        + (f"<br/>유튜브 참고: <a href='{htmlmod.escape(yt)}' target='_blank' rel='nofollow noopener'>{htmlmod.escape(yt)}</a>" if yt else "")
        + "</p>"
    )

    html = (
        disclosure
        + head
        + meta
        + intro
        + img_block
        + "<h2>재료</h2>"
        + ing_html
        + "<h2>만드는 법</h2>"
        + "<p style='opacity:.85;'>아래 순서대로만 따라가면 됩니다. (원문 레시피 흐름을 최대한 살렸어요.)</p>"
        + step_html
        + "<h2>맛있게 만드는 팁</h2>"
        + tips
        + refs
    )

    return rid, title, slug, html


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_key = now.strftime("%Y-%m-%d")

    init_db(cfg.sqlite_path)

    # 레시피 하나 뽑기(최근 N일 중복 방지)
    meal = fetch_unique_recipe(cfg)
    rid = str(meal.get("idMeal") or "")
    name = clean_text(str(meal.get("strMeal") or "오늘의 레시피"))
    thumb = clean_text(str(meal.get("strMealThumb") or ""))

    # 이미지 업로드(선택)
    media_id = 0
    media_url = ""

    if cfg.run.upload_thumb and thumb and not cfg.run.dry_run:
        try:
            img_bytes, mime = download_image(thumb)
            filename = safe_filename(f"recipe_{date_key}_{rid or name}", mime)
            media_title = f"{date_key} {name} 썸네일"
            alt_text = f"{name} 레시피 이미지"
            media_id, media_url = wp_upload_media(cfg.wp, img_bytes, filename, mime, media_title, alt_text)
            if cfg.run.debug:
                print("[DEBUG] media uploaded:", media_id, media_url)
        except Exception as e:
            # 이미지 업로드 실패해도 글 발행은 진행
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))
            media_id = 0
            media_url = ""

    # 블로거톤 HTML 생성 (업로드된 media_url 있으면 본문 이미지로 사용)
    rid2, title, slug, html = build_recipe_html(cfg, now, meal, media_url=media_url)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. HTML 미리보기 ↓\n")
        print(html)
        return

    # 오늘 글은 update로 유지(슬러그 고정)
    today = get_today_post(cfg.sqlite_path, date_key)
    featured = media_id if cfg.run.set_featured else 0

    if today and today[1] > 0:
        _, post_id, old_link, _, _ = today
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html, featured_media=featured)
        save_today_post(cfg.sqlite_path, date_key, rid2, wp_post_id, wp_link, media_id, media_url)
        if rid2:
            mark_used_recipe(cfg.sqlite_path, rid2)
        print("OK(updated):", wp_post_id, wp_link or old_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, html, featured_media=featured)
        save_today_post(cfg.sqlite_path, date_key, rid2, wp_post_id, wp_link, media_id, media_url)
        if rid2:
            mark_used_recipe(cfg.sqlite_path, rid2)
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
