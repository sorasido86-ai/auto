# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp.py (완전 통합 / 매일 1개 레시피 자동 발행)

- 공개 레시피 API(TheMealDB)에서 "매일 다른" 레시피 1개 랜덤 수집
- SQLite로 오늘 발행 여부/레시피 중복(최근 N일) 방지
- WordPress REST API로 생성/업데이트
- ✅ WP_CATEGORY_IDS 기본값: "7" (요청사항 반영)

필수 환경변수(GitHub Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

옵션 환경변수:
  - WP_STATUS: publish (기본 publish)
  - WP_CATEGORY_IDS: "7" (기본 7)  ← 필요하면 여기만 바꾸면 됨
  - WP_TAG_IDS: "1,2,3" (선택)
  - SQLITE_PATH: data/daily_recipe.sqlite3
  - DRY_RUN: 1이면 WP 발행 안하고 HTML 미리보기 출력
  - DEBUG: 1이면 로그 상세
  - AVOID_REPEAT_DAYS: 90 (최근 N일 내 동일 레시피 id 재사용 방지)
  - MAX_TRIES: 20 (중복 피하려고 랜덤 재시도 횟수)
"""

from __future__ import annotations

import base64
import html as htmlmod
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


def get_today_post(path: str, date_key: str) -> Optional[Tuple[str, int, str]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT recipe_id, wp_post_id, wp_link FROM daily_posts WHERE date_key = ?", (date_key,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    recipe_id = str(row[0] or "")
    wp_post_id = int(row[1] or 0)
    wp_link = str(row[2] or "")
    return recipe_id, wp_post_id, wp_link


def save_today_post(path: str, date_key: str, recipe_id: str, post_id: int, link: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_key, recipe_id, wp_post_id, wp_link, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (date_key, recipe_id, post_id, link, datetime.utcnow().isoformat()),
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-recipe-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


# -----------------------------
# Recipe fetching (TheMealDB)
# -----------------------------
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; daily-recipe-bot/1.0)"})
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

    # 1) 줄바꿈 기반
    lines = [x.strip() for x in re.split(r"[\r\n]+", t) if x.strip()]
    if len(lines) >= 3:
        return lines[:25]

    # 2) 문장 기반(너무 잘게 쪼개지 않게)
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


# -----------------------------
# Rendering
# -----------------------------
DISCLOSURE = "※ 본 글은 공개 레시피 데이터(TheMealDB) 기반으로 자동 생성되었습니다. 원문/출처는 하단 링크를 참고하세요."


def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")


def build_recipe_html(now: datetime, meal: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    returns: (recipe_id, title, slug, html)
    """
    date_str = now.strftime("%Y-%m-%d")
    rid = str(meal.get("idMeal") or "")
    name = clean_text(str(meal.get("strMeal") or "오늘의 레시피"))
    area = clean_text(str(meal.get("strArea") or ""))
    cat = clean_text(str(meal.get("strCategory") or ""))
    thumb = clean_text(str(meal.get("strMealThumb") or ""))
    source_url = clean_text(str(meal.get("strSource") or ""))
    yt = clean_text(str(meal.get("strYoutube") or ""))

    ingredients = extract_ingredients(meal)
    steps = pick_instructions_steps(str(meal.get("strInstructions") or ""))

    title = f"{date_str} 오늘의 레시피: {name}" + (f" ({area})" if area else "")
    slug = f"daily-recipe-{date_str}"  # ✅ 하루 1개 고정(오늘 재실행하면 update)

    mealdb_link = f"https://www.themealdb.com/meal/{rid}" if rid else "https://www.themealdb.com/"
    ref = source_url or mealdb_link

    disclosure = f'<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{htmlmod.escape(DISCLOSURE)}</p>'
    head = f"<p>기준시각: <b>{htmlmod.escape(fmt_dt(now))}</b></p>"

    meta_bits = []
    if cat:
        meta_bits.append(f"카테고리: <b>{htmlmod.escape(cat)}</b>")
    if area:
        meta_bits.append(f"지역/스타일: <b>{htmlmod.escape(area)}</b>")
    meta = f"<p>{' · '.join(meta_bits)}</p>" if meta_bits else ""

    img = (
        f'<p><img src="{htmlmod.escape(thumb)}" alt="{htmlmod.escape(name)}" '
        f'style="max-width:100%;height:auto;border-radius:10px;" /></p>'
        if thumb
        else ""
    )

    ing_html = "<ul>" + "".join(f"<li>{htmlmod.escape(x)}</li>" for x in ingredients) + "</ul>" if ingredients else "<p>-</p>"
    step_html = "<ol>" + "".join(f"<li>{htmlmod.escape(x)}</li>" for x in steps) + "</ol>" if steps else "<p>-</p>"

    refs = f"""
    <hr/>
    <p style="font-size:13px;opacity:.85;">
      출처/원문 링크: <a href="{htmlmod.escape(ref)}" target="_blank" rel="nofollow noopener">{htmlmod.escape(ref)}</a><br/>
      데이터 제공: <a href="{htmlmod.escape(mealdb_link)}" target="_blank" rel="nofollow noopener">TheMealDB</a>
      {("<br/>유튜브: <a href='"+htmlmod.escape(yt)+"' target='_blank' rel='nofollow noopener'>"+htmlmod.escape(yt)+"</a>") if yt else ""}
    </p>
    """

    html = disclosure + head + meta + img + "<h2>재료</h2>" + ing_html + "<h2>만드는 법</h2>" + step_html + refs
    return rid, title, slug, html


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_key = now.strftime("%Y-%m-%d")

    init_db(cfg.sqlite_path)

    meal = fetch_unique_recipe(cfg)
    rid, title, slug, html = build_recipe_html(now, meal)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. HTML 미리보기 ↓\n")
        print(html)
        return

    today = get_today_post(cfg.sqlite_path, date_key)
    if today and today[1] > 0:
        _, post_id, old_link = today
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html)
        save_today_post(cfg.sqlite_path, date_key, rid, wp_post_id, wp_link)
        if rid:
            mark_used_recipe(cfg.sqlite_path, rid)
        print("OK(updated):", wp_post_id, wp_link or old_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, html)
        save_today_post(cfg.sqlite_path, date_key, rid, wp_post_id, wp_link)
        if rid:
            mark_used_recipe(cfg.sqlite_path, rid)
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
