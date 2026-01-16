# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle.py
- ✅ 항상 새 글 발행(Create Only) 모드 (업데이트 로직 제거)
- ✅ 실행마다 slug/run_id 고유값 부여 -> 중복 실행해도 항상 새 글 생성
- ✅ 본문 상단에 <!-- run_id: ... --> 주석 삽입 -> 생성 여부 추적 가능
- ✅ Actions에서 실행된 파일/해시 출력 -> “옛 파일 실행” 문제 방지

필수 env:
  WP_BASE_URL, WP_USER, WP_APP_PASS, OPENAI_API_KEY

권장 env:
  WP_STATUS=publish
  WP_CATEGORY_IDS="7"
  WP_TAG_IDS=""
  SQLITE_PATH=data/daily_recipe.sqlite3
  RUN_SLOT=day|am|pm
  DRY_RUN=0|1
  DEBUG=0|1
  OPENAI_MODEL=gpt-4.1-mini
  NAVER_KEYWORDS="키워드1,키워드2,..."
"""

from __future__ import annotations

import base64
import hashlib
import html as _html
import json
import os
import random
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import openai
from openai import OpenAI

SCRIPT_VERSION = "2026-01-16-create-only-v3"
KST = timezone(timedelta(hours=9))

THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"


# -----------------------------
# ENV helpers
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
# Text sanitize
# -----------------------------
def sanitize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\uFFFD", "")
    s = "".join(ch for ch in s if ch in ("\n", "\t") or (ord(ch) >= 32 and ord(ch) != 127))
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


def _contains_english(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]", s or ""))


def _strip_periods(s: str) -> str:
    s = s or ""
    return s.replace(".", "").replace("。", "").replace("．", "")


def _strip_bullets_and_numbers(s: str) -> str:
    s = s or ""
    s = re.sub(r"[•●◦∙·]", "", s)
    s = re.sub(r"^\s*[-–—*]\s*", "", s, flags=re.M)
    s = re.sub(r"^\s*(\d+)\s*(단계|step)\s*[:：]?\s*", "", s, flags=re.I | re.M)
    return s.strip()


def clean_korean_style(s: str) -> str:
    s = sanitize_text(s)
    s = _strip_bullets_and_numbers(_strip_periods(s))
    return s.strip()


def _dedupe_paragraphs(text: str) -> str:
    t = sanitize_text(text or "")
    if not t:
        return ""
    paras = [p.strip() for p in re.split(r"\n{2,}", t) if p.strip()]
    seen = set()
    kept = []
    for p in paras:
        key = re.sub(r"\s+", " ", p).strip()
        if len(key) >= 40:
            if key in seen:
                continue
            seen.add(key)
        kept.append(p)
    return "\n\n".join(kept).strip()


# -----------------------------
# OpenAI helpers
# -----------------------------
def _openai_call_with_retry(
    client: OpenAI,
    model: str,
    instructions: str,
    input_text: str,
    max_retries: int,
    debug: bool = False,
):
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(model=model, instructions=instructions, input=input_text)
        except openai.RateLimitError:
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())
        except openai.APIError:
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep((2 ** attempt) + random.random())


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
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    openai_max_retries: int = 3


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4.1-mini"


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    sqlite_path: str
    openai: OpenAIConfig
    naver_keywords_csv: str = ""


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "7"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    sqlite_path = _env("SQLITE_PATH", "data/daily_recipe.sqlite3")

    run_slot = (_env("RUN_SLOT", "day") or "day").lower()
    if run_slot not in ("day", "am", "pm"):
        run_slot = "day"

    cfg_run = RunConfig(
        run_slot=run_slot,
        dry_run=_env_bool("DRY_RUN", False),
        debug=_env_bool("DEBUG", False),
        avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
        max_tries=_env_int("MAX_TRIES", 20),
        upload_thumb=_env_bool("UPLOAD_THUMB", True),
        set_featured=_env_bool("SET_FEATURED", True),
        embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
    )

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"
    naver_keywords = _env("NAVER_KEYWORDS", "")

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=cat_ids,
            tag_ids=tag_ids,
        ),
        run=cfg_run,
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
        naver_keywords_csv=naver_keywords,
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

    print("[VERSION]", SCRIPT_VERSION)
    print("[CFG] WP_BASE_URL:", cfg.wp.base_url or "MISSING")
    print("[CFG] WP_USER:", ok(cfg.wp.user))
    print("[CFG] WP_APP_PASS:", ok(cfg.wp.app_pass))
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot)
    print("[CFG] DRY_RUN:", int(cfg.run.dry_run), "| DEBUG:", int(cfg.run.debug))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model, "| OPENAI_KEY:", ok(cfg.openai.api_key))
    print("[CFG] NAVER_KEYWORDS:", cfg.naver_keywords_csv or "(empty)")


# -----------------------------
# SQLite (history)
# -----------------------------
TABLE_SQL = """
CREATE TABLE IF NOT EXISTS daily_posts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  run_id TEXT,
  slot TEXT,
  wp_post_id INTEGER,
  wp_link TEXT,
  recipe_id TEXT,
  recipe_title_en TEXT,
  slug TEXT
);
"""


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(TABLE_SQL)
    con.commit()
    con.close()


def save_post_meta(
    path: str,
    run_id: str,
    slot: str,
    wp_post_id: int,
    wp_link: str,
    recipe_id: str,
    recipe_title_en: str,
    slug: str,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO daily_posts(created_at, run_id, slot, wp_post_id, wp_link, recipe_id, recipe_title_en, slug)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (datetime.utcnow().isoformat(), run_id, slot, wp_post_id, wp_link, recipe_id, recipe_title_en, slug),
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
# WordPress REST (CREATE ONLY)
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-recipe-bot/1.0"}


def wp_create_post(
    cfg: WordPressConfig,
    title: str,
    slug: str,
    html: str,
    featured_media: Optional[int],
    now_kst: datetime,
) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json; charset=utf-8"}

    # ✅ 날짜도 강제로 현재로(새 글이 목록에 잘 뜨게)
    payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": cfg.status,
        "date": now_kst.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = int(featured_media)

    r = requests.post(url, headers=headers, json=payload, timeout=50)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:800]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str) -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=50)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code} url={image_url}")

    content = r.content
    ctype = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip().lower()
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename_hint).strip("-") or "recipe.jpg"
    if "." not in safe_name:
        safe_name += ".jpg"

    headers["Content-Type"] = ctype
    headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    up = requests.post(media_endpoint, headers=headers, data=content, timeout=80)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {up.status_code} body={up.text[:800]}")
    data = up.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# Recipe fetch
# -----------------------------
def fetch_random_recipe() -> Dict[str, Any]:
    r = requests.get(THEMEALDB_RANDOM, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Recipe API failed: {r.status_code}")
    j = r.json()
    meals = j.get("meals") or []
    if not meals:
        raise RuntimeError("Recipe API returned empty meals")
    m = meals[0]

    recipe_id = sanitize_text(m.get("idMeal") or "")
    title = sanitize_text(m.get("strMeal") or "")
    instructions = sanitize_text(m.get("strInstructions") or "")
    thumb = sanitize_text(m.get("strMealThumb") or "")

    ingredients: List[Dict[str, str]] = []
    for i in range(1, 21):
        ing = sanitize_text(m.get(f"strIngredient{i}") or "")
        mea = sanitize_text(m.get(f"strMeasure{i}") or "")
        if ing:
            ingredients.append({"name": ing, "measure": mea})

    return {
        "id": recipe_id,
        "title": title,
        "instructions": instructions,
        "ingredients": ingredients,
        "thumb": thumb,
        "source": sanitize_text(m.get("strSource") or ""),
        "youtube": sanitize_text(m.get("strYoutube") or ""),
    }


def split_steps(instructions: str) -> List[str]:
    t = sanitize_text(instructions or "")
    if not t:
        return []
    parts = [p.strip() for p in re.split(r"\n+", t) if p.strip()]
    if len(parts) <= 2 and len(t) > 400:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    return [p for p in parts if p]


# -----------------------------
# HTML builders (가독성 + 불릿/단계 제거)
# -----------------------------
def _wrap_div(inner: str) -> str:
    style = (
        "font-size:16px;"
        "line-height:2.05;"
        "letter-spacing:-0.2px;"
        "word-break:keep-all;"
        "max-width:760px;"
        "margin:0 auto;"
    )
    return f"<div style='{style}'>{inner}</div>"


def _p(text: str) -> str:
    safe = _html.escape(sanitize_text(text)).replace("\n", "<br/><br/>")
    return f"<p style='margin:0 0 18px 0;line-height:2.05;'>{safe}</p>"


def _h2(title: str) -> str:
    t = _html.escape(sanitize_text(title))
    return f"<h2 style='margin:28px 0 12px 0;'><b>{t}</b></h2>"


def _ingredients_block(ingredients: List[Dict[str, str]]) -> str:
    # ✅ 특문 점/불릿 없이 줄바꿈 형태로
    out = []
    for it in ingredients:
        name = clean_korean_style(it.get("name_ko") or it.get("name") or "")
        mea = clean_korean_style(it.get("measure") or "")
        if not name:
            continue
        line = f"{name}   {mea}".strip() if mea else name
        out.append(f"<div style='margin:0 0 10px 0;opacity:.92;'>{_html.escape(line)}</div>")
    if not out:
        return _p("재료는 본문 기준으로 준비하시면 됩니다")
    return "<div style='margin:0 0 6px 0;'>" + "".join(out) + "</div>"


def _method_block(steps: List[str]) -> str:
    # ✅ 1단계 2단계 같은 번호 제거 + 문단 형태
    out = []
    for s in steps:
        s2 = clean_korean_style(s)
        if not s2:
            continue
        out.append(f"<p style='margin:0 0 16px 0;line-height:2.05;'>{_html.escape(s2)}</p>")
    if not out:
        return _p("만드는 법은 제공된 조리 순서대로  상태를 보면서 진행하시면 됩니다")
    return "<div>" + "".join(out) + "</div>"


def tags_html(keywords_csv: str, extra: List[str]) -> str:
    kws: List[str] = []
    for x in (keywords_csv or "").split(","):
        x = sanitize_text(x)
        if x and x not in kws:
            kws.append(x)
    for x in extra:
        x = sanitize_text(x)
        if x and x not in kws:
            kws.append(x)

    base_add = ["레시피", "집밥", "오늘의요리", "간단요리"]
    for x in base_add:
        if x not in kws:
            kws.append(x)

    kws = [k for k in kws if k and not _contains_english(k)]
    kws = kws[:12]
    if not kws:
        return ""

    line = " ".join([f"#{k.replace(' ', '')}" for k in kws])
    return f"<div style='margin:24px 0 0 0;opacity:.82;font-size:14px;line-height:1.95;'>태그  {line}</div>"


def closing_random() -> str:
    pool = [
        "요리는 완벽하게 하려는 순간부터 부담이 커지더라고요\n\n오늘은 기준만 잡는 느낌으로  편하게 가져가셔도 충분해요",
        "한 번 감만 잡히면 다음부터는 속도가 확 줄어요\n\n다음엔 내 입맛 포인트 하나만 살짝 조절해보셔도 좋겠어요",
        "요리는 결국 내 컨디션을 배려하는 쪽이 오래 가더라고요\n\n급하지 않게  편한 속도로 해보셔도 충분해요",
        "오늘 정리한 방식은 메뉴 고민될 때 다시 꺼내 보기 좋아요\n\n특히 갈리는 지점만 기억해두셔도 도움이 될 거예요",
    ]
    return clean_korean_style(random.choice(pool))


# -----------------------------
# Content generation (해요체 강제)
# -----------------------------
def generate_homefeed_text(cfg: AppConfig, keyword: str) -> Dict[str, Any]:
    client = OpenAI(api_key=cfg.openai.api_key)

    prompt = {
        "keyword": keyword,
        "format": {
            "intro": "200~300자",
            "sections": 3,
            "each_section_min_chars": 1500,
            "total_min_chars": 2300,
        },
    }

    instructions = (
        "너는 한국어 블로그 글을 쓰는 사람이다\n"
        "출력은 반드시 JSON 하나만  코드블럭 금지\n"
        "- 말투는 친구에게 수다 떠는 해요체 존댓말로만\n"
        "- 반말 금지\n"
        "- 마침표 문자 사용 금지  대신 줄바꿈과 여백으로 호흡\n"
        "- 번호 매기기 금지  1단계 2단계 step 같은 표현 금지\n"
        "- 같은 문장 반복 금지\n"
        "{\"intro\":\"...\",\"sections\":[{\"h\":\"...\",\"body\":\"...\"},{\"h\":\"...\",\"body\":\"...\"},{\"h\":\"...\",\"body\":\"...\"}]}\n"
    )

    resp = _openai_call_with_retry(
        client=client,
        model=cfg.openai.model,
        instructions=instructions,
        input_text=json.dumps(prompt, ensure_ascii=False),
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    raw = sanitize_text((resp.output_text or "").strip())
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1]) if ("{" in raw and "}" in raw) else json.loads(raw)

    intro = _dedupe_paragraphs(clean_korean_style(str(data.get("intro") or "")))
    sections = data.get("sections") or []

    fixed = []
    for s in sections[:3]:
        h = clean_korean_style(str((s or {}).get("h") or "오늘 이야기"))
        body = _dedupe_paragraphs(clean_korean_style(str((s or {}).get("body") or "")))
        fixed.append({"h": h, "body": body})

    return {"intro": intro, "sections": fixed}


# -----------------------------
# Run
# -----------------------------
def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)

    # ✅ 실행 파일/해시를 찍어서 “옛 파일 실행”을 확실히 잡는다
    try:
        me = Path(__file__)
        md5 = hashlib.md5(me.read_bytes()).hexdigest()
        print("[FILE]", str(me.resolve()))
        print("[MD5 ]", md5)
    except Exception:
        pass

    init_db(cfg.sqlite_path)

    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else ("오후" if slot == "pm" else "오늘")

    run_id = now.strftime("%Y%m%d-%H%M%S") + "-" + "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    slug = f"daily-recipe-{date_str}-{slot}-{run_id}"

    # 레시피 선택(중복 회피)
    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))
    recipe = None
    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = cand.get("id") or ""
        if rid and rid not in recent_ids:
            recipe = cand
            break
    if not recipe:
        recipe = fetch_random_recipe()

    recipe_id = recipe.get("id") or ""
    recipe_title_en = recipe.get("title") or "Daily Recipe"
    thumb_url = recipe.get("thumb") or ""
    steps_en = split_steps(recipe.get("instructions") or "")
    ingredients = recipe.get("ingredients") or []

    # 키워드(제목용)
    kw_list = [k.strip() for k in (cfg.naver_keywords_csv or "").split(",") if k.strip()]
    keyword_main = kw_list[0] if kw_list else "오늘의 레시피"

    # 해요체 본문 생성
    narrative = generate_homefeed_text(cfg, keyword_main)

    # 썸네일 업로드
    media_id: Optional[int] = None
    media_url: str = ""
    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"recipe-{date_str}-{slot}-{run_id}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    featured = media_id if (cfg.run.set_featured and media_id) else None

    # 본문 조립(가독성 + 순서 고정)
    blocks: List[str] = []

    # 추적용 run_id 주석(새 글인지 확인용)
    blocks.append(f"<!-- run_id: {run_id} version: {SCRIPT_VERSION} -->")

    if cfg.run.embed_image_in_body:
        img = media_url or thumb_url
        if img:
            blocks.append(
                "<p style='margin:0 0 20px 0;'>"
                f"<img src='{_html.escape(img)}' alt='{_html.escape(keyword_main)}' "
                "style='max-width:100%;height:auto;border-radius:16px;display:block;'/>"
                "</p>"
            )

    intro = narrative.get("intro") or ""
    if intro:
        blocks.append(_p(intro))

    for sec in (narrative.get("sections") or [])[:3]:
        blocks.append(_h2(sec.get("h") or "오늘 이야기"))
        blocks.append(_p(sec.get("body") or ""))

    blocks.append(_h2("재료"))
    # (재료는 지금 영문 기반인데, 너가 이미 번역 잘 나오게 만든 버전이 따로 있다면 그쪽 번역 결과를 name_ko에 넣어주면 됨)
    blocks.append(_ingredients_block(ingredients))

    blocks.append(_h2("만드는 법"))
    blocks.append(_method_block(steps_en))

    blocks.append(_h2("마무리"))
    blocks.append(_p(closing_random()))

    blocks.append(tags_html(cfg.naver_keywords_csv, extra=[]))

    body_html = _wrap_div("".join(blocks))

    title = f"{date_str} {slot_label}  {keyword_main}"
    title = clean_korean_style(title)

    print("[CREATE ONLY] slug =", slug)
    print("[CREATE ONLY] run_id =", run_id)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", title)
        print("SLUG:", slug)
        print(body_html[:2000])
        return

    # ✅ 무조건 새 글 생성만 수행
    post_id, link = wp_create_post(cfg.wp, title, slug, body_html, featured_media=featured, now_kst=now)
    save_post_meta(cfg.sqlite_path, run_id, slot, post_id, link, recipe_id, recipe_title_en, slug)
    print("OK(created):", post_id, link)


if __name__ == "__main__":
    main()
