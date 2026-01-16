# -*- coding: utf-8 -*-
"""
daily_recipe_to_wp_naverstyle_FINAL.py (통합 최종본 / 들여쓰기 오류 제거 / HTML 깨짐 방지)
- 제목: 4가지 후킹 유형(이득/위협/궁금/비교논쟁) 중 랜덤 1개 + 키워드 포함
- 재료/만드는 흐름: 번역 필수(JSON) + 영어 남으면 1회 재번역
- 글: 네이버 홈피드형(도입 200~300자 + 굵은 소제목 3개 + 충분한 분량)
- 톤: 친구에게 수다떠는 존댓말(반말 섞이면 재생성 시도)
- 마침표/불릿/단계번호: "텍스트 노드"에서만 제거 (HTML 태그/속성/스타일 절대 보호)
- WP: 하루 중복 실행돼도 '항상 새 글 발행'(업데이트 X) -> slug에 시간+랜덤
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import sqlite3
import time
import html as _html
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import openai  # 예외 타입 호환용
from openai import OpenAI

KST = timezone(timedelta(hours=9))

THEMEALDB_RANDOM = "https://www.themealdb.com/api/json/v1/1/random.php"
PEXELS_SEARCH = "https://api.pexels.com/v1/search"

TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")


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


def _parse_str_list(csv: str) -> List[str]:
    out: List[str] = []
    for x in (csv or "").split(","):
        x = x.strip()
        if x:
            out.append(x)
    return out


def _now_kst() -> datetime:
    return datetime.now(tz=KST)


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
    run_slot: str = "day"     # day / am / pm
    dry_run: bool = False
    debug: bool = False
    force_new: bool = True    # 항상 새 글
    avoid_repeat_days: int = 90
    max_tries: int = 20
    upload_thumb: bool = True
    set_featured: bool = True
    embed_image_in_body: bool = True
    openai_max_retries: int = 3


@dataclass
class NaverStyleConfig:
    main_keyword: str = ""      # 제목 키워드 강제 (비우면 레시피명 사용)
    keywords_csv: str = ""      # 태그 힌트
    prefer_areas: List[str] = field(default_factory=list)


@dataclass
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4.1-mini"


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    naver: NaverStyleConfig
    sqlite_path: str
    openai: OpenAIConfig
    pexels_api_key: str = ""


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
        force_new=_env_bool("FORCE_NEW", True),
        avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
        max_tries=_env_int("MAX_TRIES", 20),
        upload_thumb=_env_bool("UPLOAD_THUMB", True),
        set_featured=_env_bool("SET_FEATURED", True),
        embed_image_in_body=_env_bool("EMBED_IMAGE_IN_BODY", True),
        openai_max_retries=_env_int("OPENAI_MAX_RETRIES", 3),
    )

    openai_key = _env("OPENAI_API_KEY", "")
    openai_model = _env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"

    main_kw = _env("NAVER_MAIN_KEYWORD", "")
    keywords_csv = _env("NAVER_KEYWORDS", "")
    prefer_areas = _parse_str_list(_env("PREFER_AREAS", ""))

    pexels_key = _env("PEXELS_API_KEY", "")

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
        naver=NaverStyleConfig(
            main_keyword=main_kw,
            keywords_csv=keywords_csv,
            prefer_areas=prefer_areas,
        ),
        sqlite_path=sqlite_path,
        openai=OpenAIConfig(api_key=openai_key, model=openai_model),
        pexels_api_key=pexels_key,
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new))
    print("[CFG] DRY_RUN:", int(cfg.run.dry_run), "| DEBUG:", int(cfg.run.debug))
    print("[CFG] OPENAI_MODEL:", cfg.openai.model, "| OPENAI_KEY:", ok(cfg.openai.api_key))
    print("[CFG] NAVER_MAIN_KEYWORD:", cfg.naver.main_keyword or "(auto)")
    print("[CFG] NAVER_KEYWORDS:", cfg.naver.keywords_csv or "(empty)")
    print("[CFG] PREFER_AREAS:", ",".join(cfg.naver.prefer_areas) if cfg.naver.prefer_areas else "(any)")
    print("[CFG] PEXELS_API_KEY:", ok(cfg.pexels_api_key))


# -----------------------------
# SQLite (history)
# -----------------------------
DB_SQL = """
CREATE TABLE IF NOT EXISTS daily_posts_v2 (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT,
  recipe_id TEXT,
  recipe_title_en TEXT,
  recipe_title_ko TEXT,
  wp_post_id INTEGER,
  wp_link TEXT
)
"""


def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(DB_SQL)
    con.commit()
    con.close()


def save_post_meta_v2(
    path: str,
    recipe_id: str,
    recipe_title_en: str,
    recipe_title_ko: str,
    wp_post_id: int,
    wp_link: str,
) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO daily_posts_v2(created_at, recipe_id, recipe_title_en, recipe_title_ko, wp_post_id, wp_link) VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            recipe_id or "",
            recipe_title_en or "",
            recipe_title_ko or "",
            int(wp_post_id),
            wp_link or "",
        ),
    )
    con.commit()
    con.close()


def get_recent_recipe_ids(path: str, days: int) -> List[str]:
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        con = sqlite3.connect(path)
        cur = con.cursor()
        cur.execute("SELECT recipe_id FROM daily_posts_v2 WHERE created_at >= ? AND recipe_id IS NOT NULL", (cutoff,))
        rows = cur.fetchall()
        con.close()
        return [str(r[0]) for r in rows if r and r[0]]
    except Exception:
        return []


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "naverstyle-recipe-bot/1.0"}


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

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:600]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, image_url: str, filename_hint: str = "thumb.jpg") -> Tuple[int, str]:
    media_endpoint = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass).copy()

    r = requests.get(image_url, timeout=60)
    if r.status_code != 200 or not r.content:
        raise RuntimeError(f"Image download failed: {r.status_code} url={image_url}")

    content = r.content
    ctype = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip().lower()
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", filename_hint).strip("-") or "thumb.jpg"
    if "." not in safe_name:
        safe_name += ".jpg"

    headers["Content-Type"] = ctype
    headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    up = requests.post(media_endpoint, headers=headers, data=content, timeout=90)
    if up.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {up.status_code} body={up.text[:600]}")

    data = up.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# TheMealDB
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
        "title_en": title,
        "category_en": category,
        "area_en": area,
        "instructions_en": instructions,
        "ingredients_en": ingredients,
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
# Pexels thumbnail
# -----------------------------
def pexels_pick_photo_url(api_key: str, query: str, debug: bool = False) -> Optional[str]:
    if not api_key:
        return None
    q = (query or "").strip()
    if not q:
        return None
    try:
        headers = {"Authorization": api_key}
        params = {"query": q, "per_page": 10, "orientation": "landscape"}
        r = requests.get(PEXELS_SEARCH, headers=headers, params=params, timeout=25)
        if r.status_code != 200:
            if debug:
                print("[PEXELS] non-200:", r.status_code, r.text[:200])
            return None
        j = r.json()
        photos = j.get("photos") or []
        if not photos:
            return None
        p = random.choice(photos)
        src = (p.get("src") or {})
        for k in ("large2x", "original", "large", "medium"):
            u = (src.get(k) or "").strip()
            if u:
                return u
        return None
    except Exception as e:
        if debug:
            print("[PEXELS] failed:", repr(e))
        return None


# -----------------------------
# Safe HTML text-node processing (태그 보호)
# -----------------------------
def _split_html_keep_tags(html: str) -> List[str]:
    return TAG_SPLIT_RE.split(html or "")


def _html_text_only(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "")


def _has_english_in_text(html: str) -> bool:
    return bool(re.search(r"[A-Za-z]", _html_text_only(html)))


def _text_remove_bullets_and_stepnums(s: str) -> str:
    s = re.sub(r"[•·●◦▪▫■□▶►➤]\s*", "", s)
    s = re.sub(r"\b\d+\s*단계\b", "", s)
    s = re.sub(r"\bstep\s*\d+\b", "", s, flags=re.I)
    s = re.sub(r"\b\d+\)\s*", "", s)
    s = re.sub(r"^\s*\d+\s*[):-]\s*", "", s, flags=re.M)
    return s


def _text_remove_periods(s: str) -> str:
    # 숫자 소수점은 유지하고 문장 마침표 성격만 제거
    s = re.sub(r"(?<!\d)\.(?!\d)", "", s)
    s = s.replace("。", "")
    return s


def _text_remove_english_words(s: str) -> str:
    # 2글자 이상 영단어만 제거(ml, g 같은 단위는 남을 수 있음)
    s = re.sub(r"\b[A-Za-z]{2,}\b", "", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def clean_html_text_nodes(html: str) -> str:
    parts = _split_html_keep_tags(html)
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        if p.startswith("<") and p.endswith(">"):
            out.append(p)  # 태그는 그대로
        else:
            t = p
            t = _text_remove_bullets_and_stepnums(t)
            t = _text_remove_periods(t)
            t = _text_remove_english_words(t)
            out.append(t)
    cleaned = "".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _dedupe_lines_text(s: str) -> str:
    lines = [ln.rstrip() for ln in (s or "").splitlines()]
    out = []
    seen = set()
    for ln in lines:
        key = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", ln).strip())
        if not key:
            out.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    merged = "\n".join(out)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()


def _polite_enough(html: str) -> bool:
    t = _html_text_only(html)
    yo = len(re.findall(r"요(\s|$)", t))
    seumnida = len(re.findall(r"습니다(\s|$)", t))
    return (yo + seumnida) >= 25


# -----------------------------
# OpenAI helpers (들여쓰기/탭 문제 없는 버전)
# -----------------------------
def _openai_call_with_retry(
    client: OpenAI,
    model: str,
    instructions: str,
    input_text: str,
    max_retries: int,
    debug: bool = False,
):
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return client.responses.create(
                model=model,
                instructions=instructions,
                input=input_text,
            )
        except openai.RateLimitError as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] RateLimit -> retry in {sleep_s:.2f}s")
            time.sleep(sleep_s)
        except openai.APIError as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] APIError -> retry in {sleep_s:.2f}s | {repr(e)}")
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            sleep_s = (2 ** attempt) + random.random()
            if debug:
                print(f"[OPENAI] UnknownError -> retry in {sleep_s:.2f}s | {repr(e)}")
            time.sleep(sleep_s)
    if last_err:
        raise last_err
    raise RuntimeError("OpenAI call failed")


# -----------------------------
# Title hooks (4 types random)
# -----------------------------
def make_hook_title(keyword: str) -> str:
    kw = (keyword or "").strip() or "오늘 메뉴"

    benefit = [
        f"{kw}  이번에 이 기준 하나 잡으니까 맛이 달라지더라고요",
        f"{kw}  괜히 어렵게 느껴졌는데  의외로 이 포인트가 전부였어요",
        f"{kw}  한 번만 이렇게 해두면  다음부터는 편하게 갑니다",
    ]
    threat = [
        f"{kw}  여기서 흔들리면  전체가 밍밍해지는 경우가 많아요",
        f"{kw}  이 타이밍 놓치면  식감이 확 달라져서 아쉬워지더라고요",
        f"{kw}  대충 넘어가면  마지막에 간이 안 잡히는 패턴이 자주 나와요",
    ]
    curious = [
        f"{kw}  왜 똑같이 했는데 결과가 갈릴까요  딱 이 지점이에요",
        f"{kw}  평소에 애매했던 포인트  오늘은 기준으로 정리해볼게요",
        f"{kw}  어렵게 느껴지는 이유가 있더라고요  그 부분만 풀어볼게요",
    ]
    compare = [
        f"{kw}  대충 따라가기 vs 기준 잡고 하기  차이가 은근 크게 납니다",
        f"{kw}  같은 재료인데도  순서만 바꾸면 결과가 달라져요",
        f"{kw}  한 번은 급하게  한 번은 기준대로  뭐가 달랐을까요",
    ]

    pool = random.choice([benefit, threat, curious, compare])
    title = random.choice(pool)

    title = _text_remove_periods(_text_remove_english_words(_text_remove_bullets_and_stepnums(title)))
    title = re.sub(r"\s{2,}", " ", title).strip()
    return title


# -----------------------------
# Translate recipe parts (JSON output, KO only)
# -----------------------------
def translate_recipe_to_korean(
    client: OpenAI,
    model: str,
    recipe: Dict[str, Any],
    max_retries: int,
    debug: bool = False,
) -> Dict[str, Any]:
    steps_en = split_steps(recipe.get("instructions_en", ""))

    payload = {
        "title_en": recipe.get("title_en", ""),
        "category_en": recipe.get("category_en", ""),
        "area_en": recipe.get("area_en", ""),
        "ingredients_en": recipe.get("ingredients_en", []),
        "steps_en": steps_en,
    }

    instructions = (
        "너는 번역가다\n"
        "반드시 한국어로만 번역한다\n"
        "영어 단어를 남기지 않는다  고유명사는 한글로 음역한다\n"
        "불릿 기호를 쓰지 않는다\n"
        "1단계 2단계 같은 단계 번호를 만들지 않는다\n"
        "마침표 기호를 쓰지 않는다\n"
        "출력은 JSON만\n"
        "{\"title_ko\":\"...\",\"category_ko\":\"...\",\"area_ko\":\"...\",\"ingredients_ko\":[{\"name_ko\":\"...\",\"measure_ko\":\"...\"}],\"steps_ko\":[\"...\",...]}\n"
    )

    def parse_json(txt: str) -> Dict[str, Any]:
        txt = (txt or "").strip()
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
        obj = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])

        obj["title_ko"] = _text_remove_periods(_text_remove_bullets_and_stepnums(str(obj.get("title_ko", "")))).strip()
        obj["category_ko"] = _text_remove_periods(_text_remove_bullets_and_stepnums(str(obj.get("category_ko", "")))).strip()
        obj["area_ko"] = _text_remove_periods(_text_remove_bullets_and_stepnums(str(obj.get("area_ko", "")))).strip()

        clean_ings = []
        for it in (obj.get("ingredients_ko") or []):
            n = _text_remove_periods(_text_remove_bullets_and_stepnums(str(it.get("name_ko", "")))).strip()
            m = _text_remove_periods(_text_remove_bullets_and_stepnums(str(it.get("measure_ko", "")))).strip()
            if n:
                clean_ings.append({"name_ko": n, "measure_ko": m})
        obj["ingredients_ko"] = clean_ings

        clean_steps = []
        for s in (obj.get("steps_ko") or []):
            s = _text_remove_periods(_text_remove_bullets_and_stepnums(str(s))).strip()
            if s:
                clean_steps.append(s)
        obj["steps_ko"] = clean_steps

        return obj

    resp = _openai_call_with_retry(client, model, instructions, json.dumps(payload, ensure_ascii=False), max_retries, debug)
    obj = parse_json(resp.output_text or "")

    # 영어가 남으면 1회 재번역
    if re.search(r"[A-Za-z]", json.dumps(obj, ensure_ascii=False)):
        if debug:
            print("[KO_CHECK] english remains -> re-translate once")
        payload["retry_note"] = "영어가 남았다  영어를 절대 남기지 말고 다시 번역"
        resp2 = _openai_call_with_retry(client, model, instructions, json.dumps(payload, ensure_ascii=False), max_retries, debug)
        obj = parse_json(resp2.output_text or "")

    return obj


# -----------------------------
# Article generation (homefeed style)
# -----------------------------
def generate_naver_homefeed_article(
    client: OpenAI,
    model: str,
    keyword: str,
    dish: str,
    max_retries: int,
    debug: bool = False,
) -> str:
    payload = {
        "keyword": keyword,
        "dish": dish,
        "rules": {
            "polite_friend_tone": True,
            "intro_chars": "200~300",
            "three_sections": 3,
            "no_bullets": True,
            "no_step_numbers": True,
            "no_periods": True,
            "no_repeat_sentences": True,
        },
        "style": [
            "친구에게 진심 담아 수다떠는 존댓말",
            "마침표 없이 띄어쓰기와 여백으로 호흡 두기",
            "실패 포인트와 상태 기준을 자연스럽게 섞기",
            "체험담 느낌을 내되 실제로 해봤다고 단정하지 않기  보통 겪는 포인트처럼",
            "뻔한 결론 문장 반복 금지",
        ],
    }

    instructions = (
        "너는 네이버 홈피드에 올릴 글을 쓰는 사람이다\n"
        "출력은 HTML만\n"
        "\n"
        "필수 형식\n"
        "도입부는 200~300자 정도\n"
        "굵은 소제목 3개를 반드시 만든다  각 소제목 아래는 충분히 길게\n"
        "친구에게 수다떠는 존댓말로만\n"
        "마침표 기호를 쓰지 마\n"
        "불릿 기호 금지\n"
        "1단계 2단계 같은 단계번호 금지\n"
        "특정 문장 반복 금지\n"
        "\n"
        "출력 구조\n"
        "<p>도입</p>\n"
        "<h2><b>소제목1</b></h2>\n"
        "<p>본문</p>\n"
        "<h2><b>소제목2</b></h2>\n"
        "<p>본문</p>\n"
        "<h2><b>소제목3</b></h2>\n"
        "<p>본문</p>\n"
    )

    def ask(note: str = "") -> str:
        inp = dict(payload)
        if note:
            inp["retry_note"] = note
        resp = _openai_call_with_retry(
            client=client,
            model=model,
            instructions=instructions,
            input_text=json.dumps(inp, ensure_ascii=False),
            max_retries=max_retries,
            debug=debug,
        )
        html = (resp.output_text or "").strip()
        html = _dedupe_lines_text(html)
        html = clean_html_text_nodes(html)
        return html

    html = ask()

    # 2회 정도 보정 시도
    for _ in range(2):
        plain_len = len(_html_text_only(html))
        if not _polite_enough(html):
            html = ask("반말이 섞였다  친구에게 수다떠는 존댓말로만 다시")
            continue
        if plain_len < 2600:  # 최소 2300보다 여유 있게
            html = ask("분량이 부족하다  도입과 각 소제목 아래를 더 길게  여백과 호흡 유지")
            continue
        if _has_english_in_text(html):
            html = ask("영어가 남았다  한국어만 사용  고유명사는 한글 음역")
            continue
        break

    return html.strip()


# -----------------------------
# Ingredient / Steps rendering
# -----------------------------
def render_ingredients_block(ingredients_ko: List[Dict[str, str]]) -> str:
    lines = []
    for it in ingredients_ko or []:
        n = (it.get("name_ko") or "").strip()
        m = (it.get("measure_ko") or "").strip()
        if not n:
            continue
        line = f"{n}  {m}".strip() if m else n
        line = _text_remove_periods(_text_remove_english_words(_text_remove_bullets_and_stepnums(line)))
        line = re.sub(r"\s{2,}", " ", line).strip()
        lines.append(f"<p style='margin:0 0 12px 0;line-height:2.0;'>{_html.escape(line)}</p>")
    if not lines:
        return "<p style='margin:0 0 12px 0;line-height:2.0;'>재료는 집에 있는 것 기준으로 편하게 맞추셔도 됩니다</p>"
    return "<div style='margin:0 0 8px 0;'>" + "\n".join(lines) + "</div>"


def render_steps_block(steps_ko: List[str]) -> str:
    blocks = []
    for s in steps_ko or []:
        s = str(s).strip()
        if not s:
            continue
        s = _text_remove_periods(_text_remove_english_words(_text_remove_bullets_and_stepnums(s)))
        s = re.sub(r"\s{2,}", " ", s).strip()
        blocks.append(f"<p style='margin:0 0 16px 0;line-height:2.05;'>{_html.escape(s)}</p>")
    if not blocks:
        return "<p style='margin:0 0 16px 0;line-height:2.05;'>만드는 흐름은 제공된 순서대로  상태 보면서 진행하시면 됩니다</p>"
    return "<div>" + "\n".join(blocks) + "</div>"


def tags_html(keywords_csv: str, main_keyword: str, dish_keyword: str) -> str:
    kws = []
    for k in [main_keyword, dish_keyword]:
        k = (k or "").strip()
        if k:
            kws.append(k)
    for x in (keywords_csv or "").split(","):
        x = x.strip()
        if x and x not in kws:
            kws.append(x)
    kws = kws[:12]
    if not kws:
        return ""
    line = " ".join([f"#{k.replace(' ', '')}" for k in kws if k])
    return f"<div style='margin:22px 0 0 0;opacity:.82;font-size:14px;line-height:1.9;'>태그  {_html.escape(line)}</div>"


def closing_random(dish: str) -> str:
    d = (dish or "").strip() or "이 메뉴"
    pool = [
        f"오늘은 {d}를 완벽하게 하려는 것보다  실패 확률만 줄이는 기준으로 정리해봤어요  이렇게 기준만 잡아두면  다음에는 훨씬 마음이 편해지더라고요",
        f"{d}는 한 번만 감이 잡히면  그 다음부터는 손이 빨라져요  다음번에는 내 입맛에 맞는 포인트 하나만 살짝 조절해보셔도 좋아요",
        f"요리는 정답 찾기보다  내가 반복 가능한 방식으로 가져가는 게 제일 오래 가더라고요  {d}도 그렇게 편하게 가져가시면 됩니다",
        f"메뉴 고민될 때  이런 글 하나가 있으면 진짜 편해요  {d}는 특히 이 기준만 기억해두셔도  결과가 안정됩니다",
    ]
    t = random.choice(pool)
    t = _text_remove_periods(_text_remove_english_words(_text_remove_bullets_and_stepnums(t)))
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def wrap_html(inner: str) -> str:
    inner = (inner or "").strip()
    return (
        "<div style='max-width:760px;margin:0 auto;padding:6px 4px;'>\n"
        f"{inner}\n"
        "</div>"
    )


# -----------------------------
# Pick recipe (avoid repeats)
# -----------------------------
def pick_recipe(cfg: AppConfig) -> Dict[str, Any]:
    recent_ids = set(get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days))
    prefer = set([a.lower() for a in (cfg.naver.prefer_areas or [])])

    for _ in range(max(1, cfg.run.max_tries)):
        cand = fetch_random_recipe()
        rid = (cand.get("id") or "").strip()
        if not rid:
            continue
        if rid in recent_ids:
            continue
        if prefer:
            area = (cand.get("area_en") or "").strip().lower()
            if area and (area not in prefer):
                continue
        return cand

    return fetch_random_recipe()


# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = _now_kst()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H%M%S")

    init_db(cfg.sqlite_path)

    recipe = pick_recipe(cfg)
    recipe_id = recipe.get("id", "")
    recipe_title_en = recipe.get("title_en", "") or "Daily Recipe"

    client = OpenAI(api_key=cfg.openai.api_key)

    # 1) 번역(재료/만드는 흐름/레시피명 번역 필수)
    recipe_ko = translate_recipe_to_korean(
        client=client,
        model=cfg.openai.model,
        recipe=recipe,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    dish_ko = (recipe_ko.get("title_ko") or "").strip() or "오늘 메뉴"

    # 2) 제목 키워드
    main_kw = (cfg.naver.main_keyword or "").strip()
    title_kw = main_kw if main_kw else dish_ko

    # 3) 제목 4유형 랜덤
    hook_title = make_hook_title(title_kw)

    # 4) 홈피드형 본문 생성(수다 존댓말)
    article_html = generate_naver_homefeed_article(
        client=client,
        model=cfg.openai.model,
        keyword=title_kw,
        dish=dish_ko,
        max_retries=cfg.run.openai_max_retries,
        debug=cfg.run.debug,
    )

    # 5) 재료/만드는 흐름/마무리/태그는 "맨 뒤"에 고정 추가
    ingredients_block = render_ingredients_block(recipe_ko.get("ingredients_ko", []))
    steps_block = render_steps_block(recipe_ko.get("steps_ko", []))

    tail: List[str] = []
    tail.append("<h2><b>재료 목록</b></h2>")
    tail.append(ingredients_block)
    tail.append("<h2><b>만드는 흐름</b></h2>")
    tail.append(steps_block)
    tail.append("<h2><b>마무리</b></h2>")
    tail.append(f"<p style='margin:0 0 14px 0;line-height:2.0;'>{_html.escape(closing_random(dish_ko))}</p>")
    tail.append(tags_html(cfg.naver.keywords_csv, main_kw, dish_ko))

    full_body = (article_html.strip() + "\n\n" + "\n".join(tail)).strip()

    # 최종 후처리: 태그/스타일은 보호하고 텍스트만 정리(깨짐 방지)
    full_body = _dedupe_lines_text(full_body)
    full_body = clean_html_text_nodes(full_body)

    # 6) 썸네일 (Pexels 우선, 없으면 MealDB thumb)
    pex_query = f"{dish_ko} 음식"
    pex_url = pexels_pick_photo_url(cfg.pexels_api_key, pex_query, debug=cfg.run.debug)
    thumb_url = (pex_url or (recipe.get("thumb") or "").strip())

    media_id: Optional[int] = None
    media_url: str = ""
    if cfg.run.upload_thumb and thumb_url:
        try:
            media_id, media_url = wp_upload_media(cfg.wp, thumb_url, filename_hint=f"thumb-{date_str}-{time_str}.jpg")
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] media upload failed:", repr(e))

    # 7) 본문 상단 이미지 삽입
    if cfg.run.embed_image_in_body:
        img = (media_url or thumb_url or "").strip()
        if img:
            img_tag = (
                "<p style='margin:0 0 20px 0;'>"
                f"<img src='{_html.escape(img)}' alt='{_html.escape(hook_title)}' "
                "style='max-width:100%;height:auto;border-radius:14px;'/>"
                "</p>"
            )
            full_body = img_tag + "\n" + full_body

    full_html = wrap_html(full_body)

    # 8) 항상 새 글 발행: slug에 시간+랜덤
    rand6 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6))
    slug = f"naverstyle-{date_str}-{cfg.run.run_slot}-{time_str}-{rand6}"

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print("TITLE:", hook_title)
        print("SLUG:", slug)
        print(full_html[:2500] + ("\n...(truncated)" if len(full_html) > 2500 else ""))
        return

    featured = media_id if (cfg.run.set_featured and media_id) else None
    post_id, link = wp_create_post(cfg.wp, hook_title, slug, full_html, featured_media=featured)
    save_post_meta_v2(cfg.sqlite_path, recipe_id, recipe_title_en, dish_ko, post_id, link)
    print("OK(created):", post_id, link)


def main():
    cfg = load_cfg()
    print_safe_cfg(cfg)
    validate_cfg(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
