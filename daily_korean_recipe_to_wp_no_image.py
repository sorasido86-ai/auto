# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp_no_image.py (네이버형 홈피드 톤 + 레파토리 랜덤화 + 이미지 완전 제거)
- "한식 레시피만" 매일 자동 업로드 (WordPress)
- 1순위: 식품안전나라(식약처) COOKRCP01 OpenAPI 레시피 DB (MFDS_API_KEY 필요)
- 2순위(폴백): 내장 한식 레시피(한국어)

핵심 요구 반영
- 레시피(재료/순서)는 항상 포함
- 특수문자(불릿/체크/점찍힌 문자) 제거: ul/li/•/✅/※/－ 등 사용 안 함
- 해시태그 12~20개 자동 생성
- 중복 업로드 허용: ALLOW_DUPLICATE_POSTS=1 이면 매번 새 글 생성
- 글자수: MIN_TOTAL_CHARS(기본 1200) 이상 보장 + MAX_TOTAL_CHARS(기본 1900) 넘으면 자동 축약
- 이미지 완전 제거(업로드/삽입/대표이미지 모두 안 함)
- 출처/기준시각/슬롯/자동생성 문구 제거

필수 환경변수(Secrets):
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

권장 환경변수:
  - WP_STATUS=publish
  - WP_CATEGORY_IDS=7
  - WP_TAG_IDS= (선택)
  - SQLITE_PATH=data/daily_korean_recipe.sqlite3
  - MFDS_API_KEY=... (없으면 내장 레시피만)
  - RUN_SLOT=day|am|pm
  - FORCE_NEW=0|1 (기본 0)
  - ALLOW_DUPLICATE_POSTS=0|1 (기본 1)
  - AVOID_REPEAT_DAYS=90
  - MAX_TRIES=20
  - MFDS_TIMEOUT_SEC=10
  - MFDS_BUDGET_SEC=18
  - MIN_TOTAL_CHARS=1200
  - MAX_TOTAL_CHARS=1900
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
# 내장 한식 레시피(폴백)
# -----------------------------
LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
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
            "냄비에 돼지고기를 넣고 중불에서 기름이 살짝 돌 때까지 볶아요",
            "김치를 넣고 2분에서 3분 더 볶아서 신맛을 살짝 눌러요",
            "고춧가루 다진마늘 국간장을 넣고 30초만 볶아서 향을 내요",
            "육수를 붓고 10분에서 12분 끓여요",
            "양파를 넣고 3분 두부를 넣고 2분 더 끓이고 대파로 마무리해요",
        ],
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1큰술 반"),
            ("고추장", "반 큰술 선택"),
            ("애호박", "1/3개"),
            ("양파", "1/3개"),
            ("두부", "1/2모"),
            ("대파", "1/2대"),
            ("다진마늘", "1작은술"),
            ("육수 또는 물", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 5분 정도 끓여요",
            "양파 애호박 두부를 넣고 5분에서 6분 더 끓여요",
            "대파를 넣고 한 번만 더 끓인 뒤 간을 보고 마무리해요",
        ],
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
            "간장 설탕 다진마늘 참기름 물 후추를 섞어서 양념장을 만들어요",
            "고기에 양념장을 넣고 15분 이상 재워요",
            "팬에 고기를 볶다가 양파 대파를 넣고 숨이 죽을 때까지 볶아요",
        ],
    },
]

KOREAN_NEGATIVE_KEYWORDS = ["파스타", "피자", "타코", "스시", "커리", "샌드위치", "버거", "샐러드"]

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
    run_slot: str = "day"
    force_new: bool = False
    allow_duplicate_posts: bool = True
    dry_run: bool = False
    debug: bool = False
    avoid_repeat_days: int = 90
    max_tries: int = 20
    mfds_timeout_sec: int = 10
    mfds_budget_sec: int = 18
    min_total_chars: int = 1200
    max_total_chars: int = 1900

@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""
    strict_korean: bool = True

@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    recipe: RecipeSourceConfig
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
        ),
        run=RunConfig(
            run_slot=run_slot,
            force_new=_env_bool("FORCE_NEW", False),
            allow_duplicate_posts=_env_bool("ALLOW_DUPLICATE_POSTS", True),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 90),
            max_tries=_env_int("MAX_TRIES", 20),
            mfds_timeout_sec=_env_int("MFDS_TIMEOUT_SEC", 10),
            mfds_budget_sec=_env_int("MFDS_BUDGET_SEC", 18),
            min_total_chars=_env_int("MIN_TOTAL_CHARS", 1200),
            max_total_chars=_env_int("MAX_TOTAL_CHARS", 1900),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
            strict_korean=_env_bool("STRICT_KOREAN", True),
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
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| FORCE_NEW:", int(cfg.run.force_new), "| ALLOW_DUPLICATE_POSTS:", int(cfg.run.allow_duplicate_posts))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| STRICT_KOREAN:", cfg.recipe.strict_korean)
    print("[CFG] MFDS_TIMEOUT_SEC:", cfg.run.mfds_timeout_sec, "| MFDS_BUDGET_SEC:", cfg.run.mfds_budget_sec)
    print("[CFG] MIN_TOTAL_CHARS:", cfg.run.min_total_chars, "| MAX_TOTAL_CHARS:", cfg.run.max_total_chars)

# -----------------------------
# SQLite (history)
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

def get_recent_recipe_ids(sqlite_path: str, days: int) -> List[Tuple[str, str]]:
    since = datetime.utcnow() - timedelta(days=days)
    con = sqlite3.connect(sqlite_path)
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

def save_post_meta(sqlite_path: str, meta: Dict[str, Any]) -> None:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.get("date_slot", ""),
            meta.get("recipe_source", ""),
            meta.get("recipe_id", ""),
            meta.get("recipe_title", ""),
            int(meta.get("wp_post_id", 0) or 0),
            meta.get("wp_link", ""),
            meta.get("created_at", datetime.utcnow().isoformat()),
        ),
    )
    con.commit()
    con.close()

# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/2.2"}

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
    return int(data["id"]), str(data.get("link") or "")

# -----------------------------
# Recipe model / MFDS provider
# -----------------------------
@dataclass
class Recipe:
    source: str
    recipe_id: str
    title: str
    ingredients: List[str]
    steps: List[str]

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

def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int, end: int, timeout_sec: int) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    r = requests.get(url, timeout=(5, max(3, int(timeout_sec))))
    if r.status_code != 200:
        return []
    data = r.json()
    co = data.get("COOKRCP01") or {}
    rows = co.get("row") or []
    return rows if isinstance(rows, list) else []

def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or ""
    title = str(row.get("RCP_NM") or "").strip()

    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()
    ingredients: List[str] = []
    for p in re.split(r"\s*,\s*", parts):
        p = p.strip()
        if p:
            ingredients.append(p)

    steps: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        if s:
            s = re.sub(r"[a-zA-Z]\s*$", "", s).strip()
            if len(s) >= 2:
                steps.append(s)

    return Recipe(
        source="mfds",
        recipe_id=rid or hashlib.sha1(title.encode("utf-8")).hexdigest()[:8],
        title=title,
        ingredients=ingredients,
        steps=steps,
    )

def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡", "갈비", "무침"]
    t0 = time.time()

    for _ in range(max(3, int(cfg.run.max_tries))):
        if (time.time() - t0) > max(8, int(cfg.run.mfds_budget_sec)):
            return None

        kw = random.choice(keywords)
        try:
            rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, 1, 60, cfg.run.mfds_timeout_sec)
        except Exception:
            continue

        random.shuffle(rows)
        for row in rows[:40]:
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
    pool = [x for x in LOCAL_KOREAN_RECIPES if ("local", str(x["id"])) not in used]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    pick = random.choice(pool)
    ing = [f"{a} {b}".strip() for a, b in pick.get("ingredients", [])]
    steps = [str(s).strip() for s in pick.get("steps", []) if str(s).strip()]
    return Recipe(source="local", recipe_id=str(pick["id"]), title=str(pick["title"]), ingredients=ing, steps=steps)

# -----------------------------
# Text / HTML builders (no bullets, no periods)
# -----------------------------
def _esc(s: str) -> str:
    return html.escape(s or "")

def _strip_html(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text or "")
    s = s.replace("&nbsp;", " ").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _no_periods(s: str) -> str:
    return (s or "").replace(".", "").replace("!", "")

def _safe_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for x in lines:
        x = (x or "").strip()
        if not x:
            continue
        x = x.replace("•", " ").replace("✅", " ").replace("※", " ").replace("·", " ")
        x = x.replace("—", " ").replace("–", " ").replace("ㆍ", " ")
        x = _no_periods(x)
        x = re.sub(r"\s+", " ", x).strip()
        if x:
            out.append(x)
    return out

def _pick(pool: List[str]) -> str:
    return random.choice(pool).strip()

def normalize_recipe_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("•", " ").replace("✅", " ").replace("※", " ").replace("·", " ")
    s = s.replace("\u2022", " ").replace("\u25cf", " ").replace("\u00b7", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return _no_periods(s)

def build_intro(title: str) -> List[str]:
    hooks = [
        f"왜인지 {title}는 자꾸 생각나는 날이 있어요",
        f"요즘은 이상하게 {title} 같은 메뉴가 더 끌리더라고요",
        f"{title}는 한 번 해먹고 나면 다음에 또 찾게 되는 편이에요",
        f"오늘 같은 날에는 {title} 쪽으로 마음이 가요",
    ]
    moods = [
        "밥 한 끼가 그냥 끼니가 아니라 마음 정리 같을 때가 있잖아요",
        "하루가 정신없으면 음식도 복잡한 건 피하게 되더라고요",
        "컨디션이 애매할수록 익숙한 맛이 제일 편해요",
        "냉장고 열어봤는데 딱히 떠오르는 게 없을 때가 있잖아요",
    ]
    personal = [
        "저도 예전에 급하게 하다가 순서가 꼬여서 맛이 애매했던 날이 있어요",
        "처음엔 대충 했다가 간이 흔들려서 아쉬웠던 적이 있었어요",
        "불을 왔다 갔다 하다가 끝맛이 거칠어져서 속상했던 적도 있고요",
        "양념을 한 번에 세게 넣었다가 향이 뭉개진 느낌이 난 적이 있어요",
    ]
    resolve = [
        "근데 포인트만 정해두니까 훨씬 편해졌어요",
        "그 뒤로는 실패가 확 줄더라고요",
        "요즘은 부담 없이 해먹는 방식으로 고정해뒀어요",
        "저는 이 흐름으로 하면 마음이 편해서 자주 하게 돼요",
    ]
    rules = [
        "불은 중간불로 안정적으로 가져가고",
        "간은 끝에서만 한 번 보고",
        "양념은 조금씩 올리고",
        "순서는 욕심내지 말고 그대로 따라가요",
    ]
    close = [
        "오늘은 그 포인트만 깔끔하게 남겨둘게요",
        "읽으시다가 마음에 드는 부분만 가져가셔도 충분해요",
        "부담 없이 따라가도 되는 버전으로요",
        "진짜 해먹을 수 있게만 정리해볼게요",
    ]
    lines = [_pick(hooks), _pick(moods), _pick(personal), _pick(resolve), _pick(rules), _pick(close)]
    return _safe_lines(lines)

def build_story_block(title: str) -> List[str]:
    vibe_blocks = [
        ["퇴근하고 나면 머리가 멍할 때가 있잖아요", "그럴 때는 손이 익은 메뉴가 제일 든든해요"],
        ["비 오는 날엔 따뜻한 맛이 더 당기더라고요", "끓는 소리 듣고 있으면 마음이 좀 풀리기도 하고요"],
        ["입맛이 왔다 갔다 할 때는 메뉴 고민부터 피곤해요", "그럴수록 검증된 흐름이 편하더라고요"],
        ["냉장고에 재료가 애매하게 남았을 때", "그럴 때 이런 메뉴가 진짜 효자 같아요"],
        ["주말에 한 번 해두면 평일이 훨씬 편해요", "남은 재료도 깔끔하게 정리되고요"],
    ]
    mistakes = [
        "저는 중간에 간을 확 잡아버렸다가 나중에 졸아들면서 짜졌던 적이 있어요",
        "불이 너무 세면 향은 좋아도 끝맛이 거칠어질 때가 있더라고요",
        "재료를 한 번에 다 넣으면 익는 타이밍이 엇갈려서 식감이 아쉬워요",
        "양념을 크게 한 번에 넣으면 내 입맛이 어디로 가는지 놓치게 돼요",
    ]
    learn = [
        "그래서 저는 마지막에만 간을 봐요 그러면 마음이 진짜 편해져요",
        "불은 중간불로 안정적으로 가는 쪽이 결과가 더 깔끔하더라고요",
        "순서만 지키면 맛이 크게 안 흔들린다는 걸 알게 됐어요",
        "완벽하게 하려는 마음 내려놓으니까 오히려 더 잘 되더라고요",
    ]
    extras = [
        "양파나 대파 같은 기본 재료가 있으면 맛이 훨씬 안정적이에요",
        "육수는 있으면 좋고 없으면 물로 해도 괜찮아요",
        "다만 간은 늦게 잡는 게 훨씬 안전해요",
        "중간에 조급해지면 한 번 숨 고르고 불부터 확인하는 게 도움이 돼요",
        "향이 딱 좋아지기 시작할 때가 있잖아요 그때가 타이밍이더라고요",
        "소리가 일정해지면 그때부터는 흔들지 않는 게 좋아요",
    ]
    comfort = [
        "맛이 한 번만 잡히면 다음부터는 몸이 기억하더라고요",
        "그래서 저는 어렵게 생각 안 하고 그냥 이 흐름으로 갑니다",
        "저장해두고 필요할 때 꺼내 쓰기 좋은 느낌이에요",
    ]

    lines: List[str] = []
    lines += random.choice(vibe_blocks)
    lines.append(_pick(mistakes))
    lines.append(_pick(learn))
    lines += random.sample(extras, k=random.randint(2, 4))
    lines.append(_pick(comfort))
    return _safe_lines(lines)

def build_recipe_block(recipe: Recipe) -> Tuple[List[str], List[str]]:
    ings = [normalize_recipe_text(x) for x in (recipe.ingredients or []) if normalize_recipe_text(x)]
    if not ings:
        ings = ["재료는 집에 있는 것 위주로 준비해도 좋아요", "양파 대파 마늘 같은 기본 재료가 있으면 훨씬 편해요"]
    if len(ings) > 16:
        ings = ings[:16]

    steps = [normalize_recipe_text(x) for x in (recipe.steps or []) if normalize_recipe_text(x)]
    if not steps:
        steps = ["재료 손질부터 차분히 시작해요", "불 조절은 중간불로 안정적으로 가요", "간은 마지막에만 한 번 보고 마무리해요"]
    if len(steps) > 10:
        steps = steps[:10]

    steps_num = [f"{i+1} {s}" for i, s in enumerate(steps)]
    return ings, steps_num

def build_tips_block(title: str) -> List[str]:
    intros = [
        "여기부터는 제가 해먹으면서 남긴 메모 같은 거예요",
        "맛이 흔들릴 때 확인할 포인트만 모아둘게요",
        "실패 줄이는 쪽으로만 간단히 정리해볼게요",
    ]
    cores = [
        "간은 마지막에만 살짝 조절하는 게 제일 안전해요",
        "불은 중간불로 안정적으로 가는 게 결과가 깔끔해요",
        "양념은 처음부터 세게 말고 끝에서 조금씩 올리는 쪽이 편해요",
    ]
    pool = [
        "향이 아쉬우면 대파나 마늘을 마지막에 아주 조금만 더해도 확 살아나요",
        "볶음은 불이 너무 세면 금방 타면서 맛이 거칠어져요",
        "단맛이 필요하면 한 번에 올리지 말고 아주 소량씩만 더해보세요",
        "아이랑 같이 먹을 땐 매운 양념을 줄이고 대신 육수나 간장으로 맞추면 편해요",
        "버섯이나 양파를 더하면 실패가 적고 향이 안정적이에요",
        "다음날 먹을 거면 처음부터 간을 세게 잡지 않는 게 더 좋아요",
        "기름이 부담이면 양을 줄이고 대신 물이나 육수로 농도를 맞춰도 좋아요",
        "냄새가 딱 좋아지기 시작할 때가 있잖아요 그때가 타이밍이더라고요",
    ]
    questions = [
        f"{title} 하실 때 어떤 재료를 더 넣는 편이세요",
        f"{title}는 매운맛 올리는 쪽이 좋으세요 담백한 쪽이 좋으세요",
        f"{title} 만들 때 가장 어려운 포인트가 뭐예요",
    ]
    closes = [
        "한 번만 성공하면 다음부터는 진짜 쉬워져요",
        "오늘 해보시고 간과 불만 메모해두시면 다음번이 훨씬 편해요",
        "해보시고 느낌 한 줄만 남겨주시면 저도 다음 메뉴 고를 때 도움이 돼요",
    ]
    lines = [_pick(intros), _pick(cores)]
    lines += random.sample(pool, k=random.randint(3, 5))
    lines += [_pick(questions), _pick(closes)]
    return _safe_lines(lines)

def extract_tokens(text: str) -> List[str]:
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", text or "")
    toks = [x.strip() for x in t.split() if x.strip()]
    out: List[str] = []
    seen = set()
    for x in toks:
        if len(x) <= 1:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def build_hashtags(recipe: Recipe, ing_lines: List[str], min_n: int, max_n: int) -> str:
    base = [
        "한식레시피", "집밥", "오늘뭐먹지", "간단요리", "자취요리",
        "초간단레시피", "밥도둑", "국물요리", "반찬", "요리기록",
        "저녁메뉴", "집밥일기", "레시피공유", "한식", "요리",
        "맛있는집밥", "집밥메뉴", "혼밥", "가정식",
    ]
    toks = extract_tokens(recipe.title)
    for x in ing_lines[:8]:
        toks += extract_tokens(x)
    random.shuffle(toks)

    candidates: List[str] = []
    candidates.extend(base)
    for tok in toks:
        tok = tok.replace(" ", "").replace("·", "")
        if tok:
            candidates.append(tok)

    uniq: List[str] = []
    seen = set()
    for x in candidates:
        x = re.sub(r"\s+", "", x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    n = random.randint(max(min_n, 12), min(max_n, 20))
    return " ".join([f"#{x}" for x in uniq[:n]])

def ensure_length(cfg: AppConfig, html_body: str, filler_lines: List[str]) -> str:
    min_chars = max(400, int(cfg.run.min_total_chars))
    max_chars = max(min_chars + 200, int(cfg.run.max_total_chars))

    body = html_body
    tries = 0
    while len(_strip_html(body)) < min_chars and tries < 30:
        tries += 1
        extra = _pick(filler_lines)
        body = body.replace("<!--FILLER-->", f"<p>{_esc(extra)}</p>\n<!--FILLER-->")
    body = body.replace("<!--FILLER-->", "")

    if len(_strip_html(body)) > max_chars:
        parts = re.split(r"(?i)</p>\s*", body)
        parts = [p for p in parts if p.strip()]
        rebuilt = []
        for p in parts:
            candidate = "</p>\n".join(rebuilt + [p]) + "</p>"
            if len(_strip_html(candidate)) > max_chars:
                break
            rebuilt.append(p)
        if rebuilt:
            body = "</p>\n".join(rebuilt) + "</p>"
        txt = _strip_html(body)
        if len(txt) > max_chars:
            txt = txt[:max_chars].strip()
            body = "<p>" + _esc(txt).replace("\n", "<br/>") + "</p>"
    return body

def build_body_html(cfg: AppConfig, recipe: Recipe) -> Tuple[str, str]:
    seed_src = f"{recipe.source}|{recipe.recipe_id}|{datetime.now(tz=KST).strftime('%Y%m%d%H%M%S')}"
    random.seed(hashlib.sha1(seed_src.encode("utf-8")).hexdigest())

    title = normalize_recipe_text(recipe.title)

    intro_lines = build_intro(title)
    story_lines = build_story_block(title)
    ing_lines, step_lines = build_recipe_block(recipe)
    tips_lines = build_tips_block(title)
    hashtags = build_hashtags(recipe, ing_lines, 12, 20)

    h1 = "<p><strong>왜 이 레시피가 자꾸 생각나는지</strong></p>"
    p1 = "<p>" + "<br/>".join([_esc(x) for x in intro_lines]) + "</p>"

    h2 = "<p><strong>제가 해먹으면서 편해졌던 흐름</strong></p>"
    p2 = "<p>" + "<br/>".join([_esc(x) for x in story_lines]) + "</p>"

    h3 = "<p><strong>레시피</strong></p>"
    recipe_text = ["재료"] + ing_lines + ["만드는 순서"] + step_lines
    p3 = "<p>" + "<br/>".join([_esc(x) for x in _safe_lines(recipe_text)]) + "</p>"

    h4 = "<p><strong>실패 줄이는 팁과 응용</strong></p>"
    p4 = "<p>" + "<br/>".join([_esc(x) for x in tips_lines]) + "</p>"

    h5 = "<p><strong>해시태그</strong></p>"
    p5 = "<p>" + _esc(hashtags) + "</p>"

    excerpt = f"{title} 레시피 재료와 순서만 깔끔하게 정리했어요"
    excerpt = _no_periods(excerpt)[:140]

    body = "\n".join([h1, p1, h2, p2, h3, p3, h4, p4, "<!--FILLER-->", h5, p5])

    filler_pool = _safe_lines([
        f"{title}는 재료가 조금 달라도 흐름만 지키면 맛이 크게 안 흔들리더라고요",
        "중간에 조급해지면 불부터 확인하고 한 번 숨 고르는 게 도움이 돼요",
        "저는 다음번을 위해 간과 불만 메모해두는 편인데 그게 진짜 편해요",
        "처음부터 완벽하게 하려는 마음을 내려놓으면 오히려 결과가 좋아질 때가 있어요",
        "오늘은 딱 여기까지 해도 충분해요 다음에는 내 입맛 쪽으로만 조금씩 바꿔보셔도 좋아요",
        "해보시고 맛이 마음에 들면 다음엔 재료 하나만 바꿔서 비교해보는 것도 재미있어요",
    ])

    body = ensure_length(cfg, body, filler_pool)
    return body, excerpt

def build_post_title(date_str: str, slot_label: str, recipe_title: str) -> str:
    title = normalize_recipe_text(recipe_title)
    tail = random.choice(["집에서 편하게", "실패 덜하게", "부담 없이", "따라 하기 쉽게"])
    return f"{title} 레시피 {tail} {date_str} {slot_label}"

def build_slug(date_str: str, slot: str, allow_duplicate: bool, recipe: Recipe) -> str:
    base = f"korean-recipe-{date_str}-{slot}"
    if allow_duplicate:
        rid = hashlib.sha1(f"{time.time()}|{recipe.source}|{recipe.recipe_id}|{random.random()}".encode("utf-8")).hexdigest()[:6]
        hhmmss = datetime.now(tz=KST).strftime("%H%M%S")
        return f"{base}-{hhmmss}-{rid}"
    return base

def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = {"day": "오늘", "am": "오전", "pm": "오후"}.get(slot, "오늘")

    init_db(cfg.sqlite_path)
    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, cfg.run.avoid_repeat_days)

    run_id = hashlib.sha1(f"{time.time()}|{random.random()}".encode("utf-8")).hexdigest()[:6]
    date_slot = f"{date_str}_{slot}_{run_id}" if cfg.run.allow_duplicate_posts else f"{date_str}_{slot}"

    print(f"[RUN] slot={slot} allow_dup={int(cfg.run.allow_duplicate_posts)} date_slot={date_slot}")

    chosen: Optional[Recipe] = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(recent_pairs)
    assert chosen is not None

    post_title = build_post_title(date_str, slot_label, chosen.title)
    slug = build_slug(date_str, slot, cfg.run.allow_duplicate_posts, chosen)

    body_html, excerpt = build_body_html(cfg, chosen)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print(post_title)
        print(slug)
        print(body_html[:1600])
        return

    wp_post_id, wp_link = wp_create_post(cfg.wp, post_title, slug, body_html, excerpt=excerpt)
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
