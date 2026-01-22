# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp_no_image.py (홈피드형 글 + 무이미지 안정화)

요구사항 반영
- 레시피(재료/순서) 항상 포함
- 특수문자(불릿/체크표시/점찍힌 문자 등) 완전 제거
- 해시태그 12~20개 자동 생성
- 중복 업로드 허용: ALLOW_DUPLICATE_POSTS=1이면 매번 새 글 생성(업데이트 X)
- 글자수: 최소 1200자 보장, 너무 길어지지 않게 상한(기본 1800자) 적용
- 문장 흐름: 랜덤 문장 “섞기” 대신 문단 템플릿 + 연결어로 자연스럽게

필수 환경변수(Secrets)
- WP_BASE_URL
- WP_USER
- WP_APP_PASS

선택
- MFDS_API_KEY (식품안전나라 COOKRCP01 OpenAPI, 없으면 내장 레시피)
- WP_STATUS=publish
- WP_CATEGORY_IDS=7
- WP_TAG_IDS=1,2,3
- SQLITE_PATH=data/daily_korean_recipe.sqlite3

동작 옵션
- RUN_SLOT=day|am|pm
- ALLOW_DUPLICATE_POSTS=1|0 (기본 1)
- DRY_RUN=1|0
- DEBUG=1|0
- MIN_TOTAL_CHARS=1200
- MAX_TOTAL_CHARS=1800
- HASHTAG_MIN=12
- HASHTAG_MAX=20
- MFDS_TIMEOUT_SEC=8
- MFDS_BUDGET_SEC=15
- MAX_TRIES=18
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

KST = timezone(timedelta(hours=9))

# -----------------------------
# 내장 레시피(폴백)
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
            "신김치를 넣고 2분 정도 더 볶아서 신맛을 한 번 눌러줘요",
            "고춧가루 다진마늘 국간장을 넣고 30초만 볶아 향을 살려요",
            "육수를 붓고 10분 정도 끓여요",
            "양파를 넣고 3분 더 끓인 뒤 두부와 대파로 마무리해요",
        ],
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "1.5큰술"),
            ("고추장", "1/2큰술 선택"),
            ("애호박", "1/3개"),
            ("양파", "1/3개"),
            ("두부", "1/2모"),
            ("대파", "1/2대"),
            ("다진마늘", "1작은술"),
            ("육수 또는 물", "700ml"),
        ],
        "steps": [
            "끓는 육수에 된장을 풀고 5분 정도 먼저 끓여요",
            "양파 애호박을 넣고 4분 정도 더 끓여요",
            "두부를 넣고 2분 정도 끓인 뒤 대파로 마무리해요",
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
            "간장 설탕 다진마늘 참기름 물 후추를 섞어 양념을 만들어요",
            "고기에 양념을 넣고 15분 정도 재워요",
            "팬에 고기를 볶다가 양파 대파를 넣고 숨이 죽을 때까지만 볶아요",
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
    allow_duplicate_posts: bool = True
    dry_run: bool = False
    debug: bool = False
    min_total_chars: int = 1200
    max_total_chars: int = 1800
    hashtag_min: int = 12
    hashtag_max: int = 20
    strict_korean: bool = True
    mfds_timeout_sec: int = 8
    mfds_budget_sec: int = 15
    max_tries: int = 18

@dataclass
class RecipeSourceConfig:
    mfds_api_key: str = ""

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
            allow_duplicate_posts=_env_bool("ALLOW_DUPLICATE_POSTS", True),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            min_total_chars=_env_int("MIN_TOTAL_CHARS", 1200),
            max_total_chars=_env_int("MAX_TOTAL_CHARS", 1800),
            hashtag_min=_env_int("HASHTAG_MIN", 12),
            hashtag_max=_env_int("HASHTAG_MAX", 20),
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout_sec=_env_int("MFDS_TIMEOUT_SEC", 8),
            mfds_budget_sec=_env_int("MFDS_BUDGET_SEC", 15),
            max_tries=_env_int("MAX_TRIES", 18),
        ),
        recipe=RecipeSourceConfig(
            mfds_api_key=_env("MFDS_API_KEY", ""),
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot)
    print("[CFG] ALLOW_DUPLICATE_POSTS:", int(cfg.run.allow_duplicate_posts))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] MFDS_API_KEY:", ok(cfg.recipe.mfds_api_key), "| STRICT_KOREAN:", cfg.run.strict_korean)
    print("[CFG] MFDS_TIMEOUT_SEC:", cfg.run.mfds_timeout_sec, "| MFDS_BUDGET_SEC:", cfg.run.mfds_budget_sec)
    print("[CFG] MIN_TOTAL_CHARS:", cfg.run.min_total_chars, "| MAX_TOTAL_CHARS:", cfg.run.max_total_chars)
    print("[CFG] HASHTAG_MIN/MAX:", cfg.run.hashtag_min, cfg.run.hashtag_max)

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
          created_at TEXT,
          style_sig TEXT
        )
        """
    )
    con.commit()

    cur.execute("PRAGMA table_info(daily_posts)")
    cols = {row[1] for row in cur.fetchall()}
    if "style_sig" not in cols:
        cur.execute("ALTER TABLE daily_posts ADD COLUMN style_sig TEXT")
        con.commit()

    con.close()

def get_recent_style_sigs(sqlite_path: str, limit: int = 20) -> List[str]:
    try:
        con = sqlite3.connect(sqlite_path)
        cur = con.cursor()
        cur.execute(
            """
            SELECT style_sig
            FROM daily_posts
            WHERE style_sig IS NOT NULL AND style_sig != ''
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()
        con.close()
        return [str(r[0]) for r in rows if r and r[0]]
    except Exception:
        return []

def save_post_meta(sqlite_path: str, meta: Dict[str, Any]) -> None:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_slot, recipe_source, recipe_id, recipe_title, wp_post_id, wp_link, created_at, style_sig)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            meta.get("date_slot", ""),
            meta.get("recipe_source", ""),
            meta.get("recipe_id", ""),
            meta.get("recipe_title", ""),
            int(meta.get("wp_post_id", 0) or 0),
            meta.get("wp_link", ""),
            meta.get("created_at", datetime.utcnow().isoformat()),
            meta.get("style_sig", ""),
        ),
    )
    con.commit()
    con.close()

def get_recent_recipe_ids(sqlite_path: str, limit: int = 120) -> List[Tuple[str, str]]:
    # 중복 회피용(최근 N개)
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT recipe_source, recipe_id
        FROM daily_posts
        WHERE recipe_source IS NOT NULL AND recipe_source != ''
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
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
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/1.3"}

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

    r = requests.post(url, headers=headers, json=payload, timeout=40)
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
    for bad in KOREAN_NEGATIVE_KEYWORDS:
        if bad in n:
            return False
    return True

def mfds_fetch_by_param(api_key: str, param: str, value: str, start: int, end: int, timeout_sec: int) -> List[Dict[str, Any]]:
    base = f"https://openapi.foodsafetykorea.go.kr/api/{api_key}/COOKRCP01/json/{start}/{end}"
    url = f"{base}/{param}={quote(value)}"
    try:
        r = requests.get(url, timeout=timeout_sec)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []
    co = data.get("COOKRCP01") or {}
    rows = co.get("row") or []
    return rows if isinstance(rows, list) else []

def _sanitize_text(s: str) -> str:
    """본문에서 '불릿/체크/점찍힌 특수문자'를 제거하고, 과도한 공백을 정리한다.

    ⚠️ 중요: 빈 문자열(""")을 replace 대상으로 쓰면 글자 사이에 공백이 삽입되므로 절대 사용하지 않는다.
    """
    s = (s or "").strip()
    if not s:
        return ""

    # 보이지 않는 문자/nbsp 정리
    s = (
        s.replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u2060", "")
        .replace("\xa0", " ")
    )

    # 불릿/체크/박스/점찍힌 문자 제거(요청사항)
    bad_chars = [
        "•", "·", "∙", "‧", "ㆍ",
        "●", "○", "◆", "◇",
        "■", "□",
        "▶", "▷", "►", "▸",
        "✔", "✓", "✅", "☑",
        "※",
        "*",
    ]
    for ch in bad_chars:
        s = s.replace(ch, " ")

    # 마침표/느낌표/물음표 제거(요청사항: 마침표 없이)
    s = re.sub(r"[\.!\?]+", " ", s)

    # 여백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or ""
    title = _sanitize_text(str(row.get("RCP_NM") or ""))
    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()

    ingredients: List[str] = []
    for p in re.split(r"\s*,\s*", parts):
        p = _sanitize_text(p)
        if p:
            ingredients.append(p)

    steps: List[str] = []
    for i in range(1, 21):
        s = _sanitize_text(str(row.get(f"MANUAL{str(i).zfill(2)}") or ""))
        if s:
            steps.append(s)

    # 최소한의 안전장치
    if not rid:
        rid = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]

    return Recipe(
        source="mfds",
        recipe_id=rid,
        title=title,
        ingredients=ingredients,
        steps=steps,
    )

def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None

    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡", "무침", "구이"]

    t0 = time.time()
    for _ in range(cfg.run.max_tries):
        if (time.time() - t0) >= max(3, cfg.run.mfds_budget_sec):
            break

        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(
            cfg.recipe.mfds_api_key,
            "RCP_NM",  #  오타 방지
            kw,
            start=1,
            end=60,
            timeout_sec=cfg.run.mfds_timeout_sec,
        )
        if not rows:
            continue

        random.shuffle(rows)
        for row in rows[:40]:
            try:
                rcp = mfds_row_to_recipe(row)
            except Exception:
                continue
            if cfg.run.strict_korean and not _is_korean_recipe_name(rcp.title, strict=True):
                continue
            if (rcp.source, rcp.recipe_id) in used:
                continue
            if not rcp.title or len(rcp.steps) < 2:
                continue
            return rcp

    return None

def pick_recipe_local(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Recipe:
    used = set(recent_pairs)
    pool = [x for x in LOCAL_KOREAN_RECIPES if ("local", str(x["id"])) not in used]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    pick = random.choice(pool)
    ing = [_sanitize_text(f"{a} {b}") for a, b in pick.get("ingredients", [])]
    steps = [_sanitize_text(s) for s in pick.get("steps", []) if _sanitize_text(s)]
    return Recipe(
        source="local",
        recipe_id=str(pick["id"]),
        title=_sanitize_text(str(pick["title"])),
        ingredients=[x for x in ing if x],
        steps=[x for x in steps if x],
    )

# -----------------------------
# Homefeed content generator (no bullets)
# -----------------------------
_ORD = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덟째", "아홉째", "열째", "열한째", "열둘째"]

def _dish_type(title: str) -> str:
    t = title or ""
    if any(k in t for k in ["찌개", "국", "탕", "전골"]):
        return "국물"
    if any(k in t for k in ["볶음", "볶아", "덮밥"]):
        return "볶음"
    if any(k in t for k in ["조림", "찜"]):
        return "조림"
    if any(k in t for k in ["전", "부침"]):
        return "전"
    if any(k in t for k in ["구이", "불고기"]):
        return "구이"
    return "집밥"

def _seeded_rng(seed_text: str) -> random.Random:
    h = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:12]
    return random.Random(int(h, 16))

def _no_period_text(s: str) -> str:
    s = _sanitize_text(s)
    # 콜론/따옴표도 너무 많으면 기계처럼 보여서 최소화
    s = s.replace(":", " ").replace("“", " ").replace("”", " ").replace('"', " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _intro_200_300(title: str, rng: random.Random) -> str:
    """
    홈피드형 도입부 200~300자
    - 마침표 없이
    - 연결어로 자연스럽게 이어지게
    """
    t = _no_period_text(title)

    templates = [
        lambda: f"요즘은 한 끼가 그냥 밥이 아니라 하루 리듬을 다시 잡아주는 느낌이더라고요\n\n그래서 저는 {t}처럼 손이 많이 가지 않는데도 만족도가 높은 메뉴가 자꾸 생각나요\n\n오늘은 어렵게 말하지 않고 흐름만 잡으실 수 있게 정리해봤어요",
        lambda: f"{t}는 괜히 떠올라서 한 번 만들기 시작하면 끝까지 가게 되는 메뉴예요\n\n재료가 완벽하지 않아도 방향만 맞추면 맛이 크게 흔들리지 않더라고요\n\n제가 자주 하는 순서대로 차분히 풀어볼게요",
        lambda: f"바쁜 날엔 뭘 해먹을지 고민하는 시간도 아깝잖아요\n\n그럴 때 {t}처럼 기준이 되는 메뉴가 하나 있으면 선택이 훨씬 쉬워져요\n\n오늘은 그 기준을 딱 잡아드리는 쪽으로 적어볼게요",
        lambda: f"{t}는 ‘지금 먹으면 딱이다’ 싶은 순간이 있어요\n\n처음 해보실 때는 간과 타이밍만 잡으면 되고 나머지는 유연하게 가셔도 돼요\n\n저는 그런 방식이 제일 현실적이더라고요",
        lambda: f"저도 예전에 {t}를 대충 했다가 간이 흔들려서 아쉬웠던 적이 있었어요\n\n그 뒤로는 순서 하나와 간 타이밍 하나만 지키니까 결과가 훨씬 안정적이더라고요\n\n그 포인트를 중심으로 오늘 버전으로 정리해드릴게요",
        lambda: f"{t}는 복잡한 기술보다 기본이 더 크게 먹히는 메뉴예요\n\n중간에 욕심만 안 내면 맛이 자연스럽게 따라오더라고요\n\n그래서 지금부터 딱 필요한 부분만 깔끔하게 적어볼게요",
        lambda: f"{t}는 만들고 나서 ‘아 이 맛이었지’ 하고 고개가 끄덕여지는 편이더라고요\n\n그래서 저는 기분이 들쭉날쭉할 때 이런 메뉴로 중심을 잡는 편이에요\n\n오늘도 그 느낌 그대로 따라가기 쉬운 형태로 정리해봤어요",
        lambda: f"저는 요리할 때 거창한 목표를 세우기보다 오늘의 컨디션에 맞추는 편이에요\n\n그래서 {t}처럼 부담이 덜한 메뉴가 자연스럽게 손에 잡히더라고요\n\n재료는 있는 만큼만 쓰시고 흐름만 따라오시면 돼요",
    ]

    out = templates[rng.randrange(len(templates))]()
    out = _no_period_text(out)

    if len(out) < 200:
        out += "\n\n처음 하시는 분도 흐름만 따라가시면 충분히 괜찮게 나와요"
        out = _no_period_text(out)
    if len(out) > 360:
        parts = out.split("\n\n")
        out = "\n\n".join(parts[:3])

    return out

def _fmt_lines(lines: List[str]) -> str:
    safe = [html.escape(_no_period_text(x)) for x in lines if _no_period_text(x)]
    return "<br/>".join(safe)

def _fmt_paras(paras: List[str]) -> str:
    safe = [html.escape(_no_period_text(x)) for x in paras if _no_period_text(x)]
    return "<br/><br/>".join(safe)

def _clean_title_tokens(title: str) -> List[str]:
    t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", title or "")
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

def build_hashtags(cfg: AppConfig, title: str, dish: str) -> str:
    base = ["한식", "집밥", "오늘뭐먹지", "간단요리", "요리기록", "저녁메뉴", "자취요리", "홈쿡", "레시피", "맛있는집밥"]
    if dish == "국물":
        base += ["국물요리", "따뜻한한끼", "찌개"]
    elif dish == "볶음":
        base += ["볶음요리", "밥반찬", "한그릇"]
    elif dish == "조림":
        base += ["조림요리", "밥도둑", "반찬"]
    elif dish == "전":
        base += ["전", "부침", "막걸리안주"]
    elif dish == "구이":
        base += ["구이", "불맛", "반찬"]

    for tok in _clean_title_tokens(title)[:6]:
        base.append(tok.replace(" ", ""))

    # 중복 제거
    uniq: List[str] = []
    seen = set()
    for x in base:
        x = re.sub(r"\s+", "", x)
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    n = max(cfg.run.hashtag_min, min(cfg.run.hashtag_max, 16))
    # #은 해시태그로 필요(요청사항에서 태그 여러 개)
    return " ".join([f"#{x}" for x in uniq[:n]])

def _strip_html_tags(html_text: str) -> str:
    return re.sub(r"<[^>]+>", "", html_text or "")

def ensure_min_chars(body_html: str, min_chars: int, max_chars: int, rng: random.Random, dish: str) -> str:
    """
    본문이 너무 짧으면 자연스러운 문단을 추가해 최소 글자수를 맞춤
    """
    base = body_html
    cur = len(_strip_html_tags(base))
    if cur >= min_chars:
        # 너무 길면 잘라내기
        if cur > max_chars:
            # 해시태그는 남기고 앞부분만 줄임
            parts = base.split("<br/><br/>")
            trimmed = []
            for p in parts:
                trimmed.append(p)
                if len(_strip_html_tags("<br/><br/>".join(trimmed))) >= max_chars:
                    break
            return "<br/><br/>".join(trimmed)
        return base

    extra_pool_common = [
        "저는 요리할 때 재료를 완벽하게 맞추는 것보다 흐름이 끊기지 않게 잡는 편이에요 그러면 결과가 훨씬 안정적이더라고요",
        "맛이 애매하게 느껴지면 양념을 더하기 전에 한 번만 더 끓여보세요 국물이나 소스가 정리되면서 맛이 또렷해질 때가 많아요",
        "시간이 촉박한 날엔 준비는 단순하게 하고 마지막에 향을 살리는 쪽으로 가면 만족도가 높아요 대파 마늘 참기름 같은 것들이요",
        "오늘 한 번 해보고 내 입맛 기준으로 매운맛이나 단맛만 살짝 메모해두면 다음엔 훨씬 빨리 끝나요",
    ]
    extra_pool_soup = [
        "국물 쪽은 센 불로 오래 끓이기보다 중불로 일정하게 가져가면 맛이 더 정돈돼요 끓는 동안은 건드리지 말고 마지막에만 간을 보셔도 좋아요",
        "찌개나 국은 한 번 끓인 뒤 3분만 쉬게 두면 맛이 더 붙을 때가 있어요 그때 간을 보면 과하게 넣을 일이 줄어들어요",
    ]
    extra_pool_pan = [
        "팬 요리는 재료가 쏟아지듯 들어가면 물이 생겨서 맛이 흐려질 수 있어요 조금씩 나눠 넣는 것만으로도 분위기가 달라져요",
        "볶거나 구울 땐 마지막 1분이 맛을 결정하는 날이 많아요 불을 세게 바꾸기보다 중불을 유지하고 마무리 향을 살리는 쪽이 안정적이에요",
    ]
    pool = extra_pool_common[:]
    if dish == "국물":
        pool += extra_pool_soup
    elif dish in ("볶음", "구이", "전"):
        pool += extra_pool_pan

    # 해시태그 전에 추가하도록 위치 찾기
    marker = "<br/><br/><strong>해시태그</strong>"
    if marker in base:
        before, after = base.split(marker, 1)
        while cur < min_chars and len(_strip_html_tags(before)) < max_chars:
            before += "<br/><br/>" + html.escape(_no_period_text(rng.choice(pool)))
            cur = len(_strip_html_tags(before)) + len(_strip_html_tags(marker + after))
        return before + marker + after

    # 마커가 없다면 끝에 붙임
    while cur < min_chars and cur < max_chars:
        base += "<br/><br/>" + html.escape(_no_period_text(rng.choice(pool)))
        cur = len(_strip_html_tags(base))
    return base

def build_body_html(cfg: AppConfig, recipe: Recipe, seed_key: str, recent_style_sigs: Optional[set] = None) -> Tuple[str, str, str]:
    """
    홈피드형 글 생성
    - 도입부 200~300자
    - 굵은 소제목 3개
    - 레시피(재료/순서) 항상 포함
    - 불릿/체크/점찍힌 특수문자 제거
    - 총 글자수 MIN_TOTAL_CHARS 이상 보장(과도한 장문 방지)
    - 레퍼토리 반복을 줄이기 위해 style_sig를 만들고 최근과 겹치면 재시도
    """
    title = _no_period_text(recipe.title)
    dish = _dish_type(title)

    used_style = set(recent_style_sigs or set())

    def compose(rng: random.Random) -> Tuple[str, str, str]:
        intro = _intro_200_300(title, rng)

        # 재료/순서 정리
        ing_lines = [_no_period_text(x) for x in (recipe.ingredients or []) if _no_period_text(x)]
        if not ing_lines:
            ing_lines = ["집에 있는 재료 기준으로 편하게 맞춰주세요"]

        step_src = [x for x in (recipe.steps or []) if _no_period_text(x)]
        if not step_src:
            step_src = ["재료를 먼저 준비해요", "중불로 익히고 마지막에 간을 맞춰요"]
        step_src = step_src[:10]

        step_lines: List[str] = []
        for i, s in enumerate(step_src, start=1):
            ord_k = _ORD[i - 1] if i - 1 < len(_ORD) else f"{i}번째"
            step_lines.append(f"순서 {ord_k}  {s}")

        # 섹션 1  왜 이 메뉴가 떠오르는지(문장이 이어지게)
        sec1_templates: List[List[str]] = [
            [
                f"{title}는 이상하게도 컨디션이 애매한 날에 먼저 떠오르더라고요 그래서 저는 이런 날엔 메뉴를 새로 고민하기보다 익숙한 쪽으로 마음을 돌려요",
                "예전에 한 번은 재료를 대충 맞췄다가 맛이 흐릿해서 아쉬웠던 적이 있었는데요 그때 느낀 게 방향만 잡아도 반은 성공이라는 거였어요 그래서 오늘은 그 방향을 먼저 잡아드릴게요",
            ],
            [
                f"저는 {title}를 생각하면 그날의 분위기까지 같이 따라오는 편이에요 그래서 한 번 만들기 시작하면 괜히 정리를 끝까지 하고 싶어지더라고요",
                "처음 하실 때는 욕심내지 않고 흐름만 따라가도 충분해요 특히 간과 시간은 뒤로 갈수록 달라질 수 있으니까 초반엔 가볍게 시작하시는 게 좋아요",
            ],
            [
                f"{title}는 그냥 해도 괜찮을 것 같은데 싶으면서도 막상 해보면 포인트가 있는 메뉴예요 그래서 저는 포인트를 두세 개만 정해두고 그 외는 유연하게 가요",
                "그 방식이 좋은 게 재료가 조금 달라도 결과가 크게 흔들리지 않더라고요 오늘도 같은 흐름으로 정리해드릴게요",
            ],
            [
                f"요즘처럼 정신 없는 날엔 메뉴를 결정하는 것부터가 에너지잖아요 그래서 저는 {title}처럼 기준이 되는 메뉴를 하나 정해두는 편이에요",
                "오늘은 그 기준을 그대로 따라가실 수 있게 재료와 순서를 한 번에 보이도록 적어볼게요",
            ],
        ]
        sec1_idx = rng.randrange(len(sec1_templates))
        sec1 = _fmt_paras(sec1_templates[sec1_idx])

        # 섹션 2  재료와 준비(도입 문단 다양화)
        sec2_open_pool = [
            "재료는 완벽할 필요 없어요 다만 순서가 꼬이지 않게 미리 손질만 해두면 훨씬 수월해요",
            "요리할 때 제일 피곤한 건 중간에 재료를 찾는 순간이더라고요 그래서 저는 시작 전에 재료부터 한 번 쫙 꺼내두는 편이에요",
            "맛을 좌우하는 건 재료 종류보다 준비 순서인 경우가 많아요 오늘은 그 부분을 기준으로 정리해볼게요",
            "재료가 조금 달라도 괜찮아요 핵심은 간을 언제 잡느냐와 불을 어떻게 유지하느냐 쪽이에요",
            "처음 하실수록 준비가 반이에요 손질만 끝내두면 조리는 생각보다 금방 따라와요",
            "냉장고 상황 따라 바꾸실 부분은 바꾸셔도 돼요 대신 기본 흐름만 지키면 맛은 따라오더라고요",
            "재료를 다 갖추지 못해도 괜찮아요 대신 양념 비율은 조금씩만 조절해보시면 좋아요",
        ]
        sec2_open_idx = rng.randrange(len(sec2_open_pool))
        sec2_open = html.escape(_no_period_text(sec2_open_pool[sec2_open_idx]))

        sec2_body = (
            f"{sec2_open}"
            f"<br/><br/><strong>레시피 재료 목록</strong><br/>{_fmt_lines(ing_lines)}"
        )

        # 섹션 3  순서 + 맛 포인트
        tip_pool_soup = [
            "국물은 센 불로 밀기보다 중불을 유지하면 맛이 더 안정적으로 나와요",
            "간은 중간에 확 잡지 말고 마지막에 한 번만 정리하는 쪽이 안전해요",
            "향이 아쉬우면 대파나 마늘을 마지막에 조금만 더해도 충분히 살아나요",
            "육수가 없다면 물로 시작해도 괜찮고 대신 끓이는 시간을 조금만 더 주세요",
            "재료가 익는 속도가 다르니 단단한 것부터 넣는 순서만 챙겨주시면 돼요",
            "국물은 마지막에 농도가 잡히니까 초반엔 조금 심심해도 괜찮아요",
        ]
        tip_pool_pan = [
            "팬 요리는 한 번에 너무 많이 넣지 않으면 물이 덜 생기고 결과가 깔끔해요",
            "불을 올렸다 내렸다 하기보다 한 단계로 유지하면 맛이 흔들리지 않아요",
            "마지막에 후추나 참기름을 아주 소량만 더해도 향이 또렷해져요",
            "양념은 먼저 섞어두고 들어가면 중간에 헤매지 않아서 좋아요",
            "수분이 많아 보이면 잠깐 뚜껑을 열고 정리해주면 식감이 좋아져요",
            "겉면만 먼저 잡아두면 안이 더 촉촉하게 남더라고요",
        ]
        tip_pool_common = [
            "단맛이나 매운맛은 한 번에 넣지 말고 아주 소량씩만 조절해보세요",
            "오늘 만든 뒤에 내 입맛 기준으로 간만 메모해두면 다음이 훨씬 쉬워요",
            "재료가 하나 빠져도 괜찮아요 흐름만 무너지지 않게 잡아주면 돼요",
            "조리는 길게 끌기보다 필요한 구간만 정확히 잡는 게 오히려 맛이 좋아요",
            "중간에 간을 보더라도 확정은 마지막에 하시는 게 안전하더라고요",
        ]
        tip_pool = tip_pool_common[:]
        tip_pool += (tip_pool_soup if dish == "국물" else tip_pool_pan)

        rng.shuffle(tip_pool)
        tips: List[str] = []
        for t in tip_pool:
            if len(tips) >= 3:
                break
            key = t[:10]
            if any(key == x[:10] for x in tips):
                continue
            tips.append(t)
        tips_html = "<br/>".join([html.escape(_no_period_text(t)) for t in tips])

        ask_pool = [
            f"{title} 만들 때 자주 쓰는 재료 조합이 있으세요",
            f"{title}는 어떤 버전이 제일 취향이세요",
            f"오늘 {title} 드신다면 다음엔 어떤 재료를 더해보고 싶으세요",
            "집에 있는 재료 중에 가장 잘 어울렸던 조합이 뭐였나요",
            "이 메뉴는 간을 어떤 스타일로 잡으시는 편이세요",
        ]
        closing_pool = [
            "말씀 한 번만 들려주시면 다음 글에 더 자연스럽게 녹여볼게요",
            "짧게라도 남겨주시면 제가 다음 메뉴 고를 때도 큰 도움이 돼요",
            "편하게 얘기해주시면 그 방향으로 다음 버전도 만들어볼게요",
            "읽으시면서 떠오른 팁이 하나라도 있으면 살짝만 공유해주셔도 좋아요",
            "오늘 드신 느낌이 어땠는지 가볍게만 알려주셔도 충분해요",
            "취향이 어디로 가는지 알면 다음 글이 훨씬 정확해지더라고요",
            "재료 한 가지만 추천해주셔도 저한텐 진짜 힌트가 돼요",
            "다음엔 더 짧게 갈지 더 디테일하게 갈지 방향도 같이 맞춰보고 싶어요",
            "혹시 다른 분들이 자주 하는 팁이 있으면 같이 나눠주셔도 좋아요",
        ]
        ask_idx = rng.randrange(len(ask_pool))
        closing_idx = rng.randrange(len(closing_pool))
        ask = html.escape(_no_period_text(ask_pool[ask_idx]))
        closing = html.escape(_no_period_text(closing_pool[closing_idx]))

        sec3_intro_pool = [
            "이제는 흐름만 따라가시면 돼요",
            "순서는 길어 보여도 실제로는 금방 끝나요",
            "여기서부터는 타이밍만 잡아주시면 돼요",
            "아래 순서대로 가시면 중간에 덜 헤매요",
            "한 단계씩만 넘기면 생각보다 단순해요",
            "처음부터 끝까지 한 번에 보이면 더 쉬워지더라고요",
            "지금부터는 단계만 따라가시면 맛이 자연스럽게 따라와요",
        ]
        sec3_intro_idx = rng.randrange(len(sec3_intro_pool))
        sec3_intro = html.escape(_no_period_text(sec3_intro_pool[sec3_intro_idx]))

        sec3_body = (
            f"{sec3_intro}"
            f"<br/><br/><strong>만드는 순서</strong><br/>{_fmt_lines(step_lines)}"
            f"<br/><br/><strong>맛 포인트</strong><br/>{tips_html}"
            f"<br/><br/>{ask}  {closing}"
        )

        hashtags = build_hashtags(cfg, title, dish)
        tags_block = f"<strong>해시태그</strong><br/>{html.escape(hashtags)}"

        intro_html = html.escape(_no_period_text(intro)).replace("\n\n", "<br/><br/>")

        body = (
            f"{intro_html}"
            f"<br/><br/><strong>왜 이 메뉴가 떠오르는지</strong><br/>{sec1}"
            f"<br/><br/><strong>재료와 준비</strong><br/>{sec2_body}"
            f"<br/><br/><strong>순서와 맛 포인트</strong><br/>{sec3_body}"
            f"<br/><br/>{tags_block}"
        )

        style_meta = {
            "sec1": sec1_idx,
            "sec2": sec2_open_idx,
            "sec3i": sec3_intro_idx,
            "ask": ask_idx,
            "close": closing_idx,
            "tips": "|".join([t[:12] for t in tips]),
        }
        style_sig = hashlib.sha1(json.dumps(style_meta, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:10]

        excerpt = f"{title} 레시피  재료와 순서  맛 포인트까지 한 번에 정리했어요"
        excerpt = excerpt[:140]

        body = ensure_min_chars(
            body_html=body,
            min_chars=cfg.run.min_total_chars,
            max_chars=cfg.run.max_total_chars,
            rng=rng,
            dish=dish,
        )

        return body, excerpt, style_sig

    for k in range(6):
        rng = _seeded_rng(f"{seed_key}|{recipe.uid()}|{k}")
        body, excerpt, style_sig = compose(rng)
        if style_sig not in used_style:
            return body, excerpt, style_sig

    return compose(_seeded_rng(f"{seed_key}|{recipe.uid()}|fallback"))

def build_post_title(recipe_title: str, date_str: str, slot_label: str) -> str:
    rng = _seeded_rng(recipe_title + "|" + date_str + "|" + slot_label)
    hooks = [
        "오늘 저녁 메뉴로 편하게",
        "실수 줄이는 흐름으로",
        "재료랑 순서만 딱 정리",
        "집에 있는 재료로도 충분",
        "마음 편한 한 끼로",
    ]
    return _no_period_text(f"{recipe_title} {rng.choice(hooks)} {date_str} {slot_label}")

def slugify(text: str) -> str:
    t = re.sub(r"[^0-9a-zA-Z\-]+", "-", (text or "").lower()).strip("-")
    t = re.sub(r"-{2,}", "-", t)
    return t or "post"

# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = {"day": "오늘", "am": "오전", "pm": "오후"}.get(slot, "오늘")

    init_db(cfg.sqlite_path)
    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, limit=120)

    # 중복 업로드면 run_id로 slug/date_slot을 매번 유니크하게
    run_id = now.strftime("%H%M%S") + "-" + hashlib.sha1(os.urandom(16)).hexdigest()[:6]
    date_slot = f"{date_str}_{slot}_{run_id}" if cfg.run.allow_duplicate_posts else f"{date_str}_{slot}"

    print(f"[RUN] slot={slot} allow_duplicate={int(cfg.run.allow_duplicate_posts)} date_slot={date_slot}")

    chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(cfg, recent_pairs)

    title = build_post_title(chosen.title, date_str, slot_label)
    slug = slugify(f"korean-recipe-{date_str}-{slot}-{run_id}" if cfg.run.allow_duplicate_posts else f"korean-recipe-{date_str}-{slot}")

    recent_style_sigs = set(get_recent_style_sigs(cfg.sqlite_path, limit=20))
    body_html, excerpt, style_sig = build_body_html(cfg, chosen, seed_key=date_slot, recent_style_sigs=recent_style_sigs)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략  HTML 미리보기 일부")
        print(body_html[:1800])
        return

    wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt)
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
            "style_sig": style_sig,
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
