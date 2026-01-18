# -*- coding: utf-8 -*-
"""
daily_korean_recipe_to_wp_no_image.py (홈피드형 레시피 자동 업로드 / 이미지 미사용 버전)

반영
- 이미지 완전 미사용(본문 img 태그 생성/대표이미지 업로드 없음)
- 레시피(재료/순서) 항상 포함
- 불릿/체크/점찍힌 문자 제거(줄바꿈만 사용)
- 해시태그 12~20개 자동 생성
- 중복 업로드 허용(ALLOW_DUPLICATE_POSTS=1이면 매번 새 글 생성)
- 총 글자수 1200 이상 보장, 기본 상한 1800

필수 Secrets
- WP_BASE_URL, WP_USER, WP_APP_PASS
옵션
- MFDS_API_KEY, WP_STATUS, WP_CATEGORY_IDS, WP_TAG_IDS, SQLITE_PATH
- RUN_SLOT(day|am|pm), ALLOW_DUPLICATE_POSTS(1|0)
- MIN_TOTAL_CHARS, MAX_TOTAL_CHARS, HASHTAG_MIN, HASHTAG_MAX
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

LOCAL_KOREAN_RECIPES: List[Dict[str, Any]] = [
    {
        "id": "kimchi-jjigae",
        "title": "돼지고기 김치찌개",
        "ingredients": [
            ("신김치", "2컵"), ("돼지고기", "200g"), ("양파", "반 개"), ("대파", "한 대"),
            ("두부", "반 모"), ("고춧가루", "한 큰술"), ("다진마늘", "한 큰술"),
            ("국간장", "한 큰술"), ("물 또는 육수", "700ml"),
        ],
        "steps": [
            "냄비에 돼지고기를 넣고 중불에서 기름이 살짝 돌 때까지 볶아주세요",
            "신김치를 넣고 2분 정도 더 볶아서 신맛을 눌러주세요",
            "고춧가루 다진마늘 국간장을 넣고 향이 올라오면 물 또는 육수를 부어주세요",
            "10분 정도 끓인 뒤 양파를 넣고 3분 더 끓여주세요",
            "두부를 넣고 2분 더 끓인 뒤 대파로 마무리해주세요",
        ],
    },
    {
        "id": "doenjang-jjigae",
        "title": "구수한 된장찌개",
        "ingredients": [
            ("된장", "한 큰술 반"), ("고추장", "반 큰술 선택"), ("애호박", "삼 분의 일 개"),
            ("양파", "삼 분의 일 개"), ("두부", "반 모"), ("대파", "반 대"),
            ("다진마늘", "한 작은술"), ("물 또는 육수", "700ml"),
        ],
        "steps": [
            "끓는 물 또는 육수에 된장을 풀고 5분 끓여주세요",
            "양파 애호박 두부를 넣고 6분 정도 더 끓여주세요",
            "대파와 다진마늘을 넣고 한 번 더 끓인 뒤 간을 보고 마무리해주세요",
        ],
    },
    {
        "id": "bulgogi",
        "title": "간장 불고기",
        "ingredients": [
            ("소고기 불고기용", "300g"), ("양파", "반 개"), ("대파", "한 대"),
            ("간장", "네 큰술"), ("설탕", "한 큰술"), ("다진마늘", "한 큰술"),
            ("참기름", "한 큰술"), ("후추", "약간"), ("물 또는 배즙", "세 큰술"),
        ],
        "steps": [
            "간장 설탕 다진마늘 참기름 물 후추로 양념을 만들어주세요",
            "고기에 양념을 넣고 15분 이상 재워주세요",
            "팬을 달군 뒤 고기를 볶고 양파 대파를 넣어 숨이 죽을 때까지 볶아주세요",
        ],
    },
    {
        "id": "bibimbap",
        "title": "비빔밥",
        "ingredients": [
            ("밥", "한 공기"), ("계란", "한 개"), ("시금치", "한 줌"), ("당근", "약간"),
            ("콩나물", "한 줌"), ("고추장", "한 큰술"), ("참기름", "한 큰술"), ("간장", "약간"),
        ],
        "steps": [
            "시금치 콩나물 당근을 각각 간단히 볶거나 데쳐서 준비해주세요",
            "그릇에 밥을 담고 준비한 나물을 보기 좋게 올려주세요",
            "계란후라이를 올리고 고추장 참기름으로 비벼서 드셔주세요",
        ],
    },
    {
        "id": "gyeran-jjim",
        "title": "폭신한 계란찜",
        "ingredients": [
            ("계란", "세 개"), ("물", "계란의 한 배 정도"), ("소금", "약간"), ("대파", "약간"), ("참기름", "약간"),
        ],
        "steps": [
            "계란을 풀고 물 소금을 넣어 고르게 섞어주세요",
            "약불에서 천천히 저어가며 익혀주세요",
            "대파를 넣고 마무리로 참기름을 아주 살짝 둘러주세요",
        ],
    },
]

KOREAN_NEGATIVE_KEYWORDS = ["파스타", "피자", "타코", "스시", "커리", "샌드위치", "버거", "샐러드"]

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
    allow_duplicate_posts: bool = True
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

    hashtag_min = max(8, min(30, _env_int("HASHTAG_MIN", 12)))
    hashtag_max = max(hashtag_min, min(40, _env_int("HASHTAG_MAX", 20)))

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
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            allow_duplicate_posts=_env_bool("ALLOW_DUPLICATE_POSTS", True),
            min_total_chars=_env_int("MIN_TOTAL_CHARS", 1200),
            max_total_chars=_env_int("MAX_TOTAL_CHARS", 1800),
            hashtag_min=hashtag_min,
            hashtag_max=hashtag_max,
            strict_korean=_env_bool("STRICT_KOREAN", True),
            mfds_timeout_sec=_env_int("MFDS_TIMEOUT_SEC", 8),
            mfds_budget_sec=_env_int("MFDS_BUDGET_SEC", 15),
            max_tries=_env_int("MAX_TRIES", 18),
        ),
        recipe=RecipeSourceConfig(mfds_api_key=_env("MFDS_API_KEY", "")),
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
        return "OK" if v else "MISSING"
    print("[CFG] WP_BASE_URL:", ok(cfg.wp.base_url))
    print("[CFG] WP_USER:", ok(cfg.wp.user))
    print("[CFG] WP_APP_PASS:", ok(cfg.wp.app_pass))
    print("[CFG] WP_STATUS:", cfg.wp.status)
    print("[CFG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CFG] WP_TAG_IDS:", cfg.wp.tag_ids)
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] RUN_SLOT:", cfg.run.run_slot)
    print("[CFG] ALLOW_DUPLICATE_POSTS:", int(cfg.run.allow_duplicate_posts))
    print("[CFG] MIN_TOTAL_CHARS:", cfg.run.min_total_chars, "| MAX_TOTAL_CHARS:", cfg.run.max_total_chars)

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
          created_at TEXT
        )
        """
    )
    con.commit()
    con.close()

def get_recent_recipe_ids(path: str, days: int = 90) -> List[Tuple[str, str]]:
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

def save_post_meta(path: str, meta: Dict[str, Any]) -> None:
    con = sqlite3.connect(path)
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

def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-korean-recipe-bot/2.0"}

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
    s = (s or "").replace(".", " ")
    s = re.sub(r"[•·●◦▪▫■□◆◇▶▷✓✔✅☑️★☆※→⇒➜➤]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def mfds_row_to_recipe(row: Dict[str, Any]) -> Recipe:
    rid = str(row.get("RCP_SEQ") or "").strip() or ""
    title = _sanitize_text(str(row.get("RCP_NM") or "").strip())

    parts = str(row.get("RCP_PARTS_DTLS") or "").strip()
    ingredients: List[str] = []
    for p in re.split(r"\s*,\s*", parts):
        p = _sanitize_text(p)
        if p:
            ingredients.append(p)

    steps: List[str] = []
    for i in range(1, 21):
        s = str(row.get(f"MANUAL{str(i).zfill(2)}") or "").strip()
        s = _sanitize_text(s)
        if s:
            steps.append(s)

    if not rid:
        rid = hashlib.sha1(title.encode("utf-8")).hexdigest()[:10]
    if not ingredients:
        ingredients = ["재료는 집에 있는 것으로 편하게 맞춰주세요"]
    if not steps:
        steps = ["재료를 준비하고 간은 마지막에 맞춰주세요", "중불에서 익히고 마지막에 한 번만 간을 조절해주세요"]
    return Recipe(source="mfds", recipe_id=rid, title=title, ingredients=ingredients, steps=steps)

def pick_recipe_mfds(cfg: AppConfig, recent_pairs: List[Tuple[str, str]]) -> Optional[Recipe]:
    if not cfg.recipe.mfds_api_key:
        return None
    used = set(recent_pairs)
    keywords = ["김치", "된장", "고추장", "국", "찌개", "볶음", "전", "조림", "비빔", "나물", "탕", "죽", "김밥", "떡", "제육", "불고기"]
    start_ts = time.time()
    for _ in range(cfg.run.max_tries):
        if (time.time() - start_ts) > cfg.run.mfds_budget_sec:
            break
        kw = random.choice(keywords)
        rows = mfds_fetch_by_param(cfg.recipe.mfds_api_key, "RCP_NM", kw, start=1, end=60, timeout_sec=cfg.run.mfds_timeout_sec)
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
            if not rcp.title:
                continue
            return rcp
    return None

def pick_recipe_local(recent_pairs: List[Tuple[str, str]]) -> Recipe:
    used = set(recent_pairs)
    pool = [x for x in LOCAL_KOREAN_RECIPES if ("local", str(x["id"])) not in used]
    if not pool:
        pool = LOCAL_KOREAN_RECIPES[:]
    pick = random.choice(pool)
    ing = [f"{_sanitize_text(a)} {_sanitize_text(b)}".strip() for a, b in pick.get("ingredients", [])]
    steps = [_sanitize_text(str(s)) for s in pick.get("steps", []) if _sanitize_text(str(s))]
    return Recipe(source="local", recipe_id=str(pick["id"]), title=_sanitize_text(str(pick["title"])), ingredients=ing, steps=steps)

def _esc(s: str) -> str:
    return html.escape(s or "")

def _strip_tags(html_text: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", html_text or "")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _intro_200_300(title: str) -> str:
    base = [
        f"오늘은 {title}로 마음을 좀 풀어보려고 해요",
        "하루가 길었던 날일수록 따뜻한 한 끼가 은근히 사람을 살려주더라고요",
        "저는 요리를 잘하는 편은 아닌데요 그래도 이 메뉴는 따라가기가 편해서 자주 해먹게 돼요",
        "실수 줄이는 포인트만 잡아두면 맛이 안정적으로 나오는 쪽이라 추천드리고 싶어요",
        "혹시 오늘 뭐 먹지 고민 중이시면 같이 천천히 해보셔도 좋아요",
    ]
    random.shuffle(base)
    s = " ".join(base)
    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < 200:
        s += " " + " ".join(["재료도 부담스럽지 않게 잡아드릴게요", "간은 마지막에만 보면 훨씬 마음이 편해요"])
    if len(s) > 300:
        s = s[:300].rsplit(" ", 1)[0]
    return s

def _korean_ord(n: int) -> str:
    ords = ["첫째", "둘째", "셋째", "넷째", "다섯째", "여섯째", "일곱째", "여덟째", "아홉째", "열째"]
    return ords[n-1] if 1 <= n <= len(ords) else f"{n}번째"

def build_hashtags(cfg: AppConfig, recipe: Recipe) -> str:
    base = [
        "한식레시피", "집밥", "오늘뭐먹지", "간단요리", "자취요리",
        "국물요리", "밥도둑", "저녁메뉴", "한끼", "요리기록",
        "초보요리", "요리스타그램", "레시피공유", "집밥스타그램",
        "한국요리", "간편식", "맛있는집밥", "한식",
    ]
    title = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", recipe.title or "")
    toks = [t.strip() for t in title.split() if len(t.strip()) >= 2]
    for t in toks[:6]:
        base.append(t.replace(" ", ""))
    for ing in recipe.ingredients[:6]:
        t = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", ing).split()
        if t:
            base.append(t[0])
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
    n = random.randint(cfg.run.hashtag_min, cfg.run.hashtag_max)
    tags = uniq[:n]
    return " ".join([f"#{t}" for t in tags])

def _section_title(s: str) -> str:
    return f"<p><strong>{_esc(s)}</strong></p>"

def _no_period(s: str) -> str:
    s = s.replace(".", " ")
    s = re.sub(r"[•·●◦▪▫■□◆◇▶▷✓✔✅☑️★☆※→⇒➜➤]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_body_html(cfg: AppConfig, recipe: Recipe) -> Tuple[str, str]:
    title = _no_period(recipe.title)
    intro = _intro_200_300(title)

    ing_lines = [_no_period(x) for x in recipe.ingredients if _no_period(x)]
    if not ing_lines:
        ing_lines = ["재료는 집에 있는 것으로 편하게 맞춰주세요"]
    step_lines = []
    for i, s in enumerate(recipe.steps[:12], start=1):
        s = _no_period(s)
        if s:
            step_lines.append(f"{_korean_ord(i)} {s}")
    if not step_lines:
        step_lines = ["첫째 재료를 준비하고 간은 마지막에 맞춰주세요", "둘째 중불에서 익히고 마지막에 한 번만 간을 조절해주세요"]

    sec1 = [
        "요즘은 밥 한 끼도 마음가짐이 되게 중요한 것 같아요",
        f"{title}는 괜히 든든해지는 쪽이라서요",
        "저도 처음엔 대충 만들었다가 간이 흔들려서 아쉬웠던 적이 있었거든요",
        "그 뒤로는 딱 두 가지만 지키니까 훨씬 편해졌어요",
        "불은 중간에서 흔들리지 않게 잡고 간은 끝에서만 보자는 쪽이요",
        "이렇게만 해도 실패 확률이 확 줄어들더라고요",
    ]
    sec2 = [
        "재료는 부담 없이 맞추셔도 돼요",
        "집에 있는 걸로 방향만 잡고 부족한 건 다음에 천천히 채우셔도 괜찮아요",
    ]
    sec3 = [
        "여기서 제일 자주 헷갈리는 게 간 타이밍이더라고요",
        "중간에 간을 확 잡아버리면 나중에 졸아들면서 짜질 때가 있어요",
        "그래서 저는 마지막에 한 번만 조절해요 그러면 마음이 진짜 편해져요",
        "향이 부족하면 대파나 마늘을 마지막에 아주 조금만 더해도 확 살아나요",
        "매운맛을 줄이고 싶으시면 양념은 줄이고 대신 육수 쪽을 조금 더 써보셔도 좋아요",
        "오늘 드셔보시고 다음에 더 내 입맛으로 맞추고 싶으면 간과 불만 메모해두면 정말 편해요",
        f"혹시 {title}는 어떤 재료를 더 넣는 편이신가요 버섯 두부 대파 같은 거요",
        "댓글로 한 줄만 남겨주시면 저도 다음 메뉴 고를 때 참고할게요",
    ]

    body = []
    body.append(_section_title(f"{title} 오늘은 이렇게 해볼게요"))
    body.append(f"<p>{_esc(_no_period(intro))}</p>")
    body.append(_section_title("왜 이 레시피가 자꾸 생각나는지"))
    body.append(f"<p>{'<br/><br/>'.join([_esc(_no_period(x)) for x in sec1])}</p>")
    body.append(_section_title("재료와 만드는 순서 정리"))
    body.append(f"<p>{'<br/><br/>'.join([_esc(_no_period(x)) for x in sec2])}</p>")
    body.append(_section_title("재료"))
    body.append(f"<p>{'<br/>'.join([_esc(x) for x in ing_lines])}</p>")
    body.append(_section_title("만드는 순서"))
    body.append(f"<p>{'<br/>'.join([_esc(x) for x in step_lines])}</p>")
    body.append(_section_title("실패 줄이는 팁과 응용"))
    body.append(f"<p>{'<br/><br/>'.join([_esc(_no_period(x)) for x in sec3])}</p>")
    body.append(f"<p>{_esc(build_hashtags(cfg, recipe))}</p>")
    html_body = "\n".join(body)

    plain = _strip_tags(html_body)
    min_chars = max(600, cfg.run.min_total_chars)
    max_chars = max(min_chars, cfg.run.max_total_chars)

    fillers = [
        "저는 요리할 때 완벽하게 하려는 마음을 내려놓는 게 제일 도움 되더라고요",
        "오늘은 있는 재료로만 해도 괜찮아요 흐름만 익히면 다음부터는 더 쉬워져요",
        "처음엔 시간 감각이 없어서 오래 끓이곤 했는데요 두 번만 해보면 감이 바로 와요",
        "맛이 애매하면 소금부터 더하지 말고 향을 먼저 살짝 올려보는 쪽이 안정적이에요",
        "한 번 성공하면 그 다음부터는 진짜 자신감이 붙어요 저도 그랬거든요",
    ]
    used = set()
    while len(plain) < min_chars:
        cand = random.choice(fillers)
        if cand in used:
            cand = random.choice([x for x in fillers if x not in used] or fillers)
        used.add(cand)
        insert = f"<p>{_esc(_no_period(cand))}</p>"
        html_body = re.sub(r"(<p>#)", insert + r"\n\1", html_body, count=1)
        plain = _strip_tags(html_body)
        if len(used) >= len(fillers):
            break

    if len(plain) > max_chars:
        for cand in list(used):
            patt = rf"<p>{re.escape(_esc(_no_period(cand)))}</p>\s*"
            html_body2 = re.sub(patt, "", html_body, count=1)
            if len(_strip_tags(html_body2)) >= min_chars:
                html_body = html_body2
            if len(_strip_tags(html_body)) <= max_chars:
                break

    excerpt = _no_period(f"{title} 레시피 정리  재료와 순서를 편하게 따라가실 수 있게 적어두었어요")[:140]
    return html_body, excerpt

def build_post_title(recipe_title: str, date_str: str, slot_label: str) -> str:
    hooks = ["오늘 저녁 메뉴로 딱이에요", "간단한데 든든해요", "생각보다 쉽게 됩니다", "실수 줄이는 포인트만 정리했어요"]
    return _no_period(f"{recipe_title} {random.choice(hooks)} {date_str} {slot_label}")

def slugify(text: str) -> str:
    t = re.sub(r"[^0-9a-zA-Z\-]+", "-", (text or "").lower()).strip("-")
    t = re.sub(r"-{2,}", "-", t)
    return t or "post"

def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = {"day": "오늘", "am": "오전", "pm": "오후"}.get(slot, "오늘")

    init_db(cfg.sqlite_path)
    run_id = datetime.utcnow().strftime("%H%M%S") + "_" + hashlib.sha1(os.urandom(16)).hexdigest()[:6]

    recent_pairs = get_recent_recipe_ids(cfg.sqlite_path, 90)
    chosen = pick_recipe_mfds(cfg, recent_pairs) or pick_recipe_local(recent_pairs)

    title = build_post_title(chosen.title, date_str, slot_label)
    slug = slugify(f"korean-recipe-{date_str}-{slot}-{run_id}" if cfg.run.allow_duplicate_posts else f"korean-recipe-{date_str}-{slot}")
    body_html, excerpt = build_body_html(cfg, chosen)

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략")
        print(body_html[:1500])
        return

    post_id, link = wp_create_post(cfg.wp, title, slug, body_html, excerpt=excerpt)
    print("OK(created):", post_id, link)

    date_slot = f"{date_str}_{slot}_{run_id}" if cfg.run.allow_duplicate_posts else f"{date_str}_{slot}"
    save_post_meta(cfg.sqlite_path, {
        "date_slot": date_slot,
        "recipe_source": chosen.source,
        "recipe_id": chosen.recipe_id,
        "recipe_title": chosen.title,
        "wp_post_id": post_id,
        "wp_link": link,
        "created_at": datetime.utcnow().isoformat(),
    })

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
