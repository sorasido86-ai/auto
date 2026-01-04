# -*- coding: utf-8 -*-
"""
daily_issue_keywords_to_wp.py (완전 통합/최적화)
- Google Trends(geo=KR) 트렌딩 검색어 수집
- Google News RSS(검색/토픽) + 커뮤니티 RSS 수집 (총 FEEDS_MAX까지)
- 최근 WINDOW_HOURS 시간 기준 "데일리 이슈 키워드 TOP N" 집계
- WordPress REST API로 오전/오후(am/pm) 글을 별도로 생성/업데이트 (slug 분리)
- GitHub Actions Secrets(환경변수)만으로 실행

필수 Secrets
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

카테고리
  - WP_CATEGORY_IDS="4"

권장 env
  - FEEDS_MAX=100
  - FEED_TIMEOUT=8
  - MAX_WORKERS=20
  - WINDOW_HOURS=12
  - LIMIT=20
  - RUN_SLOT=am|pm
  - DEBUG=1
"""

from __future__ import annotations

import base64
import html as htmlmod
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import feedparser
import xml.etree.ElementTree as ET

KST = timezone(timedelta(hours=9))


# -----------------------------
# env helpers
# -----------------------------
def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    s = _env(name, str(default))
    try:
        return int(s)
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = (_env(name, "1" if default else "0")).lower()
    return v in ("1", "true", "yes", "y", "on")


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for p in (s or "").split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out


def _split_lines_env(name: str) -> List[str]:
    raw = _env(name, "")
    if not raw:
        return []
    feeds = []
    for line in raw.splitlines():
        u = line.strip()
        if not u or u.startswith("#"):
            continue
        feeds.append(u)
    seen = set()
    out = []
    for u in feeds:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


# -----------------------------
# config
# -----------------------------
@dataclass
class WordPressConfig:
    base_url: str
    username: str
    app_password: str
    status: str = "publish"
    category_ids: List[int] = field(default_factory=lambda: [4])


@dataclass
class RunConfig:
    limit: int = 20
    window_hours: int = 12
    google_geo: str = "KR"
    feeds_max: int = 100
    max_workers: int = 20
    feed_timeout: int = 8
    dry_run: bool = False
    debug: bool = False
    slot: str = "am"  # am|pm


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    blocklist: Set[str]


def load_cfg() -> AppConfig:
    wp = WordPressConfig(
        base_url=_env("WP_BASE_URL").rstrip("/"),
        username=_env("WP_USER"),
        app_password=_env("WP_APP_PASS"),
        status=_env("WP_STATUS", "publish") or "publish",
        category_ids=_parse_int_list(_env("WP_CATEGORY_IDS", "4")) or [4],
    )

    run = RunConfig(
        limit=_env_int("LIMIT", 20),
        window_hours=_env_int("WINDOW_HOURS", 12),
        google_geo=_env("GOOGLE_GEO", "KR") or "KR",
        feeds_max=_env_int("FEEDS_MAX", 100),
        max_workers=_env_int("MAX_WORKERS", 20),
        feed_timeout=_env_int("FEED_TIMEOUT", 8),
        dry_run=_env_bool("DRY_RUN", False),
        debug=_env_bool("DEBUG", False),
        slot=(_env("RUN_SLOT", "am") or "am").lower(),
    )
    if run.slot not in ("am", "pm"):
        run.slot = "am"

    base_block = {
        "net", "com", "co", "kr", "www", "m", "amp",
        "daum", "naver", "google", "youtube", "tiktok", "instagram", "facebook",
        "twitter", "reddit", "blog", "cafe", "post", "news",
        "기사", "기자", "단독", "속보", "영상", "사진",
        "breaking", "exclusive",
    }
    extra = set()
    raw = _env("BLOCKLIST", "")
    if raw:
        for p in re.split(r"[,\n]+", raw):
            p = p.strip().lower()
            if p:
                extra.add(p)

    return AppConfig(wp=wp, run=run, blocklist=base_block | extra)


def validate_cfg(cfg: AppConfig) -> None:
    missing = []
    if not cfg.wp.base_url:
        missing.append("WP_BASE_URL")
    if not cfg.wp.username:
        missing.append("WP_USER")
    if not cfg.wp.app_password:
        missing.append("WP_APP_PASS")
    if missing:
        raise RuntimeError("필수 설정 누락:\n- " + "\n- ".join(missing))


def print_safe(cfg: AppConfig) -> None:
    def ok(v: str) -> str:
        return f"OK(len={len(v)})" if v else "MISSING"

    print("[CONFIG] WP_BASE_URL:", cfg.wp.base_url)
    print("[CONFIG] WP_USER:", ok(cfg.wp.username))
    print("[CONFIG] WP_APP_PASS:", ok(cfg.wp.app_password))
    print("[CONFIG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CONFIG] STATUS:", cfg.wp.status)
    print("[CONFIG] LIMIT:", cfg.run.limit, "WINDOW_HOURS:", cfg.run.window_hours, "SLOT:", cfg.run.slot)
    print("[CONFIG] GEO:", cfg.run.google_geo, "FEEDS_MAX:", cfg.run.feeds_max)
    print("[CONFIG] TIMEOUT:", cfg.run.feed_timeout, "MAX_WORKERS:", cfg.run.max_workers)
    print("[CONFIG] DRY_RUN:", cfg.run.dry_run, "DEBUG:", cfg.run.debug)
    print("[CONFIG] BLOCKLIST size:", len(cfg.blocklist))


# -----------------------------
# Google Trends RSS
# -----------------------------
def fetch_google_trends(geo: str, limit: int) -> List[Dict[str, Any]]:
    url = f"https://trends.google.com/trending/rss?geo={geo}"
    r = requests.get(url, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Google Trends RSS failed: {r.status_code} body={r.text[:200]}")
    root = ET.fromstring(r.content)
    items = root.findall(".//item")

    out: List[Dict[str, Any]] = []
    for it in items[:max(1, limit)]:
        title_el = it.find("title")
        link_el = it.find("link")
        title = (title_el.text or "").strip() if title_el is not None else ""
        link = (link_el.text or "").strip() if link_el is not None else ""
        approx = ""
        for child in list(it):
            if child.tag.lower().endswith("approx_traffic"):
                approx = (child.text or "").strip()
        if title:
            out.append({"keyword": title, "link": link, "traffic": approx})
    return out


# -----------------------------
# Feed builders
# -----------------------------
def google_news_rss_top() -> str:
    return "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"


def google_news_rss_topic(topic: str) -> str:
    return f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=ko&gl=KR&ceid=KR:ko"


def google_news_rss_search(q: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=ko&gl=KR&ceid=KR:ko"


def default_news_keywords() -> List[str]:
    return [
        "선거", "국회", "검찰", "법원", "대통령",
        "환율", "금리", "물가", "부동산", "전세", "코스피", "비트코인",
        "삼성전자", "네이버", "카카오", "테슬라", "엔비디아",
        "AI", "반도체", "전기차", "배터리", "로봇",
        "사고", "화재", "산불", "지진", "태풍",
        "독감", "백신", "의료",
        "축구", "야구", "손흥민",
        "드라마", "영화", "넷플릭스", "K팝",
        "해킹", "개인정보", "보안",
    ]


def build_default_news_feeds(max_n: int) -> List[str]:
    """
    ✅ 수정 포인트(중요):
    - 예전 버전은 채우기 키워드가 10개뿐이라 max_n(100)을 못 채우면 무한루프 가능
    - 이 버전은 (분야 키워드 × 사건 키워드) 조합으로 수백개 후보를 만들어 max_n까지 채움
    - 그래도 못 채우면 그냥 있는 만큼 반환(무한루프 없음)
    """
    urls: List[str] = [google_news_rss_top()]

    topics = ["WORLD", "NATION", "BUSINESS", "TECHNOLOGY", "ENTERTAINMENT", "SCIENCE", "SPORTS", "HEALTH"]
    for t in topics:
        urls.append(google_news_rss_topic(t))

    for kw in default_news_keywords():
        urls.append(google_news_rss_search(kw))

    # 조합 생성(충분히 많이)
    bases = [
        "논란", "구속", "사퇴", "파면", "기소", "선고", "재판", "수사", "압수수색", "검거",
        "사고", "화재", "추락", "폭발", "정전", "침수", "붕괴", "산불", "태풍", "지진",
        "감염", "독감", "확진", "백신", "주의보", "특보",
        "해킹", "유출", "장애", "점검", "업데이트", "출시",
        "인상", "인하", "상승",igger = "폭등"
    ]
    # 위 라인 오타 방지: (깃헙에 올릴 때 혹시 편집 중 깨질까봐) 아래처럼 정상화
    bases = [
        "논란", "구속", "사퇴", "파면", "기소", "선고", "재판", "수사", "압수수색", "검거",
        "사고", "화재", "추락", "폭발", "정전", "침수", "붕괴", "산불", "태풍", "지진",
        "감염", "독감", "확진", "백신", "주의보", "특보",
        "해킹", "유출", "장애", "점검", "업데이트", "출시",
        "인상", "인하", "상승", "하락", "폭등", "폭락",
        "파업", "시위", "협상", "합의", "불매", "리콜", "환불",
    ]
    mods = [
        "정치", "경제", "사회", "연예", "스포츠", "IT", "과학", "건강",
        "부동산", "금융", "코인", "주식", "환율", "국제", "교육",
        "검찰", "경찰", "법원", "국회",
    ]

    # 먼저 모듈×베이스 조합으로 유니크 URL을 많이 생성
    for m in mods:
        for b in bases:
            urls.append(google_news_rss_search(f"{m} {b}"))
            if len(urls) >= max_n * 3:
                break
        if len(urls) >= max_n * 3:
            break

    # dedup + max_n까지 채우기
    seen = set()
    out: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= max_n:
            break

    # ✅ 안전장치: max_n 못 채워도 무한루프 없이 그대로 반환
    return out


def default_community_feeds() -> List[str]:
    return [
        "https://www.reddit.com/r/korea/.rss",
        "https://www.reddit.com/r/southkorea/.rss",
        "https://www.reddit.com/r/kpop/.rss",
    ]


def build_feed_list(cfg: AppConfig) -> List[str]:
    community = _split_lines_env("COMMUNITY_FEEDS")
    if not community:
        community = default_community_feeds()

    news = _split_lines_env("NEWS_FEEDS")
    if not news:
        news = build_default_news_feeds(cfg.run.feeds_max)

    urls = community + news
    seen = set()
    out = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


# -----------------------------
# RSS fetching
# -----------------------------
@dataclass
class FeedItem:
    title: str
    link: str
    source: str
    published: datetime


def _entry_dt(e: Any) -> Optional[datetime]:
    for k in ("published_parsed", "updated_parsed"):
        v = getattr(e, k, None)
        if v:
            try:
                return datetime(*v[:6], tzinfo=timezone.utc).astimezone(KST)
            except Exception:
                pass
    return None


def fetch_feed(session: requests.Session, url: str, timeout: int) -> Tuple[str, List[FeedItem], Optional[str]]:
    try:
        # 짧은 HEAD 체크
        try:
            h = session.head(url, timeout=3, allow_redirects=True, headers={"User-Agent": "daily-issue-bot/1.0"})
            if h.status_code == 405:
                pass
        except requests.exceptions.Timeout:
            return url, [], "HEAD timeout"
        except requests.exceptions.RequestException:
            return url, [], "HEAD failed"

        r = session.get(url, timeout=timeout, headers={"User-Agent": "daily-issue-bot/1.0"})
        if r.status_code != 200:
            return url, [], f"HTTP {r.status_code}"

        parsed = feedparser.parse(r.content)
        src = (parsed.feed.get("title") or "").strip() or re.sub(r"^https?://", "", url).split("/")[0]
        now = datetime.now(KST)

        items: List[FeedItem] = []
        for e in parsed.entries[:40]:
            t = (getattr(e, "title", "") or "").strip()
            if not t:
                continue
            link = (getattr(e, "link", "") or "").strip()
            dt = _entry_dt(e) or now
            items.append(FeedItem(title=t, link=link, source=src, published=dt))
        return url, items, None

    except Exception as ex:
        return url, [], f"{type(ex).__name__}: {ex}"


def fetch_all(cfg: AppConfig, urls: List[str]) -> Tuple[List[FeedItem], Dict[str, str]]:
    errors: Dict[str, str] = {}
    items: List[FeedItem] = []

    t0 = datetime.now(KST)
    total = len(urls)
    print(f"[FETCH] start feeds={total} workers={cfg.run.max_workers} timeout={cfg.run.feed_timeout}s")

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as ex:
            futs = [ex.submit(fetch_feed, session, u, cfg.run.feed_timeout) for u in urls]
            done = 0

            for f in as_completed(futs):
                u, got, err = f.result()
                done += 1
                if cfg.run.debug or done % 10 == 0:
                    print(f"[FETCH] {done}/{total} got={len(got)} err={'Y' if err else 'N'}")

                if err:
                    errors[u] = err
                else:
                    items.extend(got)

    t1 = datetime.now(KST)
    print(f"[FETCH] done in {(t1 - t0).total_seconds():.1f}s items={len(items)} errors={len(errors)}")
    return items, errors


# -----------------------------
# Keyword extraction
# -----------------------------
TAIL_PATTERNS = [
    r"\s+[-–—]\s+[^-–—]{1,30}$",
    r"\s+\|\s+[^|]{1,30}$",
    r"\s+:\s*네이버\s*뉴스$",
    r"\s+:\s*Daum$",
    r"\s+\(\s*네이버\s*뉴스\s*\)\s*$",
    r"\s+\(\s*다음\s*\)\s*$",
]


def normalize_title(title: str) -> str:
    t = htmlmod.unescape(title or "").strip()
    t = re.sub(r"\s+", " ", t)
    for pat in TAIL_PATTERNS:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def is_domainy(tok: str) -> bool:
    if "." in tok:
        return True
    if re.fullmatch(r"(co|com|net|org|kr|tv|io|me|ai)", tok.lower() or ""):
        return True
    return False


def tokenize(cfg: AppConfig, title: str) -> List[str]:
    t = normalize_title(title)
    t = re.sub(r"[^\w가-힣]+", " ", t, flags=re.UNICODE)
    raw = re.findall(r"[A-Za-z]{2,}|[0-9]{2,}|[가-힣]{2,}", t)

    out: List[str] = []
    for tok in raw:
        tok = tok.strip()
        if not tok:
            continue
        low = tok.lower()

        if low in cfg.blocklist:
            continue
        if is_domainy(low):
            continue
        if re.fullmatch(r"[0-9]{2,}", low):
            continue
        if re.fullmatch(r"[a-z]{2}", low) and low not in ("ai", "us", "eu"):
            continue

        out.append(tok if re.search(r"[가-힣]", tok) else low)
    return out


def phrases_from_tokens(tokens: List[str]) -> List[str]:
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
    return bigrams + tokens


def score_keywords(cfg: AppConfig, items: List[FeedItem], trends: List[str]) -> List[Dict[str, Any]]:
    now = datetime.now(KST)
    cutoff = now - timedelta(hours=cfg.run.window_hours)

    trend_set = {normalize_title(x).lower() for x in trends if x}

    counts: Dict[str, int] = {}
    sources: Dict[str, Set[str]] = {}
    examples: Dict[str, List[FeedItem]] = {}

    recent = [it for it in items if it.published >= cutoff]

    for it in recent:
        toks = tokenize(cfg, it.title)
        if not toks:
            continue
        phs = phrases_from_tokens(toks)

        seen = set()
        for ph in phs:
            key = ph.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)

            # phrase 내부에 블록리스트 포함되면 제거
            if any(w.lower() in cfg.blocklist for w in key.split()):
                continue

            counts[key] = counts.get(key, 0) + 1
            sources.setdefault(key, set()).add(it.source)
            examples.setdefault(key, [])
            if len(examples[key]) < 4:
                examples[key].append(it)

    scored: List[Tuple[float, str]] = []
    for k, cnt in counts.items():
        src = len(sources.get(k, set()))

        if cnt < 2 and k not in trend_set:
            continue

        trend_bonus = 0.0
        if k in trend_set:
            trend_bonus += 7.0
        else:
            for tk in trend_set:
                if tk and (tk in k or k in tk):
                    trend_bonus += 3.0
                    break

        score = cnt * 1.0 + max(0, src - 1) * 1.8 + trend_bonus
        scored.append((score, k))

    scored.sort(reverse=True, key=lambda x: x[0])

    def jacc(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / max(1, len(a | b))

    final: List[Dict[str, Any]] = []
    used: List[Set[str]] = []

    for score, k in scored:
        toks = set(k.split())
        if any(jacc(toks, ut) >= 0.6 for ut in used):
            continue

        used.append(toks)
        final.append(
            {
                "keyword": k,
                "score": round(score, 2),
                "mentions": counts.get(k, 0),
                "sources": len(sources.get(k, set())),
                "examples": examples.get(k, [])[:3],
                "is_trend": (k in trend_set),
            }
        )
        if len(final) >= cfg.run.limit:
            break

    return final


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-issue-bot/1.0"}


def wp_find_post_by_slug(cfg: WordPressConfig, slug: str) -> Optional[Tuple[int, str]]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = wp_auth_header(cfg.username, cfg.app_password)
    r = requests.get(url, relaxed=True) if False else requests.get(url, headers=headers, params={"slug": slug, "per_page": 1}, timeout=25)
    if r.status_code != 200:
        return None
    arr = r.json()
    if not arr:
        return None
    return int(arr[0]["id"]), str(arr[0].get("link") or "")


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.username, cfg.app_password), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": cfg.status,
        "categories": cfg.category_ids,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:400]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, slug: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.username, cfg.app_password), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": cfg.status,
        "categories": cfg.category_ids,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:400]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


# -----------------------------
# Render
# -----------------------------
def esc(s: str) -> str:
    return htmlmod.escape(s or "")


def slot_label(slot: str) -> str:
    return "오전" if slot == "am" else "오후"


def build_html(cfg: AppConfig, date_str: str, trends: List[Dict[str, Any]], keywords: List[Dict[str, Any]], stats: Dict[str, Any]) -> str:
    disclosure = (
        '<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">'
        "※ Google 트렌딩 검색어 + 커뮤니티/뉴스 헤드라인을 기반으로 '데일리 이슈 키워드'를 자동 집계했습니다."
        "</p>"
    )
    head = (
        f"<p>기준일: <b>{esc(date_str)}</b> / 구분: <b>{esc(slot_label(cfg.run.slot))}</b><br/>"
        f"집계범위: 최근 <b>{cfg.run.window_hours}시간</b> / 수집헤드라인: <b>{stats.get('items',0)}</b>개 / "
        f"피드성공: <b>{stats.get('ok_feeds',0)}</b> / 실패: <b>{stats.get('err_feeds',0)}</b></p>"
    )

    tr_rows = []
    for i, t in enumerate(trends[:cfg.run.limit], start=1):
        kw = esc(t.get("keyword", ""))
        link = esc(t.get("link", ""))
        traffic = esc(t.get("traffic", ""))
        kw_html = f'<a href="{link}" target="_blank" rel="nofollow noopener">{kw}</a>' if link else kw
        tr_rows.append(
            f"<tr>"
            f"<td style='padding:8px;border:1px solid #e5e5e5;text-align:center;'>{i}</td>"
            f"<td style='padding:8px;border:1px solid #e5e5e5;'>{kw_html}</td>"
            f"<td style='padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;'>{traffic}</td>"
            f"</tr>"
        )

    trends_html = f"""
    <h2>Google 트렌딩 검색어</h2>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">순위</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">키워드</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">트래픽</th>
        </tr>
      </thead>
      <tbody>{''.join(tr_rows) if tr_rows else "<tr><td colspan='3' style='padding:8px;border:1px solid #e5e5e5;'>데이터 없음</td></tr>"}</tbody>
    </table>
    """

    rows = []
    for i, k in enumerate(keywords, start=1):
        label = esc(k["keyword"])
        mentions = k.get("mentions", 0)
        srcs = k.get("sources", 0)
        is_tr = "✅" if k.get("is_trend") else ""
        ex_lines = []
        for it in k.get("examples", []):
            tt = esc(normalize_title(it.title))
            lk = esc(it.link)
            sc = esc(it.source)
            if lk:
                ex_lines.append(f'• <a href="{lk}" target="_blank" rel="nofollow noopener">{tt}</a> <span style="opacity:.7;">({sc})</span>')
            else:
                ex_lines.append(f"• {tt} <span style='opacity:.7;'>({sc})</span>")

        rows.append(
            f"<tr>"
            f"<td style='padding:8px;border:1px solid #e5e5e5;text-align:center;'>{i}</td>"
            f"<td style='padding:8px;border:1px solid #e5e5e5;'><b>{label}</b> {is_tr}</td>"
            f"<td style='padding:8px;border:1px solid #e5e5e5;text-align:center;'>{mentions}</td>"
            f"<td style='padding:8px;border:1px solid #e5e5e5;text-align:center;'>{srcs}</td>"
            f"<td style='padding:8px;border:1px solid #e5e5e5;font-size:13px;line-height:1.55;'>{'<br/>'.join(ex_lines)}</td>"
            f"</tr>"
        )

    kw_html = f"""
    <h2>데일리 이슈 키워드 TOP {len(keywords)}</h2>
    <p style="font-size:12px;opacity:.75;">※ ✅ 표시 = Google 트렌딩에도 걸린 키워드(가중치 적용)</p>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">순위</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">키워드</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">언급</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">출처수</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">대표 링크</th>
        </tr>
      </thead>
      <tbody>{''.join(rows) if rows else "<tr><td colspan='5' style='padding:8px;border:1px solid #e5e5e5;'>키워드 부족(피드/커뮤니티 소스 확대 필요)</td></tr>"}</tbody>
    </table>
    """
    return disclosure + head + trends_html + "<hr/>" + kw_html + "<hr/><p style='font-size:12px;opacity:.7;'>자동 포스팅 봇</p>"


# -----------------------------
# CLI
# -----------------------------
def parse_args(argv: List[str]) -> Dict[str, Any]:
    out = {"dry_run": False, "debug": False, "slot": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--dry-run":
            out["dry_run"] = True
            i += 1
            continue
        if a == "--debug":
            out["debug"] = True
            i += 1
            continue
        if a == "--slot" and i + 1 < len(argv):
            out["slot"] = argv[i + 1].strip().lower()
            i += 2
            continue
        i += 1
    return out


def main() -> None:
    args = parse_args(sys.argv[1:])
    cfg = load_cfg()

    if args["dry_run"]:
        cfg.run.dry_run = True
    if args["debug"]:
        cfg.run.debug = True
    if args["slot"] in ("am", "pm"):
        cfg.run.slot = args["slot"]

    validate_cfg(cfg)
    print_safe(cfg)

    now = datetime.now(KST)
    date_str = now.strftime("%Y-%m-%d")

    slug = f"daily-issue-keywords-{date_str}-{cfg.run.slot}"
    title = f"{date_str} 데일리 이슈 키워드 TOP{cfg.run.limit} ({slot_label(cfg.run.slot)})"

    # 1) Google Trends
    try:
        trends = fetch_google_trends(cfg.run.google_geo, cfg.run.limit)
        print(f"[TRENDS] ok {len(trends)} items")
    except Exception as ex:
        trends = []
        print("[TRENDS] failed:", ex)

    trend_keywords = [t.get("keyword", "") for t in trends if t.get("keyword")]

    # 2) feeds (여기서 예전 코드가 무한루프였음)
    print("[FEEDS] building feed list...")
    feeds = build_feed_list(cfg)
    print(f"[FEEDS] total urls={len(feeds)} (community+news)")

    items, errors = fetch_all(cfg, feeds)

    ok_feeds = len(feeds) - len(errors)
    err_feeds = len(errors)
    if cfg.run.debug and errors:
        print("[FEEDS] sample errors:")
        for u, e in list(errors.items())[:15]:
            print(" -", u, "=>", e)

    # 3) scoring
    keywords = score_keywords(cfg, items, trend_keywords)
    print(f"[SCORE] keywords={len(keywords)}")

    stats = {"items": len(items), "ok_feeds": ok_feeds, "err_feeds": err_feeds}
    html = build_html(cfg, date_str, trends, keywords, stats)

    if cfg.run.dry_run:
        print("[DRY_RUN] Posting skipped. HTML preview:\n")
        print(html)
        return

    # 4) WP upsert by slug
    existing = wp_find_post_by_slug(cfg.wp, slug)
    if existing:
        post_id, old_link = existing
        new_id, new_link = wp_update_post(cfg.wp, post_id, title, slug, html)
        print(f"OK(updated): {new_id} {new_link or old_link}")
    else:
        new_id, new_link = wp_create_post(cfg.wp, title, slug, html)
        print(f"OK(created): {new_id} {new_link}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
