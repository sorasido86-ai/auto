# -*- coding: utf-8 -*-
"""
hot_issue_ranking_daily_to_wp.py (완전 통합)
- RSS 100개(기본: Google News RSS 주제+키워드 검색 RSS)에서 헤드라인 수집
- 최근 N시간(window) 기준으로 "핫이슈 TOP20" 자동 집계
- 오전/오후(AM/PM) 별로 "서로 다른 글"로 워드프레스에 발행/업데이트(슬러그 기반)
- GitHub Actions에서 Secrets(환경변수)로 실행 가능

✅ 너의 워드프레스 Secrets 이름 그대로 사용:
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

✅ 카테고리 번호: 기본 4번
  - WP_CATEGORY_IDS="4" (콤마로 여러개 가능)

(옵션)
- RSS_FEEDS: 직접 RSS 목록(줄바꿈 구분) 제공하면 기본 100개 대신 이걸 사용
- WINDOW_HOURS: 몇 시간치로 집계할지(기본 12)
- LIMIT: 이슈 TOP N (기본 20)
"""

from __future__ import annotations

import base64
import html as htmlmod
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import feedparser


KST = timezone(timedelta(hours=9))


# -----------------------------
# Config helpers
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
    s = _env(name, "1" if default else "0").lower()
    return s in ("1", "true", "yes", "y", "on")


def _parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


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
    feed_timeout: int = 15
    max_workers: int = 12
    dry_run: bool = False
    debug: bool = False
    slot: str = "am"  # am|pm
    feeds_max: int = 100


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig


def load_cfg_from_env() -> AppConfig:
    wp_base = (_env("WP_BASE_URL") or _env("WP_SITE_URL")).rstrip("/")
    wp_user = _env("WP_USER") or _env("WP_USERNAME")
    wp_pass = _env("WP_APP_PASS") or _env("WP_APP_PASSWORD")
    wp_status = _env("WP_STATUS", "publish") or "publish"
    wp_cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "4")) or [4]

    run = RunConfig(
        limit=_env_int("LIMIT", 20),
        window_hours=_env_int("WINDOW_HOURS", 12),
        feed_timeout=_env_int("FEED_TIMEOUT", 15),
        max_workers=_env_int("MAX_WORKERS", 12),
        dry_run=_env_bool("DRY_RUN", False),
        debug=_env_bool("DEBUG", False),
        slot=(_env("RUN_SLOT", "am") or "am").lower(),
        feeds_max=_env_int("FEEDS_MAX", 100),
    )
    if run.slot not in ("am", "pm"):
        run.slot = "am"

    wp = WordPressConfig(
        base_url=wp_base,
        username=wp_user,
        app_password=wp_pass,
        status=wp_status,
        category_ids=wp_cat_ids,
    )
    return AppConfig(wp=wp, run=run)


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


def print_safe_settings(cfg: AppConfig) -> None:
    def ok(v: str) -> str:
        return f"OK(len={len(v)})" if v else "MISSING"
    print("[CONFIG] WP_BASE_URL:", cfg.wp.base_url or "MISSING")
    print("[CONFIG] WP_USER:", ok(cfg.wp.username))
    print("[CONFIG] WP_APP_PASS:", ok(cfg.wp.app_password))
    print("[CONFIG] WP_STATUS:", cfg.wp.status)
    print("[CONFIG] WP_CATEGORY_IDS:", cfg.wp.category_ids)
    print("[CONFIG] LIMIT:", cfg.run.limit, "WINDOW_HOURS:", cfg.run.window_hours, "SLOT:", cfg.run.slot)
    print("[CONFIG] FEEDS_MAX:", cfg.run.feeds_max, "MAX_WORKERS:", cfg.run.max_workers, "TIMEOUT:", cfg.run.feed_timeout)
    print("[CONFIG] DRY_RUN:", cfg.run.dry_run, "DEBUG:", cfg.run.debug)


# -----------------------------
# RSS feed list (default 100)
# -----------------------------
def google_news_rss_top() -> str:
    return "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko"


def google_news_rss_topic(topic: str) -> str:
    # topic: WORLD/NATION/BUSINESS/TECHNOLOGY/ENTERTAINMENT/SCIENCE/SPORTS/HEALTH
    return f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=ko&gl=KR&ceid=KR:ko"


def google_news_rss_search(query: str) -> str:
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"


def default_keywords() -> List[str]:
    # 91개(= 100 - (top 1 + topic 8)) 정도로 맞춤
    # 너무 흔한 조사/단어 대신 이슈성 키워드를 섞어둠. 필요하면 여기만 바꿔도 됨.
    return [
        "속보", "긴급", "단독", "실시간",
        "정치", "국회", "대통령", "총리", "선거", "검찰", "법원",
        "경제", "증시", "코스피", "코스닥", "환율", "금리", "물가", "부동산", "아파트", "전세",
        "삼성전자", "SK하이닉스", "LG에너지솔루션", "현대차", "네이버", "카카오",
        "애플", "테슬라", "엔비디아", "마이크로소프트", "구글",
        "AI", "챗GPT", "OpenAI", "반도체", "배터리", "전기차", "로봇", "빅테크",
        "우크라이나", "러시아", "중국", "일본", "북한", "미국", "중동", "이스라엘", "가자",
        "유가", "원유", "천연가스", "비트코인", "이더리움", "가상자산",
        "사고", "화재", "산불", "지진", "태풍", "미세먼지", "날씨",
        "의료", "병원", "독감", "백신",
        "교육", "수능", "대학", "취업",
        "스포츠", "축구", "손흥민", "야구", "KBO", "농구",
        "연예", "드라마", "영화", "넷플릭스", "K팝", "BTS",
        "보안", "해킹", "개인정보", "랜섬웨어",
        "기후변화", "탄소", "원전", "재생에너지", "수소",
    ]


def default_feed_urls(feeds_max: int = 100) -> List[str]:
    urls: List[str] = []
    urls.append(google_news_rss_top())
    for t in ["WORLD", "NATION", "BUSINESS", "TECHNOLOGY", "ENTERTAINMENT", "SCIENCE", "SPORTS", "HEALTH"]:
        urls.append(google_news_rss_topic(t))

    # 나머지는 키워드 검색 RSS로 채움
    for kw in default_keywords():
        urls.append(google_news_rss_search(kw))

    # 중복 제거 + max 제한
    seen = set()
    out: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= feeds_max:
            break
    return out


def load_feed_urls(cfg: AppConfig) -> List[str]:
    raw = _env("RSS_FEEDS", "")
    if raw:
        feeds = []
        for line in raw.splitlines():
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            feeds.append(u)
        # 중복 제거
        dedup = []
        seen = set()
        for u in feeds:
            if u in seen:
                continue
            seen.add(u)
            dedup.append(u)
        if cfg.run.debug:
            print(f"[RSS] Using RSS_FEEDS override: {len(dedup)} feeds")
        return dedup[: cfg.run.feeds_max]

    feeds = default_feed_urls(cfg.run.feeds_max)
    if cfg.run.debug:
        print(f"[RSS] Using default feeds: {len(feeds)} feeds")
    return feeds


# -----------------------------
# RSS fetching / parsing
# -----------------------------
@dataclass
class Item:
    title: str
    link: str
    source: str
    published: datetime


def _to_dt(entry: Any) -> Optional[datetime]:
    # feedparser gives time.struct_time in published_parsed / updated_parsed
    for key in ("published_parsed", "updated_parsed"):
        v = getattr(entry, key, None)
        if v:
            try:
                return datetime(*v[:6], tzinfo=timezone.utc).astimezone(KST)
            except Exception:
                pass
    return None


def fetch_one_feed(session: requests.Session, url: str, timeout: int) -> Tuple[str, List[Item], Optional[str]]:
    headers = {"User-Agent": "hot-issue-bot/1.0 (+rss)"}
    try:
        r = session.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return url, [], f"HTTP {r.status_code}"
        parsed = feedparser.parse(r.content)
        feed_title = (parsed.feed.get("title") or "").strip() or "RSS"
        items: List[Item] = []
        now_kst = datetime.now(KST)

        for e in parsed.entries[:30]:  # 한 RSS에서 너무 많이 안가져오도록 상한
            title = (getattr(e, "title", "") or "").strip()
            link = (getattr(e, "link", "") or "").strip()
            if not title:
                continue
            dt = _to_dt(e) or now_kst
            items.append(Item(title=title, link=link, source=feed_title, published=dt))
        return url, items, None
    except Exception as ex:
        return url, [], f"{type(ex).__name__}: {ex}"


def fetch_all_items(cfg: AppConfig, feed_urls: List[str]) -> Tuple[List[Item], Dict[str, str]]:
    errors: Dict[str, str] = {}
    items: List[Item] = []
    with requests.Session() as session:
        futures = []
        with ThreadPoolExecutor(max_workers=cfg.run.max_workers) as ex:
            for u in feed_urls:
                futures.append(ex.submit(fetch_one_feed, session, u, cfg.run.feed_timeout))

            for f in as_completed(futures):
                url, got, err = f.result()
                if err:
                    errors[url] = err
                else:
                    items.extend(got)
    return items, errors


# -----------------------------
# Issue extraction
# -----------------------------
STOPWORDS = set([
    "기자", "뉴스", "속보", "단독", "오늘", "내일", "어제", "오전", "오후", "관련", "논란",
    "영상", "사진", "현장", "분석", "전망", "이유", "확인", "공개", "발표", "최신", "업데이트",
    "the", "and", "for", "with", "from", "into", "this", "that", "are", "was",
])


def normalize_title(s: str) -> str:
    s = htmlmod.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(title: str) -> List[str]:
    t = normalize_title(title)
    t = re.sub(r"[^\w가-힣]+", " ", t, flags=re.UNICODE)
    raw = re.findall(r"[A-Za-z0-9]+|[가-힣]{2,}", t)
    out: List[str] = []
    for tok in raw:
        tok = tok.strip()
        if not tok:
            continue
        if re.fullmatch(r"[A-Za-z0-9]+", tok):
            tok2 = tok.lower()
        else:
            tok2 = tok
        if tok2 in STOPWORDS:
            continue
        if len(tok2) < 2:
            continue
        out.append(tok2)
    return out


def build_ngrams(tokens: List[str]) -> Tuple[List[str], List[str]]:
    unigrams = tokens
    bigrams: List[str] = []
    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]
        if a in STOPWORDS or b in STOPWORDS:
            continue
        bigrams.append(f"{a} {b}")
    return unigrams, bigrams


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


@dataclass
class Issue:
    label: str
    count: int
    sources: int
    examples: List[Item]


def extract_issues(cfg: AppConfig, items: List[Item]) -> Tuple[List[Issue], Dict[str, Any]]:
    now_kst = datetime.now(KST)
    window_start = now_kst - timedelta(hours=cfg.run.window_hours)

    # 시간 필터 + 중복 타이틀 제거
    dedup_titles = set()
    recent: List[Item] = []
    for it in items:
        if it.published < window_start:
            continue
        key = normalize_title(it.title).lower()
        if key in dedup_titles:
            continue
        dedup_titles.add(key)
        recent.append(it)

    # phrase stats
    phrase_count: Dict[str, int] = {}
    phrase_sources: Dict[str, set] = {}
    phrase_examples: Dict[str, List[Item]] = {}

    for it in recent:
        toks = tokenize(it.title)
        if not toks:
            continue
        uni, bi = build_ngrams(toks)

        # bigram 우선 + unigram도 같이 카운트
        phrases = bi + uni

        # 한 제목에서 같은 phrase 중복 카운트 방지
        seen_in_title = set()
        for ph in phrases:
            if ph in seen_in_title:
                continue
            seen_in_title.add(ph)

            phrase_count[ph] = phrase_count.get(ph, 0) + 1
            phrase_sources.setdefault(ph, set()).add(it.source)
            phrase_examples.setdefault(ph, [])
            if len(phrase_examples[ph]) < 6:
                phrase_examples[ph].append(it)

    # 후보: 최소 2회 이상 언급
    candidates = []
    for ph, cnt in phrase_count.items():
        if cnt < 2:
            continue
        srcs = len(phrase_sources.get(ph, set()))
        # 점수: 언급수 + 출처 다양성 가중
        score = cnt * (1.0 + 0.25 * max(0, srcs - 1))
        candidates.append((score, ph, cnt, srcs))

    candidates.sort(reverse=True, key=lambda x: x[0])

    selected: List[Issue] = []
    selected_tokens: List[List[str]] = []

    for _, ph, cnt, srcs in candidates:
        ph_tokens = ph.split()
        # 너무 흔한 단일 토큰은 컷(예: '정치' 같은 일반어) → 그래도 남길 수 있게 완전 금지는 안함
        if len(ph_tokens) == 1 and ph_tokens[0] in STOPWORDS:
            continue

        # 기존 선택과 과도하게 겹치면 스킵(중복 이슈 방지)
        is_dup = False
        for prev in selected_tokens:
            if jaccard(prev, ph_tokens) >= 0.6:
                is_dup = True
                break
        if is_dup:
            continue

        ex = phrase_examples.get(ph, [])[:3]
        selected.append(Issue(label=ph, count=cnt, sources=srcs, examples=ex))
        selected_tokens.append(ph_tokens)

        if len(selected) >= cfg.run.limit:
            break

    meta = {
        "now_kst": now_kst,
        "window_start": window_start,
        "recent_items": len(recent),
        "unique_phrases": len(phrase_count),
    }
    return selected, meta


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "hot-issue-bot/1.0"}


def wp_get_post_by_slug(cfg: WordPressConfig, slug: str) -> Optional[Dict[str, Any]]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts"
    headers = wp_auth_header(cfg.username, cfg.app_password)
    r = requests.get(url, headers=headers, params={"slug": slug, "per_page": 1}, timeout=25)
    if r.status_code != 200:
        return None
    arr = r.json()
    if isinstance(arr, list) and arr:
        return arr[0]
    return None


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
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:400]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.username, cfg.app_password), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "content": html,
        "status": cfg.status,
        "categories": cfg.category_ids,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:400]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


# -----------------------------
# Rendering
# -----------------------------
def slot_label(slot: str) -> str:
    return "오전" if slot == "am" else "오후"


def esc(s: str) -> str:
    return htmlmod.escape(s or "")


def build_post_html(
    cfg: AppConfig,
    issues: List[Issue],
    meta: Dict[str, Any],
    ok_feeds: int,
    err_feeds: int,
    errors: Dict[str, str],
) -> str:
    now_kst: datetime = meta["now_kst"]
    window_start: datetime = meta["window_start"]

    disclosure = (
        '<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">'
        "※ 이 글은 <b>RSS 헤드라인</b>을 자동 수집해, 최근 기사 제목에서 반복 등장한 키워드를 기반으로 "
        "<b>핫이슈 TOP</b>을 집계한 결과입니다. (완전 자동)"
        "</p>"
    )

    stats = (
        f"<p>기준시각(KST): <b>{now_kst.strftime('%Y-%m-%d %H:%M')}</b><br/>"
        f"집계범위: <b>{cfg.run.window_hours}시간</b> ( {window_start.strftime('%m/%d %H:%M')} ~ {now_kst.strftime('%m/%d %H:%M')} )<br/>"
        f"RSS: 성공 <b>{ok_feeds}</b> / 실패 <b>{err_feeds}</b></p>"
    )

    rows = []
    for i, iss in enumerate(issues, start=1):
        ex_html = []
        for it in iss.examples:
            title = esc(normalize_title(it.title))
            link = esc(it.link)
            src = esc(it.source)
            if link:
                ex_html.append(f'• <a href="{link}" target="_blank" rel="nofollow noopener">{title}</a> <span style="opacity:.7;">({src})</span>')
            else:
                ex_html.append(f"• {title} <span style='opacity:.7;'>({src})</span>")

        rows.append(
            f"""
            <tr>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;"><b>{esc(iss.label)}</b></td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{iss.count}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{iss.sources}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;font-size:13px;line-height:1.55;">{"<br/>".join(ex_html) if ex_html else ""}</td>
            </tr>
            """
        )

    table = f"""
    <h2>핫이슈 TOP {cfg.run.limit} ({esc(slot_label(cfg.run.slot))})</h2>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">순위</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">이슈</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">언급</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">출처수</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">대표 헤드라인</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """

    err_block = ""
    if errors and cfg.run.debug:
        # debug일 때만 상세 노출(너무 길어질 수 있어서)
        lines = []
        for u, e in list(errors.items())[:20]:
            lines.append(f"<li><code>{esc(u)}</code> — {esc(e)}</li>")
        err_block = "<h3>RSS 실패(일부)</h3><ul>" + "".join(lines) + "</ul>"

    footer = (
        '<hr/><p style="font-size:12px;opacity:.75;">'
        "자동 포스팅 봇 · RSS 기반 집계"
        "</p>"
    )

    return disclosure + stats + table + err_block + footer


# -----------------------------
# Main
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
        if a in ("--slot",) and i + 1 < len(argv):
            out["slot"] = argv[i + 1].strip().lower()
            i += 2
            continue
        i += 1
    return out


def main() -> None:
    args = parse_args(sys.argv[1:])
    cfg = load_cfg_from_env()
    if args["dry_run"]:
        cfg.run.dry_run = True
    if args["debug"]:
        cfg.run.debug = True
    if args["slot"] in ("am", "pm"):
        cfg.run.slot = args["slot"]

    validate_cfg(cfg)
    print_safe_settings(cfg)

    # 날짜/슬롯 기준(한국시간)
    now_kst = datetime.now(KST)
    date_str = now_kst.strftime("%Y-%m-%d")

    slug = f"hot-issues-{date_str}-{cfg.run.slot}"
    title = f"{date_str} 핫이슈 TOP{cfg.run.limit} ({slot_label(cfg.run.slot)})"

    feed_urls = load_feed_urls(cfg)

    items, errors = fetch_all_items(cfg, feed_urls)
    ok_feeds = len(feed_urls) - len(errors)
    err_feeds = len(errors)

    issues, meta = extract_issues(cfg, items)
    html = build_post_html(cfg, issues, meta, ok_feeds, err_feeds, errors)

    if cfg.run.dry_run:
        print("[DRY_RUN] Posting skipped. HTML preview:\n")
        print(html)
        return

    # slug로 기존 글 있으면 update, 없으면 create
    existing = wp_get_post_by_slug(cfg.wp, slug)
    if existing:
        post_id = int(existing.get("id"))
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html)
        print(f"OK(updated): {wp_post_id} {wp_link}")
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, html)
        print(f"OK(created): {wp_post_id} {wp_link}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
