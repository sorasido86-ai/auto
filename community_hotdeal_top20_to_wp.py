# -*- coding: utf-8 -*-
"""
community_hotdeal_top20_to_wp.py (완전 통합/하루 2번 슬롯 지원)

- 커뮤니티 핫딜(루리웹/뽐뿌RSS/FM코리아) 수집 → TOP N 스코어링
- RUN_SLOT(am/pm/auto/both) 지원
  - auto: KST 기준 12시 이전 am, 이후 pm
  - both: 한 번 실행으로 am/pm 글을 각각 따로 생성/업데이트(같은 시각 기준이라 내용은 유사할 수 있음)
- 오전/오후 RUN_SLOT 별로 글을 "따로" 생성(슬롯별 slug 고정)
- WordPress REST API로 생성/업데이트
- SQLite로 발행 이력 저장(같은 슬롯은 업데이트)

✅ GitHub Secrets(환경변수) 이름 그대로 사용:
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

✅ 옵션 환경변수:
  - WP_STATUS: publish (기본 publish)
  - WP_CATEGORY_IDS: "4" (기본 4)
  - WP_TAG_IDS: "1,2,3" (선택)
  - SQLITE_PATH: data/community_hotdeal.sqlite3
  - WINDOW_HOURS: 18 (기본 18시간 내 글만)
  - LIMIT: 20 (기본 20)
  - RUN_SLOT: am / pm / auto / both (기본 auto)
  - DRY_RUN: 1이면 워드프레스 발행 안하고 미리보기 출력
  - DEBUG: 1이면 상세 로그/테스트 출력
  - SOURCES_JSON: {"key":"url"} 형태로 소스 교체 가능
"""

from __future__ import annotations

import base64
import hashlib
import html as htmlmod
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import feedparser
from bs4 import BeautifulSoup


KST = timezone(timedelta(hours=9))

DEFAULT_SOURCES = {
    "ruliweb_hotdeal": "https://bbs.ruliweb.com/market/board/1020",
    "ppomppu_rss_domestic": "http://www.ppomppu.co.kr/rss.php?id=ppomppu",
    "ppomppu_rss_oversea": "http://www.ppomppu.co.kr/rss.php?id=ppomppu4",
    "fmkorea_hotdeal": "https://m.fmkorea.com/hotdeal",
}


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
    run_slot: str = "auto"  # am / pm / auto / both
    window_hours: int = 18
    limit: int = 20
    dry_run: bool = False
    debug: bool = False


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    sqlite_path: str
    sources: Dict[str, str]


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "4"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    run_slot = (_env("RUN_SLOT", "auto") or "auto").lower()
    if run_slot not in ("am", "pm", "auto", "both"):
        run_slot = "auto"

    window_hours = _env_int("WINDOW_HOURS", 18)
    limit = _env_int("LIMIT", 20)

    sqlite_path = _env("SQLITE_PATH", "data/community_hotdeal.sqlite3")
    dry_run = _env_bool("DRY_RUN", False)
    debug = _env_bool("DEBUG", False)

    sources_json = _env("SOURCES_JSON", "")
    sources = DEFAULT_SOURCES.copy()
    if sources_json:
        try:
            j = json.loads(sources_json)
            if isinstance(j, dict):
                for k, v in j.items():
                    if isinstance(k, str) and isinstance(v, str) and v.startswith("http"):
                        sources[k] = v
        except Exception:
            pass

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
            window_hours=window_hours,
            limit=limit,
            dry_run=dry_run,
            debug=debug,
        ),
        sqlite_path=sqlite_path,
        sources=sources,
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot, "| WINDOW_HOURS:", cfg.run.window_hours, "| LIMIT:", cfg.run.limit)
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)
    print("[CFG] SQLITE_PATH:", cfg.sqlite_path)
    print("[CFG] SOURCES:", list(cfg.sources.keys()))


# -----------------------------
# SQLite (post history)
# -----------------------------
def init_db(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_posts (
          date_slot TEXT PRIMARY KEY,
          wp_post_id INTEGER,
          wp_link TEXT,
          created_at TEXT
        )
        """
    )
    con.commit()
    con.close()


def get_existing_post(path: str, date_slot: str) -> Optional[Tuple[int, str]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("SELECT wp_post_id, wp_link FROM daily_posts WHERE date_slot = ?", (date_slot,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return int(row[0]), str(row[1] or "")


def save_post_meta(path: str, date_slot: str, post_id: int, link: str) -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_slot, wp_post_id, wp_link, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (date_slot, post_id, link, datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "community-hotdeal-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": cfg.status,
    }
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
    payload: Dict[str, Any] = {
        "title": title,
        "content": html,
        "status": cfg.status,
    }
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=25)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_debug_check(cfg: WordPressConfig) -> None:
    base = cfg.base_url.rstrip("/")
    r = requests.get(base + "/wp-json/", timeout=20)
    print("[DEBUG] wp-json:", r.status_code)

    r2 = requests.get(base + "/wp-json/wp/v2/users/me", headers=wp_auth_header(cfg.user, cfg.app_pass), timeout=20)
    print("[DEBUG] users/me:", r2.status_code)
    if r2.status_code != 200:
        print("[DEBUG] users/me body:", r2.text[:500])
        raise RuntimeError("워드프레스 인증 실패(401/403). WP_USER/WP_APP_PASS 또는 보안플러그인 REST 차단 확인.")


# -----------------------------
# Deal model + scoring
# -----------------------------
@dataclass
class Deal:
    source: str
    title: str
    link: str
    created_at: datetime
    reco: int = 0
    views: int = 0
    comments: int = 0
    shop: str = ""
    price: str = ""
    score: float = 0.0

    def uid(self) -> str:
        h = hashlib.sha1((self.source + "|" + self.link).encode("utf-8")).hexdigest()
        return h[:16]


def clean_text(s: str) -> str:
    s = htmlmod.unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_shop_and_price(title: str) -> Tuple[str, str]:
    t = title

    shop = ""
    m = re.match(r"^\s*\[([^\]]{1,30})\]\s*", t)
    if m:
        shop = clean_text(m.group(1))

    price = ""
    m2 = re.search(r"([0-9]{1,3}(?:,[0-9]{3})+)\s*원", t)
    if m2:
        price = m2.group(1) + "원"
    else:
        m3 = re.search(r"\b([0-9]{4,7})\s*원\b", t)
        if m3:
            price = m3.group(1) + "원"

    return shop, price


def compute_score(d: Deal, now: datetime, window_hours: int) -> float:
    age_h = max(0.0, (now - d.created_at).total_seconds() / 3600.0)
    recency = max(0.0, (window_hours - age_h) / window_hours)  # 0~1
    s = 0.0
    s += recency * 10.0
    s += min(d.reco, 300) * 0.25
    s += min(d.comments, 300) * 0.10
    s += (min(d.views, 300000) ** 0.5) * 0.02
    return s


# -----------------------------
# Fetchers
# -----------------------------
def _requests_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; community-hotdeal-bot/1.0)"})
    return s


def fetch_ppomppu_rss(url: str, now: datetime, window_hours: int, source_name: str) -> List[Deal]:
    deals: List[Deal] = []
    feed = feedparser.parse(url)
    if getattr(feed, "bozo", 0) and not feed.entries:
        return deals

    for e in feed.entries[:80]:
        title = clean_text(getattr(e, "title", "") or "")
        link = clean_text(getattr(e, "link", "") or "")
        if not title or not link:
            continue

        dt = None
        if getattr(e, "published_parsed", None):
            try:
                dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc).astimezone(KST)
            except Exception:
                dt = None
        if dt is None:
            dt = now

        if (now - dt) > timedelta(hours=window_hours):
            continue

        shop, price = parse_shop_and_price(title)
        deals.append(
            Deal(
                source=source_name,
                title=title,
                link=link,
                created_at=dt,
                shop=shop,
                price=price,
            )
        )
    return deals


def fetch_ruliweb_hotdeal(url: str, now: datetime, window_hours: int) -> List[Deal]:
    deals: List[Deal] = []
    with _requests_session() as s:
        r = s.get(url, timeout=20)
        if r.status_code != 200:
            return deals

    soup = BeautifulSoup(r.text, "html.parser")
    anchors = soup.find_all("a", href=True)
    seen = set()

    for a in anchors:
        href = a.get("href", "")
        if not href:
            continue
        if "/market/board/1020/read/" not in href:
            continue

        link = href if href.startswith("http") else urljoin(url, href)
        if link in seen:
            continue
        seen.add(link)

        title = clean_text(a.get_text(" ", strip=True))
        if not title or len(title) < 3:
            continue

        tr = a.find_parent("tr")
        if not tr:
            continue

        tr_text = clean_text(tr.get_text(" ", strip=True))

        comments = 0
        m_c = re.search(r"\((\d{1,4})\)", tr_text)
        if m_c:
            try:
                comments = int(m_c.group(1))
            except Exception:
                comments = 0

        reco = 0
        views = 0
        date_token = ""

        tds = tr.find_all("td")
        nums: List[int] = []
        for td in tds:
            tx = clean_text(td.get_text(" ", strip=True))
            if re.fullmatch(r"\d{1,7}", tx or ""):
                try:
                    nums.append(int(tx))
                except Exception:
                    pass
            if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", tx) or re.fullmatch(r"\d{2}:\d{2}", tx):
                date_token = tx

        if len(nums) >= 2:
            reco, views = nums[0], nums[1]

        created_at = now
        if date_token:
            try:
                if ":" in date_token and "." not in date_token:
                    hh, mm = date_token.split(":")
                    created_at = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
                elif "." in date_token:
                    y, m, d = date_token.split(".")
                    created_at = datetime(int(y), int(m), int(d), 0, 0, tzinfo=KST)
            except Exception:
                created_at = now

        if (now - created_at) > timedelta(hours=window_hours):
            continue

        shop, price = parse_shop_and_price(title)
        deals.append(
            Deal(
                source="ruliweb",
                title=title,
                link=link,
                created_at=created_at,
                reco=reco,
                views=views,
                comments=comments,
                shop=shop,
                price=price,
            )
        )

    return deals


def fetch_fmkorea_hotdeal(url: str, now: datetime, window_hours: int) -> List[Deal]:
    deals: List[Deal] = []
    with _requests_session() as s:
        r = s.get(url, timeout=20)
        if r.status_code != 200:
            return deals

    soup = BeautifulSoup(r.text, "html.parser")
    anchors = soup.find_all("a", href=True)
    seen = set()

    for a in anchors:
        href = a.get("href", "")
        title = clean_text(a.get_text(" ", strip=True))
        if not href or not title:
            continue

        if "document_srl=" not in href and "/hotdeal" not in href:
            continue

        link = href if href.startswith("http") else urljoin(url, href)
        if link in seen:
            continue
        seen.add(link)

        created_at = now  # 목록에서 정확 시간 추출이 어려워 현재 시각 처리

        shop, price = parse_shop_and_price(title)
        deals.append(
            Deal(
                source="fmkorea",
                title=title,
                link=link,
                created_at=created_at,
                shop=shop,
                price=price,
            )
        )

    return deals


# -----------------------------
# Rendering
# -----------------------------
DISCLOSURE = "※ 이 포스팅은 자료를 모으기 위해 만든 글입니다."
POST_NOTE = "데이터 출처: 커뮤니티 핫딜(루리웹/뽐뿌RSS/기타). 일부 사이트는 봇 차단으로 누락될 수 있습니다."


def fmt_dt(dt: datetime) -> str:
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")


def build_html(slot_label: str, now: datetime, deals: List[Deal]) -> str:
    disclosure = f'<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{htmlmod.escape(DISCLOSURE)}</p>'
    head = f"<p>기준시각: <b>{htmlmod.escape(fmt_dt(now))}</b> / 슬롯: <b>{htmlmod.escape(slot_label)}</b></p>"
    note = f'<p style="font-size:13px;opacity:.8;">{htmlmod.escape(POST_NOTE)}</p>'

    rows = []
    for i, d in enumerate(deals, start=1):
        title = htmlmod.escape(d.title)
        link = htmlmod.escape(d.link)
        src = htmlmod.escape(d.source)
        shop = htmlmod.escape(d.shop or "-")
        price = htmlmod.escape(d.price or "-")
        meta = f"추천 {d.reco} · 댓글 {d.comments} · 조회 {d.views}" if (d.reco or d.comments or d.views) else ""
        meta_html = f'<div style="font-size:12px;opacity:.7;margin-top:4px;">{htmlmod.escape(meta)}</div>' if meta else ""

        rows.append(
            f"""
            <tr>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{src}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">{shop}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">
                <a href="{link}" target="_blank" rel="nofollow sponsored noopener">{title}</a>
                {meta_html}
              </td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;">{price}</td>
            </tr>
            """
        )

    table = f"""
    <h2>커뮤니티 핫딜 TOP {len(deals)}</h2>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">순위</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">출처</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">쇼핑몰</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">제목</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">가격</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """
    return disclosure + head + note + table


# -----------------------------
# Main logic
# -----------------------------
def resolve_slots(now: datetime, run_slot: str) -> List[str]:
    rs = (run_slot or "auto").lower()
    if rs == "both":
        return ["am", "pm"]
    if rs == "auto":
        return ["am" if now.hour < 12 else "pm"]
    if rs in ("am", "pm"):
        return [rs]
    return ["am" if now.hour < 12 else "pm"]


def collect_deals(cfg: AppConfig, now: datetime) -> List[Deal]:
    deals_all: List[Deal] = []

    try:
        deals_all += fetch_ruliweb_hotdeal(cfg.sources["ruliweb_hotdeal"], now, cfg.run.window_hours)
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] ruliweb failed:", repr(e))

    try:
        deals_all += fetch_ppomppu_rss(cfg.sources["ppomppu_rss_domestic"], now, cfg.run.window_hours, "ppomppu")
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] ppomppu domestic rss failed:", repr(e))

    try:
        deals_all += fetch_ppomppu_rss(cfg.sources["ppomppu_rss_oversea"], now, cfg.run.window_hours, "ppomppu_oversea")
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] ppomppu oversea rss failed:", repr(e))

    try:
        deals_all += fetch_fmkorea_hotdeal(cfg.sources["fmkorea_hotdeal"], now, cfg.run.window_hours)
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] fmkorea failed:", repr(e))

    uniq: Dict[str, Deal] = {}
    for d in deals_all:
        if not d.link:
            continue
        if d.link not in uniq:
            uniq[d.link] = d

    deals = list(uniq.values())
    for d in deals:
        d.score = compute_score(d, now, cfg.run.window_hours)

    deals.sort(key=lambda x: x.score, reverse=True)
    return deals[: cfg.run.limit]


def publish_one_slot(cfg: AppConfig, slot: str, now: datetime, deals: List[Deal]) -> None:
    date_str = now.strftime("%Y-%m-%d")
    slot_label = "오전" if slot == "am" else "오후"
    date_slot = f"{date_str}_{slot}"

    title = f"{date_str} 커뮤니티 핫딜 TOP{len(deals)} ({slot_label})"
    slug = f"community-hotdeal-{date_str}-{slot}"
    html = build_html(slot_label, now, deals)

    if cfg.run.dry_run:
        print(f"[DRY_RUN] 슬롯={slot_label} 발행 생략. 미리보기 HTML ↓\n")
        print(html)
        print("\n" + "-" * 80 + "\n")
        return

    existing = get_existing_post(cfg.sqlite_path, date_slot)
    if existing:
        post_id, old_link = existing
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html)
        save_post_meta(cfg.sqlite_path, date_slot, wp_post_id, wp_link)
        print("OK(updated):", slot, wp_post_id, wp_link or old_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, html)
        save_post_meta(cfg.sqlite_path, date_slot, wp_post_id, wp_link)
        print("OK(created):", slot, wp_post_id, wp_link)


def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    slots = resolve_slots(now, cfg.run.run_slot)

    init_db(cfg.sqlite_path)

    # (수집은 한 번만) — 하루 2번은 보통 Actions에서 두 번 실행 권장
    deals = collect_deals(cfg, now)

    for slot in slots:
        publish_one_slot(cfg, slot, now, deals)


def main():
    cfg = load_cfg()
    validate_cfg(cfg)
    print_safe_cfg(cfg)

    if cfg.run.debug:
        wp_debug_check(cfg.wp)

    run(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
