# -*- coding: utf-8 -*-
"""
daily_pet_trending_to_wp.py (완전 통합/안정화)
- YouTube Data API v3: "Pets & Animals" 인기영상(차트 mostPopular) TOP N 수집
- am/pm 슬롯별로 글을 "따로" 생성/업데이트 (슬롯별 slug 고정)
- WordPress REST API로 생성/업데이트
- 대표이미지(Featured): 1등 영상 썸네일을 WP 미디어로 업로드 후 featured_media 설정(옵션)
- SQLite로 발행 이력 저장(같은 슬롯은 업데이트)
- YouTube API 실패 시: data/last_success_{slot}.json 캐시를 재사용(안정성)

필수 Secrets/환경변수:
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS
  - YOUTUBE_API_KEY

옵션 환경변수:
  - WP_STATUS=publish (기본 publish)
  - WP_CATEGORY_IDS="3" (기본 3)
  - WP_TAG_IDS="1,2,3" (선택)
  - SQLITE_PATH=data/daily_pet_trending.sqlite3
  - RUN_SLOT=am|pm (yml에서 자동 세팅)
  - REGION_CODE=KR (기본 KR)
  - VIDEO_CATEGORY_ID=15 (기본 15 = Pets & Animals)
  - LIMIT=12 (기본 12)
  - UPLOAD_THUMB=1 (기본 1)  : 대표이미지 업로드 시도
  - SET_FEATURED=1 (기본 1)  : featured_media 설정
  - REUSE_MEDIA_BY_SEARCH=1 (기본 1) : 미디어 중복 업로드 방지(검색 재사용)
  - DRY_RUN=1 (발행 대신 미리보기 출력)
  - DEBUG=1 (상세 로그)

주의:
- 이 방식은 "영상 링크/임베드 큐레이션"이라 가장 안정적이며 저작권 리스크가 낮습니다.
"""

from __future__ import annotations

import base64
import hashlib
import html as htmlmod
import json
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

KST = timezone(timedelta(hours=9))

# -----------------------------
# Env helpers
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
class YouTubeConfig:
    api_key: str
    region_code: str = "KR"
    video_category_id: str = "15"  # Pets & Animals
    limit: int = 12

@dataclass
class RunConfig:
    run_slot: str = "am"  # am / pm
    dry_run: bool = False
    debug: bool = False
    upload_thumb: bool = True
    set_featured: bool = True
    reuse_media_by_search: bool = True

@dataclass
class AppConfig:
    wp: WordPressConfig
    yt: YouTubeConfig
    run: RunConfig
    sqlite_path: str

def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"
    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "3"))  # 요청: 카테고리 3
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    yt_key = _env("YOUTUBE_API_KEY")
    region = _env("REGION_CODE", "KR") or "KR"
    cat = _env("VIDEO_CATEGORY_ID", "15") or "15"
    limit = _env_int("LIMIT", 12)

    slot = (_env("RUN_SLOT", "am") or "am").lower()
    if slot not in ("am", "pm"):
        slot = "am"

    sqlite_path = _env("SQLITE_PATH", "data/daily_pet_trending.sqlite3")

    return AppConfig(
        wp=WordPressConfig(
            base_url=wp_base,
            user=wp_user,
            app_pass=wp_pass,
            status=wp_status,
            category_ids=cat_ids,
            tag_ids=tag_ids,
        ),
        yt=YouTubeConfig(
            api_key=yt_key,
            region_code=region,
            video_category_id=cat,
            limit=limit,
        ),
        run=RunConfig(
            run_slot=slot,
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            reuse_media_by_search=_env_bool("REUSE_MEDIA_BY_SEARCH", True),
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
    if not cfg.yt.api_key:
        missing.append("YOUTUBE_API_KEY")
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot)
    print("[CFG] YOUTUBE_API_KEY:", ok(cfg.yt.api_key))
    print("[CFG] REGION_CODE:", cfg.yt.region_code, "| VIDEO_CATEGORY_ID:", cfg.yt.video_category_id, "| LIMIT:", cfg.yt.limit)
    print("[CFG] UPLOAD_THUMB:", cfg.run.upload_thumb, "| SET_FEATURED:", cfg.run.set_featured, "| REUSE_MEDIA_BY_SEARCH:", cfg.run.reuse_media_by_search)
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)

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
          featured_media INTEGER,
          featured_url TEXT,
          created_at TEXT
        )
        """
    )
    con.commit()
    con.close()

def get_existing_post(path: str, date_slot: str) -> Optional[Dict[str, Any]]:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "SELECT wp_post_id, wp_link, featured_media, featured_url, created_at FROM daily_posts WHERE date_slot = ?",
        (date_slot,),
    )
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return {
        "wp_post_id": int(row[0] or 0),
        "wp_link": str(row[1] or ""),
        "featured_media": int(row[2] or 0),
        "featured_url": str(row[3] or ""),
        "created_at": str(row[4] or ""),
    }

def save_post_meta(path: str, date_slot: str, post_id: int, link: str, featured_media: int = 0, featured_url: str = "") -> None:
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_slot, wp_post_id, wp_link, featured_media, featured_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (date_slot, int(post_id), str(link or ""), int(featured_media or 0), str(featured_url or ""), datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()

# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "daily-pet-trending-bot/1.0"}

def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html: str, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": cfg.status,
    }
    if featured_media:
        payload["featured_media"] = featured_media
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")

def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html: str, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "content": html,
        "status": cfg.status,
    }
    if featured_media:
        payload["featured_media"] = featured_media
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids

    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")

def wp_find_media_by_search(cfg: WordPressConfig, search: str) -> Optional[Tuple[int, str]]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = wp_auth_header(cfg.user, cfg.app_pass)
    params = {"search": search, "per_page": 10}
    r = requests.get(url, headers=headers, params=params, timeout=25)
    if r.status_code != 200:
        return None
    try:
        items = r.json()
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    it = items[0]
    mid = int(it.get("id") or 0)
    src = str(it.get("source_url") or "")
    if mid and src:
        return mid, src
    return None

def wp_upload_media_from_url(cfg: WordPressConfig, image_url: str, filename: str) -> Tuple[int, str]:
    rr = requests.get(image_url, timeout=35)
    if rr.status_code != 200 or not rr.content:
        raise RuntimeError(f"Image download failed: {rr.status_code}")
    content = rr.content
    ctype = (rr.headers.get("Content-Type", "") or "").split(";")[0].strip().lower() or "image/jpeg"

    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": ctype,
    }
    r = requests.post(url, headers=headers, data=content, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("source_url") or "")

# -----------------------------
# YouTube fetch
# -----------------------------
def yt_most_popular(cfg: YouTubeConfig) -> List[Dict[str, Any]]:
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "key": cfg.api_key,
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": cfg.region_code,
        "videoCategoryId": cfg.video_category_id,
        "maxResults": min(max(cfg.limit, 1), 50),
    }
    r = requests.get(url, params=params, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"YouTube API failed: {r.status_code} body={r.text[:300]}")
    data = r.json()
    items = data.get("items") or []
    out: List[Dict[str, Any]] = []
    for it in items:
        vid = str(it.get("id") or "")
        sn = it.get("snippet") or {}
        st = it.get("statistics") or {}
        title = str(sn.get("title") or "").strip()
        channel = str(sn.get("channelTitle") or "").strip()
        published = str(sn.get("publishedAt") or "")
        thumbs = (sn.get("thumbnails") or {})
        thumb_url = ""
        # maxres > standard > high > medium > default
        for k in ("maxres", "standard", "high", "medium", "default"):
            if k in thumbs and isinstance(thumbs[k], dict) and thumbs[k].get("url"):
                thumb_url = str(thumbs[k]["url"])
                break
        view_count = int(st.get("viewCount") or 0) if str(st.get("viewCount") or "").isdigit() else 0
        out.append(
            {
                "video_id": vid,
                "title": title,
                "channel": channel,
                "publishedAt": published,
                "views": view_count,
                "thumb": thumb_url,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "embed": f"https://www.youtube.com/embed/{vid}",
            }
        )
    return out[: cfg.limit]

def cache_path_for(slot: str) -> Path:
    return Path("data") / f"last_success_{slot}.json"

def save_cache(slot: str, items: List[Dict[str, Any]]) -> None:
    Path("data").mkdir(parents=True, exist_ok=True)
    p = cache_path_for(slot)
    p.write_text(json.dumps({"items": items, "saved_at": datetime.utcnow().isoformat()}, ensure_ascii=False, indent=2), encoding="utf-8")

def load_cache(slot: str) -> Optional[List[Dict[str, Any]]]:
    p = cache_path_for(slot)
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text(encoding="utf-8"))
        items = j.get("items")
        if isinstance(items, list) and items:
            return items
    except Exception:
        return None
    return None

# -----------------------------
# Rendering
# -----------------------------
DISCLOSURE = "※ 이 글은 유튜브 인기 영상을 모아 소개(임베드)하는 자동 포스팅입니다."
NOTE = "영상은 원본 소유자에게 저작권이 있으며, 본 포스팅은 링크/임베드 형태로 큐레이션합니다."

def esc(s: str) -> str:
    return htmlmod.escape(s or "")

def fmt_views(n: int) -> str:
    if n >= 100000000:
        return f"{n/100000000:.1f}억"
    if n >= 10000:
        return f"{n/10000:.1f}만"
    return str(n)

def build_html(now: datetime, slot_label: str, items: List[Dict[str, Any]], used_cache: bool = False) -> str:
    head = f"<p>기준시각: <b>{esc(now.astimezone(KST).strftime('%Y-%m-%d %H:%M'))}</b> / 슬롯: <b>{esc(slot_label)}</b></p>"
    badge = ""
    if used_cache:
        badge = '<p style="color:#b00;"><b>⚠️ YouTube API 오류로 최근 캐시 데이터를 재사용했습니다.</b></p>'

    disclosure = f'<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">{esc(DISCLOSURE)}</p>'
    note = f'<p style="font-size:13px;opacity:.8;">{esc(NOTE)}</p>'

    rows = []
    for i, it in enumerate(items, start=1):
        title = esc(it.get("title", ""))
        channel = esc(it.get("channel", ""))
        url = esc(it.get("url", ""))
        embed = esc(it.get("embed", ""))
        thumb = esc(it.get("thumb", ""))
        views = int(it.get("views") or 0)
        vtxt = esc(fmt_views(views))

        # 임베드(반응형)
        iframe = f"""
        <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:12px;background:#000;margin:10px 0;">
          <iframe src="{embed}" title="{title}" loading="lazy"
            style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
          </iframe>
        </div>
        """

        rows.append(
            f"""
            <div style="padding:14px;border:1px solid #eee;border-radius:14px;margin:12px 0;">
              <div style="font-size:15px;line-height:1.4;">
                <b>{i}. {title}</b><br/>
                <span style="opacity:.8;">채널: {channel} · 조회수: {vtxt}</span>
              </div>
              {iframe}
              <div style="font-size:13px;opacity:.8;">
                <a href="{url}" target="_blank" rel="nofollow sponsored noopener">유튜브에서 보기</a>
              </div>
              {"<img src='"+thumb+"' style='max-width:100%;height:auto;border-radius:10px;margin-top:10px;' alt='"+title+"'/>" if thumb else ""}
            </div>
            """
        )

    body = f"""
    {disclosure}
    {head}
    {badge}
    {note}
    <h2>반려동물 인기영상 TOP {len(items)}</h2>
    {''.join(rows) if rows else "<p>오늘은 데이터를 가져오지 못했습니다.</p>"}
    <hr/>
    <p style="opacity:.85;">다음 업데이트도 자동으로 올라옵니다 🙂</p>
    """
    return body

# -----------------------------
# Featured image helper
# -----------------------------
def ensure_featured_media(cfg: AppConfig, thumb_url: str, stable_name: str) -> Tuple[int, str]:
    if not thumb_url:
        return 0, ""
    h = hashlib.sha1(thumb_url.encode("utf-8")).hexdigest()[:12]
    filename = f"{stable_name}_{h}.jpg"

    if cfg.run.reuse_media_by_search:
        found = wp_find_media_by_search(cfg.wp, search=f"{stable_name}_{h}")
        if found:
            return found

    mid, murl = wp_upload_media_from_url(cfg.wp, thumb_url, filename)
    return mid, murl

# -----------------------------
# Main
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    date_str = now.strftime("%Y-%m-%d")
    slot = cfg.run.run_slot
    slot_label = "오전" if slot == "am" else "오후"
    date_slot = f"{date_str}_{slot}"

    init_db(cfg.sqlite_path)

    used_cache = False
    items: List[Dict[str, Any]] = []
    try:
        items = yt_most_popular(cfg.yt)
        if not items:
            raise RuntimeError("YouTube API returned empty items")
        save_cache(slot, items)
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] YouTube fetch failed:", repr(e))
        cached = load_cache(slot)
        if cached:
            items = cached
            used_cache = True
        else:
            raise RuntimeError("YouTube 수집 실패 + 캐시도 없음(첫 실행). DEBUG=1로 원인 확인 필요.") from e

    title = f"{date_str} 반려동물 인기영상 TOP{len(items)} ({slot_label})"
    slug = f"pet-trending-{date_str}-{slot}"  # 슬롯별 분리
    html = build_html(now, slot_label, items, used_cache=used_cache)

    # Featured image = 1등 썸네일
    featured_media = 0
    featured_url = ""
    if cfg.run.upload_thumb and cfg.run.set_featured and items:
        top_thumb = str(items[0].get("thumb") or "").strip()
        if top_thumb:
            try:
                featured_media, featured_url = ensure_featured_media(cfg, top_thumb, stable_name="pet_trending_thumb")
            except Exception as e:
                if cfg.run.debug:
                    print("[WARN] featured upload failed:", repr(e))
                featured_media, featured_url = 0, ""

    if cfg.run.dry_run:
        print("[DRY_RUN] 발행 생략. 미리보기 HTML 일부 ↓")
        print(html[:2000])
        print("... (truncated)")
        return

    existing = get_existing_post(cfg.sqlite_path, date_slot)
    if existing and existing.get("wp_post_id"):
        post_id = int(existing["wp_post_id"])
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, html, featured_media=featured_media)
        save_post_meta(cfg.sqlite_path, date_slot, wp_post_id, wp_link, featured_media, featured_url)
        print("OK(updated):", wp_post_id, wp_link)
    else:
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, html, featured_media=featured_media)
        save_post_meta(cfg.sqlite_path, date_slot, wp_post_id, wp_link, featured_media, featured_url)
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
        sys.exit(1)
