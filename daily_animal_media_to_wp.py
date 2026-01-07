# -*- coding: utf-8 -*-
"""
daily_animal_media_to_wp.py (완전 통합 / Tenor 불필요 / Pexels(사진) + Commons(GIF))
- 사진: PEXELS_API_KEY 있으면 Pexels Search API로 수집 (Authorization 헤더 필요)
- 움짤: Wikimedia Commons(MediaWiki API)에서 애니 GIF 수집 (키 필요 없음)
- 하루 2회(am/pm) 슬롯별로 글을 따로 생성/업데이트 (slug 고정)
- SQLite: 슬롯별 발행이력 + 최근 N일 중복 방지
- 대표이미지: 1번 아이템 다운로드 → WP 미디어 업로드 → featured_media 설정 (옵션)
- DRY_RUN/DEBUG 지원

필수 Secrets:
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

선택 Secrets:
  - PEXELS_API_KEY (있으면 사진 퀄리티/안정성 ↑)

옵션 env:
  - WP_STATUS=publish
  - WP_CATEGORY_IDS="3"
  - WP_TAG_IDS="1,2"
  - SQLITE_PATH=data/daily_animal_media.sqlite3
  - RUN_SLOT=am|pm (없으면 KST 기준 자동)
  - LIMIT=20
  - GIF_COUNT=8
  - PHOTO_COUNT=12
  - AVOID_REPEAT_DAYS=14
  - FORCE_NEW=0
  - UPLOAD_THUMB=1
  - SET_FEATURED=1
  - DRY_RUN=0
  - DEBUG=0
"""

from __future__ import annotations

import base64
import html
import json
import os
import random
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests


KST = timezone(timedelta(hours=9))

# ---- Wikimedia Commons (MediaWiki API)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_WIKI = "https://commons.wikimedia.org/wiki/"
GIF_CATS = ["Category:Animations_of_cats", "Category:Animations_of_dogs"]
PHOTO_CATS = ["Category:Cats", "Category:Dogs"]

# ---- Pexels API
PEXELS_SEARCH = "https://api.pexels.com/v1/search"
PEXELS_QUERIES = ["cute cat", "kitten", "cute dog", "puppy", "pet dog", "pet cat"]


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
    run_slot: str = ""  # am/pm or blank(auto)
    limit: int = 20
    gif_count: int = 8
    photo_count: int = 12
    avoid_repeat_days: int = 14
    force_new: bool = False
    upload_thumb: bool = True
    set_featured: bool = True
    dry_run: bool = False
    debug: bool = False


@dataclass
class AppConfig:
    wp: WordPressConfig
    run: RunConfig
    sqlite_path: str
    pexels_api_key: str = ""


def load_cfg() -> AppConfig:
    wp_base = _env("WP_BASE_URL").rstrip("/")
    wp_user = _env("WP_USER")
    wp_pass = _env("WP_APP_PASS")
    wp_status = _env("WP_STATUS", "publish") or "publish"

    cat_ids = _parse_int_list(_env("WP_CATEGORY_IDS", "3"))
    tag_ids = _parse_int_list(_env("WP_TAG_IDS", ""))

    run_slot = (_env("RUN_SLOT", "") or "").lower()
    if run_slot and run_slot not in ("am", "pm"):
        run_slot = ""

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
            limit=_env_int("LIMIT", 20),
            gif_count=_env_int("GIF_COUNT", 8),
            photo_count=_env_int("PHOTO_COUNT", 12),
            avoid_repeat_days=_env_int("AVOID_REPEAT_DAYS", 14),
            force_new=_env_bool("FORCE_NEW", False),
            upload_thumb=_env_bool("UPLOAD_THUMB", True),
            set_featured=_env_bool("SET_FEATURED", True),
            dry_run=_env_bool("DRY_RUN", False),
            debug=_env_bool("DEBUG", False),
        ),
        sqlite_path=_env("SQLITE_PATH", "data/daily_animal_media.sqlite3"),
        pexels_api_key=_env("PEXELS_API_KEY", ""),
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
    print("[CFG] RUN_SLOT:", cfg.run.run_slot or "(auto)")
    print("[CFG] LIMIT:", cfg.run.limit, "| GIF_COUNT:", cfg.run.gif_count, "| PHOTO_COUNT:", cfg.run.photo_count)
    print("[CFG] AVOID_REPEAT_DAYS:", cfg.run.avoid_repeat_days, "| FORCE_NEW:", cfg.run.force_new)
    print("[CFG] UPLOAD_THUMB:", cfg.run.upload_thumb, "| SET_FEATURED:", cfg.run.set_featured)
    print("[CFG] PEXELS_API_KEY:", ok(cfg.pexels_api_key))
    print("[CFG] DRY_RUN:", cfg.run.dry_run, "| DEBUG:", cfg.run.debug)


# -----------------------------
# SQLite
# -----------------------------
def _db(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(path: str) -> None:
    con = _db(path)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_posts (
          date_slot TEXT PRIMARY KEY,
          wp_post_id INTEGER,
          wp_link TEXT,
          featured_media_id INTEGER,
          featured_media_url TEXT,
          created_at TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS seen_items (
          item_uid TEXT PRIMARY KEY,
          seen_at TEXT
        )
        """
    )

    con.commit()
    con.close()


def get_existing_post(path: str, date_slot: str) -> Optional[Tuple[int, str]]:
    con = _db(path)
    cur = con.cursor()
    cur.execute("SELECT wp_post_id, wp_link FROM daily_posts WHERE date_slot = ?", (date_slot,))
    row = cur.fetchone()
    con.close()
    if not row:
        return None
    return int(row[0]), str(row[1] or "")


def save_post_meta(path: str, date_slot: str, post_id: int, link: str, featured_media_id: int = 0, featured_media_url: str = "") -> None:
    con = _db(path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO daily_posts(date_slot, wp_post_id, wp_link, featured_media_id, featured_media_url, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (date_slot, post_id, link, int(featured_media_id or 0), str(featured_media_url or ""), datetime.utcnow().isoformat()),
    )
    con.commit()
    con.close()


def seen_recent(path: str, uid: str, days: int) -> bool:
    con = _db(path)
    cur = con.cursor()
    cur.execute("SELECT seen_at FROM seen_items WHERE item_uid = ?", (uid,))
    row = cur.fetchone()
    con.close()
    if not row:
        return False
    try:
        dt = datetime.fromisoformat(row[0])
    except Exception:
        return False
    return (datetime.utcnow() - dt) < timedelta(days=days)


def mark_seen(path: str, uids: List[str]) -> None:
    if not uids:
        return
    con = _db(path)
    cur = con.cursor()
    now = datetime.utcnow().isoformat()
    for uid in uids:
        cur.execute("INSERT OR REPLACE INTO seen_items(item_uid, seen_at) VALUES (?, ?)", (uid, now))
    con.commit()
    con.close()


# -----------------------------
# WordPress REST
# -----------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "animal-media-bot/1.0"}


def wp_create_post(cfg: WordPressConfig, title: str, slug: str, html_body: str, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "slug": slug, "content": html_body, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = featured_media

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(cfg: WordPressConfig, post_id: int, title: str, html_body: str, featured_media: int = 0) -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + f"/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(cfg.user, cfg.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"title": title, "content": html_body, "status": cfg.status}
    if cfg.category_ids:
        payload["categories"] = cfg.category_ids
    if cfg.tag_ids:
        payload["tags"] = cfg.tag_ids
    if featured_media:
        payload["featured_media"] = featured_media

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_upload_media(cfg: WordPressConfig, file_bytes: bytes, filename: str, mime: str, alt_text: str = "") -> Tuple[int, str]:
    url = cfg.base_url.rstrip("/") + "/wp-json/wp/v2/media"
    headers = {
        **wp_auth_header(cfg.user, cfg.app_pass),
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Content-Type": mime or "application/octet-stream",
    }
    r = requests.post(url, headers=headers, data=file_bytes, timeout=60)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP media upload failed: {r.status_code} body={r.text[:500]}")
    data = r.json()
    return int(data["id"]), str(data.get("source_url") or "")


# -----------------------------
# Download helper
# -----------------------------
def download_bytes(url: str, max_bytes: int = 15 * 1024 * 1024) -> Tuple[bytes, str]:
    if not url:
        return b"", ""
    with requests.get(url, stream=True, timeout=50) as r:
        if r.status_code != 200:
            return b"", ""
        ctype = str(r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        chunks = []
        total = 0
        for ch in r.iter_content(chunk_size=65536):
            if not ch:
                continue
            chunks.append(ch)
            total += len(ch)
            if total > max_bytes:
                break
        return b"".join(chunks), ctype


def guess_ext(mime: str) -> str:
    if mime == "image/jpeg":
        return "jpg"
    if mime == "image/png":
        return "png"
    if mime == "image/gif":
        return "gif"
    if mime == "image/webp":
        return "webp"
    return "bin"


def slot_auto(now: datetime, cfg_slot: str) -> str:
    if cfg_slot in ("am", "pm"):
        return cfg_slot
    return "am" if now.hour < 12 else "pm"


# -----------------------------
# Pexels (photos)
# -----------------------------
@dataclass
class PexelsPhoto:
    pid: int
    page_url: str
    photographer: str
    src_large: str
    src_original: str


def pexels_search(api_key: str, query: str, per_page: int = 80, page: int = 1) -> List[PexelsPhoto]:
    # Pexels는 Authorization 헤더로 키를 전달 :contentReference[oaicite:2]{index=2}
    headers = {"Authorization": api_key, "User-Agent": "animal-media-bot/1.0"}
    params = {"query": query, "per_page": per_page, "page": page}
    r = requests.get(PEXELS_SEARCH, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        return []
    j = r.json()
    out: List[PexelsPhoto] = []
    for p in (j.get("photos") or []):
        try:
            pid = int(p.get("id"))
            page_url = str(p.get("url") or "")
            photographer = str(p.get("photographer") or "")
            src = p.get("src") or {}
            large = str(src.get("large") or src.get("large2x") or src.get("original") or "")
            orig = str(src.get("original") or large or "")
            if pid and large:
                out.append(PexelsPhoto(pid=pid, page_url=page_url, photographer=photographer, src_large=large, src_original=orig))
        except Exception:
            continue
    return out


# -----------------------------
# Commons (GIF + fallback photos)
# -----------------------------
def commons_get(params: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    base = {"format": "json", "formatversion": 2}
    r = requests.get(COMMONS_API, params={**base, **params}, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Commons API failed: {r.status_code} body={r.text[:200]}")
    return r.json()


def list_category_files(cmtitle: str, limit: int = 200) -> List[str]:
    titles: List[str] = []
    cont = None
    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",  # Categorymembers 문서 :contentReference[oaicite:3]{index=3}
            "cmtitle": cmtitle,
            "cmtype": "file",
            "cmlimit": min(200, limit - len(titles)),
        }
        if cont:
            params["cmcontinue"] = cont
        data = commons_get(params)
        cms = (data.get("query") or {}).get("categorymembers") or []
        for it in cms:
            t = str(it.get("title") or "")
            if t.startswith("File:"):
                titles.append(t)
        cont = (data.get("continue") or {}).get("cmcontinue")
        if not cont or not cms:
            break
    return titles


def strip_tags(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class CommonsItem:
    file_title: str
    page_url: str
    embed_url: str
    download_url: str
    mime: str
    license_short: str = ""
    artist: str = ""


def fetch_commons_items(file_titles: List[str], thumb_width: int = 900) -> List[CommonsItem]:
    if not file_titles:
        return []
    joined = "|".join(file_titles[:50])
    data = commons_get(
        {
            "action": "query",
            "prop": "imageinfo",  # Imageinfo 문서 :contentReference[oaicite:4]{index=4}
            "titles": joined,
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": thumb_width,
        }
    )
    pages = (data.get("query") or {}).get("pages") or []
    out: List[CommonsItem] = []
    for p in pages:
        title = str(p.get("title") or "")
        infos = p.get("imageinfo") or []
        if not infos:
            continue
        ii = infos[0]
        page_url = COMMONS_WIKI + quote(title.replace(" ", "_"), safe=":/")
        original = str(ii.get("url") or "").strip()
        thumb = str(ii.get("thumburl") or "").strip()
        mime = str(ii.get("mime") or "").strip()

        ext = (ii.get("extmetadata") or {})
        lic = strip_tags(((ext.get("LicenseShortName") or {}) or {}).get("value", ""))
        artist = strip_tags(((ext.get("Artist") or {}) or {}).get("value", ""))

        embed = original if mime == "image/gif" else (thumb or original)
        if not embed:
            continue
        out.append(
            CommonsItem(
                file_title=title,
                page_url=page_url,
                embed_url=embed,
                download_url=original or embed,
                mime=mime,
                license_short=lic,
                artist=artist,
            )
        )
    return out


# -----------------------------
# Render
# -----------------------------
def build_html(now: datetime, slot_label: str, gifs: List[Dict[str, str]], photos: List[Dict[str, str]]) -> str:
    dt = now.astimezone(KST).strftime("%Y-%m-%d %H:%M")
    intro = f"""
    <p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">
      오늘의 멍냥짤/사진 모음 🐶🐱<br/>
      기준시각: <b>{html.escape(dt)}</b> / 슬롯: <b>{html.escape(slot_label)}</b>
    </p>
    <p style="opacity:.85;">이미지 출처 링크는 각 카드에 함께 표기했습니다.</p>
    """

    def card(it: Dict[str, str], idx: int, kind: str) -> str:
        img = html.escape(it.get("embed_url", ""))
        link = html.escape(it.get("page_url", ""))
        credit = it.get("credit", "")
        credit_html = f'<div style="font-size:12px;opacity:.75;margin-top:8px;">{html.escape(credit)}</div>' if credit else ""
        return f"""
        <figure style="margin:16px 0;padding:12px;border:1px solid #eee;border-radius:12px;">
          <div style="font-size:13px;opacity:.75;margin-bottom:8px;">{kind} #{idx}</div>
          <a href="{link}" target="_blank" rel="nofollow noopener">
            <img src="{img}" alt="{kind} {idx}" loading="lazy" style="max-width:100%;height:auto;border-radius:10px;" />
          </a>
          <figcaption style="font-size:12px;opacity:.75;margin-top:8px;">
            출처: <a href="{link}" target="_blank" rel="nofollow noopener">{html.escape(it.get("source",""))}</a>
          </figcaption>
          {credit_html}
        </figure>
        """

    gif_html = "".join(card(it, i, "움짤") for i, it in enumerate(gifs, start=1))
    photo_html = "".join(card(it, i, "사진") for i, it in enumerate(photos, start=1))

    return (
        intro
        + f"<h2>🔥 오늘의 움짤 {len(gifs)}개</h2>"
        + (gif_html or "<p>오늘은 움짤을 충분히 가져오지 못했습니다.</p>")
        + f"<h2>📸 오늘의 사진 {len(photos)}장</h2>"
        + (photo_html or "<p>오늘은 사진을 충분히 가져오지 못했습니다.</p>")
    )


# -----------------------------
# Main run
# -----------------------------
def run(cfg: AppConfig) -> None:
    now = datetime.now(tz=KST)
    slot = slot_auto(now, cfg.run.run_slot)
    slot_label = "오전" if slot == "am" else "오후"
    date_str = now.strftime("%Y-%m-%d")
    date_slot = f"{date_str}_{slot}"

    init_db(cfg.sqlite_path)

    # 1) GIF: Commons
    gif_items: List[CommonsItem] = []
    try:
        titles = []
        for c in GIF_CATS:
            titles += list_category_files(c, limit=250)
        random.shuffle(titles)

        picked = []
        for t in titles:
            uid = "commons:" + t
            if seen_recent(cfg.sqlite_path, uid, cfg.run.avoid_repeat_days):
                continue
            picked.append(t)
            if len(picked) >= 80:
                break

        # 50개씩
        for i in range(0, len(picked), 50):
            gif_items += fetch_commons_items(picked[i:i+50], thumb_width=900)
        gif_items = [x for x in gif_items if x.mime == "image/gif"][: max(0, cfg.run.gif_count)]
    except Exception as e:
        if cfg.run.debug:
            print("[WARN] commons gif fetch failed:", repr(e))

    # 2) PHOTOS: Pexels(있으면) → 없으면 Commons
    photo_cards: List[Dict[str, str]] = []

    if cfg.pexels_api_key:
        try:
            pool: List[PexelsPhoto] = []
            for _ in range(6):
                q = random.choice(PEXELS_QUERIES)
                pool += pexels_search(cfg.pexels_api_key, q, per_page=80, page=random.randint(1, 3))
            random.shuffle(pool)

            for p in pool:
                uid = f"pexels:{p.pid}"
                if seen_recent(cfg.sqlite_path, uid, cfg.run.avoid_repeat_days):
                    continue
                photo_cards.append(
                    {
                        "source": "Pexels",
                        "page_url": p.page_url or "https://www.pexels.com",
                        "embed_url": p.src_large,
                        "download_url": p.src_large,
                        "uid": uid,
                        "credit": f"Photo by {p.photographer} (Pexels)",
                    }
                )
                if len(photo_cards) >= max(0, cfg.run.photo_count):
                    break
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] pexels fetch failed:", repr(e))

    if len(photo_cards) < max(0, cfg.run.photo_count):
        # Commons fallback photos
        try:
            titles = []
            for c in PHOTO_CATS:
                titles += list_category_files(c, limit=350)
            random.shuffle(titles)

            picked = []
            for t in titles:
                uid = "commons:" + t
                if seen_recent(cfg.sqlite_path, uid, cfg.run.avoid_repeat_days):
                    continue
                picked.append(t)
                if len(picked) >= 120:
                    break

            commons_photos: List[CommonsItem] = []
            for i in range(0, len(picked), 50):
                commons_photos += fetch_commons_items(picked[i:i+50], thumb_width=900)
            commons_photos = [x for x in commons_photos if x.mime != "image/gif"]

            for it in commons_photos:
                photo_cards.append(
                    {
                        "source": "Wikimedia Commons",
                        "page_url": it.page_url,
                        "embed_url": it.embed_url,
                        "download_url": it.download_url,
                        "uid": "commons:" + it.file_title,
                        "credit": f"License: {it.license_short} | Artist: {it.artist}".strip(" |"),
                    }
                )
                if len(photo_cards) >= max(0, cfg.run.photo_count):
                    break
        except Exception as e:
            if cfg.run.debug:
                print("[WARN] commons photo fetch failed:", repr(e))

    # LIMIT 맞추기
    gif_cards = [
        {
            "source": "Wikimedia Commons",
            "page_url": it.page_url,
            "embed_url": it.embed_url,
            "download_url": it.download_url,
            "uid": "commons:" + it.file_title,
            "credit": f"License: {it.license_short} | Artist: {it.artist}".strip(" |"),
        }
        for it in gif_items
    ]

    # 합쳐서 LIMIT로 자르기(사진 우선으로 부족분 채우기)
    total_target = max(1, cfg.run.limit)
    combined = gif_cards + photo_cards
    combined = combined[:total_target]

    gifs_out = [x for x in combined if x.get("embed_url","").lower().endswith(".gif") or x.get("uid","").startswith("commons:File") and x.get("source")=="Wikimedia Commons"][: cfg.run.gif_count]
    photos_out = [x for x in combined if x not in gifs_out]

    title = f"{date_str} 멍냥짤/사진 TOP{len(gifs_out)+len(photos_out)} ({slot_label})"
    slug = f"animal-media-{date_str}-{slot}"
    body_html = build_html(now, slot_label, gifs_out, photos_out)

    # 대표이미지: 사진 1번(우선) → gif → 없음
    featured_id = 0
    featured_url = ""
    pick = (photos_out[0] if photos_out else (gifs_out[0] if gifs_out else None))

    if cfg.run.upload_thumb and cfg.run.set_featured and pick:
        b, mime = download_bytes(pick.get("download_url",""))
        if b:
            mime = mime or "image/jpeg"
            fname = f"{slug}-thumb.{guess_ext(mime)}"
            try:
                featured_id, featured_url = wp_upload_media(cfg.wp, b, fname, mime, alt_text=title)
            except Exception as e:
                if cfg.run.debug:
                    print("[WARN] featured upload failed:", repr(e))
                featured_id, featured_url = 0, ""

    if cfg.run.dry_run:
        print("[DRY_RUN] title:", title)
        print("[DRY_RUN] slug:", slug)
        print("[DRY_RUN] featured:", bool(featured_id))
        print(body_html[:2500] + "\n... (truncated)")
        return

    existing = None if cfg.run.force_new else get_existing_post(cfg.sqlite_path, date_slot)
    if existing:
        post_id, old_link = existing
        wp_post_id, wp_link = wp_update_post(cfg.wp, post_id, title, body_html, featured_media=featured_id)
        save_post_meta(cfg.sqlite_path, date_slot, wp_post_id, wp_link or old_link, featured_id, featured_url)
        print("OK(updated):", wp_post_id, wp_link or old_link)
    else:
        if cfg.run.force_new:
            slug = f"{slug}-{now.strftime('%H%M%S')}"
        wp_post_id, wp_link = wp_create_post(cfg.wp, title, slug, body_html, featured_media=featured_id)
        save_post_meta(cfg.sqlite_path, date_slot, wp_post_id, wp_link, featured_id, featured_url)
        print("OK(created):", wp_post_id, wp_link)

    # 중복 방지 기록
    seen = [x.get("uid","") for x in (gifs_out + photos_out) if x.get("uid")]
    mark_seen(cfg.sqlite_path, seen)


def main() -> None:
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
