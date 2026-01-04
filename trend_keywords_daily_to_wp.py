# -*- coding: utf-8 -*-
"""
trend_keywords_daily_to_wp.py (통합, 네이버 안정화)
- Google Trends Trending RSS(geo=KR)에서 트렌딩 키워드 TOP N 수집
- Naver DataLab Search Trend API:
    NAVER_KEYWORD_POOL(후보 키워드) 내에서
    "전일(어제) vs 전전일" 상대지수 상승폭(delta) 기준 TOP N 산출
  * DataLab은 실시간이 아니라 집계 지연이 있을 수 있어 '어제 기준'으로 계산해야 안정적
- WordPress REST API로 데일리 포스트 생성/업데이트 (slug 기준)
- GitHub Actions Secrets(환경변수)로 실행

✅ WordPress Secrets
  - WP_BASE_URL
  - WP_USER
  - WP_APP_PASS

✅ Naver Secrets (네가 쓰는 이름 그대로)
  - XNAVERCLIENT_ID
  - XNAVERCLIENT_SECRET

✅ 네이버 후보 키워드 풀(필수)
  - NAVER_KEYWORD_POOL: 콤마로 구분된 후보 키워드들

✅ 카테고리
  - WP_CATEGORY_ID 기본 31
"""

from __future__ import annotations

import base64
import html as htmlmod
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
import xml.etree.ElementTree as ET

KST = timezone(timedelta(hours=9))

NAVER_DATALAB_ENDPOINT = "https://openapi.naver.com/v1/datalab/search"


# -------------------------
# Env helpers
# -------------------------
def _env(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or "").strip()


def _env_int(name: str, default: int) -> int:
    v = _env(name, str(default))
    try:
        return int(v)
    except Exception:
        return default


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def _safe_len(s: str) -> str:
    return f"OK(len={len(s)})" if s else "MISSING"


# -------------------------
# Config
# -------------------------
@dataclass
class WPConfig:
    base_url: str
    user: str
    app_pass: str
    status: str = "publish"
    category_id: int = 31


@dataclass
class RunConfig:
    limit: int = 20
    google_geo: str = "KR"
    dry_run: bool = False
    debug: bool = False


@dataclass
class NaverCfg:
    client_id: str
    client_secret: str
    keyword_pool: List[str]
    pool_limit: int = 50
    lookback_days: int = 7  # 어제 기준으로 최근 n일 요청


def load_configs() -> Tuple[WPConfig, RunConfig, NaverCfg]:
    wp = WPConfig(
        base_url=_env("WP_BASE_URL").rstrip("/"),
        user=_env("WP_USER"),
        app_pass=_env("WP_APP_PASS"),
        status=_env("WP_STATUS", "publish") or "publish",
        category_id=_env_int("WP_CATEGORY_ID", 31),
    )

    run = RunConfig(
        limit=_env_int("LIMIT", 20),
        google_geo=_env("GOOGLE_GEO", "KR") or "KR",
        dry_run=_env("DRY_RUN", "0").lower() in ("1", "true", "yes"),
        debug=_env("DEBUG", "0").lower() in ("1", "true", "yes"),
    )

    # ✅ 네가 쓰는 이름 우선
    naver_id = _env("XNAVERCLIENT_ID") or _env("NAVER_CLIENT_ID")
    naver_secret = _env("XNAVERCLIENT_SECRET") or _env("NAVER_CLIENT_SECRET")

    pool_raw = _env("NAVER_KEYWORD_POOL", "")
    pool = [p.strip() for p in pool_raw.split(",") if p.strip()]

    # 중복 제거(순서 유지)
    seen = set()
    pool2 = []
    for x in pool:
        if x not in seen:
            seen.add(x)
            pool2.append(x)

    naver = NaverCfg(
        client_id=naver_id,
        client_secret=naver_secret,
        keyword_pool=pool2,
        pool_limit=_env_int("NAVER_POOL_LIMIT", 50),
        lookback_days=_env_int("NAVER_LOOKBACK_DAYS", 7),
    )
    return wp, run, naver


def validate_wp(wp: WPConfig) -> None:
    missing = []
    if not wp.base_url:
        missing.append("WP_BASE_URL")
    if not wp.user:
        missing.append("WP_USER")
    if not wp.app_pass:
        missing.append("WP_APP_PASS")
    if missing:
        raise RuntimeError("필수 WP 설정 누락: " + ", ".join(missing))


# -------------------------
# WordPress REST
# -------------------------
def wp_auth_header(user: str, app_pass: str) -> Dict[str, str]:
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": "trend-keywords-bot/1.0"}


def wp_find_post_by_slug(wp: WPConfig, slug: str) -> Optional[Tuple[int, str]]:
    url = f"{wp.base_url}/wp-json/wp/v2/posts"
    r = requests.get(url, params={"slug": slug, "per_page": 1}, timeout=25)
    if r.status_code != 200:
        return None
    arr = r.json()
    if not arr:
        return None
    post = arr[0]
    return int(post["id"]), str(post.get("link") or "")


def wp_create_post(wp: WPConfig, title: str, slug: str, html: str) -> Tuple[int, str]:
    url = f"{wp.base_url}/wp-json/wp/v2/posts"
    headers = {**wp_auth_header(wp.user, wp.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": wp.status,
        "categories": [wp.category_id],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP create failed: {r.status_code} body={r.text[:400]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


def wp_update_post(wp: WPConfig, post_id: int, title: str, slug: str, html: str) -> Tuple[int, str]:
    url = f"{wp.base_url}/wp-json/wp/v2/posts/{post_id}"
    headers = {**wp_auth_header(wp.user, wp.app_pass), "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": wp.status,
        "categories": [wp.category_id],
    }
    r = requests.post(url, headers=headers, json=payload, timeout=35)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"WP update failed: {r.status_code} body={r.text[:400]}")
    data = r.json()
    return int(data["id"]), str(data.get("link") or "")


# -------------------------
# Google Trends RSS
# -------------------------
def _xml_text(el: Optional[ET.Element]) -> str:
    return (el.text or "").strip() if el is not None else ""


def fetch_google_trending(geo: str, limit: int) -> List[Dict[str, Any]]:
    url = f"https://trends.google.com/trending/rss?geo={geo}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Google Trends RSS failed: {r.status_code} body={r.text[:200]}")
    root = ET.fromstring(r.content)
    items = root.findall(".//item")

    out: List[Dict[str, Any]] = []
    for it in items[: max(1, limit)]:
        title = _xml_text(it.find("title"))
        link = _xml_text(it.find("link"))
        pub = _xml_text(it.find("pubDate"))

        approx_traffic = ""
        for child in list(it):
            if child.tag.lower().endswith("approx_traffic"):
                approx_traffic = _xml_text(child)

        out.append(
            {"keyword": title, "link": link, "pubDate": pub, "traffic": approx_traffic}
        )
    return out


# -------------------------
# Naver DataLab (Search Trend)
# -------------------------
def fetch_naver_datalab_rank(nc: NaverCfg, limit: int) -> List[Dict[str, Any]]:
    """
    NAVER_KEYWORD_POOL 내에서 '어제 vs 그 전날' 상승폭(delta) TOP.
    ✅ 네이버가 실패해도 전체 실패시키지 않고 note로 대체.
    """
    if not nc.client_id or not nc.client_secret:
        return [{"note": "XNAVERCLIENT_ID / XNAVERCLIENT_SECRET 미설정 → 네이버 섹션 스킵"}]

    pool = nc.keyword_pool[: max(0, nc.pool_limit)]
    if not pool:
        return [{"note": "NAVER_KEYWORD_POOL 미설정 → 네이버 섹션 스킵"}]

    # ✅ 오늘 데이터 지연 가능 → '어제'를 끝으로 잡고 최근 n일 요청
    end = (now_kst().date() - timedelta(days=1))
    start = end - timedelta(days=max(2, nc.lookback_days))
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    headers = {
        "X-Naver-Client-Id": nc.client_id,
        "X-Naver-Client-Secret": nc.client_secret,
        "Content-Type": "application/json",
        "User-Agent": "trend-keywords-bot/1.0",
    }

    def chunks(lst: List[str], n: int) -> List[List[str]]:
        return [lst[i : i + n] for i in range(0, len(lst), n)]

    ranked: List[Dict[str, Any]] = []

    with requests.Session() as s:
        for ch in chunks(pool, 5):
            payload = {
                "startDate": start_s,
                "endDate": end_s,
                "timeUnit": "date",
                "keywordGroups": [{"groupName": kw, "keywords": [kw]} for kw in ch],
            }

            r = s.post(NAVER_DATALAB_ENDPOINT, headers=headers, json=payload, timeout=30)

            if r.status_code == 401:
                return [{
                    "note": (
                        "네이버 DataLab 인증 실패(401). "
                        "Secrets의 XNAVERCLIENT_ID/XNAVERCLIENT_SECRET가 맞는지, "
                        "네이버 개발자센터에서 'DataLab 검색어트렌드 API' 사용 설정이 되어있는지 확인 필요."
                    )
                }]

            if r.status_code != 200:
                return [{"note": f"네이버 DataLab 오류({r.status_code}). 응답: {r.text[:200]}"}]

            data = r.json()
            results = data.get("results", []) or []
            for res in results:
                kw = str(res.get("title") or "")
                series = res.get("data", []) or []
                if len(series) < 2:
                    continue

                # ✅ 마지막 2포인트로 delta 계산(어제 vs 전전일이 보통 됨)
                prev = float(series[-2].get("ratio", 0) or 0)
                last = float(series[-1].get("ratio", 0) or 0)
                delta = last - prev

                ranked.append(
                    {
                        "keyword": kw,
                        "prev": prev,
                        "last": last,
                        "delta": delta,
                        "link": f"https://search.naver.com/search.naver?query={quote(kw)}",
                    }
                )

    if not ranked:
        return [{"note": f"네이버 DataLab 응답은 받았지만 랭킹 계산용 데이터(2일치)가 부족합니다. 기간: {start_s}~{end_s}"}]

    ranked.sort(key=lambda x: (x["delta"], x["last"]), reverse=True)
    return ranked[: max(1, limit)]


# -------------------------
# Render HTML
# -------------------------
def esc(s: str) -> str:
    return htmlmod.escape(s or "")


def build_google_table(items: List[Dict[str, Any]]) -> str:
    rows = []
    for i, it in enumerate(items, start=1):
        kw = esc(it.get("keyword", ""))
        link = it.get("link", "")
        traffic = esc(it.get("traffic", ""))

        kw_html = f'<a href="{esc(link)}" target="_blank" rel="nofollow noopener">{kw}</a>' if link else kw
        rows.append(
            f"""
            <tr>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">{kw_html}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;white-space:nowrap;">{traffic}</td>
            </tr>
            """
        )
    return f"""
    <h2>Google 트렌딩 키워드 TOP {len(items)}</h2>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">순위</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">키워드</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">대략 트래픽</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def build_naver_table(items: List[Dict[str, Any]], end_date_str: str) -> str:
    if items and "note" in items[0]:
        return f"""
        <h2>네이버 트렌드 TOP</h2>
        <p style="opacity:.85;">{esc(items[0]["note"])}</p>
        """

    rows = []
    for i, it in enumerate(items, start=1):
        kw = esc(it.get("keyword", ""))
        link = it.get("link", "")
        delta = float(it.get("delta", 0.0) or 0.0)
        last = float(it.get("last", 0.0) or 0.0)
        kw_html = f'<a href="{esc(link)}" target="_blank" rel="nofollow noopener">{kw}</a>' if link else kw

        rows.append(
            f"""
            <tr>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:center;">{i}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;">{kw_html}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;">{delta:.2f}</td>
              <td style="padding:8px;border:1px solid #e5e5e5;text-align:right;">{last:.2f}</td>
            </tr>
            """
        )

    return f"""
    <h2>네이버 트렌드 TOP {len(items)}</h2>
    <p style="font-size:13px;opacity:.75;margin-top:-6px;">
      기준일: <b>{esc(end_date_str)}</b> (DataLab은 실시간이 아니라 집계 기반이어서 '어제 기준'으로 계산합니다)
    </p>
    <table style="border-collapse:collapse;width:100%;font-size:14px;">
      <thead>
        <tr>
          <th style="padding:8px;border:1px solid #e5e5e5;">순위</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">키워드</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">상승폭(Δ)</th>
          <th style="padding:8px;border:1px solid #e5e5e5;">지수</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def build_post_html(date_str: str, google_items: List[Dict[str, Any]], naver_items: List[Dict[str, Any]], naver_end: str) -> str:
    disclosure = (
        '<p style="padding:10px;border-left:4px solid #111;background:#f7f7f7;">'
        "※ 네이버 섹션은 NAVER_KEYWORD_POOL(후보 키워드 풀) 내에서 DataLab 상대지수 기반으로 계산됩니다."
        "</p>"
    )
    head = f"<p>기준일: <b>{esc(date_str)}</b></p>"
    return f"""
    {disclosure}
    {head}
    {build_google_table(google_items)}
    <hr/>
    {build_naver_table(naver_items, naver_end)}
    """


# -------------------------
# Main
# -------------------------
def main():
    wp, run, naver = load_configs()
    validate_wp(wp)

    date_str = now_kst().strftime("%Y-%m-%d")
    slug = f"realtime-keywords-{date_str}".lower()
    title = f"{date_str} 실시간(트렌딩) 검색어 TOP{run.limit} - 구글/네이버"

    if run.debug:
        print("[DEBUG] WP_BASE_URL:", wp.base_url)
        print("[DEBUG] WP_USER:", _safe_len(wp.user))
        print("[DEBUG] WP_APP_PASS:", _safe_len(wp.app_pass))
        print("[DEBUG] WP_CATEGORY_ID:", wp.category_id)
        print("[DEBUG] GOOGLE_GEO:", run.google_geo, "LIMIT:", run.limit)
        print("[DEBUG] XNAVERCLIENT_ID:", _safe_len(naver.client_id))
        print("[DEBUG] XNAVERCLIENT_SECRET:", _safe_len(naver.client_secret))
        print("[DEBUG] NAVER_KEYWORD_POOL size:", len(naver.keyword_pool), "POOL_LIMIT:", naver.pool_limit)
        print("[DEBUG] NAVER_LOOKBACK_DAYS:", naver.lookback_days)

    google_items = fetch_google_trending(run.google_geo, run.limit)

    # 네이버 기준일(어제)
    naver_end_date = (now_kst().date() - timedelta(days=1)).strftime("%Y-%m-%d")
    naver_items = fetch_naver_datalab_rank(naver, run.limit)

    html = build_post_html(date_str, google_items, naver_items, naver_end_date)

    if run.dry_run:
        print("[DRY_RUN] 아래 HTML을 WP에 올리지 않고 출력만 합니다.\n")
        print(html)
        return

    existed = wp_find_post_by_slug(wp, slug)
    if existed:
        post_id, old_link = existed
        new_id, new_link = wp_update_post(wp, post_id, title, slug, html)
        print(f"OK(updated): {new_id} {new_link or old_link}")
    else:
        new_id, new_link = wp_create_post(wp, title, slug, html)
        print(f"OK(created): {new_id} {new_link}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
