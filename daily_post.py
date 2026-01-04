# -*- coding: utf-8 -*-
"""
DailyReportBot - WordPress 자동 글 발행 (무료/키 없이 동작)
- USD/KRW: FRED CSV (DEXKOUS)
- Brent Oil: FRED CSV (DCOILBRENTEU)
- KOSPI: 네이버 모바일 주식 API (m.stock.naver.com)
- 뉴스: Google News RSS (한국어/한국지역)

✅ TEST_MODE=1 이면 5분마다 "새 글" 생성 (slug에 시간 포함)
✅ TEST_MODE=0(기본) 이면 날짜 기준 1일 1글(upsert: 있으면 업데이트)
✅ WP_POST_STATUS로 publish/draft 선택 가능 (기본 publish)

필수 설정(Secrets 또는 bot_config.json):
- WP_BASE_URL, WP_USER, WP_APP_PASS
선택:
- KAKAO_REST_KEY, KAKAO_REFRESH_TOKEN (카톡 알림)
"""

from __future__ import annotations

import os
import json
import base64
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import feedparser
from bs4 import BeautifulSoup

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore


# -----------------------------
# 기본 설정
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "bot_config.json"

KST = ZoneInfo("Asia/Seoul") if ZoneInfo else None

FRED_USDKRW = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS"
FRED_BRENT = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"

NAVER_KOSPI = "https://m.stock.naver.com/api/index/KOSPI/price"

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"


@dataclass
class Indicator:
    name: str
    value: Optional[float]
    prev: Optional[float]
    asof_date: Optional[str]  # YYYY-MM-DD
    unit: str = ""

    @property
    def change(self) -> Optional[float]:
        if self.value is None or self.prev is None:
            return None
        return self.value - self.prev

    @property
    def change_pct(self) -> Optional[float]:
        if self.value is None or self.prev is None or self.prev == 0:
            return None
        return (self.value - self.prev) / self.prev * 100.0


# -----------------------------
# 공통 유틸
# -----------------------------
def _now_kst() -> datetime:
    if KST:
        return datetime.now(KST)
    return datetime.now()


def _env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def load_config() -> Dict[str, Any]:
    """
    우선순위:
    1) 환경변수(Secrets) > 2) bot_config.json(스크립트와 같은 폴더)
    """
    cfg: Dict[str, Any] = {}

    # WordPress (필수)
    if _env("WP_BASE_URL"):
        cfg["wp_base_url"] = _env("WP_BASE_URL")
    if _env("WP_USER"):
        cfg["wp_user"] = _env("WP_USER")
    if _env("WP_APP_PASS"):
        cfg["wp_app_pass"] = _env("WP_APP_PASS")

    # 선택 옵션
    if _env("WP_POST_STATUS"):
        cfg["wp_post_status"] = _env("WP_POST_STATUS")
    if _env("KAKAO_REST_KEY"):
        cfg["kakao_rest_key"] = _env("KAKAO_REST_KEY")
    if _env("KAKAO_REFRESH_TOKEN"):
        cfg["kakao_refresh_token"] = _env("KAKAO_REFRESH_TOKEN")

    # env로 필수 3개가 다 있으면 파일 안 봄
    if cfg.get("wp_base_url") and cfg.get("wp_user") and cfg.get("wp_app_pass"):
        return cfg

    # 파일 로드(로컬 실행용)
    cfg_path = Path(_env("BOT_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"bot_config.json을 찾을 수 없습니다.\n"
            f"- 찾는 위치: {cfg_path.resolve()}\n"
            f"- 해결: daily_post.py와 같은 폴더에 bot_config.json을 두거나, "
            f"환경변수(WP_BASE_URL/WP_USER/WP_APP_PASS)를 설정하세요."
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        file_cfg = json.load(f)

    # 파일 cfg를 기본으로, env cfg로 덮어쓰기(있으면)
    file_cfg.update({k: v for k, v in cfg.items() if v is not None})
    return file_cfg


def _requests_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None, timeout: int = 30) -> requests.Response:
    h = {"User-Agent": "Mozilla/5.0 (DailyReportBot; +https://example.com)"}
    if headers:
        h.update(headers)
    r = requests.get(url, params=params, headers=h, timeout=timeout)
    r.raise_for_status()
    return r


def _requests_post(url: str, data: Any = None, json_body: Any = None, headers: Optional[dict] = None, timeout: int = 30) -> requests.Response:
    h = {"User-Agent": "Mozilla/5.0 (DailyReportBot; +https://example.com)"}
    if headers:
        h.update(headers)
    r = requests.post(url, data=data, json=json_body, headers=h, timeout=timeout)
    r.raise_for_status()
    return r


def _requests_put(url: str, json_body: Any = None, headers: Optional[dict] = None, timeout: int = 30) -> requests.Response:
    h = {"User-Agent": "Mozilla/5.0 (DailyReportBot; +https://example.com)"}
    if headers:
        h.update(headers)
    r = requests.put(url, json=json_body, headers=h, timeout=timeout)
    r.raise_for_status()
    return r


def _parse_fred_last_two(csv_text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    FRED CSV 형태:
    DATE,VALUE
    2025-12-26,1430.12
    ...
    마지막에서 '.'(결측) 제외하고 2개 추출
    """
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return None, None, None

    data_rows = []
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) < 2:
            continue
        d, v = parts[0], parts[1]
        if v == ".":
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        data_rows.append((d, fv))

    if len(data_rows) < 1:
        return None, None, None

    # 최신 2개
    latest_d, latest_v = data_rows[-1]
    prev_v = data_rows[-2][1] if len(data_rows) >= 2 else None
    return latest_v, prev_v, latest_d


def fetch_usdkrw() -> Indicator:
    try:
        r = _requests_get(FRED_USDKRW)
        latest, prev, asof = _parse_fred_last_two(r.text)
        return Indicator(name="USD/KRW", value=latest, prev=prev, asof_date=asof, unit="KRW")
    except Exception:
        return Indicator(name="USD/KRW", value=None, prev=None, asof_date=None, unit="KRW")


def fetch_brent() -> Indicator:
    try:
        r = _requests_get(FRED_BRENT)
        latest, prev, asof = _parse_fred_last_two(r.text)
        return Indicator(name="Brent Oil", value=latest, prev=prev, asof_date=asof, unit="USD")
    except Exception:
        return Indicator(name="Brent Oil", value=None, prev=None, asof_date=None, unit="USD")


def fetch_kospi() -> Indicator:
    """
    네이버 모바일 주식 API:
    GET https://m.stock.naver.com/api/index/KOSPI/price?pageSize=10&page=1
    반환 JSON 리스트(최근순). closePrice/localTradedAt 사용.
    """
    try:
        r = _requests_get(NAVER_KOSPI, params={"pageSize": 10, "page": 1})
        arr = r.json()
        if not isinstance(arr, list) or len(arr) == 0:
            raise ValueError("empty")
        rows = []
        for it in arr:
            cp = it.get("closePrice")
            dt = it.get("localTradedAt") or it.get("localDate")
            if cp is None or dt is None:
                continue
            try:
                v = float(str(cp).replace(",", ""))
            except Exception:
                continue
            # dt 예: 2026-01-03T00:00:00+09:00 혹은 2026-01-03
            d = str(dt)[:10]
            rows.append((d, v))
        if len(rows) == 0:
            raise ValueError("no_rows")
        # 이미 최근순일 확률 높지만 안전하게 날짜 정렬
        rows = sorted(rows, key=lambda x: x[0])
        latest_d, latest_v = rows[-1]
        prev_v = rows[-2][1] if len(rows) >= 2 else None
        return Indicator(name="KOSPI", value=latest_v, prev=prev_v, asof_date=latest_d, unit="pt")
    except Exception:
        return Indicator(name="KOSPI", value=None, prev=None, asof_date=None, unit="pt")


# -----------------------------
# 뉴스(한국어 RSS)
# -----------------------------
def fetch_news(query: str, max_items: int = 3) -> List[Dict[str, str]]:
    """
    Google News RSS (한국/한국어):
    https://news.google.com/rss/search?q=...&hl=ko&gl=KR&ceid=KR:ko
    """
    try:
        params = {"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
        r = _requests_get(GOOGLE_NEWS_RSS, params=params, timeout=20)
        feed = feedparser.parse(r.text)
        items = []
        for e in feed.entries[: max_items * 2]:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            if not title or not link:
                continue
            # 한글이 거의 없는 제목은 뒤로
            if not re.search(r"[가-힣]", title):
                continue
            items.append({"title": title, "link": link})
            if len(items) >= max_items:
                break

        # 한글 필터로 부족하면 그냥 채움
        if len(items) < max_items:
            for e in feed.entries:
                title = (e.get("title") or "").strip()
                link = (e.get("link") or "").strip()
                if not title or not link:
                    continue
                if {"title": title, "link": link} in items:
                    continue
                items.append({"title": title, "link": link})
                if len(items) >= max_items:
                    break
        return items[:max_items]
    except Exception:
        return []


def extract_keywords_from_titles(titles: List[str], topk: int = 5) -> List[str]:
    # 아주 가벼운 키워드 추출(헤드라인 기반 참고용)
    stop = set([
        "오늘", "정부", "한국", "미국", "전망", "주가", "환율", "유가", "코스피",
        "달러", "원화", "상승", "하락", "급등", "급락", "관련", "속보", "시장",
        "가격", "금리", "경제", "증시", "외환"
    ])
    tokens: List[str] = []
    for t in titles:
        # 특수문자 제거
        clean = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", t)
        for w in clean.split():
            if len(w) < 2:
                continue
            if w in stop:
                continue
            tokens.append(w)

    freq: Dict[str, int] = {}
    for w in tokens:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:topk]]


# -----------------------------
# WordPress
# -----------------------------
def wp_posts_api(cfg: Dict[str, Any]) -> str:
    return cfg["wp_base_url"].rstrip("/") + WP_POSTS_API_SUFFIX


def wp_auth_headers(cfg: Dict[str, Any]) -> Dict[str, str]:
    user = str(cfg["wp_user"]).strip()
    app_pass = str(cfg["wp_app_pass"]).replace(" ", "").strip()
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


def wp_find_post_by_slug(cfg: Dict[str, Any], slug: str) -> Optional[Dict[str, Any]]:
    try:
        r = _requests_get(
            wp_posts_api(cfg),
            params={"slug": slug, "per_page": 1, "status": "any"},
            headers=wp_auth_headers(cfg),
            timeout=30,
        )
        arr = r.json()
        if isinstance(arr, list) and len(arr) > 0:
            return arr[0]
        return None
    except Exception:
        return None


def wp_create_post(cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    r = _requests_post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    return r.json()


def wp_update_post(cfg: Dict[str, Any], post_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{wp_posts_api(cfg).rstrip('/')}/{post_id}"
    r = _requests_put(
        url,
        json_body=payload,
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        timeout=30,
    )
    return r.json()


# -----------------------------
# Kakao 알림(선택)
# -----------------------------
def refresh_access_token(cfg: Dict[str, Any]) -> str:
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": cfg["kakao_rest_key"],
        "refresh_token": cfg["kakao_refresh_token"],
    }
    r = _requests_post(url, data=data, timeout=30)
    tokens = r.json()
    if "refresh_token" in tokens and tokens["refresh_token"]:
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        # 로컬 파일이 있을 때만 갱신 저장(깃헙 액션에서는 의미 없음)
        try:
            if DEFAULT_CONFIG_PATH.exists():
                DEFAULT_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return tokens["access_token"]


def kakao_send_to_me(cfg: Dict[str, Any], text: str) -> None:
    if not (cfg.get("kakao_rest_key") and cfg.get("kakao_refresh_token")):
        return
    access_token = refresh_access_token(cfg)
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {"Authorization": f"Bearer {access_token}"}
    template_object = {
        "object_type": "text",
        "text": text[:1000],
        "link": {"web_url": cfg["wp_base_url"], "mobile_web_url": cfg["wp_base_url"]},
        "button_title": "사이트 열기",
    }
    data = {"template_object": json.dumps(template_object, ensure_ascii=False)}
    _requests_post(url, headers=headers, data=data, timeout=30)


# -----------------------------
# 글 내용(HTML)
# -----------------------------
def fmt_num(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "-"
    if abs(x) >= 1000:
        return f"{x:,.{digits}f}"
    return f"{x:.{digits}f}"


def fmt_change(x: Optional[float]) -> str:
    if x is None:
        return "-"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:,.2f}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f}%"


def build_html_report(usdkrw: Indicator, brent: Indicator, kospi: Indicator,
                      news_usd: List[Dict[str, str]],
                      news_brent: List[Dict[str, str]],
                      news_kospi: List[Dict[str, str]],
                      warnings: List[str]) -> str:

    # 키워드(헤드라인 기반 참고)
    kw_usd = extract_keywords_from_titles([x["title"] for x in news_usd]) if news_usd else []
    kw_brent = extract_keywords_from_titles([x["title"] for x in news_brent]) if news_brent else []
    kw_kospi = extract_keywords_from_titles([x["title"] for x in news_kospi]) if news_kospi else []

    def news_list_html(items: List[Dict[str, str]]) -> str:
        if not items:
            return "<li>관련 뉴스 수집 실패(또는 결과 없음)</li>"
        lis = []
        for it in items:
            t = BeautifulSoup(it["title"], "html.parser").get_text(" ", strip=True)
            l = it["link"]
            lis.append(f'<li><a href="{l}" target="_blank" rel="noopener noreferrer">{t}</a></li>')
        return "\n".join(lis)

    def kw_html(kws: List[str]) -> str:
        if not kws:
            return ""
        return '<div style="margin-top:8px;color:#6b7280;font-size:13px;">' \
               + "헤드라인 키워드: " + ", ".join(kws) + "</div>"

    warn_box = ""
    if warnings:
        warn_items = "".join([f"<li>{BeautifulSoup(w,'html.parser').get_text(' ', strip=True)}</li>" for w in warnings])
        warn_box = f"""
        <div style="border:1px solid #f5c2c7;background:#fff5f5;padding:14px;border-radius:10px;margin:16px 0;">
          <div style="font-weight:700;margin-bottom:8px;">⚠ 일부 데이터 수집 실패</div>
          <ul style="margin:0;padding-left:18px;">{warn_items}</ul>
        </div>
        """

    rows = [
        ("USD/KRW", usdkrw, "KRW"),
        ("Brent Oil", brent, "USD"),
        ("KOSPI", kospi, "pt"),
    ]
    tr = []
    for label, ind, unit in rows:
        tr.append(f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;">{label}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right;">{fmt_num(ind.value, 2)} {unit}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right;">{fmt_change(ind.change)}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right;">{fmt_pct(ind.change_pct)}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:center;">{ind.asof_date or "-"}</td>
        </tr>
        """)

    html = f"""
    <div style="max-width:900px;margin:0 auto;font-family:system-ui,-apple-system,Segoe UI,Roboto,Apple SD Gothic Neo,Malgun Gothic,sans-serif;">
      {warn_box}

      <h2 style="margin:18px 0 8px;">핵심 요약</h2>
      <ul style="margin:0;padding-left:18px;line-height:1.7;">
        <li>원/달러(USD/KRW), 브렌트유, 코스피의 <b>전일 대비 변화</b>를 한 번에 정리했습니다.</li>
        <li>변동 원인(참고용)으로 <b>관련 뉴스 헤드라인</b>과 <b>키워드</b>를 함께 첨부했습니다.</li>
      </ul>

      <h2 style="margin:22px 0 10px;">주요 지표</h2>
      <div style="border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;">
        <table style="width:100%;border-collapse:collapse;">
          <thead>
            <tr style="background:#111827;color:white;">
              <th style="padding:10px 12px;text-align:left;">지표</th>
              <th style="padding:10px 12px;text-align:right;">현재</th>
              <th style="padding:10px 12px;text-align:right;">전일대비</th>
              <th style="padding:10px 12px;text-align:right;">변동률</th>
              <th style="padding:10px 12px;text-align:center;">기준일</th>
            </tr>
          </thead>
          <tbody>
            {''.join(tr)}
          </tbody>
        </table>
      </div>

      <h2 style="margin:26px 0 10px;">왜 움직였나? (원인 참고)</h2>

      <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin-bottom:12px;">
        <h3 style="margin:0 0 10px;">USD/KRW 변동 관련 뉴스</h3>
        <ul style="margin:0;padding-left:18px;line-height:1.7;">
          {news_list_html(news_usd)}
        </ul>
        {kw_html(kw_usd)}
      </div>

      <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin-bottom:12px;">
        <h3 style="margin:0 0 10px;">Brent 유가 변동 관련 뉴스</h3>
        <ul style="margin:0;padding-left:18px;line-height:1.7;">
          {news_list_html(news_brent)}
        </ul>
        {kw_html(kw_brent)}
      </div>

      <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin-bottom:12px;">
        <h3 style="margin:0 0 10px;">KOSPI 변동 관련 뉴스</h3>
        <ul style="margin:0;padding-left:18px;line-height:1.7;">
          {news_list_html(news_kospi)}
        </ul>
        {kw_html(kw_kospi)}
      </div>

      <h2 style="margin:26px 0 10px;">내일 체크포인트</h2>
      <ul style="margin:0;padding-left:18px;line-height:1.7;">
        <li>주요 지표 발표 일정(미국 CPI/고용, 연준 발언 등)</li>
        <li>유가: OPEC/재고/지정학 이슈 헤드라인</li>
        <li>환율: 달러 강세/위험자산 선호 변화 여부</li>
      </ul>

      <div style="margin-top:22px;color:#6b7280;font-size:12px;">
        데이터: FRED(USD/KRW, Brent) · Naver(KOSPI) · Google News RSS(헤드라인)
      </div>
    </div>
    """
    return html


# -----------------------------
# 메인
# -----------------------------
def main() -> None:
    cfg = load_config()

    test_mode = (_env("TEST_MODE") == "1")
    status = cfg.get("wp_post_status") or "publish"
    # 테스트 모드에서는 기본 draft 권장(원하면 workflow에서 WP_POST_STATUS=publish)
    if test_mode and not _env("WP_POST_STATUS"):
        status = "draft"

    now = _now_kst()
    today = now.strftime("%Y-%m-%d")
    hm = now.strftime("%H%M")

    title = f"오늘의 지표 리포트 ({today})"
    slug = f"daily-indicator-report-{today}"

    if test_mode:
        title = f"[TEST {hm}] 오늘의 지표 리포트 ({today})"
        slug = f"daily-indicator-report-{today}-{hm}"

    # 1) 지표 수집
    warnings: List[str] = []

    usdkrw = fetch_usdkrw()
    if usdkrw.value is None:
        warnings.append("USD/KRW 수집 실패")

    brent = fetch_brent()
    if brent.value is None:
        warnings.append("Brent 수집 실패")

    kospi = fetch_kospi()
    if kospi.value is None:
        warnings.append("KOSPI 수집 실패")

    # 2) 뉴스 수집(한국어)
    news_usd = fetch_news("원달러 환율 원인", 3)
    news_brent = fetch_news("브렌트유 유가 하락 상승 원인", 3)
    news_kospi = fetch_news("코스피 상승 하락 원인", 3)

    # 3) 글 생성(HTML)
    html = build_html_report(
        usdkrw=usdkrw,
        brent=brent,
        kospi=kospi,
        news_usd=news_usd,
        news_brent=news_brent,
        news_kospi=news_kospi,
        warnings=warnings
    )

    # 4) WP upsert
    payload = {
        "title": title,
        "slug": slug,
        "content": html,
        "status": status,
    }

    try:
        if test_mode:
            post = wp_create_post(cfg, payload)
            link = post.get("link", cfg["wp_base_url"])
            msg = f"✅ (TEST) 글 생성 성공!\n날짜: {today}\n상태: {status}\n링크: {link}"
            print(msg)
            kakao_send_to_me(cfg, msg)
            return

        # 운영 모드: 같은 날짜 slug 있으면 업데이트
        existing = wp_find_post_by_slug(cfg, slug)
        if existing and isinstance(existing.get("id"), int):
            post_id = int(existing["id"])
            post = wp_update_post(cfg, post_id, payload)
            link = post.get("link", cfg["wp_base_url"])
            msg = f"✅ 글 업데이트 성공!\n날짜: {today}\n상태: {status}\n링크: {link}"
        else:
            post = wp_create_post(cfg, payload)
            link = post.get("link", cfg["wp_base_url"])
            msg = f"✅ 글 생성 성공!\n날짜: {today}\n상태: {status}\n링크: {link}"

        print(msg)
        kakao_send_to_me(cfg, msg)

    except Exception as e:
        err = f"❌ 자동발행 실패 ({today})\n{type(e).__name__}: {e}"
        print(err)
        try:
            kakao_send_to_me(cfg, err)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
