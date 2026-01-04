# -*- coding: utf-8 -*-
"""
daily_post.py (WordPress 자동 글 발행)
- 무료/키 없이 동작하도록 구성 (AlphaVantage 등 API Key 불필요)
- 지표: USD/KRW, Brent(선물), KOSPI
- 변동 원인(참고): Google News RSS(한국어) 헤드라인 3개씩
- TEST_MODE=1 이면 5분 테스트용으로 매번 새 글(슬러그 고유) + draft 발행

필수 환경변수/설정(bot_config.json 또는 GitHub Secrets)
- WP_BASE_URL: 예) https://rainsow.com
- WP_USER: 워드프레스 사용자명
- WP_APP_PASS: 워드프레스 "애플리케이션 비밀번호"

선택
- KAKAO_REST_KEY, KAKAO_REFRESH_TOKEN (카톡 나에게 보내기 알림)
- WP_POST_STATUS: publish/draft (TEST_MODE=1이면 draft가 우선)
- TEST_MODE: "1"이면 테스트 모드
"""

import os
import json
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import requests
import feedparser

WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

KST = timezone(timedelta(hours=9))


# -----------------------------
# Config
# -----------------------------
def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def load_config() -> dict:
    """
    1) bot_config.json(스크립트 옆) 읽고
    2) 환경변수로 덮어쓰기
    """
    cfg: dict = {}

    cfg_path = Path(os.getenv("BOT_CONFIG_PATH", _script_dir() / "bot_config.json"))
    if cfg_path.exists():
        cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))

    # env overlay
    mapping = {
        "WP_BASE_URL": "wp_base_url",
        "WP_USER": "wp_user",
        "WP_APP_PASS": "wp_app_pass",
        "WP_POST_STATUS": "wp_post_status",
        "KAKAO_REST_KEY": "kakao_rest_key",
        "KAKAO_REFRESH_TOKEN": "kakao_refresh_token",
    }
    for env_k, cfg_k in mapping.items():
        v = os.getenv(env_k)
        if v:
            cfg[cfg_k] = v

    # sanitize
    if "wp_base_url" in cfg and isinstance(cfg["wp_base_url"], str):
        cfg["wp_base_url"] = cfg["wp_base_url"].strip().rstrip("/")
    if "wp_user" in cfg and isinstance(cfg["wp_user"], str):
        cfg["wp_user"] = cfg["wp_user"].strip()
    if "wp_app_pass" in cfg and isinstance(cfg["wp_app_pass"], str):
        # 애플리케이션 비번은 공백이 섞이면 인증 실패함
        cfg["wp_app_pass"] = cfg["wp_app_pass"].replace(" ", "").strip()

    return cfg


def require_keys(cfg: dict, keys: list[str]) -> None:
    missing = [k for k in keys if not cfg.get(k)]
    if missing:
        raise ValueError("필수 설정 누락: " + ", ".join(missing))


def now_kst() -> datetime:
    return datetime.now(KST)


# -----------------------------
# HTTP helpers
# -----------------------------
def http_get(url: str, *, params=None, headers=None, timeout=25) -> requests.Response:
    h = {"User-Agent": DEFAULT_UA, "Accept": "*/*"}
    if headers:
        h.update(headers)
    r = requests.get(url, params=params, headers=h, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


# -----------------------------
# Kakao (optional)
# -----------------------------
def refresh_access_token(cfg: dict) -> str:
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": cfg["kakao_rest_key"],
        "refresh_token": cfg["kakao_refresh_token"],
    }
    r = requests.post(url, data=data, timeout=30)
    r.raise_for_status()
    tokens = r.json()

    # 새 refresh_token이 오면 갱신 저장
    if tokens.get("refresh_token"):
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        # 가능하면 같은 위치에 저장(없으면 무시)
        try:
            cfg_path = Path(os.getenv("BOT_CONFIG_PATH", _script_dir() / "bot_config.json"))
            cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    return tokens["access_token"]


def kakao_send_to_me(cfg: dict, text: str) -> bool:
    if not (cfg.get("kakao_rest_key") and cfg.get("kakao_refresh_token")):
        return False

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
    r = requests.post(url, headers=headers, data=data, timeout=30)
    r.raise_for_status()
    return True


# -----------------------------
# WordPress
# -----------------------------
def wp_posts_api(cfg: dict) -> str:
    return cfg["wp_base_url"] + WP_POSTS_API_SUFFIX


def wp_get(cfg: dict, url: str, *, params=None):
    return requests.get(
        url,
        params=params,
        auth=(cfg["wp_user"], cfg["wp_app_pass"]),
        headers={"User-Agent": DEFAULT_UA, "Accept": "application/json"},
        timeout=30,
    )


def wp_post(cfg: dict, url: str, *, payload: dict):
    return requests.post(
        url,
        json=payload,
        auth=(cfg["wp_user"], cfg["wp_app_pass"]),
        headers={"User-Agent": DEFAULT_UA, "Accept": "application/json"},
        timeout=30,
    )


def wp_post_exists(cfg: dict, slug: str) -> bool:
    r = wp_get(cfg, wp_posts_api(cfg), params={"slug": slug, "per_page": 1})
    r.raise_for_status()
    return len(r.json()) > 0


def wp_create_post(cfg: dict, title: str, slug: str, content_html: str, status: str) -> dict:
    payload = {"title": title, "slug": slug, "content": content_html, "status": status}
    r = wp_post(cfg, wp_posts_api(cfg), payload=payload)
    r.raise_for_status()
    return r.json()


# -----------------------------
# Data models
# -----------------------------
@dataclass
class Indicator:
    name: str
    value: float | None
    prev: float | None
    date: str | None
    unit: str = ""
    source: str = ""

    @property
    def delta(self) -> float | None:
        if self.value is None or self.prev is None:
            return None
        return self.value - self.prev

    @property
    def pct(self) -> float | None:
        if self.value is None or self.prev in (None, 0):
            return None
        return (self.value - self.prev) / self.prev * 100.0


# -----------------------------
# Data fetching (no API keys)
# -----------------------------
def _parse_float(x: str) -> float | None:
    try:
        return float(str(x).strip())
    except Exception:
        return None


def fetch_stooq_last_two_closes(symbol: str) -> tuple[float, float, str]:
    """
    Stooq CSV: https://stooq.com/q/d/l/?s=<symbol>&i=d
    - 반환: (latest_close, prev_close, latest_date)
    """
    url = "https://stooq.com/q/d/l/"
    r = http_get(
        url,
        params={"s": symbol, "i": "d"},
        headers={"Accept": "text/csv,application/csv;q=0.9,*/*;q=0.8"},
        timeout=30,
    )
    text = r.text.strip()

    # 가끔 HTML/빈값이 올 수 있음 (차단/오류)
    if not text or "<html" in text.lower():
        raise ValueError(f"Stooq CSV 수집 실패: {symbol} (CSV가 아닌 응답)")

    reader = csv.DictReader(text.splitlines())
    rows = []
    for row in reader:
        c = _parse_float(row.get("Close"))
        d = row.get("Date")
        if c is not None and d:
            rows.append((d, c))

    if len(rows) < 2:
        raise ValueError(f"Stooq 데이터가 부족합니다: {symbol}")

    latest_d, latest_c = rows[-1]
    prev_d, prev_c = rows[-2]
    return latest_c, prev_c, latest_d


def fetch_usdkrw() -> tuple[float, float, str]:
    """
    1) 무료 환율 JSON(날짜 폴더 기반)로 시도
    2) 실패하면 Stooq(usdkrw)로 폴백
    """
    base = "https://cdn.jsdelivr.net/gh/fawazahmed0/currency-api@1"
    latest_url = f"{base}/latest/currencies/usd/krw.json"

    try:
        r = http_get(latest_url, timeout=20)
        data = r.json()
        latest_date = data.get("date") or data.get("Date")
        latest_rate = data.get("krw")
        if latest_rate is None and isinstance(data.get("usd"), dict):
            latest_rate = data["usd"].get("krw")

        if latest_date and latest_rate is not None:
            d0 = datetime.fromisoformat(str(latest_date)).date()
            prev_rate = None
            prev_date = None
            for i in range(1, 8):
                dd = (d0 - timedelta(days=i)).isoformat()
                url = f"{base}/{dd}/currencies/usd/krw.json"
                try:
                    r2 = http_get(url, timeout=20)
                    data2 = r2.json()
                    pr = data2.get("krw")
                    if pr is None and isinstance(data2.get("usd"), dict):
                        pr = data2["usd"].get("krw")
                    if pr is not None:
                        prev_rate = float(pr)
                        prev_date = dd
                        break
                except Exception:
                    continue

            if prev_rate is None or prev_date is None:
                raise ValueError("환율 전일 데이터 확보 실패")

            return float(latest_rate), float(prev_rate), str(latest_date)

        raise ValueError("환율 응답 포맷 파싱 실패")

    except Exception:
        latest, prev, d = fetch_stooq_last_two_closes("usdkrw")
        return latest, prev, d


def fetch_brent() -> tuple[float, float, str, str]:
    """Brent 선물: stooq에서 cb.f 또는 brn.f 중 성공하는 심볼 사용"""
    for sym in ("cb.f", "brn.f"):
        try:
            latest, prev, d = fetch_stooq_last_two_closes(sym)
            return latest, prev, d, sym
        except Exception:
            pass
    raise ValueError("Brent 데이터를 가져오지 못했습니다 (cb.f/brn.f 모두 실패)")


def fetch_kospi() -> tuple[float, float, str, str]:
    """KOSPI: stooq에서 ^kospi 우선, 실패 시 kospi 폴백"""
    for sym in ("^kospi", "kospi"):
        try:
            latest, prev, d = fetch_stooq_last_two_closes(sym)
            return latest, prev, d, sym
        except Exception:
            pass
    raise ValueError("KOSPI 데이터를 가져오지 못했습니다 (^kospi/kospi 모두 실패)")


# -----------------------------
# News (Google News RSS)
# -----------------------------
def _looks_non_ko(title: str) -> bool:
    """중국어/일본어 위주 제목을 대충 걸러내기(완벽X)."""
    if not title:
        return True
    s = title.strip()
    if any("\uac00" <= ch <= "\ud7a3" for ch in s):  # Hangul present
        return False

    han = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    kana = sum(1 for ch in s if "\u3040" <= ch <= "\u30ff")
    denom = max(len(s), 1)
    if han / denom > 0.15 or kana / denom > 0.15:
        return True
    return False


def fetch_google_news(query: str, max_items: int = 3) -> list[dict]:
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=ko&gl=KR&ceid=KR:ko"
    )
    feed = feedparser.parse(url)

    items: list[dict] = []
    for e in getattr(feed, "entries", [])[:20]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        if not title or not link:
            continue
        if _looks_non_ko(title):
            continue
        items.append({"title": title, "link": link})
        if len(items) >= max_items:
            break

    if not items:
        for e in getattr(feed, "entries", [])[:max_items]:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()
            if title and link:
                items.append({"title": title, "link": link})
    return items


def extract_keywords(titles: list[str]) -> list[str]:
    dict_terms = [
        "연준", "금리", "CPI", "물가", "고용", "달러", "환율", "위험자산", "안전자산",
        "OPEC", "재고", "감산", "증산", "중동", "러시아", "중국", "경기", "침체",
        "외국인", "수급", "반도체", "실적", "채권", "국채", "유가", "원화"
    ]
    found = []
    joined = " / ".join(titles)
    for t in dict_terms:
        if t.lower() in joined.lower():
            found.append(t)
    return found[:10]


# -----------------------------
# Content builder (HTML)
# -----------------------------
def fmt_num(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    return f"{x:,.{digits}f}"


def fmt_delta(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:,.{digits}f}"


def fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "-"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{digits}f}%"


def build_post_html(today: str, indicators: list[Indicator], news_map: dict, errors: list[str], test_tag: str | None) -> str:
    gen_time = now_kst().strftime("%Y-%m-%d %H:%M KST")

    html = []
    html.append(f"<h2>오늘의 지표 리포트 ({today})</h2>")
    if test_tag:
        html.append(
            "<p><span style='display:inline-block;padding:4px 10px;border-radius:999px;"
            "background:#eef2ff;color:#3730a3;font-weight:700;'>"
            f"TEST {test_tag}</span></p>"
        )

    if errors:
        items = "".join(f"<li>{e}</li>" for e in errors)
        html.append(
            "<div style='border:1px solid #fecaca;background:#fff1f2;padding:14px;border-radius:12px;margin:12px 0;'>"
            "<b>⚠ 일부 데이터 수집 실패</b>"
            f"<ul style='margin:10px 0 0 18px;'>{items}</ul>"
            "</div>"
        )

    html.append("<h3>핵심 요약</h3>")
    html.append("<ul>")
    html.append("<li>원/달러(USD/KRW), 브렌트유, 코스피의 <b>최근 거래일 기준</b> 변동을 한 번에 정리했습니다.</li>")
    html.append("<li>변동 원인(참고용)으로 관련 뉴스 헤드라인을 함께 첨부했습니다.</li>")
    html.append("</ul>")

    html.append("<h3>주요 지표</h3>")
    html.append(
        "<table style='width:100%;border-collapse:collapse;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;'>"
        "<thead>"
        "<tr style='background:#0f172a;color:#fff;'>"
        "<th style='padding:12px;text-align:left;'>지표</th>"
        "<th style='padding:12px;text-align:right;'>현재</th>"
        "<th style='padding:12px;text-align:right;'>전일대비</th>"
        "<th style='padding:12px;text-align:right;'>변동률</th>"
        "<th style='padding:12px;text-align:center;'>기준일(데이터)</th>"
        "</tr>"
        "</thead><tbody>"
    )
    for ind in indicators:
        val = fmt_num(ind.value, 2) + (f" {ind.unit}" if ind.unit else "")
        delta = fmt_delta(ind.delta, 2)
        pct = fmt_pct(ind.pct, 2)
        d = ind.date or "-"
        html.append(
            "<tr>"
            f"<td style='padding:12px;border-top:1px solid #e5e7eb;'><b>{ind.name}</b></td>"
            f"<td style='padding:12px;border-top:1px solid #e5e7eb;text-align:right;'>{val}</td>"
            f"<td style='padding:12px;border-top:1px solid #e5e7eb;text-align:right;'>{delta}</td>"
            f"<td style='padding:12px;border-top:1px solid #e5e7eb;text-align:right;'>{pct}</td>"
            f"<td style='padding:12px;border-top:1px solid #e5e7eb;text-align:center;'>{d}</td>"
            "</tr>"
        )
    html.append("</tbody></table>")

    html.append("<h3 style='margin-top:22px;'>왜 움직였나? (원인 참고)</h3>")
    html.append("<p style='color:#475569;margin-top:-8px;'><b>헤드라인 기반 참고</b>입니다. 실제 원인은 복합적일 수 있어요.</p>")

    for _, block in news_map.items():
        title = block.get("title", "")
        headlines = block.get("items", [])
        keywords = block.get("keywords", [])

        html.append(
            "<div style='border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:12px 0;'>"
            f"<h4 style='margin:0 0 8px 0;'>{title}</h4>"
        )
        if keywords:
            html.append(
                f"<div style='font-size:13px;color:#334155;margin-bottom:8px;'>"
                f"오늘 키워드(참고): <b>{', '.join(keywords)}</b></div>"
            )
        if headlines:
            html.append("<ul style='margin:0 0 0 18px;'>")
            for it in headlines:
                html.append(f"<li><a href='{it['link']}' target='_blank' rel='noopener'>{it['title']}</a></li>")
            html.append("</ul>")
        else:
            html.append("<div style='color:#64748b;'>- 관련 뉴스가 없거나 수집에 실패했습니다.</div>")
        html.append("</div>")

    html.append("<h3>내일 체크포인트</h3>")
    html.append("<ul>")
    html.append("<li>미국 CPI/고용 등 주요 지표 발표 일정</li>")
    html.append("<li>유가: OPEC+ 발언/재고 발표/지정학 이슈</li>")
    html.append("<li>환율/주식: 달러 강세 여부, 외국인 수급</li>")
    html.append("</ul>")

    html.append("<hr/>")
    html.append(f"<div style='color:#64748b;font-size:12px;'>생성 시각: {gen_time}</div>")
    html.append("<div style='color:#64748b;font-size:12px;'>데이터: Stooq / Currency-api(환율) · 뉴스: Google News RSS</div>")
    return "\n".join(html)


# -----------------------------
# Main
# -----------------------------
def fetch_indicators_and_news() -> tuple[list[Indicator], dict, list[str]]:
    errors: list[str] = []
    indicators: list[Indicator] = []

    try:
        v, p, d = fetch_usdkrw()
        indicators.append(Indicator(name="USD/KRW", value=v, prev=p, date=d))
    except Exception as e:
        errors.append(f"USD/KRW 수집 실패: {type(e).__name__}: {e}")
        indicators.append(Indicator(name="USD/KRW", value=None, prev=None, date=None))

    try:
        v, p, d, sym = fetch_brent()
        indicators.append(Indicator(name="Brent Oil", value=v, prev=p, date=d, unit="USD", source=f"stooq:{sym}"))
    except Exception as e:
        errors.append(f"Brent 수집 실패: {type(e).__name__}: {e}")
        indicators.append(Indicator(name="Brent Oil", value=None, prev=None, date=None))

    try:
        v, p, d, sym = fetch_kospi()
        indicators.append(Indicator(name="KOSPI", value=v, prev=p, date=d, source=f"stooq:{sym}"))
    except Exception as e:
        errors.append(f"KOSPI 수집 실패: {type(e).__name__}: {e}")
        indicators.append(Indicator(name="KOSPI", value=None, prev=None, date=None))

    news_map = {}

    def add_news(block_key: str, title: str, query: str):
        try:
            items = fetch_google_news(query, max_items=3)
            titles = [it["title"] for it in items]
            news_map[block_key] = {"title": title, "items": items, "keywords": extract_keywords(titles)}
        except Exception:
            news_map[block_key] = {"title": title, "items": [], "keywords": []}

    add_news("usdkrw", "USD/KRW 변동 원인(뉴스)", "원달러 환율 변동 원인")
    add_news("brent", "Brent 유가 변동 원인(뉴스)", "브렌트유 유가 하락 상승 원인")
    add_news("kospi", "KOSPI 변동 원인(뉴스)", "코스피 상승 하락 원인")

    return indicators, news_map, errors


def main():
    cfg = load_config()
    require_keys(cfg, ["wp_base_url", "wp_user", "wp_app_pass"])

    test_mode = os.getenv("TEST_MODE", "").strip() == "1" or str(cfg.get("test_mode", "")).strip() == "1"

    today = now_kst().strftime("%Y-%m-%d")
    run_tag = str(int(time.time())) if test_mode else None

    title = f"오늘의 지표 리포트 ({today})" + (f" - {run_tag}" if run_tag else "")
    slug = f"daily-indicator-report-{today}" + (f"-{run_tag}" if run_tag else "")

    status = (cfg.get("wp_post_status") or "publish").strip().lower()
    if test_mode:
        status = "draft"

    if not test_mode and wp_post_exists(cfg, slug):
        msg = f"✅ 이미 오늘 글이 있어요 ({today}) → 중복 발행 안 함"
        print(msg)
        kakao_send_to_me(cfg, msg)
        return

    indicators, news_map, errors = fetch_indicators_and_news()
    content_html = build_post_html(today, indicators, news_map, errors, run_tag)

    post = wp_create_post(cfg, title, slug, content_html, status=status)
    link = post.get("link", cfg["wp_base_url"])
    msg = f"✅ 글 작성 성공!\n날짜: {today}\n상태: {status}\n링크: {link}"
    print(msg)
    kakao_send_to_me(cfg, msg)


if __name__ == "__main__":
    main()
