# -*- coding: utf-8 -*-
import os
import re
import json
import base64
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import requests
import feedparser
from bs4 import BeautifulSoup

WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bot_config.json"


# -----------------------------
# Config
# -----------------------------
def load_config():
    """
    1) daily_post.py 파일과 같은 폴더의 bot_config.json을 우선 사용
    2) 없으면 환경변수(WP_BASE_URL/WP_USER/WP_APP_PASS)로 구성
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # fallback: env
    base = os.getenv("WP_BASE_URL", "").strip()
    user = os.getenv("WP_USER", "").strip()
    app = os.getenv("WP_APP_PASS", "").strip()

    if not (base and user and app):
        raise FileNotFoundError(
            f"bot_config.json이 없습니다: {CONFIG_PATH}\n"
            f"또는 환경변수 WP_BASE_URL/WP_USER/WP_APP_PASS가 필요합니다."
        )

    cfg = {"wp_base_url": base, "wp_user": user, "wp_app_pass": app}

    kakao_rest = os.getenv("KAKAO_REST_KEY", "").strip()
    kakao_refresh = os.getenv("KAKAO_REFRESH_TOKEN", "").strip()
    if kakao_rest and kakao_refresh:
        cfg["kakao_rest_key"] = kakao_rest
        cfg["kakao_refresh_token"] = kakao_refresh

    return cfg


def now_date_str():
    return datetime.now().strftime("%Y-%m-%d")


# -----------------------------
# Kakao (optional)
# -----------------------------
def refresh_access_token(cfg):
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": cfg["kakao_rest_key"],
        "refresh_token": cfg["kakao_refresh_token"],
    }
    r = requests.post(url, data=data, timeout=30)
    r.raise_for_status()
    tokens = r.json()

    # 새 refresh_token 내려오면 저장
    if "refresh_token" in tokens and tokens["refresh_token"]:
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return tokens["access_token"]


def kakao_send_to_me(cfg, text):
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
def wp_posts_api(cfg):
    return cfg["wp_base_url"].rstrip("/") + WP_POSTS_API_SUFFIX


def wp_auth_headers(cfg):
    user = cfg["wp_user"].strip()
    app_pass = cfg["wp_app_pass"].replace(" ", "").strip()  # 공백 제거
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


def wp_post_exists(cfg, slug):
    r = requests.get(
        wp_posts_api(cfg),
        params={"slug": slug},
        headers=wp_auth_headers(cfg),
        timeout=30,
    )
    r.raise_for_status()
    return len(r.json()) > 0


def wp_create_post(cfg, title, slug, content, status="publish"):
    payload = {"title": title, "slug": slug, "content": content, "status": status}
    r = requests.post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


# -----------------------------
# News (Google News RSS)
# -----------------------------
def _looks_like_chinese_only(title: str) -> bool:
    has_hangul = bool(re.search(r"[가-힣]", title))
    has_han = bool(re.search(r"[\u4e00-\u9fff]", title))  # 한자 범위
    has_latin = bool(re.search(r"[A-Za-z]", title))
    # 한글이 전혀 없고, 한자만(또는 한자 위주)인데 영문도 거의 없으면 제외
    if has_han and not has_hangul and not has_latin:
        return True
    return False


def fetch_google_news(query: str, max_items=3):
    # 한국 뉴스 위주
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=ko&gl=KR&ceid=KR:ko"
    )
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[: max_items * 6]:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        if _looks_like_chinese_only(title):
            continue
        out.append((title, link))
        if len(out) >= max_items:
            break
    return out


# -----------------------------
# Simple indicator fetch (안전하게: 실패해도 글은 발행)
# -----------------------------
def fetch_indicators_safe():
    """
    여기서는 '실패해도 글은 올라가게'만 보장합니다.
    값 수집 로직이 실패하면 None 처리 + errors에 기록합니다.
    """
    errors = []

    # TODO: 여기에 본인 로직(환율/유가/코스피 실제값)을 붙여도 됨
    # 일단은 예시로 비워두되, 뉴스/원인 섹션은 자동 생성되게 구성
    data = {
        "usdkrw": None,
        "usdkrw_prev": None,
        "brent": None,
        "brent_prev": None,
        "kospi": None,
        "kospi_prev": None,
        "errors": errors,
    }
    return data


# -----------------------------
# HTML Build
# -----------------------------
def _fmt(v, digits=2):
    if v is None:
        return "-"
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:
        return str(v)


def _diff(latest, prev):
    if latest is None or prev is None:
        return None, None
    d = latest - prev
    p = (d / prev * 100.0) if prev else None
    return d, p


def build_post_html(today, d, run_tag=""):
    usd_d, usd_p = _diff(d.get("usdkrw"), d.get("usdkrw_prev"))
    brent_d, brent_p = _diff(d.get("brent"), d.get("brent_prev"))
    kospi_d, kospi_p = _diff(d.get("kospi"), d.get("kospi_prev"))

    def arrow(x):
        if x is None:
            return ""
        return "▲" if x > 0 else ("▼" if x < 0 else "—")

    # 원인 참고용 뉴스
    news_usd = fetch_google_news("원달러 환율 변동 원인", 3)
    news_brent = fetch_google_news("브렌트유 하락 원인", 3)
    news_kospi = fetch_google_news("코스피 변동 원인", 3)

    err_html = ""
    if d.get("errors"):
        items = "".join(f"<li>{BeautifulSoup(e,'html.parser').get_text()}</li>" for e in d["errors"])
        err_html = f"""
        <div style="border:1px solid #f5c2c7;background:#f8d7da;padding:14px;border-radius:10px;margin:14px 0;">
          <b>⚠ 일부 데이터 수집 실패</b>
          <ul style="margin:10px 0 0 20px;">{items}</ul>
        </div>
        """

    def news_list(items):
        if not items:
            return "<li>관련 뉴스 수집 실패(또는 결과 없음)</li>"
        return "".join([f'<li><a href="{link}" target="_blank" rel="noopener">{title}</a></li>' for title, link in items])

    tag_badge = f'<span style="font-size:12px;background:#eef2ff;padding:4px 8px;border-radius:999px;">TEST {run_tag}</span>' if run_tag else ""

    html = f"""
    <div style="max-width:860px;margin:0 auto;font-family:system-ui,-apple-system,Segoe UI,Roboto,'Noto Sans KR',sans-serif;line-height:1.65;">
      {err_html}

      <h2 style="margin:10px 0;">핵심 요약 {tag_badge}</h2>
      <ul>
        <li>원/달러(USD/KRW), 브렌트유, 코스피 변동을 한 번에 정리했습니다.</li>
        <li>변동 원인 참고용으로 관련 뉴스 헤드라인을 함께 첨부했습니다.</li>
      </ul>

      <h2 style="margin:22px 0 10px;">주요 지표</h2>
      <div style="overflow:auto;border:1px solid #e5e7eb;border-radius:12px;">
        <table style="width:100%;border-collapse:collapse;min-width:720px;">
          <thead>
            <tr style="background:#0f172a;color:#fff;">
              <th style="padding:12px;text-align:left;">지표</th>
              <th style="padding:12px;text-align:right;">현재</th>
              <th style="padding:12px;text-align:right;">전일대비</th>
              <th style="padding:12px;text-align:right;">변동률</th>
              <th style="padding:12px;text-align:center;">기준일</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td style="padding:12px;border-top:1px solid #e5e7eb;">USD/KRW</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{_fmt(d.get("usdkrw"))}</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{arrow(usd_d)} {_fmt(usd_d)}</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{_fmt(usd_p)}%</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:center;">{today}</td>
            </tr>
            <tr>
              <td style="padding:12px;border-top:1px solid #e5e7eb;">Brent Oil</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{_fmt(d.get("brent"))}</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{arrow(brent_d)} {_fmt(brent_d)}</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{_fmt(brent_p)}%</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:center;">{today}</td>
            </tr>
            <tr>
              <td style="padding:12px;border-top:1px solid #e5e7eb;">KOSPI</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{_fmt(d.get("kospi"))}</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{arrow(kospi_d)} {_fmt(kospi_d)}</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:right;">{_fmt(kospi_p)}%</td>
              <td style="padding:12px;border-top:1px solid #e5e7eb;text-align:center;">{today}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <h2 style="margin:24px 0 10px;">왜 움직였나? (원인 참고)</h2>

      <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:10px 0;">
        <h3 style="margin:0 0 8px;">USD/KRW 변동 원인(뉴스)</h3>
        <ul style="margin:0 0 0 18px;">{news_list(news_usd)}</ul>
      </div>

      <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:10px 0;">
        <h3 style="margin:0 0 8px;">Brent 유가 변동 원인(뉴스)</h3>
        <ul style="margin:0 0 0 18px;">{news_list(news_brent)}</ul>
      </div>

      <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px;margin:10px 0;">
        <h3 style="margin:0 0 8px;">KOSPI 변동 원인(뉴스)</h3>
        <ul style="margin:0 0 0 18px;">{news_list(news_kospi)}</ul>
      </div>

      <h2 style="margin:24px 0 10px;">내일 체크포인트</h2>
      <ul>
        <li>주요 지표 발표 일정(미국 CPI/고용, 연준 발언 등)</li>
        <li>유가: OPEC/재고/지정학 이슈 헤드라인</li>
        <li>환율: 달러 강세/위험자산 선호 변화</li>
      </ul>

      <p style="margin-top:22px;color:#6b7280;font-size:12px;">
        * 이 글의 ‘원인’은 참고용 헤드라인 모음입니다. (투자 조언 아님)
      </p>
    </div>
    """
    return html


def main():
    cfg = load_config()

    today = now_date_str()
    run_tag = os.getenv("RUN_TAG", "").strip()
    test_mode = os.getenv("TEST_MODE", "").strip() in ("1", "true", "True", "yes", "YES")

    # ✅ 테스트 모드면 초안으로(워크플로우에서 WP_STATUS로 제어)
    status = os.getenv("WP_STATUS", "publish").strip() or "publish"

    base_title = f"오늘의 지표 리포트 ({today})"
    base_slug = f"daily-indicator-report-{today}"

    # ✅ RUN_TAG가 있으면 매 실행마다 고유 slug/title → 5분마다 새 글 생성 가능
    if run_tag:
        title = f"{base_title} - {run_tag}"
        slug = f"{base_slug}-{run_tag}"
    else:
        title = base_title
        slug = base_slug

    try:
        # 중복 발행 방지(테스트 RUN_TAG 있을 땐 스킵)
        if not run_tag and wp_post_exists(cfg, slug):
            msg = f"✅ 이미 오늘 글이 있어요 ({today})\n중복 발행 안 함"
            print(msg)
            kakao_send_to_me(cfg, msg)
            return

        data = fetch_indicators_safe()  # 실패해도 글은 발행되게
        html = build_post_html(today, data, run_tag=run_tag)

        post = wp_create_post(cfg, title, slug, html, status=status)
        link = post.get("link", cfg["wp_base_url"])

        ok = f"✅ 글 작성 성공!\n날짜: {today}\n상태: {status}\n링크: {link}"
        print(ok)
        kakao_send_to_me(cfg, ok)

    except Exception as e:
        msg = f"❌ 자동발행 실패 ({today})\n{type(e).__name__}: {e}"
        print(msg)
        try:
            kakao_send_to_me(cfg, msg)
        except Exception:
            pass


if __name__ == "__main__":
    main()
