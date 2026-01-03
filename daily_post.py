# -*- coding: utf-8 -*-
import json
import base64
import csv
import io
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

import requests

WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) daily-post-bot/1.0"
HEADERS_COMMON = {"User-Agent": UA}

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "bot_config.json"


# -----------------------------
# Config
# -----------------------------
def ensure_config_file():
    if CONFIG_PATH.exists():
        return

    template = {
        "wp_base_url": "https://rainsow.com",
        "wp_user": "YOUR_WP_USERNAME",
        "wp_app_pass": "xxxx xxxx xxxx xxxx xxxx xxxx",  # 워드프레스 'Application Password'
        "wp_status": "publish",  # publish 또는 draft
        "kakao_rest_key": "",
        "kakao_refresh_token": ""
    }
    CONFIG_PATH.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(
        f"[필수] {CONFIG_PATH.name} 파일이 없어서 템플릿을 생성했습니다.\n"
        f"같은 폴더에 생긴 bot_config.json을 열어 값 채운 뒤 다시 실행하세요."
    )


def load_config():
    ensure_config_file()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def now_date_str():
    return datetime.now().strftime("%Y-%m-%d")


# -----------------------------
# Kakao (선택)
# -----------------------------
def refresh_access_token(cfg):
    if not cfg.get("kakao_rest_key") or not cfg.get("kakao_refresh_token"):
        return None

    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": cfg["kakao_rest_key"],
        "refresh_token": cfg["kakao_refresh_token"],
    }
    r = requests.post(url, data=data, headers=HEADERS_COMMON, timeout=30)
    r.raise_for_status()
    tokens = r.json()

    if tokens.get("refresh_token"):
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    return tokens["access_token"]


def kakao_send_to_me(cfg, text):
    access_token = refresh_access_token(cfg)
    if not access_token:
        return False

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {**HEADERS_COMMON, "Authorization": f"Bearer {access_token}"}

    template_object = {
        "object_type": "text",
        "text": text[:1000],
        "link": {"web_url": cfg["wp_base_url"], "mobile_web_url": cfg["wp_base_url"]},
        "button_title": "사이트 열기"
    }
    data = {"template_object": json.dumps(template_object, ensure_ascii=False)}
    r = requests.post(url, headers=headers, data=data, timeout=30)
    r.raise_for_status()
    return True


# -----------------------------
# WordPress REST
# -----------------------------
def wp_posts_api(cfg):
    return cfg["wp_base_url"].rstrip("/") + WP_POSTS_API_SUFFIX


def wp_auth_headers(cfg):
    user = str(cfg["wp_user"]).strip()
    app_pass = str(cfg["wp_app_pass"]).replace(" ", "").strip()  # 공백 제거
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}", "User-Agent": UA}


def wp_request_json(method, url, cfg, params=None, payload=None):
    headers = {**wp_auth_headers(cfg), "Content-Type": "application/json"}
    r = requests.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        data=(json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload else None),
        timeout=30
    )
    r.raise_for_status()
    return r.json()


def wp_find_posts_by_search(cfg, term):
    # status=any 로 draft/publish/trashed 등도 같이 조회(권한 필요)
    params = {"search": term, "status": "any", "per_page": 100}
    return wp_request_json("GET", wp_posts_api(cfg), cfg, params=params)


def wp_upsert_daily_post(cfg, title, slug_base, content_html, status="publish"):
    # 1) 같은 날짜 글이 이미 있으면(슬러그 -2/-3 포함) 그 글을 업데이트
    candidates = wp_find_posts_by_search(cfg, slug_base)
    target = None
    for p in candidates:
        s = (p.get("slug") or "").strip()
        if s == slug_base or s.startswith(slug_base + "-"):
            target = p
            break

    payload = {
        "title": title,
        "content": content_html,
        "status": status
    }

    if target and target.get("id"):
        post_id = target["id"]
        updated = wp_request_json("POST", f"{wp_posts_api(cfg)}/{post_id}", cfg, payload=payload)
        return updated, "updated"

    # 2) 없으면 새로 생성(슬러그 지정)
    payload["slug"] = slug_base
    created = wp_request_json("POST", wp_posts_api(cfg), cfg, payload=payload)
    return created, "created"


# -----------------------------
# Data fetch (무료/키 없이)
# -----------------------------
def fetch_fred_last_two(series_id):
    # 예: https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, headers=HEADERS_COMMON, timeout=30)
    r.raise_for_status()

    text = r.text.strip()
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"FRED 데이터가 비었습니다: {series_id}")

    header = rows[0]
    # 보통: DATE, SERIESID
    value_col = 1 if len(header) >= 2 else None
    if value_col is None:
        raise ValueError(f"FRED 헤더 파싱 실패: {series_id}")

    vals = []
    for row in rows[1:]:
        if len(row) <= value_col:
            continue
        v = row[value_col].strip()
        if not v or v == ".":
            continue
        try:
            vals.append(float(v))
        except:
            continue

    if len(vals) < 1:
        raise ValueError(f"유효한 값이 없습니다: {series_id}")

    last = vals[-1]
    prev = vals[-2] if len(vals) >= 2 else None
    return last, prev


def fetch_naver_now_value(url):
    r = requests.get(url, headers=HEADERS_COMMON, timeout=30)
    r.raise_for_status()
    html = r.text

    # 자주 쓰이는 패턴: id="now_value">2,625.30</em>
    m = re.search(r'id="now_value"\s*>\s*([0-9,\.]+)\s*<', html)
    if not m:
        # fallback: class="no_today">...<span class="blind">현재</span> ... 숫자
        m = re.search(r'([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?)', html)
    if not m:
        raise ValueError(f"네이버 파싱 실패: {url}")

    return float(m.group(1).replace(",", ""))


def fetch_indicators_real():
    # 1) USD/KRW (FRED: DEXKOUS = KRW per USD)
    usdkrw, usdkrw_prev = fetch_fred_last_two("DEXKOUS")

    # 2) Brent (FRED: DCOILBRENTEU = Dollars per Barrel)
    brent, brent_prev = fetch_fred_last_two("DCOILBRENTEU")

    # 3) KOSPI (네이버)
    kospi = fetch_naver_now_value("https://finance.naver.com/sise/sise_index.naver?code=KOSPI")
    kospi_prev = None  # 페이지에서 전일 값까지 안정적으로 뽑으려면 추가 파싱이 필요(일단 생략)

    def delta_str(cur, prev):
        if prev is None:
            return ""
        d = cur - prev
        sign = "+" if d >= 0 else ""
        return f" ({sign}{d:.2f})"

    commentary = []
    # 환율: 상승 = 원 약세
    if usdkrw_prev is not None:
        if usdkrw > usdkrw_prev:
            commentary.append(f"원/달러 환율이 상승(원 약세)했습니다{delta_str(usdkrw, usdkrw_prev)}.")
        else:
            commentary.append(f"원/달러 환율이 하락(원 강세)했습니다{delta_str(usdkrw, usdkrw_prev)}.")
    else:
        commentary.append("원/달러 환율 최신값을 확인했습니다.")

    if brent_prev is not None:
        if brent > brent_prev:
            commentary.append(f"브렌트유가가 상승했습니다{delta_str(brent, brent_prev)}.")
        else:
            commentary.append(f"브렌트유가가 하락했습니다{delta_str(brent, brent_prev)}.")
    else:
        commentary.append("브렌트유 최신값을 확인했습니다.")

    commentary.append("코스피는 장중 변동이 크니(특히 이벤트/발표일) 변동 원인을 같이 체크하세요.")

    return {
        "usdkrw": usdkrw,
        "brent": brent,
        "kospi": kospi,
        "commentary": commentary,
        "sources": [
            "USD/KRW: FRED(Dexkous), Brent: FRED(Dcoilbrenteu), KOSPI: Naver Finance"
        ]
    }


def build_post_content_html(data):
    def fmt(x, digits=2):
        try:
            return f"{float(x):,.{digits}f}"
        except:
            return "-"

    lines = []
    lines.append("<h2>오늘의 한줄 요약</h2>")
    one = data["commentary"][0] if data.get("commentary") else "오늘 핵심 요약"
    lines.append(f"<ul><li>{one}</li></ul>")

    lines.append("<h2>주요 지표</h2>")
    lines.append("<ul>")
    lines.append(f"<li>USD/KRW: <b>{fmt(data.get('usdkrw'))}</b></li>")
    lines.append(f"<li>Brent Oil: <b>{fmt(data.get('brent'))}</b></li>")
    lines.append(f"<li>KOSPI: <b>{fmt(data.get('kospi'))}</b></li>")
    lines.append("</ul>")

    lines.append("<h2>코멘터리</h2>")
    lines.append("<ul>")
    for c in data.get("commentary", []):
        lines.append(f"<li>{c}</li>")
    lines.append("</ul>")

    lines.append("<h2>체크포인트(내일 확인할 것)</h2>")
    lines.append("<ul>")
    lines.append("<li>큰 변동 원인(뉴스/발표 일정)</li>")
    lines.append("<li>다음 지표 발표/이벤트 캘린더</li>")
    lines.append("</ul>")

    lines.append("<h2>출처</h2>")
    lines.append("<ul>")
    for s in data.get("sources", []):
        lines.append(f"<li>{s}</li>")
    lines.append("</ul>")

    return "\n".join(lines)


def run_once():
    cfg = load_config()
    today = now_date_str()
    title = f"오늘의 지표 리포트 ({today})"
    slug_base = f"daily-indicator-report-{today}"
    status = (cfg.get("wp_status") or "publish").strip().lower()

    try:
        data = fetch_indicators_real()
        content_html = build_post_content_html(data)

        post, mode = wp_upsert_daily_post(cfg, title, slug_base, content_html, status=status)
        link = post.get("link", cfg["wp_base_url"])

        msg = f"✅ 글 {mode} 성공!\n날짜: {today}\n상태: {status}\n링크: {link}"
        print(msg)
        kakao_send_to_me(cfg, msg)

    except Exception as e:
        msg = f"❌ 자동발행 실패 ({today})\n{type(e).__name__}: {e}"
        print(msg)
        try:
            kakao_send_to_me(cfg, msg)
        except:
            pass


def run_daemon(at_hhmm="09:00"):
    # 계속 실행(PC 켜져 있을 때만). 추천은 작업 스케줄러(아래 설명)
    hh, mm = map(int, at_hhmm.split(":"))
    print(f"[daemon] 매일 {hh:02d}:{mm:02d} 실행 모드 시작")

    last_run_date = None
    while True:
        now = datetime.now()
        if now.hour == hh and now.minute == mm:
            if last_run_date != now.date():
                run_once()
                last_run_date = now.date()
                time.sleep(60)  # 중복 방지
        time.sleep(5)


if __name__ == "__main__":
    # 기본: 그냥 실행하면 1번 발행(publish)
    run_once()

    # 계속 자동 실행(원하면 아래 주석 해제)
    # run_daemon("09:00")
