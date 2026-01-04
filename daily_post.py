# -*- coding: utf-8 -*-
"""
Daily WordPress Indicator Post (통합본)
- WordPress REST API로 글 자동 발행
- 데이터: Alpha Vantage (무료 API Key 필요)
- 뉴스: Google News RSS(한국 로케일) + 한글 제목만 필터
- GitHub Actions/로컬 둘 다 동작

필수 설정(Secrets 또는 bot_config.json):
- WP_BASE_URL, WP_USER, WP_APP_PASS
- ALPHAVANTAGE_API_KEY

선택:
- KAKAO_REST_KEY, KAKAO_REFRESH_TOKEN
- WP_STATUS: publish/draft (기본 publish)
- TEST_MODE: "1"이면 매 실행마다 새 글(슬러그에 run_id 붙임)
"""

import os
import json
import base64
import re
import time
from pathlib import Path
from datetime import datetime

import requests
import feedparser

CONFIG_NAME = "bot_config.json"
WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"

AV_ENDPOINT = "https://www.alphavantage.co/query"
DEFAULT_TIMEOUT = 30


# -----------------------------
# 공통 유틸 / 설정
# -----------------------------
def now_kst_date_str():
    # GitHub Actions에서도 TZ 주입 가능하지만, 안전하게 "오늘" 기준만 사용
    return datetime.now().strftime("%Y-%m-%d")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config():
    """
    1) 현재 작업폴더 bot_config.json
    2) 스크립트 옆 bot_config.json
    3) 환경변수로 구성
    """
    candidates = [
        Path.cwd() / CONFIG_NAME,
        Path(__file__).resolve().with_name(CONFIG_NAME),
    ]
    for p in candidates:
        if p.exists():
            cfg = _load_json(p)
            cfg["_config_path"] = str(p)
            return cfg

    # env fallback
    cfg = {
        "wp_base_url": os.getenv("WP_BASE_URL", "").strip(),
        "wp_user": os.getenv("WP_USER", "").strip(),
        "wp_app_pass": os.getenv("WP_APP_PASS", "").strip(),
        "alphavantage_api_key": os.getenv("ALPHAVANTAGE_API_KEY", "").strip(),
        "wp_status": os.getenv("WP_STATUS", "publish").strip() or "publish",
        "test_mode": os.getenv("TEST_MODE", "").strip(),
    }
    # kakao optional
    if os.getenv("KAKAO_REST_KEY") and os.getenv("KAKAO_REFRESH_TOKEN"):
        cfg["kakao_rest_key"] = os.getenv("KAKAO_REST_KEY").strip()
        cfg["kakao_refresh_token"] = os.getenv("KAKAO_REFRESH_TOKEN").strip()

    cfg["_config_path"] = "(env)"
    return cfg


def must_get(cfg, key: str):
    v = (cfg.get(key) or "").strip()
    if not v:
        raise ValueError(f"필수 설정 누락: {key}")
    return v


def session():
    s = requests.Session()
    # 일부 사이트가 UA 없으면 차단하는 경우가 있어 기본 UA 부여
    s.headers.update({"User-Agent": "Mozilla/5.0 (DailyReportBot/1.0)"})
    return s


# -----------------------------
# Kakao (선택)
# -----------------------------
def refresh_access_token(cfg):
    url = "https://kauth.kakao.com/oauth/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": cfg["kakao_rest_key"],
        "refresh_token": cfg["kakao_refresh_token"],
    }
    r = session().post(url, data=data, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    tokens = r.json()

    if tokens.get("refresh_token"):
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        # 로컬에서만 저장 가능(액션에서는 파일이 없을 수 있음)
        try:
            p = Path(__file__).resolve().with_name(CONFIG_NAME)
            if p.exists():
                with p.open("w", encoding="utf-8") as f:
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
    r = session().post(url, headers=headers, data=data, timeout=DEFAULT_TIMEOUT)
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
    # draft까지 중복 체크하려면 status=any가 안전
    r = session().get(
        wp_posts_api(cfg),
        params={"slug": slug, "status": "any"},
        headers=wp_auth_headers(cfg),
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    return len(r.json()) > 0


def wp_create_post(cfg, title, slug, html_content, status="publish"):
    payload = {"title": title, "slug": slug, "content": html_content, "status": status}
    r = session().post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


# -----------------------------
# 데이터: Alpha Vantage
# -----------------------------
def av_get(params: dict, api_key: str):
    p = dict(params)
    p["apikey"] = api_key
    r = session().get(AV_ENDPOINT, params=p, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _pick_last_two_dates(d: dict):
    # keys = "YYYY-MM-DD"
    dates = sorted(d.keys(), reverse=True)
    if len(dates) < 2:
        raise ValueError("최근 2거래일 데이터가 부족합니다.")
    return dates[0], dates[1]


def fetch_usdkrw(api_key: str):
    j = av_get(
        {
            "function": "FX_DAILY",
            "from_symbol": "USD",
            "to_symbol": "KRW",
            "outputsize": "compact",
        },
        api_key,
    )
    ts = j.get("Time Series FX (Daily)")
    if not ts:
        raise ValueError("USD/KRW: Alpha Vantage 응답에 시계열이 없습니다.")
    d0, d1 = _pick_last_two_dates(ts)
    c0 = float(ts[d0]["4. close"])
    c1 = float(ts[d1]["4. close"])
    return c0, c1, d0  # latest, prev, latest_date


def fetch_brent(api_key: str):
    # Alpha Vantage commodities: BRENT
    j = av_get({"function": "BRENT", "interval": "daily"}, api_key)
    data = j.get("data") or []
    if len(data) < 2:
        raise ValueError("Brent: Alpha Vantage data가 부족합니다.")
    # data: [{"date":"YYYY-MM-DD","value":"..."}] 보통 최신이 앞에 옴
    data_sorted = sorted(data, key=lambda x: x.get("date", ""), reverse=True)
    v0 = float(data_sorted[0]["value"])
    v1 = float(data_sorted[1]["value"])
    d0 = data_sorted[0]["date"]
    return v0, v1, d0


def fetch_kospi(api_key: str, symbol_primary="^KS11", symbol_fallback="EWY"):
    """
    KOSPI는 데이터 제공처/심볼 이슈가 있을 수 있어 2단계로 시도:
    1) ^KS11 (KOSPI)
    2) 실패 시 EWY(한국 ETF)로 대체 표기
    """
    def _daily(symbol: str):
        j = av_get(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",
            },
            api_key,
        )
        ts = j.get("Time Series (Daily)")
        if not ts:
            # Alpha Vantage는 오류 메시지가 "Error Message" / "Note"로 올 수 있음
            note = j.get("Note") or j.get("Error Message") or "시계열 없음"
            raise ValueError(f"{symbol}: {note}")
        d0, d1 = _pick_last_two_dates(ts)
        c0 = float(ts[d0]["4. close"])
        c1 = float(ts[d1]["4. close"])
        return c0, c1, d0

    try:
        c0, c1, d0 = _daily(symbol_primary)
        return {"label": "KOSPI", "symbol": symbol_primary, "latest": c0, "prev": c1, "date": d0, "proxy": False}
    except Exception:
        c0, c1, d0 = _daily(symbol_fallback)
        return {"label": "KOSPI(대체: EWY)", "symbol": symbol_fallback, "latest": c0, "prev": c1, "date": d0, "proxy": True}


# -----------------------------
# 뉴스(RSS) - 한국어만
# -----------------------------
HANGUL_RE = re.compile(r"[가-힣]")


def google_news_rss(query: str):
    # 한국 로케일 고정 → 한자/중국 매체 섞이는 것 최대한 줄임
    # (그래도 섞일 수 있어 한글 필터 추가)
    q = requests.utils.quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"


def fetch_news_headlines(query: str, limit=3):
    url = google_news_rss(query)
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[: 20]:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        if not title or not link:
            continue
        # 한글 제목만 통과(한자/중문 제거 효과 큼)
        if not HANGUL_RE.search(title):
            continue
        items.append((title, link))
        if len(items) >= limit:
            break
    return items


def infer_drivers_from_headlines(headlines):
    text = " ".join([t for t, _ in headlines])
    mapping = [
        (r"금리|연준|Fed|파월|FOMC|긴축|인하|인상", "금리/연준 관련 이슈"),
        (r"CPI|물가|인플레이션", "물가 지표"),
        (r"고용|실업|비농업|NFP", "고용 지표"),
        (r"중동|이스라엘|이란|우크라|전쟁|지정학", "지정학 리스크"),
        (r"OPEC|감산|증산", "OPEC+ 공급 이슈"),
        (r"재고|원유재고|EIA", "원유 재고/수요 지표"),
        (r"외국인|기관|순매수|순매도|수급", "수급(매수/매도)"),
        (r"반도체|수출|무역|경상수지", "한국 경기/수출 관련"),
    ]
    hits = []
    for pat, label in mapping:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(label)
    return hits[:2]


# -----------------------------
# 리포트 HTML
# -----------------------------
def fmt_num(x, digits=2):
    try:
        return f"{x:,.{digits}f}"
    except Exception:
        return "-"


def calc_change(latest, prev):
    if latest is None or prev is None:
        return None, None
    diff = latest - prev
    pct = (diff / prev) * 100 if prev != 0 else None
    return diff, pct


def build_post_html(today, report, failures, news_blocks, test_badge=None):
    # 간단한 카드형 디자인 (테마에 크게 의존하지 않게 인라인 CSS)
    css = """
    <style>
      .dr-wrap{max-width:900px;margin:0 auto}
      .dr-badge{display:inline-block;padding:4px 10px;border-radius:999px;background:#eef2ff;color:#3730a3;font-size:12px;font-weight:700}
      .dr-alert{border:1px solid #fecaca;background:#fff1f2;padding:14px;border-radius:12px;margin:16px 0}
      .dr-alert h4{margin:0 0 8px 0;font-size:15px}
      .dr-alert ul{margin:0;padding-left:18px}
      .dr-card{border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:16px 0}
      .dr-table{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border-radius:12px;border:1px solid #e5e7eb}
      .dr-table th{background:#0f172a;color:#fff;text-align:left;padding:12px;font-weight:700}
      .dr-table td{padding:12px;border-top:1px solid #e5e7eb}
      .dr-table tr td+td,.dr-table tr th+th{border-left:1px solid rgba(255,255,255,.12)}
      .dr-kicker{color:#6b7280;font-size:13px;margin-top:6px}
      .dr-h2{font-size:22px;margin:18px 0 10px 0}
      .dr-h3{font-size:18px;margin:10px 0}
      .dr-list{margin:0;padding-left:18px}
      .dr-list li{margin:6px 0}
      .dr-muted{color:#6b7280}
    </style>
    """

    title_badge = f' <span class="dr-badge">{test_badge}</span>' if test_badge else ""

    def row(label, latest, prev, date_label):
        diff, pct = calc_change(latest, prev)
        if latest is None:
            latest_s, prev_s, diff_s, pct_s, date_s = "수집 실패", "-", "-", "-", "-"
        else:
            latest_s = fmt_num(latest, 2)
            prev_s = fmt_num(prev, 2)
            diff_s = f"{diff:+,.2f}" if diff is not None else "-"
            pct_s = f"{pct:+.2f}%" if pct is not None else "-"
            date_s = date_label or "-"
        return f"""
        <tr>
          <td><b>{label}</b></td>
          <td>{latest_s}</td>
          <td>{diff_s}</td>
          <td>{pct_s}</td>
          <td>{date_s}</td>
        </tr>
        """

    # 테이블
    usd = report.get("usdkrw")
    brent = report.get("brent")
    kospi = report.get("kospi")

    table_html = f"""
    <table class="dr-table">
      <thead>
        <tr>
          <th style="width:26%">지표</th>
          <th style="width:22%">현재</th>
          <th style="width:18%">전일대비</th>
          <th style="width:18%">변동률</th>
          <th style="width:16%">기준일(데이터)</th>
        </tr>
      </thead>
      <tbody>
        {row("USD/KRW", usd.get("latest"), usd.get("prev"), usd.get("date"))}
        {row("Brent Oil", brent.get("latest"), brent.get("prev"), brent.get("date"))}
        {row(kospi.get("label","KOSPI"), kospi.get("latest"), kospi.get("prev"), kospi.get("date"))}
      </tbody>
    </table>
    """

    # 원인 코멘트(과장 없이: “보통 영향을 주는 요인” + “관련 헤드라인”)
    why_blocks = []
    for key, title in [("usdkrw", "USD/KRW 변동 원인(참고)"),
                       ("brent", "Brent 유가 변동 원인(참고)"),
                       ("kospi", "KOSPI 변동 원인(참고)")]:
        headlines = news_blocks.get(key, [])
        drivers = infer_drivers_from_headlines(headlines) if headlines else []
        driver_line = ""
        if drivers:
            driver_line = f"<p class='dr-muted'>헤드라인에서 <b>{', '.join(drivers)}</b> 관련 키워드가 많이 보여요. 이런 이슈들이 보통 가격/지수 변동에 영향을 줍니다.</p>"
        items = "".join([f"<li><a href='{link}' target='_blank' rel='noopener'>{t}</a></li>" for t, link in headlines]) or "<li class='dr-muted'>관련 뉴스 수집 실패(또는 결과 없음)</li>"
        why_blocks.append(f"""
        <div class="dr-card">
          <div class="dr-h3">{title}</div>
          {driver_line}
          <ul class="dr-list">{items}</ul>
        </div>
        """)

    # 실패 경고 박스
    alert_html = ""
    if failures:
        li = "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in failures.items()])
        alert_html = f"""
        <div class="dr-alert">
          <h4>⚠️ 일부 데이터 수집 실패</h4>
          <ul>{li}</ul>
          <div class="dr-kicker">bot_config.json/Secrets 값과 데이터 제공처 응답 상태를 확인해주세요.</div>
        </div>
        """

    html = f"""
    {css}
    <div class="dr-wrap">
      <h1>오늘의 지표 리포트 ({today}){title_badge}</h1>
      {alert_html}

      <h2 class="dr-h2">핵심 요약</h2>
      <ul class="dr-list">
        <li>원/달러(USD/KRW), 브렌트유, 코스피의 최근 거래일 기준 변동을 한 번에 정리했습니다.</li>
        <li>변동 원인 참고용으로 관련 뉴스 헤드라인을 함께 첨부했습니다.</li>
      </ul>

      <h2 class="dr-h2">주요 지표</h2>
      {table_html}

      <h2 class="dr-h2">왜 움직였나? (원인 참고)</h2>
      {''.join(why_blocks)}

      <h2 class="dr-h2">내일 체크포인트</h2>
      <ul class="dr-list">
        <li>주요 지표 발표 일정(미국 CPI/고용, 연준 발언 등)</li>
        <li>유가: OPEC/재고/지정학 이슈 헤드라인</li>
        <li>환율: 위험자산 선호/달러 강세 여부</li>
      </ul>

      <p class="dr-kicker">※ 본 글은 자동 수집 기반 요약이며, 정확한 해석은 원문/공식 발표를 확인하세요.</p>
    </div>
    """
    return html


# -----------------------------
# 메인
# -----------------------------
def main():
    cfg = load_config()

    # 필수값
    cfg["wp_base_url"] = must_get(cfg, "wp_base_url")
    cfg["wp_user"] = must_get(cfg, "wp_user")
    cfg["wp_app_pass"] = must_get(cfg, "wp_app_pass")
    cfg["alphavantage_api_key"] = must_get(cfg, "alphavantage_api_key")

    wp_status = (cfg.get("wp_status") or "publish").strip()
    test_mode = str(cfg.get("test_mode") or "").strip() == "1"

    today = now_kst_date_str()

    run_id = str(int(time.time() * 1000))
    test_badge = f"TEST {run_id}" if test_mode else None

    base_title = f"오늘의 지표 리포트 ({today})"
    title = f"{base_title} - {run_id}" if test_mode else base_title

    base_slug = f"daily-indicator-report-{today}"
    slug = f"{base_slug}-{run_id}" if test_mode else base_slug

    # 중복 방지(운영 모드에서만)
    if not test_mode:
        if wp_post_exists(cfg, slug):
            msg = f"✅ 이미 오늘 글이 있어요 ({today})\n중복 발행 안 함"
            print(msg)
            kakao_send_to_me(cfg, msg)
            return

    failures = {}
    report = {}

    # 데이터 수집
    api_key = cfg["alphavantage_api_key"]
    try:
        u0, u1, ud = fetch_usdkrw(api_key)
        report["usdkrw"] = {"latest": u0, "prev": u1, "date": ud}
    except Exception as e:
        failures["USD/KRW"] = f"{type(e).__name__}: {e}"
        report["usdkrw"] = {"latest": None, "prev": None, "date": None}

    try:
        b0, b1, bd = fetch_brent(api_key)
        report["brent"] = {"latest": b0, "prev": b1, "date": bd}
    except Exception as e:
        failures["Brent Oil"] = f"{type(e).__name__}: {e}"
        report["brent"] = {"latest": None, "prev": None, "date": None}

    try:
        kospi = fetch_kospi(api_key)
        report["kospi"] = kospi
    except Exception as e:
        failures["KOSPI"] = f"{type(e).__name__}: {e}"
        report["kospi"] = {"label": "KOSPI", "latest": None, "prev": None, "date": None}

    # 뉴스 수집(지표 방향에 따라 쿼리)
    def direction_query(name, latest, prev, up_q, down_q):
        if latest is None or prev is None:
            return name
        return up_q if latest >= prev else down_q

    usd_q = direction_query("원달러 환율", report["usdkrw"]["latest"], report["usdkrw"]["prev"],
                            "원달러 환율 상승 원인", "원달러 환율 하락 원인")
    brent_q = direction_query("브렌트유", report["brent"]["latest"], report["brent"]["prev"],
                              "브렌트유 상승 원인", "브렌트유 하락 원인")
    kospi_label = report["kospi"].get("label", "코스피")
    kospi_q = direction_query("코스피", report["kospi"].get("latest"), report["kospi"].get("prev"),
                              "코스피 상승 원인", "코스피 하락 원인")

    news_blocks = {"usdkrw": [], "brent": [], "kospi": []}
    try:
        news_blocks["usdkrw"] = fetch_news_headlines(usd_q, limit=3)
    except Exception:
        news_blocks["usdkrw"] = []
    try:
        news_blocks["brent"] = fetch_news_headlines(brent_q, limit=3)
    except Exception:
        news_blocks["brent"] = []
    try:
        news_blocks["kospi"] = fetch_news_headlines(kospi_q, limit=3)
    except Exception:
        news_blocks["kospi"] = []

    html = build_post_html(today, report, failures, news_blocks, test_badge=test_badge)

    # WP 발행
    post = wp_create_post(cfg, title, slug, html, status=wp_status)
    link = post.get("link", cfg["wp_base_url"])

    ok_msg = f"✅ 글 작성 성공!\n날짜: {today}\n상태: {wp_status}\n링크: {link}\n(설정 로드: {cfg.get('_config_path')})"
    print(ok_msg)
    kakao_send_to_me(cfg, ok_msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        today = now_kst_date_str()
        msg = f"❌ 자동발행 실패 ({today})\n{type(e).__name__}: {e}"
        print(msg)
        # config 로드가 실패할 수도 있어 예외적으로 재시도
        try:
            cfg = load_config()
            kakao_send_to_me(cfg, msg)
        except Exception:
            pass
        raise
