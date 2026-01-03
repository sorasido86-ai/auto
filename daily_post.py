# -*- coding: utf-8 -*-
import os
import json
import base64
import csv
import io
import re
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup

# -----------------------------
# 기본 설정
# -----------------------------
WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"
UA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

def script_dir() -> Path:
    return Path(__file__).resolve().parent

def _read_json_if_exists(p: Path) -> dict:
    if p.exists() and p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_config() -> dict:
    """
    ✅ 우선순위
    1) 환경변수(GitHub Actions Secrets)
    2) daily_post.py와 같은 폴더의 bot_config.json
    """
    cfg = {}
    cfg_path = Path(os.getenv("BOT_CONFIG_PATH", str(script_dir() / "bot_config.json")))
    cfg.update(_read_json_if_exists(cfg_path))

    # env overlay
    env_map = {
        "wp_base_url": "WP_BASE_URL",
        "wp_user": "WP_USER",
        "wp_app_pass": "WP_APP_PASS",
        "wp_post_status": "WP_POST_STATUS",  # publish / draft
        "kakao_rest_key": "KAKAO_REST_KEY",
        "kakao_refresh_token": "KAKAO_REFRESH_TOKEN",
    }
    for k, ev in env_map.items():
        v = os.getenv(ev)
        if v:
            cfg[k] = v

    # defaults
    cfg.setdefault("wp_post_status", "publish")

    # validate
    required = ["wp_base_url", "wp_user", "wp_app_pass", "kakao_rest_key", "kakao_refresh_token"]
    missing = [k for k in required if not str(cfg.get(k, "")).strip()]
    if missing:
        raise ValueError(f"필수 설정 누락: {', '.join(missing)} (bot_config.json 또는 GitHub Secrets 확인)")

    return cfg

def now_date_str():
    return datetime.now().strftime("%Y-%m-%d")

def fmt_num(x, digits=2):
    if x is None:
        return "-"
    try:
        return f"{x:,.{digits}f}"
    except Exception:
        return str(x)

def parse_float(s: str):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    s = s.replace(",", "")
    s = re.sub(r"[^\d\.\-]", "", s)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


# -----------------------------
# Kakao: refresh_token -> access_token 갱신
# -----------------------------
def save_config(cfg, path: Path):
    # 로컬에서만 의미(깃헙 액션은 파일 저장 X)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

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

    if tokens.get("refresh_token"):
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        cfg_path = Path(os.getenv("BOT_CONFIG_PATH", str(script_dir() / "bot_config.json")))
        save_config(cfg, cfg_path)

    return tokens["access_token"]

def kakao_send_to_me(cfg, text):
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
# WordPress: 글 생성/수정
# -----------------------------
def wp_posts_api(cfg):
    return cfg["wp_base_url"].rstrip("/") + WP_POSTS_API_SUFFIX

def wp_auth_headers(cfg):
    user = cfg["wp_user"].strip()
    app_pass = cfg["wp_app_pass"].replace(" ", "").strip()  # 공백 제거
    token = base64.b64encode(f"{user}:{app_pass}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}

def wp_get_post_by_slug(cfg, slug):
    # ✅ status=any 로 draft/publish 모두 검색 (권한 있는 경우)
    params = {"slug": slug, "status": "any", "per_page": 1}
    r = requests.get(wp_posts_api(cfg), params=params, headers=wp_auth_headers(cfg), timeout=30)
    r.raise_for_status()
    arr = r.json()
    if isinstance(arr, list) and arr:
        return arr[0]
    return None

def wp_create_post(cfg, title, slug, content_html, status="publish"):
    payload = {"title": title, "slug": slug, "content": content_html, "status": status}
    r = requests.post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def wp_update_post(cfg, post_id, title, content_html, status="publish"):
    url = wp_posts_api(cfg).rstrip("/") + f"/{post_id}"
    payload = {"title": title, "content": content_html, "status": status}
    r = requests.post(
        url,
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def wp_upsert_post(cfg, title, slug, content_html, status="publish"):
    existing = wp_get_post_by_slug(cfg, slug)
    if existing and existing.get("id"):
        return wp_update_post(cfg, existing["id"], title, content_html, status=status), "updated"
    return wp_create_post(cfg, title, slug, content_html, status=status), "created"


# -----------------------------
# 데이터 수집 (Stooq -> Naver fallback)
# -----------------------------
def fetch_stooq_last_two_closes(symbol: str):
    """
    Stooq CSV 다운로드:
    https://stooq.com/q/d/l/?s={symbol}&d1=YYYYMMDD&d2=YYYYMMDD&i=d
    """
    end = datetime.utcnow()
    start = end - timedelta(days=45)
    params = {
        "s": symbol.lower(),
        "d1": start.strftime("%Y%m%d"),
        "d2": end.strftime("%Y%m%d"),
        "i": "d",
    }
    r = requests.get("https://stooq.com/q/d/l/", params=params, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    text = (r.text or "").strip()
    if not text.startswith("Date,"):
        raise ValueError(f"Stooq CSV가 아닌 응답(HTML/빈값 가능): {symbol}")

    f = io.StringIO(text)
    reader = csv.DictReader(f)
    rows = []
    for row in reader:
        d = row.get("Date")
        c = parse_float(row.get("Close"))
        if d and c is not None:
            rows.append((d, c))

    if len(rows) < 2:
        raise ValueError(f"Stooq 데이터가 부족합니다: {symbol}")

    # 최신 2개
    rows.sort(key=lambda x: x[0], reverse=True)
    (d1, c1), (d2, c2) = rows[0], rows[1]
    return c1, c2, d1

def _naver_get_soup(url):
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def fetch_naver_exchange_usdkrw():
    # 네이버 환율 일별시세
    url = "https://finance.naver.com/marketindex/exchangeDailyQuote.naver?marketindexCd=FX_USDKRW&page=1"
    soup = _naver_get_soup(url)
    trs = soup.select("table tbody tr")
    rows = []
    for tr in trs:
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        date = tds[0].get_text(strip=True).replace(".", "-")
        close = parse_float(tds[1].get_text(strip=True))
        if date and close is not None:
            rows.append((date, close))
    if len(rows) < 2:
        raise ValueError("네이버 USD/KRW 데이터 부족")
    return rows[0][1], rows[1][1], rows[0][0]

def fetch_naver_brent():
    # 네이버 브렌트유 일별시세
    url = "https://finance.naver.com/marketindex/worldDailyQuote.naver?marketindexCd=OIL_BRT&fdtc=2&page=1"
    soup = _naver_get_soup(url)
    trs = soup.select("table tbody tr")
    rows = []
    for tr in trs:
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        date = tds[0].get_text(strip=True).replace(".", "-")
        close = parse_float(tds[1].get_text(strip=True))
        if date and close is not None:
            rows.append((date, close))
    if len(rows) < 2:
        raise ValueError("네이버 Brent 데이터 부족")
    return rows[0][1], rows[1][1], rows[0][0]

def fetch_naver_kospi():
    # 네이버 코스피 지수 일별시세
    url = "https://finance.naver.com/sise/sise_index_day.nhn?code=KOSPI&page=1"
    soup = _naver_get_soup(url)
    trs = soup.select("table.type_1 tbody tr")
    rows = []
    for tr in trs:
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        date = tds[0].get_text(strip=True).replace(".", "-")
        close = parse_float(tds[1].get_text(strip=True))
        if date and close is not None:
            rows.append((date, close))
    rows = [r for r in rows if re.match(r"\d{4}-\d{2}-\d{2}", r[0])]
    if len(rows) < 2:
        raise ValueError("네이버 KOSPI 데이터 부족")
    return rows[0][1], rows[1][1], rows[0][0]

def get_two_closes_with_fallback(kind: str):
    """
    kind: usdkrw | brent | kospi
    """
    if kind == "usdkrw":
        # 1) Stooq
        try:
            c1, c2, d = fetch_stooq_last_two_closes("usdkrw")
            return c1, c2, d, "Stooq"
        except Exception:
            # 2) Naver
            c1, c2, d = fetch_naver_exchange_usdkrw()
            return c1, c2, d, "Naver"
    if kind == "brent":
        # 1) Stooq (Brent: CB.F)
        try:
            c1, c2, d = fetch_stooq_last_two_closes("cb.f")
            return c1, c2, d, "Stooq"
        except Exception:
            c1, c2, d = fetch_naver_brent()
            return c1, c2, d, "Naver"
    if kind == "kospi":
        # 1) Stooq (KOSPI: ^KOSPI)
        try:
            c1, c2, d = fetch_stooq_last_two_closes("^kospi")
            return c1, c2, d, "Stooq"
        except Exception:
            c1, c2, d = fetch_naver_kospi()
            return c1, c2, d, "Naver"
    raise ValueError("지원하지 않는 kind")

def calc_change(latest, prev):
    if latest is None or prev is None:
        return None, None
    diff = latest - prev
    pct = (diff / prev * 100.0) if prev != 0 else None
    return diff, pct


# -----------------------------
# 뉴스(원인 참고): Google News RSS (ko-KR 고정)
# -----------------------------
def fetch_google_news_rss(query: str, max_items=3):
    """
    Google News RSS:
    https://news.google.com/rss/search?q=...&hl=ko&gl=KR&ceid=KR:ko
    """
    base = "https://news.google.com/rss/search"
    params = {
        "q": query,
        "hl": "ko",
        "gl": "KR",
        "ceid": "KR:ko",
    }
    r = requests.get(base, params=params, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    root = ET.fromstring(r.text)
    items = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        desc = item.findtext("description") or ""
        title = title.strip()
        link = link.strip()
        desc = re.sub(r"<.*?>", "", desc).strip()
        if title and link:
            items.append({"title": title, "link": link, "desc": desc})

        if len(items) >= max_items:
            break

    return items

def generic_reason_text(kind: str, diff: float):
    """
    ✅ '사실 단정'이 아니라 '일반적으로 자주 거론되는 요인' 형태로 안전하게 작성
    """
    up = (diff is not None and diff > 0)
    if kind == "usdkrw":
        if up:
            return ["달러 강세(미 금리/지표/연준 발언) 영향",
                    "위험회피 심리(글로벌 증시/지정학 이슈)로 달러 수요 증가",
                    "수급 요인(결제/수입대금/외국인 수급)"]
        return ["달러 약세(미 금리 기대 하락/지표 둔화) 영향",
                "위험선호 회복으로 신흥국 통화 강세",
                "수급 요인(외국인 순매수/달러 공급)"]
    if kind == "brent":
        if up:
            return ["공급 차질/지정학 리스크(중동 등)로 공급 우려",
                    "OPEC+ 감산 기대 또는 감산 연장",
                    "미국 재고 감소/수요 기대 개선"]
        return ["수요 둔화 우려(경기/중국 지표 등)",
                "미국 원유 재고 증가/공급 확대 전망",
                "달러 강세 시 원유 가격에 하방 압력"]
    if kind == "kospi":
        if up:
            return ["대형주(반도체 등) 실적/가이던스 기대",
                    "외국인 수급 개선",
                    "미 금리 부담 완화/위험선호 회복"]
        return ["미 금리 부담/달러 강세로 위험자산 선호 약화",
                "외국인 수급(순매도) 영향",
                "주요 업종(반도체/2차전지 등) 변동성 확대"]
    return []


# -----------------------------
# 글(HTML) 디자인
# -----------------------------
def build_post_html(report):
    today = report["date"]
    updated_at = report["generated_at"]

    errors = report.get("errors", [])
    usd = report.get("usdkrw")
    brent = report.get("brent")
    kospi = report.get("kospi")

    def badge(diff):
        if diff is None:
            return "보합/미집계"
        return "상승 ▲" if diff > 0 else ("하락 ▼" if diff < 0 else "보합 •")

    def row(name, obj, digits=2):
        if not obj or obj.get("latest") is None:
            return f"""
            <tr>
              <td>{name}</td><td>-</td><td>-</td><td>-</td><td>-</td>
            </tr>
            """
        latest = obj["latest"]
        prev = obj["prev"]
        diff = obj["diff"]
        pct = obj["pct"]
        base_date = obj.get("base_date", "-")
        return f"""
        <tr>
          <td><b>{name}</b></td>
          <td>{fmt_num(latest, digits)}</td>
          <td>{fmt_num(diff, digits)}</td>
          <td>{fmt_num(pct, 2)}%</td>
          <td>{base_date}</td>
        </tr>
        """

    def news_block(title, reasons, news_items):
        li_reasons = "".join([f"<li>{r}</li>" for r in reasons]) if reasons else "<li>원인 요약 생성 실패</li>"
        if news_items:
            li_news = "".join([
                f'<li><a href="{n["link"]}" target="_blank" rel="noopener noreferrer">{n["title"]}</a>'
                + (f'<br><span style="color:#666;font-size:13px;">{n["desc"]}</span>' if n.get("desc") else "")
                + "</li>"
                for n in news_items
            ])
        else:
            li_news = "<li>관련 뉴스 수집 실패(또는 결과 없음)</li>"

        return f"""
        <div style="border:1px solid #e5e7eb;border-radius:14px;padding:16px;margin:12px 0;background:#fff;">
          <h3 style="margin:0 0 8px 0;">{title}</h3>
          <div style="margin:10px 0 6px 0;font-weight:700;">일반적으로 거론되는 요인</div>
          <ul style="margin:6px 0 14px 18px;">{li_reasons}</ul>
          <div style="margin:10px 0 6px 0;font-weight:700;">관련 헤드라인(참고)</div>
          <ul style="margin:6px 0 0 18px;">{li_news}</ul>
        </div>
        """

    err_html = ""
    if errors:
        err_list = "".join([f"<li>{e}</li>" for e in errors])
        err_html = f"""
        <div style="border:1px solid #fecaca;background:#fff1f2;padding:14px;border-radius:14px;margin:14px 0;">
          <b>⚠ 일부 데이터 수집 실패</b>
          <ul style="margin:8px 0 0 18px;">{err_list}</ul>
        </div>
        """

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans KR', Arial, sans-serif; line-height:1.55;">
      <div style="padding:18px 18px 10px;border-radius:16px;background:linear-gradient(135deg,#111827,#374151);color:white;">
        <div style="font-size:14px;opacity:.9;">Daily Indicator Report</div>
        <div style="font-size:26px;font-weight:800;margin-top:6px;">오늘의 지표 리포트 ({today})</div>
        <div style="font-size:13px;opacity:.85;margin-top:6px;">업데이트: {updated_at}</div>
      </div>

      {err_html}

      <h2 style="margin:18px 0 10px;">핵심 요약</h2>
      <ul>
        <li>원/달러(USD/KRW), 브렌트유(Brent), 코스피(KOSPI)의 전일 대비 변화를 한 번에 정리했습니다.</li>
        <li>변동 원인은 <b>일반적으로 거론되는 요인</b> + <b>관련 뉴스 헤드라인</b>으로 함께 제공합니다.</li>
      </ul>

      <h2 style="margin:18px 0 10px;">주요 지표</h2>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;border:1px solid #e5e7eb;">
          <thead>
            <tr style="background:#111827;color:white;">
              <th style="padding:10px;border:1px solid #e5e7eb;text-align:left;">지표</th>
              <th style="padding:10px;border:1px solid #e5e7eb;text-align:left;">현재</th>
              <th style="padding:10px;border:1px solid #e5e7eb;text-align:left;">전일대비</th>
              <th style="padding:10px;border:1px solid #e5e7eb;text-align:left;">변동률</th>
              <th style="padding:10px;border:1px solid #e5e7eb;text-align:left;">기준일</th>
            </tr>
          </thead>
          <tbody>
            {row("USD/KRW", usd, digits=2)}
            {row("Brent Oil", brent, digits=2)}
            {row("KOSPI", kospi, digits=2)}
          </tbody>
        </table>
      </div>

      <h2 style="margin:22px 0 10px;">왜 움직였나? (원인 참고)</h2>
      {news_block("USD/KRW 변동 원인(뉴스)", report.get("usdkrw_reasons", []), report.get("usdkrw_news", []))}
      {news_block("Brent 유가 변동 원인(뉴스)", report.get("brent_reasons", []), report.get("brent_news", []))}
      {news_block("KOSPI 변동 원인(뉴스)", report.get("kospi_reasons", []), report.get("kospi_news", []))}

      <h2 style="margin:22px 0 10px;">내일 체크포인트</h2>
      <ul>
        <li>주요 지표 발표 일정(미국 CPI/고용, 연준 발언 등)</li>
        <li>유가: OPEC+ / 재고 / 지정학 이슈 헤드라인</li>
        <li>환율: 달러 강세/위험자산 선호 변화 여부</li>
      </ul>

      <h2 style="margin:22px 0 10px;">출처</h2>
      <ul>
        <li>지표: Stooq 또는 Naver Finance(자동 폴백)</li>
        <li>뉴스: Google News RSS (ko-KR)</li>
      </ul>

      <div style="margin-top:18px;color:#6b7280;font-size:12px;">
        ※ 원인 요약은 ‘사실 단정’이 아니라 ‘일반적으로 자주 언급되는 요인’ 형태로 제공됩니다.
      </div>
    </div>
    """
    return html


# -----------------------------
# 리포트 생성
# -----------------------------
def fetch_indicators_real():
    errors = []

    def pack(kind, label):
        try:
            latest, prev, base_date, source = get_two_closes_with_fallback(kind)
            diff, pct = calc_change(latest, prev)
            return {
                "latest": latest, "prev": prev, "diff": diff, "pct": pct,
                "base_date": base_date, "source": source
            }
        except Exception as e:
            errors.append(f"{label} 수집 실패: {e}")
            return None

    usd = pack("usdkrw", "USD/KRW")
    brent = pack("brent", "Brent")
    kospi = pack("kospi", "KOSPI")

    # 뉴스(한국어 고정)
    usd_news, brent_news, kospi_news = [], [], []
    usd_reasons, brent_reasons, kospi_reasons = [], [], []

    try:
        if usd and usd.get("diff") is not None:
            usd_reasons = generic_reason_text("usdkrw", usd["diff"])
        usd_news = fetch_google_news_rss("원달러 환율 변동 원인", max_items=3)
    except Exception:
        pass

    try:
        if brent and brent.get("diff") is not None:
            brent_reasons = generic_reason_text("brent", brent["diff"])
        brent_news = fetch_google_news_rss("브렌트유 하락 원인 OR 브렌트유 상승 원인", max_items=3)
    except Exception:
        pass

    try:
        if kospi and kospi.get("diff") is not None:
            kospi_reasons = generic_reason_text("kospi", kospi["diff"])
        kospi_news = fetch_google_news_rss("코스피 하락 원인 OR 코스피 상승 원인", max_items=3)
    except Exception:
        pass

    # 전부 실패면 글을 비우지 말고 종료(카톡으로 안내)
    if (usd is None) and (brent is None) and (kospi is None):
        raise ValueError("유효한 값이 없습니다: usdkrw/brent/kospi (데이터 소스 응답 확인 필요)")

    return {
        "date": now_date_str(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "usdkrw": usd,
        "brent": brent,
        "kospi": kospi,
        "usdkrw_reasons": usd_reasons,
        "brent_reasons": brent_reasons,
        "kospi_reasons": kospi_reasons,
        "usdkrw_news": usd_news,
        "brent_news": brent_news,
        "kospi_news": kospi_news,
        "errors": errors,
    }


# -----------------------------
# main
# -----------------------------
def main():
    cfg = load_config()

    today = now_date_str()
    title = f"오늘의 지표 리포트 ({today})"
    slug = f"daily-indicator-report-{today}"

    try:
        report = fetch_indicators_real()
        html = build_post_html(report)

        post, mode = wp_upsert_post(
            cfg,
            title=title,
            slug=slug,
            content_html=html,
            status=cfg.get("wp_post_status", "publish"),
        )

        link = post.get("link", cfg["wp_base_url"])
        kakao_send_to_me(
            cfg,
            f"✅ 글 {mode} 성공!\n날짜: {today}\n상태: {cfg.get('wp_post_status','publish')}\n링크: {link}"
        )

    except Exception as e:
        msg = f"❌ 자동발행 실패 ({today})\n{type(e).__name__}: {e}\n\nbot_config.json/Secrets 값과 데이터 수집 상태를 확인해주세요."
        print(msg)
        try:
            kakao_send_to_me(cfg, msg)
        except Exception as e2:
            print("카톡 알림까지 실패:", type(e2).__name__, e2)


if __name__ == "__main__":
    main()
