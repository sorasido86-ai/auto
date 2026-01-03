# -*- coding: utf-8 -*-
import os
import json
import base64
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from html import escape
import xml.etree.ElementTree as ET

CONFIG_PATH = "bot_config.json"
WP_POSTS_API_SUFFIX = "/wp-json/wp/v2/posts"

# -----------------------------
# 공통
# -----------------------------
def now_seoul_date_str():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")

def _ua_headers():
    return {
        "User-Agent": "Mozilla/5.0 (compatible; daily-post-bot/1.0; +https://rainsow.com)",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }

def _parse_number(text):
    if not text:
        return None
    t = text.replace(",", "").strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", t)
    return float(m.group()) if m else None

def _fmt_num(x, digits=2):
    if x is None:
        return "-"
    try:
        return f"{x:,.{digits}f}"
    except Exception:
        return str(x)

def _sign_direction(change):
    if change is None:
        return "unknown"
    if change > 0:
        return "up"
    if change < 0:
        return "down"
    return "flat"

# -----------------------------
# 설정 로드 (GitHub Secrets 우선)
# -----------------------------
def load_config():
    # GitHub Actions(Secrets) 우선
    env = os.environ
    if env.get("WP_BASE_URL") and env.get("WP_USER") and env.get("WP_APP_PASS"):
        return {
            "wp_base_url": env["WP_BASE_URL"].strip(),
            "wp_user": env["WP_USER"].strip(),
            "wp_app_pass": env["WP_APP_PASS"].strip(),
            "wp_status": env.get("WP_STATUS", "publish").strip(),  # publish 권장
            # 카톡 알림(선택)
            "kakao_rest_key": env.get("KAKAO_REST_KEY", "").strip(),
            "kakao_refresh_token": env.get("KAKAO_REFRESH_TOKEN", "").strip(),
        }

    # 로컬(PC) 실행용 bot_config.json
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("wp_status", "publish")
    return cfg

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# -----------------------------
# Kakao: (선택) 나에게 보내기
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
    r = requests.post(url, data=data, timeout=30)
    r.raise_for_status()
    tokens = r.json()

    if "refresh_token" in tokens and tokens["refresh_token"]:
        cfg["kakao_refresh_token"] = tokens["refresh_token"]
        save_config(cfg)

    return tokens.get("access_token")

def kakao_send_to_me(cfg, text):
    access_token = refresh_access_token(cfg)
    if not access_token:
        return False
    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {"Authorization": f"Bearer {access_token}"}

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
        params={"slug": slug, "per_page": 1, "status": "any"},
        headers=wp_auth_headers(cfg),
        timeout=30
    )
    r.raise_for_status()
    return len(r.json()) > 0

def wp_create_post(cfg, title, slug, content_html, status="publish"):
    payload = {"title": title, "slug": slug, "content": content_html, "status": status}
    r = requests.post(
        wp_posts_api(cfg),
        headers={**wp_auth_headers(cfg), "Content-Type": "application/json"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=30
    )
    r.raise_for_status()
    return r.json()

# -----------------------------
# 지표: 네이버에서 가져오기 (무료)
# -----------------------------
def _fetch_text(url):
    r = requests.get(url, headers=_ua_headers(), timeout=30)
    r.raise_for_status()
    # 네이버가 간혹 인코딩 추정이 필요할 때가 있어 보정
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text

def fetch_usdkrw_naver():
    # 환율 상세
    url = "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW"
    html = _fetch_text(url)

    # 현재가
    m_val = re.search(r'class="value"\s*>\s*([0-9\.,]+)\s*<', html)
    value = _parse_number(m_val.group(1)) if m_val else None

    # 전일대비(상승/하락)
    # no_exday 안에 숫자가 2개(변동폭, 변동률) 들어가는 경우가 많아서 두 개를 잡아봄
    exday_block = re.search(r'no_exday.*?</p>', html, re.DOTALL)
    change = None
    change_pct = None
    direction = "unknown"
    if exday_block:
        nums = re.findall(r'[-+]?[0-9\.,]+', exday_block.group(0))
        # 보통 [변동폭, 변동률] 순서
        if len(nums) >= 1:
            change = _parse_number(nums[0])
        if len(nums) >= 2:
            change_pct = _parse_number(nums[1])
        # up/down 힌트
        if "ico_up" in exday_block.group(0) or "up" in exday_block.group(0):
            direction = "up"
            if change is not None:
                change = abs(change)
        elif "ico_down" in exday_block.group(0) or "down" in exday_block.group(0):
            direction = "down"
            if change is not None:
                change = -abs(change)
        else:
            direction = _sign_direction(change)

    return {"value": value, "change": change, "change_pct": change_pct, "direction": direction}

def fetch_brent_naver():
    url = "https://finance.naver.com/marketindex/worldOilDetail.naver?marketindexCd=OIL_BRT"
    html = _fetch_text(url)

    m_val = re.search(r'class="value"\s*>\s*([0-9\.,]+)\s*<', html)
    value = _parse_number(m_val.group(1)) if m_val else None

    exday_block = re.search(r'no_exday.*?</p>', html, re.DOTALL)
    change = None
    change_pct = None
    direction = "unknown"
    if exday_block:
        nums = re.findall(r'[-+]?[0-9\.,]+', exday_block.group(0))
        if len(nums) >= 1:
            change = _parse_number(nums[0])
        if len(nums) >= 2:
            change_pct = _parse_number(nums[1])
        if "ico_up" in exday_block.group(0):
            direction = "up"
            if change is not None:
                change = abs(change)
        elif "ico_down" in exday_block.group(0):
            direction = "down"
            if change is not None:
                change = -abs(change)
        else:
            direction = _sign_direction(change)

    return {"value": value, "change": change, "change_pct": change_pct, "direction": direction}

def fetch_kospi_naver():
    url = "https://finance.naver.com/sise/sise_index.naver?code=KOSPI"
    html = _fetch_text(url)

    # 현재지수
    m_val = re.search(r'id="now_value"\s*>\s*([0-9\.,]+)\s*<', html)
    if not m_val:
        m_val = re.search(r'class="num"\s*>\s*([0-9\.,]+)\s*<', html)
    value = _parse_number(m_val.group(1)) if m_val else None

    # 변동폭/변동률 (change_value_and_rate 영역)
    block = re.search(r'id="change_value_and_rate".*?</span>', html, re.DOTALL)
    change = None
    change_pct = None
    direction = "unknown"
    if block:
        nums = re.findall(r'[-+]?[0-9\.,]+', block.group(0))
        if len(nums) >= 1:
            change = _parse_number(nums[0])
        if len(nums) >= 2:
            change_pct = _parse_number(nums[1])
        if "up" in block.group(0) or "상승" in block.group(0):
            direction = "up"
            if change is not None:
                change = abs(change)
        elif "down" in block.group(0) or "하락" in block.group(0):
            direction = "down"
            if change is not None:
                change = -abs(change)
        else:
            direction = _sign_direction(change)
    else:
        direction = _sign_direction(change)

    return {"value": value, "change": change, "change_pct": change_pct, "direction": direction}

def fetch_indicators_real():
    usdkrw = fetch_usdkrw_naver()
    brent = fetch_brent_naver()
    kospi = fetch_kospi_naver()

    # 값이 하나도 못 오면 실패 처리
    if usdkrw["value"] is None:
        raise ValueError("유효한 값이 없습니다: usdkrw")

    return {"usdkrw": usdkrw, "brent": brent, "kospi": kospi}

# -----------------------------
# 뉴스(RSS): 무료 (Google News RSS)
# -----------------------------
def fetch_google_news_rss(query, max_items=5):
    # Google News RSS search
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    r = requests.get(url, params=params, headers=_ua_headers(), timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if title and link:
            items.append({"title": title, "link": link, "pubDate": pub})
        if len(items) >= max_items:
            break
    return items

def build_reason_lines(kind, direction, headlines):
    # “정답”이 아니라 “가능성 높은 원인” 형태로 안전하게 구성
    # + 헤드라인 키워드를 참고해서 문장을 조금 더 현실적으로 붙임
    titles = " ".join([h["title"] for h in headlines]).lower()

    def has(*ks):
        return any(k.lower() in titles for k in ks)

    lines = []

    if kind == "fx":  # USD/KRW
        if direction == "up":
            lines += [
                "달러 강세(미 금리·연준 발언·미 지표)나 위험회피 심리가 커지면 원/달러가 오르는 흐름이 자주 나옵니다.",
                "외국인 수급/주식 변동, 수입 결제 수요가 겹치면 단기적으로 환율을 밀어올릴 수 있어요.",
            ]
        elif direction == "down":
            lines += [
                "달러 약세(금리 기대 하락)나 위험선호 회복 시 원화가 강해지며 원/달러가 내려가는 경우가 많습니다.",
                "수출 네고(달러 매도)나 외국인 순매수가 동반되면 환율 하락 압력이 생길 수 있어요.",
            ]
        else:
            lines += ["환율은 금리 기대·위험선호·수급(외국인/수출입 결제) 영향이 복합적으로 섞여 변동합니다."]

        # 헤드라인 기반 보강
        if has("fed", "연준", "금리", "cpi", "pce"):
            lines.append("오늘 관련 기사에 ‘금리/연준/물가’ 키워드가 보여서, 달러 방향성이 환율에 영향을 줬을 가능성이 있어요.")
        if has("위험", "리스크", "전쟁", "지정학"):
            lines.append("지정학/리스크 이슈가 있으면 안전자산 선호로 달러가 강해지는 패턴이 나타날 수 있습니다.")

    if kind == "oil":  # Brent
        if direction == "down":
            lines += [
                "유가는 ‘수요 둔화(경기/중국 지표)’ 또는 ‘공급 증가(OPEC+, 증산/재고)’ 뉴스에 민감하게 반응해요.",
                "재고가 예상보다 늘거나, 경기 우려가 커지면 브렌트가 밀리는 경우가 흔합니다.",
            ]
        elif direction == "up":
            lines += [
                "중동 등 공급 차질 우려, OPEC+ 감산 기대가 커지면 브렌트가 오르기 쉬워요.",
                "재고 감소/수요 개선 신호가 나오면 유가 상승으로 연결될 수 있습니다.",
            ]
        else:
            lines += ["브렌트는 공급(OPEC+, 생산/재고)과 수요(경기/중국/항공) 변수가 동시에 작용합니다."]

        if has("opec", "감산", "증산"):
            lines.append("오늘 헤드라인에 OPEC/감산·증산 키워드가 있어 공급 기대가 가격에 반영됐을 수 있어요.")
        if has("재고", "inventory"):
            lines.append("원유 재고 관련 이슈가 있으면 단기 변동 폭이 커질 수 있습니다.")
        if has("중국", "경기", "침체"):
            lines.append("수요 쪽(중국/경기) 우려가 기사에 보이면 유가 하락 논리를 강화합니다.")

    if kind == "stock":  # KOSPI
        if direction == "up":
            lines += [
                "미 증시 흐름/금리 안정, 외국인 수급이 개선되면 코스피가 반등하는 흐름이 자주 나옵니다.",
                "대형주(반도체 등) 뉴스가 긍정적이면 지수 상승에 기여할 수 있어요.",
            ]
        elif direction == "down":
            lines += [
                "미 금리 상승, 달러 강세(환율 상승), 외국인 매도 압력이 겹치면 코스피가 약해질 수 있어요.",
                "대형주 실적/가이던스 불확실성 뉴스가 나오면 지수에 부담이 됩니다.",
            ]
        else:
            lines += ["지수는 금리·환율·미 증시·외국인 수급과 업종(대형주) 뉴스의 합으로 움직입니다."]

        if has("삼성", "하이닉스", "반도체"):
            lines.append("오늘 헤드라인에 반도체/대형주 키워드가 보여 지수에 영향이 있었을 가능성이 있어요.")
        if has("금리", "채권", "fed", "연준"):
            lines.append("금리 관련 뉴스는 할인율/수급 경로로 주식시장에 직접 영향을 줍니다.")

    # 너무 길어지지 않게 제한
    return lines[:4]

# -----------------------------
# 글 콘텐츠(HTML) - 디자인 강화
# -----------------------------
def build_post_content_html(today, data, news):
    usd = data["usdkrw"]
    oil = data.get("brent", {})
    kos = data.get("kospi", {})

    def badge(direction):
        return {
            "up": "📈 상승",
            "down": "📉 하락",
            "flat": "➖ 보합",
            "unknown": "❔"
        }.get(direction, "❔")

    # 한줄요약(간단 규칙)
    summary_parts = []
    if usd.get("direction") == "up":
        summary_parts.append("원/달러 상승")
    elif usd.get("direction") == "down":
        summary_parts.append("원/달러 하락")
    if oil.get("direction") == "down":
        summary_parts.append("브렌트 하락")
    elif oil.get("direction") == "up":
        summary_parts.append("브렌트 상승")
    if kos.get("direction") == "up":
        summary_parts.append("코스피 강세")
    elif kos.get("direction") == "down":
        summary_parts.append("코스피 약세")

    one_liner = " · ".join(summary_parts) if summary_parts else "주요 지표 변동 체크"

    # 원인(가능성) + 헤드라인
    fx_lines = build_reason_lines("fx", usd.get("direction", "unknown"), news["fx"])
    oil_lines = build_reason_lines("oil", oil.get("direction", "unknown"), news["oil"])
    st_lines = build_reason_lines("stock", kos.get("direction", "unknown"), news["stock"])

    def headlines_html(items):
        if not items:
            return "<li>관련 헤드라인을 불러오지 못했습니다.</li>"
        lis = []
        for it in items[:4]:
            t = escape(it["title"])
            l = escape(it["link"])
            lis.append(f'<li><a href="{l}" target="_blank" rel="noopener noreferrer">{t}</a></li>')
        return "\n".join(lis)

    # 표(스냅샷)
    def row(name, v, ch, pct, direction, unit=""):
        vtxt = _fmt_num(v, 2)
        chtxt = _fmt_num(ch, 2)
        pcttxt = _fmt_num(pct, 2)
        return f"""
        <tr>
          <td style="padding:10px;border-bottom:1px solid #eee;"><b>{escape(name)}</b></td>
          <td style="padding:10px;border-bottom:1px solid #eee;text-align:right;">{vtxt}{unit}</td>
          <td style="padding:10px;border-bottom:1px solid #eee;text-align:right;">{chtxt}</td>
          <td style="padding:10px;border-bottom:1px solid #eee;text-align:right;">{pcttxt}%</td>
          <td style="padding:10px;border-bottom:1px solid #eee;">{badge(direction)}</td>
        </tr>
        """

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;border:1px solid #eee;border-radius:12px;overflow:hidden;">
      <thead>
        <tr style="background:#fafafa;">
          <th style="padding:10px;text-align:left;border-bottom:1px solid #eee;">지표</th>
          <th style="padding:10px;text-align:right;border-bottom:1px solid #eee;">현재</th>
          <th style="padding:10px;text-align:right;border-bottom:1px solid #eee;">전일대비</th>
          <th style="padding:10px;text-align:right;border-bottom:1px solid #eee;">등락률</th>
          <th style="padding:10px;text-align:left;border-bottom:1px solid #eee;">상태</th>
        </tr>
      </thead>
      <tbody>
        {row("USD/KRW", usd.get("value"), usd.get("change"), usd.get("change_pct"), usd.get("direction"), "")}
        {row("Brent Oil", oil.get("value"), oil.get("change"), oil.get("change_pct"), oil.get("direction"), " $")}
        {row("KOSPI", kos.get("value"), kos.get("change"), kos.get("change_pct"), kos.get("direction"), "")}
      </tbody>
    </table>
    """

    def bullets(lines):
        return "\n".join([f"<li>{escape(x)}</li>" for x in (lines or [])])

    html = f"""
    <div style="padding:18px;border:1px solid #eee;border-radius:14px;background:#fff;">
      <div style="font-size:13px;color:#666;">오늘의 지표 리포트</div>
      <h2 style="margin:6px 0 10px 0;">{escape(today)} · {escape(one_liner)}</h2>
      <div style="color:#666;line-height:1.6;">
        ※ 아래 ‘원인’은 <b>확정</b>이 아니라, <b>관련 헤드라인 + 일반적인 시장 메커니즘</b>을 합쳐 만든 “가능성 높은 해석”입니다.
      </div>
    </div>

    <h3 style="margin-top:18px;">📌 주요 지표 스냅샷</h3>
    {table_html}

    <h3 style="margin-top:22px;">🧠 왜 움직였나(가능성 높은 원인)</h3>

    <div style="display:grid;grid-template-columns:1fr;gap:14px;">
      <div style="padding:14px;border:1px solid #eee;border-radius:14px;">
        <h4 style="margin:0 0 8px 0;">1) 원/달러 환율</h4>
        <ul style="margin:0 0 10px 18px;line-height:1.7;">{bullets(fx_lines)}</ul>
        <div style="font-size:13px;color:#666;margin-top:6px;">관련 헤드라인</div>
        <ul style="margin:6px 0 0 18px;line-height:1.7;">{headlines_html(news["fx"])}</ul>
      </div>

      <div style="padding:14px;border:1px solid #eee;border-radius:14px;">
        <h4 style="margin:0 0 8px 0;">2) 브렌트 유가</h4>
        <ul style="margin:0 0 10px 18px;line-height:1.7;">{bullets(oil_lines)}</ul>
        <div style="font-size:13px;color:#666;margin-top:6px;">관련 헤드라인</div>
        <ul style="margin:6px 0 0 18px;line-height:1.7;">{headlines_html(news["oil"])}</ul>
      </div>

      <div style="padding:14px;border:1px solid #eee;border-radius:14px;">
        <h4 style="margin:0 0 8px 0;">3) 코스피</h4>
        <ul style="margin:0 0 10px 18px;line-height:1.7;">{bullets(st_lines)}</ul>
        <div style="font-size:13px;color:#666;margin-top:6px;">관련 헤드라인</div>
        <ul style="margin:6px 0 0 18px;line-height:1.7;">{headlines_html(news["stock"])}</ul>
      </div>
    </div>

    <h3 style="margin-top:22px;">✅ 체크포인트(내일 확인할 것)</h3>
    <ul style="margin:6px 0 0 18px;line-height:1.7;">
      <li>큰 변동이 있었다면: 원인 뉴스(금리/재고/OPEC/지정학/수급) 1~2개만이라도 확인</li>
      <li>환율: 달러 인덱스·미 국채금리 흐름과 동행 여부 체크</li>
      <li>유가: 재고 발표/감산·증산 관련 뉴스 확인</li>
    </ul>

    <hr style="margin:22px 0;border:none;border-top:1px solid #eee;"/>
    <div style="font-size:12px;color:#777;line-height:1.6;">
      데이터: 네이버 금융/마켓인덱스(자동 수집) · 뉴스: Google News RSS(자동 수집)
    </div>
    """
    return html

# -----------------------------
# 실행
# -----------------------------
def main():
    cfg = load_config()
    today = now_seoul_date_str()

    title = f"오늘의 지표 리포트 ({today})"
    slug = f"daily-indicator-report-{today}"

    try:
        if wp_post_exists(cfg, slug):
            kakao_send_to_me(cfg, f"✅ 이미 오늘 글이 있어요 ({today})\n중복 발행 안 함")
            return

        data = fetch_indicators_real()

        news = {
            "fx": fetch_google_news_rss("원달러 환율 금리 연준", max_items=5),
            "oil": fetch_google_news_rss("브렌트 유가 OPEC 재고", max_items=5),
            "stock": fetch_google_news_rss("코스피 외국인 금리 반도체", max_items=5),
        }

        content_html = build_post_content_html(today, data, news)

        status = cfg.get("wp_status", "publish")  # 기본 publish
        post = wp_create_post(cfg, title, slug, content_html, status=status)
        link = post.get("link", cfg["wp_base_url"])

        kakao_send_to_me(cfg, f"✅ 글 발행 성공!\n날짜: {today}\n상태: {status}\n링크: {link}")

    except Exception as e:
        msg = f"❌ 자동발행 실패 ({today})\n{type(e).__name__}: {e}"
        print(msg)
        try:
            kakao_send_to_me(cfg, msg)
        except Exception as e2:
            print("카톡 알림까지 실패:", type(e2).__name__, e2)

if __name__ == "__main__":
    main()
