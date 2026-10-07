"""Check public search metadata and actual sitemap inclusion, without credentials."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from content_quality import safe_url


def fetch(url):
    response = requests.get(url, headers={"Cache-Control": "no-cache"}, timeout=(5, 20))
    response.raise_for_status()
    return response.text


def xml_locations(text):
    root = ET.fromstring(text)
    return [x.text for x in root.findall(".//{*}loc") if x.text], [x.text for x in root.findall(".//{*}lastmod") if x.text]


def page_metadata(text, url):
    soup = BeautifulSoup(text, "html.parser")
    meta = soup.select_one('meta[name="description"]')
    canonical = soup.select_one('link[rel="canonical"]')
    robots = soup.select_one('meta[name="robots"]')
    schemas = []
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.get_text())
            schemas += data.get("@graph", [data]) if isinstance(data, dict) else data
        except (ValueError, TypeError):
            pass
    recipes = [x for x in schemas if isinstance(x, dict) and x.get("@type") == "Recipe"]
    return {"url": url, "title": soup.title.get_text() if soup.title else "",
            "description": meta.get("content", "") if meta else "",
            "canonical": canonical.get("href", "") if canonical else "",
            "robots": robots.get("content", "") if robots else "",
            "h1_count": len(soup.select("h1")), "recipe_schema": len(recipes),
            "recipe_has_image": any(x.get("image") for x in recipes),
            "intro": "\n\n".join(x.get_text(" ", strip=True) for x in soup.select(".recipe-intro")),
            "story": [x.get_text(" ", strip=True) for x in soup.select(".recipe-story")]}


def audit(base):
    base = safe_url(base).rstrip("/")
    if not base:
        raise ValueError("Valid SITE_URL required")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "site": base, "warnings": []}
    posts = json.loads(fetch(base + "/wp-json/wp/v2/posts?per_page=6&_fields=link,date,title,modified_gmt"))
    urls = [p["link"] for p in posts]
    report["pages"] = [page_metadata(fetch(url), url) for url in urls[:3]]
    robots = fetch(base + "/robots.txt")
    report["robots_sitemap_declared"] = "Sitemap:" in robots
    children, dates = xml_locations(fetch(base + "/sitemap_index.xml"))
    # Follow only the observed same-site sitemap links; never external XML locations.
    children = [u for u in children if urlsplit(u).netloc == urlsplit(base).netloc and safe_url(u)]
    sitemap_urls, errors = set(), []
    def inspect(url):
        try:
            return xml_locations(fetch(url))[0], None
        except (requests.RequestException, ValueError, ET.ParseError):
            return [], url
    with ThreadPoolExecutor(max_workers=4) as pool:
        for locations, error in pool.map(inspect, children[:20]):
            sitemap_urls.update(locations)
            if error:
                errors.append(error)
    report["sitemap_latest_modified"] = max(dates, default="")
    modified = max((p.get("modified_gmt", "") for p in posts), default="")
    report["latest_post_modified_gmt"] = modified
    report["sitemap_stale"] = bool(modified and report["sitemap_latest_modified"] and
                                  report["sitemap_latest_modified"][:19] < modified[:19])
    report["sitemap_read_errors"] = errors
    report["latest_posts_missing_from_sitemap"] = [u for u in urls if u not in sitemap_urls]
    if errors:
        report["warnings"].append("일부 사이트맵을 읽지 못해 포함 여부를 확정할 수 없습니다.")
    elif report["latest_posts_missing_from_sitemap"]:
        report["warnings"].append("최신 글이 사이트맵에 없습니다. Rank Math 사이트맵 캐시·정적 XML·서버 캐시를 점검하세요.")
    if report["sitemap_stale"]:
        report["warnings"].append("사이트맵 갱신 시각이 최신 글 수정 시각보다 오래됐습니다.")
    for page in report["pages"]:
        if "noindex" in page["robots"] or page["canonical"] != page["url"]:
            report["warnings"].append("최신 글의 색인 허용 또는 canonical을 확인하세요: " + page["url"])
        if not page["description"]:
            report["warnings"].append("검색 설명이 없습니다: " + page["url"])
    return report


if __name__ == "__main__":
    result = audit(os.environ.get("SITE_URL", "https://rainsow.com"))
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/site_search_health.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for warning in result["warnings"]:
        print("::warning::" + warning)
