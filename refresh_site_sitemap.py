"""Regenerate Rank Math's sitemap cache, restoring all original sitemap settings."""
import base64
import json
import os
from pathlib import Path

import requests
from content_quality import safe_url
from site_search_health import audit
from wp_common import find_post, transport_target, response_json


def export_sitemap(base, headers):
    endpoint = base + "/wp-json/rankmath/v1/status/exportSettings"
    target, params = transport_target(endpoint)
    response = requests.post(target, headers=headers, params=params or None,
                             data={"panels[]": "sitemap"}, timeout=(10, 45), allow_redirects=False)
    data = response_json(response)
    if isinstance(data, str):
        data = json.loads(data)
    settings = data.get("sitemap") if isinstance(data, dict) else None
    if not isinstance(settings, dict) or "items_per_page" not in settings:
        raise RuntimeError("Rank Math 사이트맵 설정을 안전하게 읽지 못했습니다.")
    return settings


def save_sitemap(base, headers, settings):
    # Submit the complete observed sitemap settings. No reset/import or role changes.
    endpoint = base + "/wp-json/rankmath/v1/updateSettings"
    target, params = transport_target(endpoint)
    response = requests.post(target, headers=headers, params=params or None,
                             json={"type": "sitemap", "settings": settings,
                                   "fieldTypes": {"items_per_page": "number"},
                                   "updated": ["items_per_page"], "isReset": False},
                             timeout=(10, 45), allow_redirects=False)
    response_json(response)


def refresh(base, headers):
    # Select the working WordPress REST route using a read before any setting write.
    find_post(base + "/wp-json/wp/v2/posts", headers, "sitemap-health-route-check")
    original = export_sitemap(base, headers)
    old = int(original["items_per_page"])
    if not 1 <= old <= 10000:
        raise RuntimeError("사이트맵 분할 수가 예상 범위를 벗어나 변경하지 않았습니다.")
    changed = {**original, "items_per_page": old + 1}
    def same(left, right):
        return ({**left, "items_per_page": int(left["items_per_page"])} ==
                {**right, "items_per_page": int(right["items_per_page"])})
    try:
        save_sitemap(base, headers, changed)
    finally:
        # Read back even after an ambiguous response; restore only if the value changed.
        observed = export_sitemap(base, headers)
        if not same(observed, original):
            if {k: v for k, v in observed.items() if k != "items_per_page"} != {k: v for k, v in original.items() if k != "items_per_page"}:
                raise RuntimeError("다른 설정이 바뀌어 복원을 중단했습니다. 관리자에서 확인하세요.")
            save_sitemap(base, headers, original)
    restored = export_sitemap(base, headers)
    if not same(restored, original):
        raise RuntimeError("원래 사이트맵 설정 복원을 확인하지 못했습니다.")
    print("[SITEMAP] Rank Math 캐시 갱신 완료. 기존 사이트맵 설정 복원 확인.")
    return {"settings_restored": True, "items_per_page": old}


def maintain(base, headers):
    before = audit(base)
    if (not before["sitemap_read_errors"] and not before["latest_posts_missing_from_sitemap"]
            and not before.get("sitemap_stale")):
        print("[SITEMAP] 최신 글 포함·갱신 확인. 설정 변경 생략.")
        return {"settings_unchanged": True, "public_audit": before}
    result = refresh(base, headers)
    result["public_audit"] = audit(base)
    return result


if __name__ == "__main__":
    base = safe_url(os.environ.get("WP_BASE_URL", "")).rstrip("/")
    user, password = os.environ.get("WP_USER", ""), os.environ.get("WP_APP_PASS", "")
    if not base or not user or not password:
        raise RuntimeError("WordPress 연결 설정이 필요합니다.")
    token = base64.b64encode((user + ":" + password).encode()).decode()
    headers = {"Authorization": "Basic " + token, "Accept": "application/json"}
    result = maintain(base, headers)
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/sitemap_refresh_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    report = result["public_audit"]
    if report["sitemap_read_errors"] or report["latest_posts_missing_from_sitemap"] or report.get("sitemap_stale"):
        raise RuntimeError("사이트맵의 최신 글 포함·갱신을 확인하지 못했습니다. 점검 결과를 확인하세요.")
