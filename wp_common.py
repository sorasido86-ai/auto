"""WordPress writes with strict responses and recovery after ambiguous creates."""
import time
from urllib.parse import urlsplit, urlunsplit
import requests
from content_quality import readable_html


class WordPressError(RuntimeError):
    pass


_QUERY_ROUTE = set()


def rest_target(endpoint):
    parsed = urlsplit(endpoint)
    prefix, separator, route = parsed.path.partition("/wp-json/")
    if not separator:
        return endpoint, {}, ""
    base = urlunsplit((parsed.scheme, parsed.netloc, prefix + "/", "", ""))
    return base, {"rest_route": "/" + route}, base


def transport_target(endpoint):
    alternate, params, base = rest_target(endpoint)
    return (alternate, params) if base in _QUERY_ROUTE else (endpoint, {})


def response_json(response):
    if response.status_code not in (200, 201):
        raise WordPressError(f"WordPress 응답 오류 HTTP {response.status_code}. 인증·REST 접근·서버 상태를 확인하세요.")
    try:
        return response.json()
    except ValueError as exc:
        # Never log raw HTML; error pages can echo credentials or infrastructure data.
        text = response.text if isinstance(response.text, str) else ""
        kind = "접근 확인 페이지" if any(x in text.lower() for x in ("just a moment", "one moment", "captcha", "cf-chl-")) else "HTML 또는 빈 응답"
        raise WordPressError(f"WordPress가 JSON 대신 {kind}을 반환했습니다 (HTTP {response.status_code}). REST 접근·캐시·보안 플러그인을 확인하세요.") from exc


def find_post(endpoint, headers, slug):
    request_headers = {**headers, "Accept": "application/json", "Cache-Control": "no-cache"}
    query = {"slug": slug, "per_page": 1, "status": "any", "context": "edit"}
    for attempt in range(3):
        target, route_params = transport_target(endpoint)
        response = requests.get(target, headers=request_headers, params={**route_params, **query}, timeout=(10, 30), allow_redirects=False)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(2 ** attempt)
            continue
        try:
            result = response_json(response)
        except WordPressError:
            alternate, params, base = rest_target(endpoint)
            # WordPress supports both REST URL forms. Only reads may try another route.
            if response.status_code not in (200, 201, 404) or not base or route_params:
                raise
            response = requests.get(alternate, headers=request_headers, params={**params, **query}, timeout=(10, 30), allow_redirects=False)
            result = response_json(response)
            if isinstance(result, list):
                _QUERY_ROUTE.add(base)
                print("[WP] REST 조회: WordPress 쿼리 경로 사용")
        if not isinstance(result, list):
            raise WordPressError("WordPress 글 조회 결과가 목록이 아닙니다.")
        return result[0] if result else None


def write_post(endpoint, headers, payload):
    payload = dict(payload)
    payload["content"] = readable_html(payload.get("content", ""))
    creating = endpoint.rstrip("/").endswith("/posts")
    slug = payload.get("slug")
    if creating and not slug:
        raise WordPressError("중복 방지를 위해 새 글에는 고정 slug가 필요합니다.")
    if creating:
        existing = find_post(endpoint, headers, slug)
        if existing:
            # Lost local history must not create another post or overwrite its recipe.
            return existing
    try:
        target, params = transport_target(endpoint)
        options = {"params": params} if params else {}
        response = requests.post(target, headers={**headers, "Accept": "application/json"}, json=payload, timeout=(10, 45), allow_redirects=False, **options)
        result = response_json(response)
        if not isinstance(result, dict) or not isinstance(result.get("id"), int) or result["id"] <= 0:
            raise WordPressError("WordPress 글 ID가 없는 응답입니다.")
        return result
    except (requests.RequestException, WordPressError) as exc:
        if creating:
            # A timeout/HTML reply does not prove that the server did not create a post.
            # Recover by slug. Never blindly repeat POST.
            try:
                existing = find_post(endpoint, headers, slug)
            except (requests.RequestException, WordPressError):
                existing = None
            if existing:
                return existing
        raise WordPressError("WordPress 게시 결과를 확인할 수 없습니다. 같은 글을 재생성하지 않았습니다. REST 응답을 확인한 뒤 재실행하세요.") from exc
