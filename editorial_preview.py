"""Read public recipes and preview new prose; never write to WordPress."""
import json
import os
from pathlib import Path

import requests
from openai import OpenAI
from content_quality import (editorial_context, generate_recipe_article,
                             recipe_response_format, recover_published_recipe,
                             render_recipe, save_preview, recipe_model_options)
from wp_common import recent_recipe_posts


def main():
    base = "https://rainsow.com"
    response = requests.get(base + "/wp-json/wp/v2/posts", params={"per_page": 20}, timeout=(5, 30))
    response.raise_for_status()
    posts = response.json()
    recent = recent_recipe_posts(base)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=120, max_retries=1)
    def call(instructions, payload):
        return client.responses.create(model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
                                       instructions=instructions, input=payload, text=recipe_response_format(payload),
                                       **recipe_model_options(os.environ.get("OPENAI_MODEL", "gpt-5.2")))
    reports, seen = [], set()
    for post in posts:
        prefix = next((p for p in ("daily-recipe-", "korean-recipe-", "naverstyle-recipe-") if post["slug"].startswith(p)), None)
        if not prefix or prefix in seen:
            continue
        snapshot = recover_published_recipe(post)
        if not snapshot:
            continue
        seen.add(prefix)
        article = generate_recipe_article(call, snapshot["dish_name"], snapshot["ingredients"], snapshot["steps"],
                                          editorial_context([], recent, post), source_is_korean=True)
        save_preview(post["slug"], article["title"], render_recipe(article))
        report = {"source_url": post["link"], "previous_title": post["title"]["rendered"], "article": article}
        reports.append(report)
        Path("artifacts/editorial_samples.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[SAMPLE] " + json.dumps(report, ensure_ascii=False), flush=True)
        recent.insert(0, article)
    if len(reports) != 3:
        raise RuntimeError("Three recipe streams must be previewed before validation is complete.")


if __name__ == "__main__":
    main()
