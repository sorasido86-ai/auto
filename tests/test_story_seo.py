import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

from bs4 import BeautifulSoup
from content_quality import (ContentQualityError, validate_article, render_recipe,
                             split_recipe_steps, split_recipe_ingredients, editorial_context,
                             review_recipe_article, generate_recipe_article, format_editorial_paragraphs,
                             recipe_response_format, normalize_mealdb_source)
from wp_common import recent_recipe_posts
from site_search_health import page_metadata, xml_locations
from refresh_site_sitemap import refresh


class StoryAndSearchTests(unittest.TestCase):
    def article(self):
        return {"title": "두부 조림, 깍둑썰기에서 시작하는 한 그릇", "dish_name": "두부 조림",
                "intro": "먼저 도마 위에 두부를 올려요. 깍둑썰기한 조각은 간장과 함께 냄비로 이어집니다.",
                "excerpt": "두부 조림의 재료와 깍둑썰기 후 5분 동안 끓이는 과정을 정리했어요.",
                "angle": "도마에서 냄비로", "ingredients": ["두부 150g", "간장 1큰술"],
                "steps": ["두부를 깍둑썰기해요.", "간장과 함께 5분 동안 끓여요."], "focus": None,
                "story": [{"heading": None, "body": "두부 조각을 만든 뒤 냄비로 옮겨요. 끓이는 시간은 5분이에요.",
                           "source_steps": [1, 2], "position": "after_steps"}]}

    def validate(self, article):
        return validate_article(article, ["Tofu 150g", "Soy sauce 1 tbsp"],
                                ["Cut tofu into cubes.", "Simmer with soy sauce for 5 minutes."])

    def test_story_is_optional_and_can_have_no_heading(self):
        article = self.article()
        self.validate(article)
        body = render_recipe(article)
        self.assertGreater(body.index("두부 조각을 만든 뒤"), body.index("</ol>"))
        article["story"] = []
        self.validate(article)
        self.assertNotIn("두부 조각을 만든 뒤", render_recipe(article))

    def test_story_cannot_invent_numbers_or_step_references(self):
        for refs, body in (([0], "두부를 옮겨요."), ([1], "5분 끓여요."), ([2], "180도에서 익혀요.")):
            article = self.article()
            article["story"][0].update(source_steps=refs, body=body)
            with self.assertRaises(ContentQualityError):
                self.validate(article)

    def test_clickable_title_still_requires_real_dish_and_no_false_promises(self):
        for title in ("오늘의 근사한 한 그릇", "두부 조림, 무조건 성공하는 비법"):
            article = self.article()
            article["title"] = title
            with self.assertRaises(ContentQualityError):
                self.validate(article)

    def test_jsonld_matches_visible_recipe_and_anchor_links(self):
        article = self.article()
        page = "https://example.com/tofu/"
        soup = BeautifulSoup(render_recipe(article, "https://source.example/", image_url="https://example.com/tofu.jpg", page_url=page), "html.parser")
        schema = json.loads(soup.select_one('script[type="application/ld+json"]').string)
        self.assertEqual(schema["name"], "두부 조림")
        self.assertEqual(schema["recipeIngredient"], [li.get_text() for li in soup.select("ul li")])
        self.assertEqual([s["text"] for s in schema["recipeInstructions"]], [li.get_text() for li in soup.select("ol li")])
        for step in schema["recipeInstructions"]:
            self.assertIsNotNone(soup.find(id=step["url"].split("#")[1]))
        self.assertFalse(set(schema) & {"aggregateRating", "nutrition", "totalTime", "cookTime"})
        self.assertIsNone(BeautifulSoup(render_recipe(article), "html.parser").select_one('script[type="application/ld+json"]'))

    def test_related_links_use_real_same_site_posts_with_shared_ingredients(self):
        recent = [{"title": "두부 김치", "link": "https://example.com/older/"},
                  {"title": "두부 구이", "link": "https://elsewhere.example/tofu/"},
                  {"title": "감자 수프", "link": "https://example.com/potato/"}]
        soup = BeautifulSoup(render_recipe(self.article(), page_url="https://example.com/tofu/", recent=recent), "html.parser")
        self.assertEqual([a["href"] for a in soup.select("aside a")], ["https://example.com/older/"])

    def test_search_audit_recognizes_recipe_beside_rankmath_graph(self):
        body = '<html><head><title>두부 조림</title><meta name="description" content="두부 조림 만드는 법"><meta name="robots" content="index,follow"><link rel="canonical" href="https://example.com/tofu/"><script type="application/ld+json">{"@graph":[{"@type":"Article"}]}</script></head><body><h1>두부 조림</h1>'
        body += render_recipe(self.article(), image_url="https://example.com/tofu.jpg") + '</body></html>'
        result = page_metadata(body, "https://example.com/tofu/")
        self.assertEqual(result["recipe_schema"], 1)
        self.assertTrue(result["recipe_has_image"])
        self.assertEqual(result["h1_count"], 1)
        self.assertEqual(result["canonical"], result["url"])

    def test_sitemap_audit_reads_real_namespace_and_locations(self):
        urls, dates = xml_locations('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://example.com/tofu/</loc><lastmod>2026-10-07</lastmod></url></urlset>')
        self.assertEqual(urls, ["https://example.com/tofu/"])
        self.assertEqual(dates, ["2026-10-07"])

    def test_source_headings_do_not_shift_recipe_instructions(self):
        source = "Prepare the Potatoes\nBoil the potatoes with skins on.\nMake the Dough\nMix potatoes and flour.\nShape the Šúĺlance\nRoll into a rope about 1.5 cm wide.\nServe and Enjoy\nServe warm."
        self.assertEqual(split_recipe_steps(source), ["Boil the potatoes with skins on.", "Mix potatoes and flour.", "Roll into a rope about 1.5 cm wide.", "Serve warm."])
        self.assertEqual(split_recipe_steps("Cook the potatoes until tender.\nUse 1.5 tbsp water."), ["Cook the potatoes until tender.", "Use 1.5 tbsp water."])

    def test_verified_provider_error_uses_linked_original_ingredients_only(self):
        recipe = {"id": "plov", "source": "https://www.thespruceeats.com/russian-lamb-pilaf-plov-recipe-1137309", "ingredients": [{"name": "Lamb", "measure": "50g"}], "instructions": "Source instructions"}
        fixed = normalize_mealdb_source(recipe)
        self.assertEqual(fixed["ingredients"][0], {"name": "Raisins", "measure": "50g"})
        self.assertIn({"name": "Ground lamb", "measure": "225g"}, fixed["ingredients"])
        self.assertEqual(fixed["instructions"], recipe["instructions"])
        other = {**recipe, "source": "https://example.com/other"}
        self.assertIs(normalize_mealdb_source(other), other)

    def test_mfds_group_headers_and_newlines_do_not_merge_or_lose_ingredients(self):
        source = "주재료\n양배추 30g, 쫄면 사리 100g\n\n양념\n올리브유 30g, 참기름 10g"
        expected = ["양배추 30g", "쫄면 사리 100g", "올리브유 30g", "참기름 10g"]
        self.assertEqual(split_recipe_ingredients(source), expected)
        bot = importlib.import_module("daily_korean_recipe_to_wp")
        self.assertEqual(bot.mfds_row_to_recipe({"RCP_PARTS_DTLS": source}).ingredients, expected)
        self.assertEqual(split_recipe_ingredients("물 1,000ml, 당근(껍질 제거, 다짐) 30g"), ["물 1,000ml", "당근(껍질 제거, 다짐) 30g"])
        self.assertEqual(split_recipe_ingredients("소금적당량 [소재료] 김치 20g, 오징어 30g [양념] 설탕 5g"), ["소금적당량", "김치 20g", "오징어 30g", "설탕 5g"])

    def test_editor_cannot_change_fixed_recipe_facts(self):
        article = self.article()
        edited = {**article, "ingredients": ["두부 50g"], "steps": ["180도에 구워요."]}
        call = Mock(return_value=SimpleNamespace(output_text=json.dumps(edited)))
        result = review_recipe_article(call, article, "Tofu", ["Tofu 150g", "Soy sauce 1 tbsp"], ["Cut tofu into cubes.", "Simmer with soy sauce for 5 minutes."], [])
        self.assertEqual(result["ingredients"], article["ingredients"])
        self.assertEqual(result["steps"], article["steps"])
        source = json.loads(call.call_args.args[1])
        self.assertEqual(source["authoring_mode"], "editorial")
        self.assertNotIn("ingredients", recipe_response_format(call.call_args.args[1])["format"]["schema"]["properties"])

    def test_korean_source_quantities_never_depend_on_model_copying(self):
        article = self.article()
        article.pop("ingredients")
        article.pop("steps")
        call = Mock(return_value=SimpleNamespace(output_text=json.dumps(article)))
        result = generate_recipe_article(call, "두부 조림", ["두부 150g", "간장 1큰술"], ["두부를 깍둑썰기해요.", "간장과 함께 5분 동안 끓여요."], source_is_korean=True)
        self.assertEqual(result["ingredients"], ["두부 150g", "간장 1큰술"])
        self.assertEqual(result["steps"], ["두부를 깍둑썰기해요.", "간장과 함께 5분 동안 끓여요."])
        self.assertEqual(call.call_count, 2)

    def test_paragraph_reflow_preserves_all_sentences(self):
        parts = ["가나다 " * 14 + ending for ending in ("써요.", "옮겨요.", "끓여요.", "담아요.")]
        original = " ".join(parts)
        formatted = format_editorial_paragraphs(original)
        self.assertEqual(formatted.replace("\n\n", " "), original)
        self.assertTrue(all(len(p) <= 180 for p in formatted.split("\n\n")))

    @patch("refresh_site_sitemap.find_post")
    @patch("refresh_site_sitemap.save_sitemap")
    @patch("refresh_site_sitemap.export_sitemap")
    def test_sitemap_refresh_restores_observed_settings_including_exclusions(self, export, save, find):
        original = {"items_per_page": "200", "exclude_posts": "12,34", "include_images": "on"}
        changed = {**original, "items_per_page": 201}
        export.side_effect = [original, changed, {**original, "items_per_page": 200}]
        result = refresh("https://example.com", {})
        self.assertTrue(result["settings_restored"])
        self.assertEqual([c.args[2] for c in save.call_args_list], [changed, original])

    @patch("refresh_site_sitemap.find_post")
    @patch("refresh_site_sitemap.save_sitemap")
    @patch("refresh_site_sitemap.export_sitemap")
    def test_failed_sitemap_write_is_read_back_and_restored(self, export, save, find):
        original = {"items_per_page": 200}
        export.side_effect = [original, {"items_per_page": 201}]
        save.side_effect = [__import__("requests").Timeout(), None]
        with self.assertRaises(__import__("requests").Timeout):
            refresh("https://example.com", {})
        self.assertEqual(save.call_args.args[2], original)

    @patch("wp_common.requests.get")
    def test_public_history_is_cross_workflow_and_ignores_non_recipe_or_scripts(self, get):
        response = Mock(status_code=200)
        response.json.return_value = [{"slug": "korean-recipe-day", "title": {"rendered": "두부 조림"}, "content": {"rendered": '<article><p class="recipe-intro">도마 위 두부.</p><script>bad</script></article>'}, "link": "https://example.com/a/"},
                                     {"slug": "news-day", "title": {"rendered": "뉴스"}}]
        get.return_value = response
        result = recent_recipe_posts("https://example.com")
        self.assertEqual(result, [{"title": "두부 조림", "intro": "도마 위 두부.", "link": "https://example.com/a/"}])
        self.assertEqual(editorial_context(result, result), result)
        get.side_effect = __import__("requests").Timeout()
        self.assertEqual(recent_recipe_posts("https://example.com"), [])


if __name__ == "__main__":
    unittest.main()
