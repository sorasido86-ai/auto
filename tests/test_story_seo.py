import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

from bs4 import BeautifulSoup
from content_quality import (ContentQualityError, validate_article, render_recipe,
                             split_recipe_steps, split_recipe_ingredients, editorial_context,
                             review_recipe_article, generate_recipe_article, format_editorial_paragraphs,
                             recipe_response_format, normalize_mealdb_source, choose_validated_recipe,
                             recover_published_recipe, translate_recipe_items, numbers, normalize_recipe_units)
from content_quality import plan_recipe_editorial, assess_recipe_editorial, EDITORIAL_CRITERIA
from wp_common import recent_recipe_posts
from site_search_health import page_metadata, xml_locations
from refresh_site_sitemap import refresh


class StoryAndSearchTests(unittest.TestCase):
    def brief(self):
        return {"reader_interest": "두부만 남은 날의 반찬", "factual_anchor": "두부와 간장",
                "development": "작은 재료 조합에서 조림으로 이어가기", "avoid_recent": "도마에서 시작하는 글 반복 피하기",
                "title_candidates": ["두부 조림으로 정한 오늘의 반찬", "두부와 간장으로 차리는 두부 조림"],
                "selected_title": "두부 조림으로 정한 오늘의 반찬"}

    def verdict(self, **changes):
        return {**dict.fromkeys(EDITORIAL_CRITERIA, True), "issues": [], **changes}

    def model_call(self, article):
        def respond(instructions, payload):
            mode = json.loads(payload)["authoring_mode"]
            value = self.brief() if mode == "planning" else self.verdict() if mode == "assessment" else article
            return SimpleNamespace(output_text=json.dumps(value, ensure_ascii=False))
        return Mock(side_effect=respond)

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

    def test_story_can_quote_a_cited_ingredient_but_cannot_invent_its_quantity(self):
        article = self.article()
        article["story"][0].update(body="두부 150g을 도마 위에 올려요.", source_steps=[], source_ingredients=[1])
        self.validate(article)
        article["story"][0]["body"] = "두부 250g을 도마 위에 올려요."
        with self.assertRaises(ContentQualityError):
            self.validate(article)

    def test_indefinite_english_pinch_may_be_equivalent_numeric_quantity(self):
        article = self.article()
        article["steps"] = ["두부를 깍둑썰기해요.", "간장과 소금 큰 1꼬집을 넣고 5분 동안 끓여요."]
        sources = ["Cut tofu into cubes.", "Add soy sauce and a large pinch of salt and simmer for 5 minutes."]
        validate_article(article, ["Tofu 150g", "Soy sauce 1 tbsp"], sources)
        article["steps"][1] = article["steps"][1].replace("1꼬집", "2꼬집")
        with self.assertRaises(ContentQualityError):
            validate_article(article, ["Tofu 150g", "Soy sauce 1 tbsp"], sources)

    def test_failed_candidate_is_replaced_only_with_a_validated_candidate(self):
        pick = Mock(side_effect=[{"id": "bad"}, {"id": "good"}])
        author = Mock(side_effect=[ContentQualityError("bad quantity"), self.article()])
        recipe, article = choose_validated_recipe(pick, author)
        self.assertEqual(recipe["id"], "good")
        self.assertEqual(article, self.article())
        self.assertEqual(author.call_count, 2)
        with self.assertRaises(ContentQualityError):
            choose_validated_recipe(Mock(side_effect=[{"id": str(i)} for i in range(3)]), Mock(side_effect=ContentQualityError("bad quantity")))

    def test_clickable_title_still_requires_real_dish_and_no_false_promises(self):
        for title in ("오늘의 근사한 한 그릇", "두부 조림, 무조건 성공하는 비법"):
            article = self.article()
            article["title"] = title
            with self.assertRaises(ContentQualityError):
                self.validate(article)

    def test_actual_generated_filler_does_not_pass_as_storytelling(self):
        for intro in ("두부 조림은 재료 조합이 분명해요.", "한 접시 완성으로 식탁이 또렷해져요.", "썰기와 끓이기의 흐름이 단순합니다."):
            article = self.article()
            article["intro"] = intro
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
        self.assertEqual(split_recipe_steps("▢\nMix flour.\n▢\nBake for 15 minutes."), ["Mix flour.", "Bake for 15 minutes."])
        self.assertEqual(split_recipe_steps("Almond base\n▢\nMix the batter.\nYellow egg cream\n▢\nHeat for 15 minutes.\nAssembly\n▢\nSpread the cream."), ["Mix the batter.", "Heat for 15 minutes.", "Spread the cream."])

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
        call = self.model_call(edited)
        result = review_recipe_article(call, article, "Tofu", ["Tofu 150g", "Soy sauce 1 tbsp"], ["Cut tofu into cubes.", "Simmer with soy sauce for 5 minutes."], [])
        self.assertEqual(result["ingredients"], article["ingredients"])
        self.assertEqual(result["steps"], article["steps"])
        source = json.loads(call.call_args_list[0].args[1])
        self.assertEqual(source["authoring_mode"], "editorial")
        self.assertNotIn("ingredients", recipe_response_format(call.call_args_list[0].args[1])["format"]["schema"]["properties"])

    def test_translation_repairs_only_failed_item_without_rewriting_valid_items(self):
        first = {"ingredients": {"item_001": "두부 150g", "item_002": "간장 1큰술"},
                 "steps": {"item_001": "두부를 깍둑썰기해요.", "item_002": "50분 동안 끓여요."}}
        repair = {"steps": {"item_002": "5분 동안 끓여요."}}
        call = Mock(side_effect=[SimpleNamespace(output_text=json.dumps(first)), SimpleNamespace(output_text=json.dumps(repair))])
        result = translate_recipe_items(call, "Tofu", ["Tofu 150g", "Soy sauce 1 tbsp"], ["Cut tofu into cubes.", "Simmer for 5 minutes."], [])
        self.assertEqual(result["ingredients"], ["두부 150g", "간장 1큰술"])
        self.assertEqual(result["steps"], ["두부를 깍둑썰기해요.", "5분 동안 끓여요."])
        source = json.loads(call.call_args.args[1])
        self.assertEqual(source["translation_fields"], {"steps": {"item_002": "Simmer for 5 minutes."}})
        props = recipe_response_format(call.call_args.args[1])["format"]["schema"]["properties"]
        self.assertEqual(set(props), {"steps"})
        self.assertEqual(set(props["steps"]["properties"]), {"item_002"})

    def test_published_recipe_recovery_requires_visible_and_structured_facts_to_match(self):
        body = render_recipe(self.article(), image_url="https://example.com/tofu.jpg")
        result = recover_published_recipe({"content": {"raw": body}})
        self.assertEqual(result["ingredients"], self.article()["ingredients"])
        self.assertEqual(result["steps"], self.article()["steps"])
        changed = body.replace("<li>두부 150g</li>", "<li>두부 250g</li>")
        self.assertIsNone(recover_published_recipe({"content": {"raw": changed}}))
        decorated = body.replace('<ul>', '<nav><ul><li>목차</li></ul></nav><ul>', 1)
        decorated = decorated.replace('</article>', '<aside><ul><li>관련 글</li></ul></aside></article>')
        self.assertEqual(recover_published_recipe({"content": {"rendered": decorated}})["ingredients"], self.article()["ingredients"])

    def test_korean_written_quantities_are_equivalent_and_wrong_values_still_differ(self):
        self.assertEqual(numbers("큰 한 꼬집을 넣고 두 분 끓여요."), numbers("1꼬집을 넣고 2분 끓여요."))
        self.assertNotEqual(numbers("큰 두 꼬집"), numbers("1꼬집"))

    def test_leftover_units_are_translated_without_converting_quantities(self):
        source = "육수 2 1/2 cups (600 milliliters), 1/2 inch (1cm), 1–2 mins, 최소 1 hour, 몇 minutes, 소금 1 tsp"
        result = normalize_recipe_units(source)
        self.assertEqual(result, "육수 2 1/2 컵 (600 ml), 1/2 인치 (1cm), 1–2 분, 최소 1 시간, 몇 분, 소금 1 작은술")
        self.assertEqual(numbers(source), numbers(result))
        self.assertEqual(normalize_recipe_units("hoursglass와 1,000ml"), "hoursglass와 1,000ml")

    def test_korean_source_quantities_never_depend_on_model_copying(self):
        article = self.article()
        article.pop("ingredients")
        article.pop("steps")
        call = self.model_call(article)
        result = generate_recipe_article(call, "두부 조림", ["두부 150g", "간장 1큰술"], ["두부를 깍둑썰기해요.", "간장과 함께 5분 동안 끓여요."], source_is_korean=True)
        self.assertEqual(result["ingredients"], ["두부 150g", "간장 1큰술"])
        self.assertEqual(result["steps"], ["두부를 깍둑썰기해요.", "간장과 함께 5분 동안 끓여요."])
        self.assertEqual(call.call_count, 4)
        self.assertEqual([json.loads(c.args[1])["authoring_mode"] for c in call.call_args_list],
                         ["planning", "editorial", "editorial", "assessment"])
        self.assertEqual(result["editorial_brief"], self.brief())

    def test_semantic_rejection_rewrites_latest_draft_with_specific_feedback(self):
        rejected = self.verdict(develops_interest=False, issues=[{"quote": "끓이는 시간은 5분이에요.", "repair": "단계를 반복하는 설명을 빼세요."}])
        revised = {**self.article(), "story": []}
        values = [self.article(), rejected, revised, self.verdict()]
        call = Mock(side_effect=[SimpleNamespace(output_text=json.dumps(x)) for x in values])
        result = review_recipe_article(call, self.article(), "두부 조림", self.article()["ingredients"], self.article()["steps"], [])
        self.assertEqual(result["story"], [])
        rewrite = json.loads(call.call_args_list[2].args[1])
        self.assertFalse(rewrite["reader_assessment"]["develops_interest"])
        self.assertEqual(rewrite["reader_assessment"]["repair_goals"], [rejected["issues"][0]["repair"]])
        self.assertNotIn("recent_editorials", rewrite)
        self.assertNotIn("draft", rewrite)
        self.assertIsNone(rewrite["editorial_brief"])
        self.assertTrue(rewrite["rebuild_from_facts"])
        self.assertTrue(all(result["editorial_assessment"][key] for key in EDITORIAL_CRITERIA))

    def test_draft_prose_is_repaired_before_publication_checks(self):
        draft = {**self.article(), "intro": "두부 조림, 한 팬 리듬으로 특별한 순간을 만들어요."}
        values = [self.brief(), draft, self.article(), self.verdict()]
        call = Mock(side_effect=[SimpleNamespace(output_text=json.dumps(x)) for x in values])
        result = generate_recipe_article(call, "두부 조림", self.article()["ingredients"], self.article()["steps"], source_is_korean=True)
        self.assertNotIn("한 팬 리듬", result["intro"])
        self.assertEqual(json.loads(call.call_args_list[2].args[1])["draft"]["intro"], draft["intro"])

    def test_persistent_dry_story_is_not_published_as_valid_fallback(self):
        rejected = self.verdict(natural_prose=False, issues=[{"quote": "도마 위에 두부를 올려요.", "repair": "조리 지시를 도입으로 반복하지 마세요."}])
        values = [self.article(), rejected] * 4
        call = Mock(side_effect=[SimpleNamespace(output_text=json.dumps(x)) for x in values])
        with self.assertRaises(ContentQualityError):
            review_recipe_article(call, self.article(), "두부 조림", self.article()["ingredients"], self.article()["steps"], [])
        self.assertEqual(call.call_count, 8)
        correction = json.loads(call.call_args_list[2].args[1])
        self.assertTrue(correction["copyedit_only"])
        self.assertFalse(correction["rebuild_from_facts"])
        self.assertIn("draft", correction)

    def test_assessment_requires_consistent_verdict_and_specific_repairs(self):
        for verdict in ({"issues": []}, self.verdict(title_interest=False), self.verdict(issues=[{"quote": "문장", "repair": "수정"}])):
            call = Mock(return_value=SimpleNamespace(output_text=json.dumps(verdict)))
            with self.assertRaises(ContentQualityError):
                assess_recipe_editorial(call, self.article(), [])

    def test_planning_uses_this_recipe_and_recent_text_not_repertoire_names(self):
        call = self.model_call(self.article())
        facts = {"ingredients": self.article()["ingredients"], "steps": self.article()["steps"]}
        recent = [self.article()]
        result = plan_recipe_editorial(call, "두부 조림", facts, recent)
        source = json.loads(call.call_args.args[1])
        self.assertEqual(source["recent_editorials"], recent)
        self.assertEqual(source["steps"], facts["steps"])
        self.assertIn(result["selected_title"], result["title_candidates"])
        bad = {**self.brief(), "selected_title": "목록에 없는 제목"}
        with self.assertRaises(ContentQualityError):
            plan_recipe_editorial(Mock(return_value=SimpleNamespace(output_text=json.dumps(bad))), "두부 조림", facts, recent)

    def test_planning_and_assessment_have_independent_strict_output_contracts(self):
        for mode, keys in (("planning", set(self.brief())), ("assessment", set(self.verdict()))):
            schema = recipe_response_format(json.dumps({"authoring_mode": mode}))["format"]["schema"]
            self.assertEqual(set(schema["required"]), keys)
            self.assertFalse(schema["additionalProperties"])

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

    def test_editing_existing_post_excludes_its_own_history_only(self):
        own = {"title": "바나나 팬케이크", "intro": "기존 도입", "link": "https://example.com/current/"}
        other = {"title": "두부 조림", "intro": "다른 도입", "link": "https://example.com/other/"}
        result = editorial_context([{"title": own["title"], "intro": own["intro"]}], [own, other],
                                   exclude={"title": {"raw": own["title"]}, "link": own["link"]})
        self.assertEqual(result, [other])
        self.assertEqual(editorial_context([], [own, other]), [own, other])

    @patch("refresh_site_sitemap.refresh")
    @patch("refresh_site_sitemap.audit")
    def test_sitemap_maintenance_skips_current_and_refreshes_missing_or_stale(self, audit_mock, refresh_mock):
        from refresh_site_sitemap import maintain
        healthy = {"sitemap_read_errors": [], "latest_posts_missing_from_sitemap": [], "sitemap_stale": False}
        audit_mock.return_value = healthy
        result = maintain("https://example.com", {})
        self.assertTrue(result["settings_unchanged"])
        refresh_mock.assert_not_called()
        for key, value in (("latest_posts_missing_from_sitemap", ["https://example.com/new/"]), ("sitemap_stale", True)):
            audit_mock.side_effect = [{**healthy, key: value}, healthy]
            refresh_mock.return_value = {"settings_restored": True}
            result = maintain("https://example.com", {})
            self.assertTrue(result["settings_restored"])
            self.assertEqual(result["public_audit"], healthy)


if __name__ == "__main__":
    unittest.main()
