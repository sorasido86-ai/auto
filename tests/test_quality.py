import copy
import importlib
import inspect
from contextlib import redirect_stdout
from io import StringIO
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from content_quality import (ContentQualityError, generate_recipe_article,
                             render_recipe, safe_url, validate_article,
                             recent_editorials, remember_editorial,
                             recipe_response_format, split_recipe_steps)
from wp_common import WordPressError, find_post, write_post


INGREDIENTS = ["Tofu 150 g", "Soy sauce 1.5 tbsp"]
STEPS = ["Cut tofu into cubes.", "Simmer for 5 minutes."]
ARTICLE = {"title": "두부 간장 조림 만드는 법", "intro": "두부와 간장을 사용하는 조림입니다.",
           "ingredients": ["두부 150 g", "간장 1.5 큰술"],
           "steps": ["두부를 깍둑썰기합니다.", "5분 동안 끓입니다."]}


def response(value=None, status=200, invalid=False):
    result = Mock(status_code=status)
    if invalid:
        result.json.side_effect = ValueError("HTML")
    else:
        result.json.return_value = value
    return result


class RecipeQualityTests(unittest.TestCase):

    def test_source_headings_not_counted_as_steps_and_decimals_kept(self):
        self.assertEqual(split_recipe_steps("STEP 1\nUse 1.5 tbsp sauce.\nSTEP 2\n2. Cook for 5 minutes."), ["Use 1.5 tbsp sauce.", "Cook for 5 minutes."])
        self.assertEqual(split_recipe_steps("1/2 cup of water is added."), ["1/2 cup of water is added."])

    def test_numbered_output_restores_source_order(self):
        article = copy.deepcopy(ARTICLE)
        for key in ("ingredients", "steps"):
            article[key] = {"item_002": article[key][1], "item_001": article[key][0]}
        call = Mock(return_value=SimpleNamespace(output_text=json.dumps(article)))
        self.assertEqual(generate_recipe_article(call, "Tofu", INGREDIENTS, STEPS), ARTICLE)

    def test_output_contract_requires_every_source_item(self):
        fmt = recipe_response_format(json.dumps({"ingredients": INGREDIENTS, "steps": STEPS}))
        self.assertTrue(fmt["format"]["strict"])
        ingredients = fmt["format"]["schema"]["properties"]["ingredients"]
        self.assertEqual(ingredients["required"], ["item_001", "item_002"])
        self.assertFalse(ingredients["additionalProperties"])

    def test_written_source_numbers_translate_without_false_alarm(self):
        article = copy.deepcopy(ARTICLE)
        article["steps"][0] = "냄비 2개에 두부를 나눕니다."
        validate_article(article, INGREDIENTS, ["Divide tofu between two saucepans.", STEPS[1]])
        article["steps"][0] = "냄비 두 개에 두부를 나눕니다."
        validate_article(article, INGREDIENTS, ["Divide tofu between two saucepans.", STEPS[1]])
        article["steps"][0] = "냄비 3개에 두부를 나눕니다."
        with self.assertRaises(ContentQualityError):
            validate_article(article, INGREDIENTS, ["Divide tofu between two saucepans.", STEPS[1]])

    def test_ingredient_amount_cannot_be_added_to_step(self):
        article = copy.deepcopy(ARTICLE)
        article["steps"][0] = "두부 150 g을 썹니다."
        with self.assertRaises(ContentQualityError):
            validate_article(article, INGREDIENTS, STEPS)

    def test_equivalent_fraction_glyphs_are_allowed(self):
        article = copy.deepcopy(ARTICLE)
        article["ingredients"][1] = "간장 1/2 큰술"
        validate_article(article, [INGREDIENTS[0], "Soy sauce ½ tbsp"], STEPS)

    def test_valid_source_translation(self):
        self.assertEqual(validate_article(ARTICLE, INGREDIENTS, STEPS), ARTICLE)

    def test_supported_time_can_appear_in_title_and_explanation(self):
        article = copy.deepcopy(ARTICLE)
        article["title"] = "5분 끓이는 두부 조림"
        article["intro"] = "두부를 썬 뒤 5분 동안 끓이는 레시피입니다."
        article["focus"] = {"heading": "5분 동안 끓이기", "body": "원문의 끓이는 시간은 5분입니다.", "source_steps": [2], "position": "before_steps"}
        self.assertEqual(validate_article(article, INGREDIENTS, STEPS), article)

    def test_changed_decimal_rejected(self):
        article = copy.deepcopy(ARTICLE)
        article["ingredients"][1] = "간장 15 큰술"
        with self.assertRaises(ContentQualityError):
            validate_article(article, INGREDIENTS, STEPS)

    def test_missing_step_rejected(self):
        article = copy.deepcopy(ARTICLE)
        article["steps"].pop()
        with self.assertRaises(ContentQualityError):
            validate_article(article, INGREDIENTS, STEPS)

    def test_invented_temperature_rejected(self):
        article = copy.deepcopy(ARTICLE)
        article["steps"][0] += " 180도에서 익힙니다."
        with self.assertRaises(ContentQualityError):
            validate_article(article, INGREDIENTS, STEPS)

    def test_false_experience_rejected(self):
        article = copy.deepcopy(ARTICLE)
        article["intro"] = "제가 직접 해봤더니 무조건 맛있습니다."
        with self.assertRaises(ContentQualityError):
            validate_article(article, INGREDIENTS, STEPS)

    def test_correction_gets_actual_source(self):
        invalid = copy.deepcopy(ARTICLE)
        invalid["steps"].pop()
        call = Mock(side_effect=[SimpleNamespace(output_text=json.dumps(invalid)), SimpleNamespace(output_text=json.dumps(ARTICLE)), SimpleNamespace(output_text=json.dumps(ARTICLE))])
        self.assertEqual(generate_recipe_article(call, "Tofu", INGREDIENTS, STEPS), ARTICLE)
        self.assertEqual(call.call_count, 3)
        self.assertEqual(json.loads(call.call_args_list[1].args[1])["previous_response"], invalid)
        for args, _ in call.call_args_list[:2]:
            source = json.loads(args[1])
            self.assertEqual(source["ingredients"], INGREDIENTS)
            self.assertEqual(source["steps"], STEPS)

    def test_invalid_generation_does_not_fallback(self):
        call = Mock(return_value=SimpleNamespace(output_text='{"title":"레시피"}'))
        with self.assertRaises(ContentQualityError):
            generate_recipe_article(call, "Tofu", INGREDIENTS, STEPS)
        self.assertEqual(call.call_count, 2)

    def test_renderer_preserves_decimal_and_semantics(self):
        rendered = render_recipe(ARTICLE, "https://example.com/recipe?a=1&b=2")
        self.assertIn("1.5 큰술", rendered)
        self.assertIn("<ol>", rendered)
        self.assertEqual(rendered.count("<li"), 4)
        self.assertIn("a=1&amp;b=2", rendered)

    def test_unsafe_urls_omitted(self):
        for url in ["javascript:alert(1)", "https://user:password@example.com", "data:text/html,hi"]:
            self.assertEqual(safe_url(url), "")
            self.assertNotIn(url, render_recipe(ARTICLE, url, image_url=url))

    def test_korean_recipe_retains_all_source_values(self):
        bot = importlib.import_module("daily_korean_recipe_to_wp")
        cfg = bot.load_cfg()
        recipe = bot.Recipe("local", "test", "두부 조림", ["두부 150g", "간장 1.5큰술"], ["두부를 썬다.", "5분 끓인다."])
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            body, excerpt = bot.build_body_html(cfg, recipe, "", None)
        self.assertIn("1.5큰술", body)
        self.assertIn("5분 끓인다.", body)
        self.assertNotIn("저는", body)
        self.assertNotIn("중불", body)
        self.assertIn("<ol>", body)
        self.assertEqual(excerpt, "")

    def test_mfds_display_labels_are_not_recipe_quantities(self):
        bot = importlib.import_module("daily_korean_recipe_to_wp")
        recipe = bot.mfds_row_to_recipe({"RCP_SEQ": "1", "RCP_NM": "두부 조림", "RCP_PARTS_DTLS": "두부 150g, 간장 1.5큰술", "MANUAL01": "1.두부를 썬다.", "MANUAL02": "2. 1.5분 동안 끓인다.", "MANUAL03": "3) 180C"})
        self.assertEqual(recipe.steps, ["두부를 썬다.", "1.5분 동안 끓인다.", "180C"])
        recipe = bot.mfds_row_to_recipe({"RCP_NM": "볶음", "MANUAL01": "1.5cm로 자른다."})
        self.assertEqual(recipe.steps, ["1.5cm로 자른다."])

    def test_missing_recipe_image_does_not_get_generic_stock_photo(self):
        bot = importlib.import_module("daily_korean_recipe_to_wp")
        cfg = bot.load_cfg()
        cfg.img.default_thumb_url = ""
        cfg.img.auto_image = True
        recipe = bot.Recipe("local", "test", "두부 조림", ["두부 150g"], ["두부를 썬다."])
        self.assertEqual(bot.choose_thumb_url(cfg, recipe), "")


class EditorialQualityTests(unittest.TestCase):

    def article(self):
        result = copy.deepcopy(ARTICLE)
        result["intro"] = "두부를 깍둑썰기한 뒤 끓이는 조림입니다.\n\n재료는 두부와 간장으로 구성됩니다."
        result["angle"] = "두부를 써는 준비 과정"
        return result

    def focus(self, position="before_steps"):
        return {"heading": "썰기에서 끓이기로", "body": "두부를 먼저 썰고 끓이는 순서로 진행합니다.",
                "source_steps": [1, 2], "position": position}

    def test_optional_focus_and_paragraphs_preserve_recipe_order(self):
        for position in ("before_steps", "after_steps"):
            article = self.article()
            article["focus"] = self.focus(position)
            validate_article(article, INGREDIENTS, STEPS)
            body = render_recipe(article)
            self.assertIn('</p>\n<p class="recipe-intro">재료는', body)
            self.assertLess(body.index(ARTICLE["steps"][0]), body.index(ARTICLE["steps"][1]))
            self.assertEqual(body.count("<li"), 4)
            if position == "before_steps":
                self.assertLess(body.index("썰기에서 끓이기로"), body.index("<ol>"))
            else:
                self.assertGreater(body.index("썰기에서 끓이기로"), body.index("</ol>"))
        self.assertNotIn("썰기에서 끓이기로", render_recipe(self.article()))

    def test_focus_requires_valid_source_references_and_numbers(self):
        for refs, body in (([], "두부를 먼저 썹니다."), ([0], "두부를 먼저 썹니다."), ([True], "두부를 먼저 썹니다."), ([1], "180도에서 익힙니다."), ([1], "5분 동안 끓입니다.")):
            article = self.article()
            article["focus"] = self.focus()
            article["focus"].update(source_steps=refs, body=body)
            with self.assertRaises(ContentQualityError):
                validate_article(article, INGREDIENTS, STEPS)

    def test_intro_cannot_invent_time_or_use_boilerplate(self):
        for intro in ("10분이면 완성하는 두부 요리입니다.", "오늘은 두부 조림을 소개합니다.", "누구나 쉽게 만들 수 있습니다."):
            article = self.article()
            article["intro"] = intro
            with self.assertRaises(ContentQualityError):
                validate_article(article, INGREDIENTS, STEPS)

    def test_long_paragraph_and_repeated_sentence_rejected(self):
        for intro in ("두부를 썰어 끓이는 순서로 재료를 준비해 만드는 조림입니다. " * 8,
                      "두부를 깍둑썰기한 뒤 간장과 함께 끓이는 조림입니다. 두부를 깍둑썰기한 뒤 간장과 함께 끓이는 조림입니다."):
            article = self.article()
            article["intro"] = intro
            with self.assertRaises(ContentQualityError):
                validate_article(article, INGREDIENTS, STEPS)

    def test_dish_name_swap_in_same_intro_rejected(self):
        previous = self.article()
        previous["intro"] = "두부를 깍둑썰기한 뒤 간장과 함께 끓이는 순서로 조리합니다. 먼저 재료를 준비하고 다음 단계에서 끓이는 과정으로 이어집니다."
        article = copy.deepcopy(previous)
        article["intro"] = article["intro"].replace("두부", "감자").replace("간장", "양념장").replace("다음 단계", "이어지는 단계")
        with self.assertRaises(ContentQualityError):
            validate_article(article, INGREDIENTS, STEPS, [previous])

    def test_different_fact_led_openings_are_allowed(self):
        previous = self.article()
        article = self.article()
        article["intro"] = "두부 150 g에 간장 1.5 큰술을 사용하는 레시피입니다. 깍둑썰기는 끓이기 전에 진행합니다."
        self.assertEqual(validate_article(article, INGREDIENTS, STEPS, [previous]), article)

    def test_recent_openings_supplied_and_repetition_corrected(self):
        previous = self.article()
        changed = self.article()
        changed["intro"] = "두부와 간장을 준비해 만드는 조림입니다. 썬 두부는 5분 동안 끓입니다."
        call = Mock(side_effect=[SimpleNamespace(output_text=json.dumps(previous)), SimpleNamespace(output_text=json.dumps(previous)), SimpleNamespace(output_text=json.dumps(changed))])
        self.assertEqual(generate_recipe_article(call, "Tofu", INGREDIENTS, STEPS, [previous]), changed)
        self.assertEqual(call.call_count, 3)
        self.assertIn("recent_editorials", json.loads(call.call_args_list[1].args[1]))
        self.assertIn("최근", call.call_args_list[2].args[0])

    def test_history_is_bounded_and_replaces_same_post(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/history.sqlite3"
            self.assertEqual(recent_editorials(path), [])
            for i in range(35):
                article = self.article()
                article["title"] = f"두부 조림 {i}"
                remember_editorial(path, str(i), article)
            self.assertEqual(len(recent_editorials(path, 100)), 30)
            self.assertEqual(recent_editorials(path)[0]["title"], "두부 조림 34")
            remember_editorial(path, "34", ARTICLE)
            self.assertEqual(len(recent_editorials(path, 100)), 30)
            self.assertEqual(recent_editorials(path)[0]["title"], ARTICLE["title"])
            self.assertNotIn("ingredients", recent_editorials(path)[0])

    def test_korean_workflow_uses_same_editorial_generator(self):
        bot = importlib.import_module("daily_korean_recipe_to_wp")
        cfg = bot.load_cfg()
        recipe = bot.Recipe("local", "test", "두부 조림", ARTICLE["ingredients"], ARTICLE["steps"])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"}), patch.object(bot, "OpenAI") as client, patch.object(bot, "generate_recipe_article", return_value=ARTICLE) as generate, patch.object(bot, "recent_editorials", return_value=[]):
            body, excerpt = bot.build_body_html(cfg, recipe, "", None)
            generate.assert_called_once()
            generate.call_args.args[0]("instructions", json.dumps({"ingredients": INGREDIENTS, "steps": STEPS}))
            self.assertEqual(client.return_value.responses.create.call_args.kwargs["model"], "test-model")
            self.assertEqual(excerpt, ARTICLE["intro"])
            self.assertEqual(cfg.run.editorial_article["title"], ARTICLE["title"])
            self.assertIn(ARTICLE["steps"][1], body)


class PublishingTests(unittest.TestCase):
    endpoint = "https://example.com/wp-json/wp/v2/posts"
    payload = {"slug": "recipe-day", "title": "두부 조림", "content": "<p>본문</p>", "status": "publish"}

    @patch("wp_common._QUERY_ROUTE", new_callable=set)
    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_html_lookup_uses_official_query_route_for_single_publish(self, get, post, modes):
        get.side_effect = [response(invalid=True), response([])]
        post.return_value = response({"id": 7}, status=201)
        with redirect_stdout(StringIO()):
            self.assertEqual(write_post(self.endpoint, {}, self.payload)["id"], 7)
        self.assertEqual(get.call_args.args[0], "https://example.com/")
        self.assertEqual(get.call_args.kwargs["params"]["rest_route"], "/wp/v2/posts")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0], "https://example.com/")
        self.assertEqual(post.call_args.kwargs["params"]["rest_route"], "/wp/v2/posts")

    @patch("wp_common._QUERY_ROUTE", new_callable=set)
    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_both_lookup_routes_fail_without_creating(self, get, post, modes):
        get.return_value = response(invalid=True)
        with self.assertRaises(WordPressError):
            write_post(self.endpoint, {}, self.payload)
        self.assertEqual(get.call_count, 2)
        post.assert_not_called()

    @patch("wp_common._QUERY_ROUTE", new_callable=set)
    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_query_route_recovers_post_timeout_without_second_write(self, get, post, modes):
        get.side_effect = [response(invalid=True), response([]), response([{"id": 8}])]
        post.side_effect = requests.Timeout("uncertain")
        with redirect_stdout(StringIO()):
            self.assertEqual(write_post(self.endpoint, {}, self.payload)["id"], 8)
        self.assertEqual(post.call_count, 1)

    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_existing_post_never_creates_again(self, get, post):
        get.return_value = response([{"id": 3, "link": "url"}])
        self.assertEqual(write_post(self.endpoint, {}, self.payload)["id"], 3)
        post.assert_not_called()

    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_successful_create_once(self, get, post):
        get.return_value = response([])
        post.return_value = response({"id": 7, "link": "url"}, status=201)
        self.assertEqual(write_post(self.endpoint, {}, self.payload)["id"], 7)
        self.assertEqual(post.call_count, 1)

    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_timeout_recovers_by_slug(self, get, post):
        get.side_effect = [response([]), response([{"id": 8}])]
        post.side_effect = requests.Timeout("uncertain")
        self.assertEqual(write_post(self.endpoint, {}, self.payload)["id"], 8)
        self.assertEqual(post.call_count, 1)

    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_html_response_does_not_repeat_create(self, get, post):
        get.return_value = response([])
        post.return_value = response(invalid=True)
        with self.assertRaises(WordPressError):
            write_post(self.endpoint, {}, self.payload)
        self.assertEqual(post.call_count, 1)

    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_lookup_failure_cannot_create(self, get, post):
        get.return_value = response(status=403)
        with self.assertRaises(WordPressError):
            write_post(self.endpoint, {}, self.payload)
        post.assert_not_called()

    @patch("wp_common.requests.post")
    @patch("wp_common.requests.get")
    def test_update_failure_cannot_create(self, get, post):
        post.return_value = response(status=503)
        with self.assertRaises(WordPressError):
            write_post(self.endpoint + "/7", {}, self.payload)
        get.assert_not_called()
        self.assertEqual(post.call_count, 1)

    def test_missing_slug_rejected(self):
        with self.assertRaises(WordPressError):
            write_post(self.endpoint, {}, {"title": "missing"})

    @patch("wp_common.time.sleep")
    @patch("wp_common.requests.get")
    def test_get_transient_failure_retries_bounded(self, get, sleep):
        get.side_effect = [response(status=503), response([])]
        self.assertIsNone(find_post(self.endpoint, {}, "recipe"))
        self.assertEqual(get.call_count, 2)


class BotIntegrationTests(unittest.TestCase):
    def test_naver_editorial_refresh_updates_verified_existing_recipe_only(self):
        bot = importlib.import_module("daily_recipe_to_wp_naverstyle_FINAL")
        with tempfile.TemporaryDirectory() as directory:
            cfg = bot.load_cfg()
            cfg.sqlite_path = directory + "/test.sqlite3"
            cfg.run.dry_run = False
            existing = {"id": 7, "content": {"raw": render_recipe(ARTICLE, image_url="https://example.com/tofu.jpg")}}
            with patch.dict(os.environ, {"REFRESH_EDITORIAL": "1"}), patch.object(bot, "find_post", return_value=existing), patch.object(bot, "OpenAI"), patch.object(bot, "recent_recipe_posts", return_value=[]), patch.object(bot, "generate_recipe_article", return_value=ARTICLE) as author, patch.object(bot, "wp_update_editorial", return_value=(7, "url")) as update, patch.object(bot, "wp_create_post") as create, patch.object(bot, "pick_recipe") as pick, patch.object(bot, "wp_upload_media") as upload, patch.object(bot, "save_preview"):
                with redirect_stdout(StringIO()):
                    bot.run(cfg)
                self.assertEqual(update.call_args.args[1], 7)
                self.assertTrue(author.call_args.kwargs["source_is_korean"])
                self.assertEqual(author.call_args.args[2], ARTICLE["ingredients"])
                create.assert_not_called()
                pick.assert_not_called()
                upload.assert_not_called()

    def test_naver_refresh_cannot_overwrite_unverified_recipe(self):
        bot = importlib.import_module("daily_recipe_to_wp_naverstyle_FINAL")
        with tempfile.TemporaryDirectory() as directory:
            cfg = bot.load_cfg()
            cfg.sqlite_path = directory + "/test.sqlite3"
            cfg.run.dry_run = False
            with patch.dict(os.environ, {"REFRESH_EDITORIAL": "1"}), patch.object(bot, "find_post", return_value={"id": 7, "content": {"raw": "<p>Old recipe</p>"}}), patch.object(bot, "OpenAI"), patch.object(bot, "recent_recipe_posts", return_value=[]), patch.object(bot, "wp_update_editorial") as update, patch.object(bot, "wp_create_post") as create:
                with self.assertRaises(ContentQualityError):
                    bot.run(cfg)
                update.assert_not_called()
                create.assert_not_called()

    def test_naver_refresh_post_preserves_existing_publication_settings(self):
        bot = importlib.import_module("daily_recipe_to_wp_naverstyle_FINAL")
        with patch.object(bot, "write_post", return_value={"id": 7, "link": "url"}) as write:
            bot.wp_update_editorial(bot.load_cfg().wp, 7, "새 제목", "<p>본문</p>", "요약")
        self.assertTrue(write.call_args.args[0].endswith("/posts/7"))
        self.assertEqual(set(write.call_args.args[2]), {"title", "content", "excerpt"})

    def test_all_ten_post_creators_pass_fixed_slug_and_content(self):
        modules = ["daily_post", "daily_recipe_to_wp", "daily_recipe_to_wp_naverstyle_FINAL", "daily_korean_recipe_to_wp", "daily_issue_keywords_to_wp", "community_hotdeal_top20_to_wp", "daily_animal_media_to_wp", "naver_fashion_daily_to_wp", "hot_keyword_naver_shop_to_wp", "trend_keywords_daily_to_wp"]
        for name in modules:
            with self.subTest(bot=name):
                bot = importlib.import_module(name)
                if name == "daily_post":
                    cfg = {"wp_base_url": "https://example.com", "wp_user": "u", "wp_app_pass": "p"}
                elif name == "naver_fashion_daily_to_wp":
                    cfg = bot.cfg_from_env().wordpress
                elif name == "trend_keywords_daily_to_wp":
                    cfg = bot.load_configs()[0]
                else:
                    cfg = bot.load_cfg().wp
                options = {"cfg": cfg, "wp": cfg, "title": "테스트 레시피", "slug": "test-day", "html": "<p>본문</p>", "html_body": "<p>본문</p>", "content_html": "<p>본문</p>", "excerpt": "요약", "tag_ids": [], "featured_media": 0}
                kwargs = {key: options[key] for key in inspect.signature(bot.wp_create_post).parameters}
                with patch.object(bot, "write_post", return_value={"id": 7, "link": "url"}) as write:
                    bot.wp_create_post(**kwargs)
                payload = write.call_args.args[2]
                self.assertEqual(payload["slug"], "test-day")
                self.assertEqual(payload["content"], "<p>본문</p>")

    def test_import_all_bots(self):
        for name in ["daily_post", "daily_recipe_to_wp", "daily_recipe_to_wp_naverstyle_FINAL", "daily_korean_recipe_to_wp", "daily_issue_keywords_to_wp", "community_hotdeal_top20_to_wp", "daily_animal_media_to_wp", "naver_fashion_daily_to_wp", "hot_keyword_naver_shop_to_wp", "trend_keywords_daily_to_wp"]:
            importlib.import_module(name)

    def test_standard_dry_run_never_writes_or_uploads(self):
        bot = importlib.import_module("daily_recipe_to_wp")
        with tempfile.TemporaryDirectory() as directory:
            cfg = bot.load_cfg()
            cfg.sqlite_path = directory + "/test.sqlite3"
            cfg.openai.api_key = "test-key"
            cfg.run.dry_run = True
            recipe = {"id": "1", "title": "Tofu", "thumb": "https://example.com/image.jpg"}
            with patch.object(bot, "pick_recipe", return_value=recipe), patch.object(bot, "generate_korean_blog_naverish", return_value=(ARTICLE["title"], render_recipe(ARTICLE))), patch.object(bot, "wp_upload_media") as upload, patch.object(bot, "wp_find_post_by_slug") as lookup, patch("requests.post") as post, patch.object(bot, "save_preview"):
                with redirect_stdout(StringIO()):
                    bot.run(cfg)
                upload.assert_not_called()
                post.assert_not_called()
                lookup.assert_not_called()

    def test_final_dry_run_never_writes_or_uploads(self):
        bot = importlib.import_module("daily_recipe_to_wp_naverstyle_FINAL")
        with tempfile.TemporaryDirectory() as directory:
            cfg = bot.load_cfg()
            cfg.sqlite_path = directory + "/test.sqlite3"
            cfg.openai.api_key = "test-key"
            cfg.run.dry_run = True
            recipe = {"id": "1", "title": "Tofu", "thumb": "https://example.com/image.jpg", "ingredients": [{"name": "Tofu", "measure": "150g"}], "instructions": "Cook tofu."}
            with patch.object(bot, "OpenAI"), patch.object(bot, "pick_recipe", return_value=recipe), patch.object(bot, "generate_recipe_article", return_value=ARTICLE), patch.object(bot, "wp_upload_media") as upload, patch("requests.post") as post, patch.object(bot, "save_preview"), patch.object(bot, "remember_editorial") as remember:
                with redirect_stdout(StringIO()):
                    bot.run(cfg)
                upload.assert_not_called()
                post.assert_not_called()
                remember.assert_not_called()

    def test_editorial_history_is_recorded_only_after_publish_success(self):
        bot = importlib.import_module("daily_recipe_to_wp_naverstyle_FINAL")
        for failure in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                cfg = bot.load_cfg()
                cfg.sqlite_path = directory + "/test.sqlite3"
                cfg.run.dry_run = False
                cfg.run.upload_thumb = False
                recipe = {"id": "1", "title": "Tofu", "ingredients": [{"name": "Tofu", "measure": "150g"}], "instructions": "Cook tofu."}
                with patch.object(bot, "find_post", return_value=None), patch.object(bot, "OpenAI"), patch.object(bot, "pick_recipe", return_value=recipe), patch.object(bot, "generate_recipe_article", return_value=ARTICLE), patch.object(bot, "save_preview"), patch.object(bot, "wp_create_post", side_effect=WordPressError("publish failed") if failure else None, return_value=(7, "url")):
                    with redirect_stdout(StringIO()):
                        if failure:
                            with self.assertRaises(WordPressError):
                                bot.run(cfg)
                        else:
                            bot.run(cfg)
                    self.assertEqual(recent_editorials(cfg.sqlite_path), [] if failure else [{"title": ARTICLE["title"], "intro": ARTICLE["intro"]}])

    def test_headline_duplicates_do_not_inflate_mentions(self):
        bot = importlib.import_module("daily_issue_keywords_to_wp")
        cfg = bot.load_cfg()
        now = datetime.now(bot.KST)
        items = [bot.FeedItem("반도체 수출 증가", "https://example.com/a", "뉴스", now)] * 5
        self.assertEqual(bot.score_keywords(cfg, items, []), [])

    def test_datalab_normalizes_two_different_batch_scales(self):
        bot = importlib.import_module("trend_keywords_daily_to_wp")
        now = datetime(2026, 10, 2, tzinfo=timezone(timedelta(hours=9)))
        def group(title, prev, last):
            return {"title": title, "data": [{"period": "2026-09-30", "ratio": prev}, {"period": "2026-10-01", "ratio": last}]}
        first = response({"results": [group("기준", 10, 20), group("가", 20, 100), group("나", 10, 30), group("다", 10, 25), group("라", 15, 25)]})
        second = response({"results": [group("기준", 50, 100), group("마", 5, 50)]})
        session = Mock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        session.post.side_effect = [first, second]
        cfg = bot.NaverCfg(client_id="id", client_secret="secret", keyword_pool=["기준", "가", "나", "다", "라", "마"], pool_limit=6, lookback_days=7)
        with patch.object(bot, "now_kst", return_value=now), patch.object(bot.requests, "Session", return_value=session):
            items = bot.fetch_naver_datalab_rank(cfg, 10)
        lookup = {x["keyword"]: x for x in items}
        self.assertAlmostEqual(lookup["가"]["last"], 500)
        self.assertAlmostEqual(lookup["마"]["last"], 50)
        self.assertEqual(len(lookup), 6)
        self.assertEqual(items[0]["keyword"], "가")
        for call in session.post.call_args_list:
            self.assertEqual(call.kwargs["json"]["keywordGroups"][0]["groupName"], "기준")

    def test_fashion_uses_actual_count_and_search_label(self):
        bot = importlib.import_module("naver_fashion_daily_to_wp")
        body = bot.build_post_html("2026-10-02", bot.PostConfig(), [], [], "sim")
        self.assertIn("여성의류 검색 결과 0개", body)
        self.assertNotIn("TOP 20", body)

    def test_new_trend_has_no_fabricated_change(self):
        bot = importlib.import_module("daily_issue_keywords_to_wp")
        body = bot.build_trends_delta_table([{"keyword": "새 소식", "traffic_num": 1000, "traffic": "1K+"}], [])
        self.assertNotIn("+1,000", body)


if __name__ == "__main__":
    unittest.main()
