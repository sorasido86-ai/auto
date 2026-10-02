import copy
import importlib
import inspect
from contextlib import redirect_stdout
from io import StringIO
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests
from content_quality import (ContentQualityError, generate_recipe_article,
                             render_recipe, safe_url, validate_article)
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
    def test_valid_source_translation(self):
        self.assertEqual(validate_article(ARTICLE, INGREDIENTS, STEPS), ARTICLE)

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
        call = Mock(side_effect=[SimpleNamespace(output_text=json.dumps(invalid)), SimpleNamespace(output_text=json.dumps(ARTICLE))])
        self.assertEqual(generate_recipe_article(call, "Tofu", INGREDIENTS, STEPS), ARTICLE)
        self.assertEqual(call.call_count, 2)
        for args, _ in call.call_args_list:
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
        body, excerpt = bot.build_body_html(cfg, recipe, "", None)
        self.assertIn("1.5큰술", body)
        self.assertIn("5분 끓인다.", body)
        self.assertNotIn("저는", body)
        self.assertNotIn("중불", body)
        self.assertIn("<ol>", body)


class PublishingTests(unittest.TestCase):
    endpoint = "https://example.com/wp-json/wp/v2/posts"
    payload = {"slug": "recipe-day", "title": "두부 조림", "content": "<p>본문</p>", "status": "publish"}

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
            with patch.object(bot, "OpenAI"), patch.object(bot, "pick_recipe", return_value=recipe), patch.object(bot, "generate_recipe_article", return_value=ARTICLE), patch.object(bot, "wp_upload_media") as upload, patch("requests.post") as post, patch.object(bot, "save_preview"):
                with redirect_stdout(StringIO()):
                    bot.run(cfg)
                upload.assert_not_called()
                post.assert_not_called()

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
