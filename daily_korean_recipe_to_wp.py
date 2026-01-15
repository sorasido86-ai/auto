name: Daily Korean Recipe to WordPress

on:
  schedule:
    - cron: "15 0 * * *" # UTC 00:15 = KST 09:15
  workflow_dispatch:

permissions:
  contents: write

jobs:
  post:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install requests openai

      - name: Run bot
        env:
          WP_BASE_URL: ${{ secrets.WP_BASE_URL }}
          WP_USER: ${{ secrets.WP_USER }}
          WP_APP_PASS: ${{ secrets.WP_APP_PASS }}

          # Optional
          MFDS_API_KEY: ${{ secrets.MFDS_API_KEY }}
          DEFAULT_THUMB_URL: ${{ secrets.DEFAULT_THUMB_URL }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

          # Behavior
          WP_STATUS: "publish"
          WP_CATEGORY_IDS: "7"
          WP_TAG_IDS: ""
          SQLITE_PATH: "data/daily_korean_recipe.sqlite3"

          RUN_SLOT: "day"
          FORCE_NEW: "0"
          DRY_RUN: "0"
          DEBUG: "0"
          AVOID_REPEAT_DAYS: "90"
          MAX_TRIES: "25"

          # Image
          UPLOAD_THUMB: "1"
          SET_FEATURED: "1"
          EMBED_IMAGE_IN_BODY: "1"
          REUSE_MEDIA_BY_SEARCH: "1"

          # Naver-friendly content
          NAVER_STYLE: "1"
          SCHEMA_MODE: "comment"   # comment|script|off
          HASHTAG_COUNT: "12"
          EMBED_STEP_IMAGES: "1"
          ADD_FAQ: "1"
          ADD_INTERNAL_LINKS: "1"

          # OpenAI (optional)
          USE_OPENAI: "0"
          OPENAI_MODEL: "gpt-4.1-mini"

        run: |
          python daily_korean_recipe_to_wp.py

      - name: Commit sqlite history
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/daily_korean_recipe.sqlite3 || true
          git commit -m "Update recipe history DB" || echo "No changes"
          git push || echo "No push"
