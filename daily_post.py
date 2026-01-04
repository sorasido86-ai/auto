name: run-daily-post

on:
  workflow_dispatch:
  schedule:
    - cron: "*/5 * * * *"

jobs:
  post:
    runs-on: ubuntu-latest
    concurrency:
      group: run-daily-post
      cancel-in-progress: true

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Create bot_config.json from Secrets
        env:
          WP_BASE_URL: ${{ secrets.WP_BASE_URL }}
          WP_USER: ${{ secrets.WP_USER }}
          WP_APP_PASS: ${{ secrets.WP_APP_PASS }}
          KAKAO_REST_KEY: ${{ secrets.KAKAO_REST_KEY }}
          KAKAO_REFRESH_TOKEN: ${{ secrets.KAKAO_REFRESH_TOKEN }}
        run: |
          python - <<'PY'
          import os, json, sys

          base = (os.getenv("WP_BASE_URL") or "").strip()
          user = (os.getenv("WP_USER") or "").strip()
          app  = (os.getenv("WP_APP_PASS") or "").replace(" ", "").strip()

          missing = [k for k,v in [("WP_BASE_URL",base),("WP_USER",user),("WP_APP_PASS",app)] if not v]
          if missing:
            sys.exit("Missing secrets: " + ", ".join(missing))

          cfg = {"wp_base_url": base, "wp_user": user, "wp_app_pass": app}

          if os.getenv("KAKAO_REST_KEY") and os.getenv("KAKAO_REFRESH_TOKEN"):
            cfg["kakao_rest_key"] = os.environ["KAKAO_REST_KEY"]
            cfg["kakao_refresh_token"] = os.environ["KAKAO_REFRESH_TOKEN"]

          with open("bot_config.json","w",encoding="utf-8") as f:
            json.dump(cfg,f,ensure_ascii=False,indent=2)

          print("bot_config.json written:", ", ".join(cfg.keys()))
          PY

      - name: Run
        env:
          TZ: Asia/Seoul
          TEST_MODE: "1"          # ✅ 테스트 끝나면 "0" 또는 줄 삭제
          WP_POST_STATUS: "draft" # ✅ 테스트는 draft 추천 (스팸 방지)
        run: |
          echo "event=${{ github.event_name }} actor=${{ github.actor }} time=$(date -Is)"
          python -u daily_post.py
