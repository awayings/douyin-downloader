# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based Douyin (TikTok China) batch downloader. Fetches videos, galleries, music, and user content without watermarks. Supports concurrent downloads with rate limiting, cookie-based authentication, and optional Whisper transcription. CLI-driven with YAML configuration.

## Common Commands

### Activate the Python environment
The project uses a local virtualenv at `.venv/`. Activate it before running any command below; otherwise `python`/`pip` will resolve to the system interpreter and miss this project's pinned dependencies.
```bash
source .venv/bin/activate          # macOS / Linux (zsh, bash)
# .venv\Scripts\activate           # Windows PowerShell

# First time only — create the venv if `.venv/` is missing:
python3 -m venv .venv && source .venv/bin/activate
```

### Run the downloader
```bash
python run.py -c config.yml
```

### Run with CLI overrides
```bash
python run.py -c config.yml -u "https://www.douyin.com/video/7604129988555574538" -t 8 -p ./Downloaded
```

### Run tests
```bash
python -m pytest tests/           # all tests
python -m pytest tests/test_<module>.py -v   # single test module
```

### Lint
```bash
ruff check .
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Fetch cookies (requires `playwright`)
```bash
pip install playwright
python -m playwright install chromium
python -m tools.cookie_fetcher --config config.yml
```

## High-Level Architecture

### Entry Point and Orchestration

`run.py` bootstraps `sys.path` and delegates to `cli.main:main()`. The main async loop (`main_async`) iterates over URLs from config or CLI, calling `download_url()` for each. `download_url()` is the per-URL pipeline:

1. Resolve short URL (`api_client.resolve_short_url()`)
2. Parse URL type (`URLParser.parse()`)
3. Factory dispatch (`DownloaderFactory.create()`)
4. Download and record history

All I/O is async using `aiohttp`, `aiofiles`, and `aiosqlite`.

### Downloader Factory Pattern

`DownloaderFactory.create(url_type, ...)` maps URL type strings to concrete downloader instances. All downloaders share an identical constructor signature:

```python
BaseDownloader(
    config, api_client, file_manager, cookie_manager,
    database, rate_limiter, retry_handler, queue_manager, progress_reporter
)
```

Supported URL types: `video`, `user`, `gallery`, `collection`, `music`, `live`. Short URLs are resolved before dispatch.

### User Download Strategy Pattern

`UserDownloader` does not handle modes directly. It delegates to mode strategies discovered by `UserModeRegistry`, which auto-scans `core/user_modes/` for `BaseUserModeStrategy` subclasses. Each strategy sets `mode_name` and `api_method_name` class attributes:

- `post_strategy.py` — published posts
- `like_strategy.py` — liked videos
- `mix_strategy.py` — mixes/collections (two-level fetch: metadata → aweme expansion)
- `music_strategy.py` — music content (two-level fetch)
- `collect_strategy.py` / `collect_mix_strategy.py` — current-login favorites

New modes: create a new `*_strategy.py` inheriting `BaseUserModeStrategy`, set `mode_name` and `api_method_name`, and it will be auto-discovered.

### Config System

`ConfigLoader` merges configuration in this order: `default_config.py` defaults → YAML file → environment variables (`DOUYIN_*` prefix). Key behaviors:

- The `mix`/`allmix` alias system keeps both keys in sync across `number` and `increase` sections. `number.allmix` / `increase.allmix` are compatibility aliases normalized at runtime to `mix`.
- Cookie resolution order: explicit string → dict → `"auto"` keyword → `auto_cookie` flag → fallback JSON files.
- `validate()` coerces types and clears invalid date formats.

### Anti-Bot Request Signing

API calls are signed by `utils/xbogus.py` (X-Bogus) and `utils/abogus.py` (A-Bogus using SM3 crypto via `gmssl`). These are used by `core/api_client.py`. Changes to signing logic affect all API calls and require careful testing.

### Deduplication

Two layers:
1. **Database** (`storage/database.py`) — SQLite records of downloaded aweme IDs, optional but required for incremental mode.
2. **Local file scanning** (`BaseDownloader._build_local_aweme_index()`) — scans existing files for aweme ID patterns (15–20 digit numbers) in filenames.

To force re-download: delete both the local files *and* the database record. Deleting only one layer will not trigger re-download.

### Concurrency Primitives

`control/` provides three async utilities instantiated per download session:
- `RateLimiter` — token-bucket, default 2 requests/second
- `RetryHandler` — exponential backoff (1s, 2s, 5s)
- `QueueManager` — async worker pool, size from `thread` config (default 5)

### No-Watermark and Quality Selection

`BaseDownloader._build_no_watermark_url()` sorts video URL candidates to prefer `watermark=0`, signs Douyin-domain URLs, and falls back to watermarked sources only if no clean source exists. For galleries, `_collect_image_url_candidates()` prefers `origin_image`/`display_image`/`url_list` before watermark fallbacks (`download_url_list`/`owner_watermark_image`). Video quality selection picks the highest `bit_rate` entry from `video.bit_rate` array.

## Important Constraints

- Python 3.8+ compatibility — avoid walrus operator, `match` statements, and `type` aliases.
- All core I/O must be async. Never use blocking I/O in download paths.
- `collect`/`collectmix` modes only support `/user/self?showTab=favorite_collection` and cannot be mixed with `post`/`like`/`mix`/`music`.
- `increase` (incremental download) only supports `post`/`like`/`mix`/`music`;收藏夹模式不支持增量截断.
- Progress display (`cli/progress_display.py`) quiets console logs during active downloads to avoid Rich redraw corruption.

## Testing Notes

- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"` — no `@pytest.mark.asyncio` decorators needed.
- Mock external HTTP calls; never hit real Douyin API in tests.
- Use `AsyncMock` for async method mocking.
- No `conftest.py` — fixtures are defined per-module.
