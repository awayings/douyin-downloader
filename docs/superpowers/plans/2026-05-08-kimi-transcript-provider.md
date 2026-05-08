# Kimi Transcript Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract existing OpenAI Whisper transcript logic into a provider strategy pattern, and add a Kimi (Moonshot) multimodal video transcript provider that auto-detects via `api_url`.

**Architecture:** Two provider classes (`OpenAIProvider`, `KimiProvider`) implement `BaseTranscriptProvider.transcribe()`, returned by a registry that inspects `api_url`. `TranscriptManager` delegates API calls to the resolved provider. Kimi uses a two-step flow: upload video to `/files`, then reference the returned `file_id` via `ms://{file_id}` in a chat completions request.

**Tech Stack:** Python 3.8+, aiohttp, pytest-asyncio, pathlib

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `core/transcript_providers/base.py` | Create | `BaseTranscriptProvider` ABC |
| `core/transcript_providers/registry.py` | Create | `resolve_provider(api_url)` factory |
| `core/transcript_providers/openai_provider.py` | Create | Whisper `multipart/form-data` implementation |
| `core/transcript_providers/kimi_provider.py` | Create | Upload + chat completions implementation |
| `core/transcript_providers/__init__.py` | Create | Package exports |
| `core/transcript_manager.py` | Modify | Remove `_call_openai_transcription` / `_guess_video_content_type`; wire provider dispatch |
| `config/default_config.py` | Modify | Add `prompt` field to `transcript` dict |
| `config.yml` | Modify | Add `prompt` under `transcript` |
| `config.example.yml` | Modify | Add `prompt` under `transcript` |
| `tests/test_transcript_providers.py` | Create | Provider + registry unit tests |

---

### Task 1: Base Provider + Registry

**Files:**
- Create: `core/transcript_providers/base.py`
- Create: `core/transcript_providers/registry.py`
- Test: `tests/test_transcript_providers.py`

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_transcript_providers.py` with the registry tests:

```python
import pytest
from core.transcript_providers.registry import resolve_provider
from core.transcript_providers.kimi_provider import KimiProvider
from core.transcript_providers.openai_provider import OpenAIProvider


def test_registry_detects_kimi():
    provider = resolve_provider("https://api.moonshot.cn/v1/chat/completions")
    assert isinstance(provider, KimiProvider)


def test_registry_detects_openai():
    provider = resolve_provider("https://api.openai.com/v1/audio/transcriptions")
    assert isinstance(provider, OpenAIProvider)


def test_registry_defaults_to_openai():
    provider = resolve_provider("https://custom-proxy.example.com/v1/audio/transcriptions")
    assert isinstance(provider, OpenAIProvider)
```

- [ ] **Step 2: Run failing tests**

```bash
source .venv/bin/activate && pytest tests/test_transcript_providers.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` for `core.transcript_providers.registry`.

- [ ] **Step 3: Implement base.py and registry.py**

Create `core/transcript_providers/base.py`:

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseTranscriptProvider(ABC):
    @abstractmethod
    async def transcribe(
        self,
        *,
        video_path: Path,
        model: str,
        api_key: str,
        api_url: str,
        prompt: str = "",
        language_hint: str = "",
    ) -> Dict[str, Any]:
        """Return standardized dict with 'text' and 'raw_response'."""
```

Create `core/transcript_providers/registry.py`:

```python
from .base import BaseTranscriptProvider
from .kimi_provider import KimiProvider
from .openai_provider import OpenAIProvider


def resolve_provider(api_url: str) -> BaseTranscriptProvider:
    url = api_url.lower()
    if "moonshot.cn" in url:
        return KimiProvider()
    return OpenAIProvider()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_transcript_providers.py::test_registry_detects_kimi tests/test_transcript_providers.py::test_registry_detects_openai tests/test_transcript_providers.py::test_registry_defaults_to_openai -v
```

Expected: 3 passes.

- [ ] **Step 5: Commit**

```bash
git add core/transcript_providers/base.py core/transcript_providers/registry.py tests/test_transcript_providers.py
git commit -m "feat(transcript): add BaseTranscriptProvider and registry"
```

---

### Task 2: OpenAI Provider

**Files:**
- Create: `core/transcript_providers/openai_provider.py`
- Modify: `tests/test_transcript_providers.py`

- [ ] **Step 1: Write failing OpenAI provider test**

Append to `tests/test_transcript_providers.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock

from core.transcript_providers.openai_provider import OpenAIProvider


def _mock_aiohttp_session(response_payload, status=200):
    """Return a mock aiohttp ClientSession context manager."""
    response = AsyncMock()
    response.status = status
    response.json.return_value = response_payload
    response.text.return_value = "error"

    post_ctx = AsyncMock()
    post_ctx.__aenter__.return_value = response
    post_ctx.__aexit__.return_value = False

    session = AsyncMock()
    session.post.return_value = post_ctx

    session_ctx = AsyncMock()
    session_ctx.__aenter__.return_value = session
    session_ctx.__aexit__.return_value = False

    return session_ctx


async def test_openai_provider_normalizes_response(tmp_path, monkeypatch):
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"fake video")

    mock_session = _mock_aiohttp_session({"text": "Hello world", "duration": 1.5})
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: mock_session)

    provider = OpenAIProvider()
    result = await provider.transcribe(
        video_path=video_path,
        model="whisper-1",
        api_key="test-key",
        api_url="https://api.openai.com/v1/audio/transcriptions",
    )

    assert result["text"] == "Hello world"
    assert result["raw_response"]["text"] == "Hello world"
```

- [ ] **Step 2: Run failing test**

```bash
source .venv/bin/activate && pytest tests/test_transcript_providers.py::test_openai_provider_normalizes_response -v
```

Expected: `ImportError` for `core.transcript_providers.openai_provider` or `NameError`.

- [ ] **Step 3: Implement openai_provider.py**

Create `core/transcript_providers/openai_provider.py`:

```python
import aiohttp
from pathlib import Path
from typing import Any, Dict

from .base import BaseTranscriptProvider


class OpenAIProvider(BaseTranscriptProvider):
    async def transcribe(
        self,
        *,
        video_path: Path,
        model: str,
        api_key: str,
        api_url: str,
        prompt: str = "",
        language_hint: str = "",
    ) -> Dict[str, Any]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        form = aiohttp.FormData()
        form.add_field("model", model)
        form.add_field("response_format", "json")
        if language_hint:
            form.add_field("language", language_hint)

        content_type = self._guess_video_content_type(video_path)
        with video_path.open("rb") as f:
            form.add_field(
                "file",
                f,
                filename=video_path.name,
                content_type=content_type,
            )
            timeout = aiohttp.ClientTimeout(total=600)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    api_url,
                    data=form,
                    headers={"Authorization": f"Bearer {api_key}"},
                ) as response:
                    if response.status != 200:
                        body = await response.text()
                        raise RuntimeError(
                            f"OpenAI transcription failed: status={response.status}, body={body}"
                        )

                    payload = await response.json(content_type=None)
                    if not isinstance(payload, dict):
                        raise RuntimeError("OpenAI transcription returned invalid payload")
                    return {
                        "text": str(payload.get("text", "")),
                        "raw_response": payload,
                    }

    @staticmethod
    def _guess_video_content_type(video_path: Path) -> str:
        suffix = video_path.suffix.lower()
        if suffix == ".mp4":
            return "video/mp4"
        if suffix == ".m4a":
            return "audio/mp4"
        if suffix == ".wav":
            return "audio/wav"
        if suffix == ".mp3":
            return "audio/mpeg"
        return "application/octet-stream"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source .venv/bin/activate && pytest tests/test_transcript_providers.py::test_openai_provider_normalizes_response -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/transcript_providers/openai_provider.py tests/test_transcript_providers.py
git commit -m "feat(transcript): add OpenAI Whisper provider"
```

---

### Task 3: Kimi Provider

**Files:**
- Create: `core/transcript_providers/kimi_provider.py`
- Modify: `tests/test_transcript_providers.py`

- [ ] **Step 1: Write failing Kimi provider tests**

Append to `tests/test_transcript_providers.py`:

```python
from core.transcript_providers.kimi_provider import KimiProvider


def _mock_aiohttp_session_sequence(response_sequence):
    """Return a mock aiohttp ClientSession for sequential POST calls."""
    contexts = []
    for payload, status in response_sequence:
        response = AsyncMock()
        response.status = status
        response.json.return_value = payload
        response.text.return_value = "error"

        ctx = AsyncMock()
        ctx.__aenter__.return_value = response
        ctx.__aexit__.return_value = False
        contexts.append(ctx)

    session = AsyncMock()
    session.post.side_effect = contexts

    session_ctx = AsyncMock()
    session_ctx.__aenter__.return_value = session
    session_ctx.__aexit__.return_value = False

    return session_ctx


async def test_kimi_provider_uploads_file(tmp_path, monkeypatch):
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"fake video")

    mock_session = _mock_aiohttp_session_sequence([
        ({"id": "file-test-123"}, 200),
        ({"choices": [{"message": {"content": "transcribed"}}]}, 200),
    ])
    monkeypatch.setattr("aiohttp.ClientSession", lambda **kwargs: mock_session)

    provider = KimiProvider()
    result = await provider.transcribe(
        video_path=video_path,
        model="kimi-k2.6",
        api_key="test-key",
        api_url="https://api.moonshot.cn/v1/chat/completions",
    )

    assert result["text"] == "transcribed"
    assert result["raw_response"]["choices"][0]["message"]["content"] == "transcribed"


async def test_kimi_provider_uses_custom_prompt(tmp_path, monkeypatch):
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"fake video")

    captured = {}

    original_post = None

    class _MockSession:
        def __init__(self, **kwargs):
            self._idx = 0
            self._responses = [
                AsyncMock(),
                AsyncMock(),
            ]
            self._responses[0].status = 200
            self._responses[0].json.return_value = {"id": "file-123"}
            self._responses[1].status = 200
            self._responses[1].json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if self._idx == 1:
                captured["json_body"] = kwargs.get("json")
            response = self._responses[self._idx]
            self._idx += 1
            ctx = AsyncMock()
            ctx.__aenter__.return_value = response
            ctx.__aexit__.return_value = False
            return ctx

    monkeypatch.setattr("aiohttp.ClientSession", _MockSession)

    provider = KimiProvider()
    await provider.transcribe(
        video_path=video_path,
        model="kimi-k2.6",
        api_key="test-key",
        api_url="https://api.moonshot.cn/v1/chat/completions",
        prompt="Custom prompt",
    )

    messages = captured["json_body"]["messages"]
    text_parts = [c for c in messages[0]["content"] if c["type"] == "text"]
    assert text_parts[0]["text"] == "Custom prompt"
```

- [ ] **Step 2: Run failing tests**

```bash
source .venv/bin/activate && pytest tests/test_transcript_providers.py::test_kimi_provider_uploads_file tests/test_transcript_providers.py::test_kimi_provider_uses_custom_prompt -v
```

Expected: `ImportError` for `core.transcript_providers.kimi_provider`.

- [ ] **Step 3: Implement kimi_provider.py**

Create `core/transcript_providers/kimi_provider.py`:

```python
import aiohttp
from pathlib import Path
from typing import Any, Dict

from .base import BaseTranscriptProvider


class KimiProvider(BaseTranscriptProvider):
    async def transcribe(
        self,
        *,
        video_path: Path,
        model: str,
        api_key: str,
        api_url: str,
        prompt: str = "",
        language_hint: str = "",
    ) -> Dict[str, Any]:
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        file_id = await self._upload_file(video_path, api_key, api_url)
        return await self._call_completions(file_id, model, api_key, api_url, prompt)

    async def _upload_file(
        self, video_path: Path, api_key: str, api_url: str
    ) -> str:
        files_url = self._derive_files_url(api_url)

        form = aiohttp.FormData()
        form.add_field("purpose", "video")
        content_type = self._guess_video_content_type(video_path)
        with video_path.open("rb") as f:
            form.add_field(
                "file",
                f,
                filename=video_path.name,
                content_type=content_type,
            )
            timeout = aiohttp.ClientTimeout(total=600)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    files_url,
                    data=form,
                    headers={"Authorization": f"Bearer {api_key}"},
                ) as response:
                    if response.status != 200:
                        body = await response.text()
                        raise RuntimeError(
                            f"Kimi file upload failed: status={response.status}, body={body}"
                        )

                    payload = await response.json(content_type=None)
                    if not isinstance(payload, dict):
                        raise RuntimeError("Kimi file upload returned invalid payload")
                    file_id = payload.get("id")
                    if not file_id:
                        raise RuntimeError("Kimi file upload response missing 'id'")
                    return str(file_id)

    async def _call_completions(
        self,
        file_id: str,
        model: str,
        api_key: str,
        api_url: str,
        prompt: str,
    ) -> Dict[str, Any]:
        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"ms://{file_id}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt or "请将这段视频的语音内容转录成文字。",
                        },
                    ],
                }
            ],
        }

        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                api_url,
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    body_text = await response.text()
                    raise RuntimeError(
                        f"Kimi transcription failed: status={response.status}, body={body_text}"
                    )

                payload = await response.json(content_type=None)
                if not isinstance(payload, dict):
                    raise RuntimeError("Kimi transcription returned invalid payload")

                choices = payload.get("choices")
                if not choices or not isinstance(choices, list):
                    raise RuntimeError("Kimi transcription response missing 'choices'")

                content = choices[0].get("message", {}).get("content", "")
                return {
                    "text": str(content),
                    "raw_response": payload,
                }

    @staticmethod
    def _derive_files_url(api_url: str) -> str:
        # e.g. https://api.moonshot.cn/v1/chat/completions -> https://api.moonshot.cn/v1/files
        base = api_url.rstrip("/").rsplit("/", 1)[0]
        return f"{base}/files"

    @staticmethod
    def _guess_video_content_type(video_path: Path) -> str:
        suffix = video_path.suffix.lower()
        if suffix == ".mp4":
            return "video/mp4"
        if suffix == ".mov":
            return "video/quicktime"
        if suffix == ".avi":
            return "video/x-msvideo"
        if suffix == ".webm":
            return "video/webm"
        if suffix == ".wmv":
            return "video/x-ms-wmv"
        if suffix == ".mpg" or suffix == ".mpeg":
            return "video/mpeg"
        return "video/mp4"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source .venv/bin/activate && pytest tests/test_transcript_providers.py::test_kimi_provider_uploads_file tests/test_transcript_providers.py::test_kimi_provider_uses_custom_prompt -v
```

Expected: 2 passes.

- [ ] **Step 5: Commit**

```bash
git add core/transcript_providers/kimi_provider.py tests/test_transcript_providers.py
git commit -m "feat(transcript): add Kimi multimodal provider with upload + completions"
```

---

### Task 4: Package `__init__.py`

**Files:**
- Create: `core/transcript_providers/__init__.py`

- [ ] **Step 1: Write `__init__.py`**

Create `core/transcript_providers/__init__.py`:

```python
from .base import BaseTranscriptProvider
from .kimi_provider import KimiProvider
from .openai_provider import OpenAIProvider
from .registry import resolve_provider

__all__ = [
    "BaseTranscriptProvider",
    "KimiProvider",
    "OpenAIProvider",
    "resolve_provider",
]
```

- [ ] **Step 2: Verify imports work**

```bash
source .venv/bin/activate && python -c "from core.transcript_providers import resolve_provider, KimiProvider, OpenAIProvider; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add core/transcript_providers/__init__.py
git commit -m "feat(transcript): add transcript_providers package exports"
```

---

### Task 5: Refactor TranscriptManager

**Files:**
- Modify: `core/transcript_manager.py`

- [ ] **Step 1: Add provider import and update `process_video`**

In `core/transcript_manager.py`, add after existing imports:

```python
from core.transcript_providers.registry import resolve_provider
```

In `process_video` (around line 118), replace:

```python
            payload = await self._call_openai_transcription(
                api_key=api_key,
                video_path=video_path,
                model=model,
            )
```

with:

```python
            provider = resolve_provider(self._api_url())
            payload = await provider.transcribe(
                video_path=video_path,
                model=model,
                api_key=api_key,
                api_url=self._api_url(),
                prompt=self._cfg().get("prompt", ""),
                language_hint=str(self._cfg().get("language_hint", "")).strip(),
            )
```

- [ ] **Step 2: Remove old methods**

Delete the `_call_openai_transcription` method (lines 176-216) and the `_guess_video_content_type` static method (lines 218-229) from `core/transcript_manager.py`.

- [ ] **Step 3: Run existing transcript tests**

```bash
source .venv/bin/activate && pytest tests/test_transcript_manager.py -v
```

Expected: All 5 existing tests pass (they don't exercise the API call path).

- [ ] **Step 4: Run new provider tests**

```bash
source .venv/bin/activate && pytest tests/test_transcript_providers.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add core/transcript_manager.py
git commit -m "refactor(transcript): delegate API calls to provider strategy"
```

---

### Task 6: Config Updates

**Files:**
- Modify: `config/default_config.py`
- Modify: `config.yml`
- Modify: `config.example.yml`
- Test: `tests/test_transcript_manager.py`

- [ ] **Step 1: Update `config/default_config.py`**

Add `"prompt": "请将这段视频的语音内容转录成文字。"` to the `transcript` dict at line 38-46:

```python
    "transcript": {
        "enabled": False,
        "model": "gpt-4o-mini-transcribe",
        "output_dir": "",
        "response_formats": ["txt", "json"],
        "api_url": "https://api.openai.com/v1/audio/transcriptions",
        "api_key_env": "OPENAI_API_KEY",
        "api_key": "",
        "prompt": "请将这段视频的语音内容转录成文字。",
    },
```

- [ ] **Step 2: Update `config.yml` and `config.example.yml`**

In both files, under `transcript:`, add after `api_key: ''`:

```yaml
  prompt: "请将这段视频的语音内容转录成文字。"
```

- [ ] **Step 3: Write config loading test**

Append to `tests/test_transcript_manager.py`:

```python
def test_transcript_config_includes_prompt():
    loader = ConfigLoader()
    transcript_cfg = loader.get("transcript", {})
    assert "prompt" in transcript_cfg
    assert transcript_cfg["prompt"] == "请将这段视频的语音内容转录成文字。"
```

- [ ] **Step 4: Run tests**

```bash
source .venv/bin/activate && pytest tests/test_transcript_manager.py -v
```

Expected: 6 passes (5 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add config/default_config.py config.yml config.example.yml tests/test_transcript_manager.py
git commit -m "feat(config): add transcript.prompt for Kimi provider"
```

---

### Task 7: Integration Verification

**Files:**
- All existing tests

- [ ] **Step 1: Run full test suite**

```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

Expected: All tests pass. No regressions.

- [ ] **Step 2: Run linter**

```bash
source .venv/bin/activate && ruff check .
```

Expected: No errors in modified/new files.

- [ ] **Step 3: Verify import paths work end-to-end**

```bash
source .venv/bin/activate && python -c "
from core.transcript_manager import TranscriptManager
from core.transcript_providers import resolve_provider, KimiProvider, OpenAIProvider
from config import ConfigLoader
from storage import FileManager
print('All imports OK')
"
```

Expected: prints `All imports OK`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test(transcript): verify full suite passes with provider refactor"
```

---

## Self-Review

### Spec Coverage Check

| Spec Requirement | Implementing Task |
|-----------------|-------------------|
| BaseTranscriptProvider ABC | Task 1 |
| Registry auto-detect by api_url | Task 1 |
| OpenAI provider (extract existing logic) | Task 2 |
| Kimi provider (upload + completions, ms://file_id) | Task 3 |
| Package `__init__.py` exports | Task 4 |
| TranscriptManager delegates to provider | Task 5 |
| Config `prompt` field | Task 6 |
| Response normalization (`text` + `raw_response`) | Task 2, 3 |
| `language_hint` forwarded to OpenAI provider | Task 5 |
| Tests for registry, both providers | Task 1, 2, 3 |
| Full test suite + lint | Task 7 |

No gaps found.

### Placeholder Scan

No TBD, TODO, "implement later", or vague requirements found. All steps contain exact file paths, code, and commands.

### Type Consistency

- `BaseTranscriptProvider.transcribe()` signature uses `video_path: Path, model: str, api_key: str, api_url: str, prompt: str = "", language_hint: str = ""` consistently across all tasks.
- `resolve_provider(api_url: str) -> BaseTranscriptProvider` signature consistent.
- `_derive_files_url` and `_guess_video_content_type` are static methods in KimiProvider, not instance methods.

Plan is clean and ready for execution.
