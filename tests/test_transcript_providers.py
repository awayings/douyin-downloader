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


from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


def _mock_aiohttp_session(response_payload, status=200):
    """Return a mock aiohttp ClientSession context manager."""
    response = AsyncMock()
    response.status = status
    response.json.return_value = response_payload
    response.text.return_value = "error"

    post_ctx = AsyncMock()
    post_ctx.__aenter__.return_value = response
    post_ctx.__aexit__.return_value = False

    session = MagicMock()
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
