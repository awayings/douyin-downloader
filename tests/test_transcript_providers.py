from unittest.mock import AsyncMock, MagicMock

from core.transcript_providers.kimi_provider import KimiProvider
from core.transcript_providers.openai_provider import OpenAIProvider
from core.transcript_providers.registry import resolve_provider


def test_registry_detects_kimi():
    provider = resolve_provider("https://api.moonshot.cn/v1/chat/completions")
    assert isinstance(provider, KimiProvider)


def test_registry_detects_openai():
    provider = resolve_provider("https://api.openai.com/v1/audio/transcriptions")
    assert isinstance(provider, OpenAIProvider)


def test_registry_defaults_to_openai():
    provider = resolve_provider("https://custom-proxy.example.com/v1/audio/transcriptions")
    assert isinstance(provider, OpenAIProvider)


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

    session = MagicMock()
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

    responses = [
        AsyncMock(),
        AsyncMock(),
    ]
    responses[0].status = 200
    responses[0].json.return_value = {"id": "file-123"}
    responses[1].status = 200
    responses[1].json.return_value = {"choices": [{"message": {"content": "ok"}}]}

    idx = [0]  # shared mutable counter across ClientSession instances

    class _MockSession:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, url, **kwargs):
            if idx[0] == 1:
                captured["json_body"] = kwargs.get("json")
            response = responses[idx[0]]
            idx[0] += 1
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
