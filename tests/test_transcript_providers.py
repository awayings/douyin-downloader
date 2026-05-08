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
