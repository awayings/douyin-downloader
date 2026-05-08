from .base import BaseTranscriptProvider
from .kimi_provider import KimiProvider
from .openai_provider import OpenAIProvider


def resolve_provider(api_url: str) -> BaseTranscriptProvider:
    url = api_url.lower()
    if "moonshot.cn" in url:
        return KimiProvider()
    return OpenAIProvider()
