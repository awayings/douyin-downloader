from .base import BaseTranscriptProvider


class OpenAIProvider(BaseTranscriptProvider):
    async def transcribe(self, *, video_path, model, api_key, api_url, prompt="", language_hint=""):
        raise NotImplementedError
