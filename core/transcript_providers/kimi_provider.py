from .base import BaseTranscriptProvider


class KimiProvider(BaseTranscriptProvider):
    async def transcribe(self, *, video_path, model, api_key, api_url, prompt="", language_hint=""):
        raise NotImplementedError
