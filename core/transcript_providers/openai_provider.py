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
