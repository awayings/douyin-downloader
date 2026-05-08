import re
from pathlib import Path
from typing import Any, Dict

import aiohttp

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
        url = api_url.rstrip("/")
        match = re.match(r"(.+?/v\d+)(?:/.*)?", url)
        if match:
            base = match.group(1)
        else:
            base = url.rsplit("/", 1)[0]
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
