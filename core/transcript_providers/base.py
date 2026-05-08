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
