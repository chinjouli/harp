from __future__ import annotations

import os
from typing import Any

from evaluate.judge.base import JudgeBase


class GeminiJudge(JudgeBase):
    def __init__(
        self,
        model_id: str = "gemini-3.1-flash-lite",
        max_tokens: int = 512,
    ) -> None:
        from google import genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_id = model_id
        self.max_tokens = max_tokens

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        from google.genai import types
        cfg = types.GenerateContentConfig(
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=cfg,
        )
        return resp.text.strip()
