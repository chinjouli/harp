from __future__ import annotations

from typing import Any, Optional

from evaluate.judge.base import JudgeBase
from inference.llm.qwen_text import QwenText


class QwenJudge(JudgeBase):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> None:
        self._llm = QwenText(
            model_id=model_id,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self._llm(prompt, **kwargs)
