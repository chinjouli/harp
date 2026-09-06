from __future__ import annotations

from typing import Any

from evaluate.judge.base import JudgeBase


class ClaudeJudge(JudgeBase):
    def __init__(
        self,
        model_id: str = "claude-opus-4-7",
        max_tokens: int = 256,
    ) -> None:
        import anthropic
        self._client = anthropic.Anthropic()
        self.model_id = model_id
        self.max_tokens = max_tokens

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        msg = self._client.messages.create(
            model=self.model_id,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
