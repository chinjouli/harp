from __future__ import annotations

import io
import json
import time
from typing import Any

from evaluate.judge.base import JudgeBase


class OpenAIJudge(JudgeBase):
    def __init__(
        self,
        model_id: str = "gpt-4o",
        max_tokens: int = 256,
    ) -> None:
        from openai import OpenAI
        self._client = OpenAI()
        self.model_id = model_id
        self.max_tokens = max_tokens

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        max_tok = kwargs.get("max_tokens", self.max_tokens)
        resp = self._client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=max_tok,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    def batch(
        self, prompts: list[str], **kwargs: Any
    ) -> list[str]:
        """Submit all prompts as a single OpenAI batch job and wait for results."""
        max_tok = kwargs.get("max_tokens", self.max_tokens)

        # Build JSONL request file in memory
        lines = []
        for i, prompt in enumerate(prompts):
            lines.append(json.dumps({
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_id,
                    "max_completion_tokens": max_tok,
                    "messages": [{"role": "user", "content": prompt}],
                },
            }))
        content = "\n".join(lines).encode()

        # Upload and submit
        file_obj = self._client.files.create(
            file=("batch.jsonl", io.BytesIO(content), "application/jsonl"),
            purpose="batch",
        )
        batch_job = self._client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # Poll until complete
        poll_interval = 30
        while True:
            job = self._client.batches.retrieve(batch_job.id)
            if job.status in ("completed", "failed", "cancelled", "expired"):
                break
            time.sleep(poll_interval)

        if job.status != "completed":
            raise RuntimeError(f"Batch job {batch_job.id} ended with status {job.status}")

        # Download and parse results
        result_bytes = self._client.files.content(job.output_file_id).read()
        results: dict[int, str] = {}
        for line in result_bytes.decode().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            idx = int(r["custom_id"])
            content = (
                r.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content") or ""
            )
            results[idx] = content.strip()

        return [results.get(i, "") for i in range(len(prompts))]
