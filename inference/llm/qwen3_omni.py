"""Qwen3-Omni via vLLM (OpenAI-compatible, input_audio / base64 WAV)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from openai import OpenAI

from inference.llm.base import LLMBase


class Qwen3Omni(LLMBase):
    """Qwen3-Omni served via vLLM; audio passed as base64 WAV.

    Call signature:
        llm(question, audio_b64=..., transcript=..., description=..., icl=...)
    Any of audio_b64 / transcript / description may be None to omit.
    icl is an optional one-shot example dict:
        {audio_b64?, transcript?, description?, question, answer}
    """

    def __init__(
        self,
        base_url: str,
        model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __call__(
        self,
        prompt: str,
        *,
        audio_b64: Optional[str] = None,
        transcript: Optional[str] = None,
        description: Optional[str] = None,
        icl: Optional[Dict[str, Any]] = None,
        audio_clips: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        messages = self._build_messages(
            prompt, audio_b64, transcript, description, icl,
            audio_clips=audio_clips,
        )
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return (resp.choices[0].message.content or "").strip()

    def _content(
        self,
        audio_b64: Optional[str],
        transcript: Optional[str],
        description: Optional[str],
        question: str,
        audio_clips: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Build content blocks.

        audio_clips, if provided, is a list of dicts with keys:
          "label"     — time label shown before the clip, e.g. "[30.0s-60.0s]"
          "audio_b64" — base64 WAV
          "transcript"— optional transcript for this clip
        When audio_clips is given, audio_b64/transcript are ignored.
        """
        blocks: List[Dict[str, Any]] = []

        if audio_clips:
            for clip in audio_clips:
                blocks.append(
                    {"type": "text", "text": clip["label"]}
                )
                blocks.append({
                    "type": "input_audio",
                    "input_audio": {"data": clip["audio_b64"], "format": "wav"},
                })
                if clip.get("transcript"):
                    blocks.append(
                        {"type": "text",
                         "text": f"Transcript: {clip['transcript']}"}
                    )
        elif audio_b64:
            blocks.append({
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            })

        parts: List[str] = []
        if not audio_clips and transcript:
            parts.append(f"Transcript: {transcript}")
        if description:
            parts.append(f"Description: {description}")
        parts.append(question)
        blocks.append({"type": "text", "text": "\n".join(parts)})
        return blocks

    def _build_messages(
        self,
        prompt: str,
        audio_b64: Optional[str],
        transcript: Optional[str],
        description: Optional[str],
        icl: Optional[Dict[str, Any]],
        audio_clips: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        msgs: List[Dict[str, Any]] = []
        if icl:
            msgs.append({
                "role": "user",
                "content": self._content(
                    icl.get("audio_b64"),
                    icl.get("transcript"),
                    icl.get("description"),
                    icl.get("question", prompt),
                ),
            })
            msgs.append({"role": "assistant", "content": icl["answer"]})
        msgs.append({
            "role": "user",
            "content": self._content(
                audio_b64, transcript, description, prompt,
                audio_clips=audio_clips,
            ),
        })
        return msgs
