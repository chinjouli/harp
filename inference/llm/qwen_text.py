from __future__ import annotations

from typing import Any, Dict, Optional

from inference.llm.base import LLMBase


class QwenText(LLMBase):
    """Qwen text LM: vLLM OpenAI-compatible server or local HF model.

    Args:
        model_id:    HuggingFace model ID.
        base_url:    vLLM server URL, e.g. "http://localhost:8000/v1".
                     None → load model locally.
        temperature: Sampling temperature.
        max_tokens:  Max generation tokens.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        if base_url:
            from openai import OpenAI
            self._client: Optional[OpenAI] = OpenAI(
                base_url=base_url, api_key="EMPTY"
            )
            self._tokenizer = None
            self._model = None
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._client = None
            self._tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        if self._client is not None:
            return self._call_server(messages, **kwargs)
        return self._call_local(messages, **kwargs)

    def _call_server(
        self, messages: list, **kwargs: Any
    ) -> str:
        schema = kwargs.pop("guided_json_schema", None)
        extra: Optional[Dict[str, Any]] = (
            {"guided_json": schema} if schema else None
        )
        resp = self._client.chat.completions.create(  # type: ignore
            model=self.model_id,
            messages=messages,
            temperature=kwargs.pop("temperature", self.temperature),
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            extra_body=extra,
        )
        return (resp.choices[0].message.content or "").strip()

    def _call_local(self, messages: list, **kwargs: Any) -> str:
        import torch
        text = self._tokenizer.apply_chat_template(  # type: ignore
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(  # type: ignore
            text, return_tensors="pt"
        ).to(self._model.device)  # type: ignore
        out = self._model.generate(  # type: ignore
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        out = out[:, inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(  # type: ignore
            out[0], skip_special_tokens=True
        ).strip()
