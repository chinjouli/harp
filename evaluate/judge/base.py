from __future__ import annotations

from abc import ABC, abstractmethod


class JudgeBase(ABC):
    @abstractmethod
    def __call__(self, prompt: str, **kwargs) -> str: ...
