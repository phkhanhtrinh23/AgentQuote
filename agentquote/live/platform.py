"""Executor interface the agent talks to, one implementation per metered platform."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..types import Call, Request


@dataclass
class RunResult:
    ok: bool
    observation: str
    calls: list[Call] = field(default_factory=list)
    plan: dict | None = None        # compiled executed code with the profiled cost per operator
    value: object = None
    outcome: str = ""               # failure outcome if the platform already knows it


class Platform(ABC):
    name = "platform"
    language = "python"

    @abstractmethod
    def catalogue(self, req: Request, data_option: str) -> tuple[str, list[Call]]:
        """Catalogue text a draft may read and the metered lookups it took."""

    @abstractmethod
    def run(self, req: Request, code: str, tag: str) -> RunResult: ...

    @abstractmethod
    def inspect(self, req: Request, code: str, tag: str) -> RunResult: ...

    @abstractmethod
    def system_prompt(self, req: Request, config: str) -> str: ...

    @abstractmethod
    def draft_prompt(self, req: Request, data_option: str, catalogue: str) -> str: ...

    def method_instruction(self, method: str) -> str:
        return f"Use the analysis method `{method}`." if method else ""

    def parse_answer(self, req: Request, text: str):
        return text.strip()
