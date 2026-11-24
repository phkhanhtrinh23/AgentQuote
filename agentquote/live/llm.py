"""OpenAI-compatible chat client (OpenRouter, or a local vLLM endpoint for the backbones in LOCAL_BACKBONES).

Tokens are always billed at the backbone's list price, so serving a backbone locally changes measured
latency only, never token spend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import BACKBONES, env, turn_usd


@dataclass
class Reply:
    text: str
    tok_in: int
    tok_out: int
    latency: float
    usd: float
    tool_calls: list | None = None


class LLM:
    def __init__(self, backbone: str, temperature: float = 0.7, max_tokens: int = 4096, retries: int = 6):
        from openai import OpenAI
        self.b, self.temperature, self.max_tokens, self.retries = backbone, temperature, max_tokens, retries
        local = [x.strip() for x in (env("LOCAL_BACKBONES", "") or "").split(",") if x.strip()]
        if backbone in local:
            self.client = OpenAI(base_url=env("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"),
                                 api_key=env("LOCAL_LLM_API_KEY", "EMPTY"))
            self.model = env("LOCAL_MODEL_" + backbone.upper().replace("-", "_").replace(".", "_"), backbone)
        else:
            key = env("OPENROUTER_API_KEY")
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY is not set (see .env.example)")
            self.client = OpenAI(base_url=env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), api_key=key)
            self.model = BACKBONES[backbone][0] if backbone in BACKBONES else backbone

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **kw) -> Reply:
        args = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        if tools:
            args["tools"] = tools
        args.update(kw)
        delay = 2.0
        for i in range(self.retries):
            try:
                t0 = time.time()
                r = self.client.chat.completions.create(**args)
                lat = time.time() - t0
                tin, tout = (r.usage.prompt_tokens, r.usage.completion_tokens) if r.usage else (0, 0)
                msg = r.choices[0].message
                return Reply(msg.content or "", tin, tout, lat, turn_usd(self.b, tin, tout), list(msg.tool_calls or []))
            except Exception:
                if i == self.retries - 1:
                    raise
                time.sleep(delay)
                delay = min(2 * delay, 60)
        raise RuntimeError("unreachable")
