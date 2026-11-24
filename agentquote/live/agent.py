"""The iterative data agent and its session recorder.

A shared low-cost backbone drafts one plan per data option in one planning turn, without analysis calls. The
executing backbone then works with four tools:
  run(code)       executes code (an Earth Engine script or SQL); the result or error is truncated
  inspect(code)   metadata, schema or a data preview; counts as exploration
  answer(value)   submits the answer of the last successful run
  give_up(reason) stops without an answer
An attempt ends at a ``run`` call; a successful run the agent does not answer from is a rejected result. The
session stops at the first answered success, a give-up, N_max attempts or the turn cap. Every turn and call
is metered and written in the session log format.
"""
from __future__ import annotations

import json
import re

from ..adapters.base import DomainAdapter
from ..config import DRAFTER, Settings
from ..types import Attempt, Call, Request, Session, Step, Turn, split_key
from .llm import LLM, Reply
from .platform import Platform

TOOLS = [
    {"type": "function", "function": {"name": "run", "description": "Execute analysis code on the platform. "
     "Billed by the platform. Returns the result or the error.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "inspect", "description": "Inspect metadata: collection sizes, "
     "band names, schemas or a few sample rows. Billed by the platform.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {"name": "answer", "description": "Submit the final answer to the request.",
     "parameters": {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}}},
    {"type": "function", "function": {"name": "give_up", "description": "Stop without an answer.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}},
]
CODE_RE = re.compile(r"```(?:python|sql)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = CODE_RE.findall(text or "")
    return m[-1].strip() if m else (text or "").strip()


def turn(kind: str, r: Reply, obs: str = "") -> Turn:
    return Turn(kind, r.tok_in, r.tok_out, max(len(obs) // 4, 0), r.latency, r.usd)


class Drafter:
    """One low-cost backbone drafts the plan of each data option; compiling it runs the metered lookups."""

    def __init__(self, platform: Platform, adapter: DomainAdapter, backbone: str = DRAFTER,
                 st: Settings | None = None):
        self.P, self.ad, self.st = platform, adapter, st or Settings()
        self.llm = LLM(backbone, self.st.temperature, self.st.max_out_tokens)

    def draft(self, req: Request, d: str) -> dict:
        cat, calls = self.P.catalogue(req, d)
        msgs = [{"role": "system", "content": "You draft complete analysis plans as code. Analysis calls are "
                 "disabled; only the catalogue below is available. Reply with one code block."},
                {"role": "user", "content": self.P.draft_prompt(req, d, cat)}]
        r = self.llm.chat(msgs)
        code = extract_code(r.text)
        calls = list(calls)
        try:
            plan = self.ad.compile(req, code)
            calls += [Call("lookup", float(x.get("cost", 0.0)), float(x.get("latency", 0.0)))
                      for x in plan.meta.get("metadata_calls", [])]
        except Exception:
            pass
        return {"code": code, "step": Step(turn("draft", r), calls)}


class DataAgent:
    def __init__(self, platform: Platform, adapter: DomainAdapter, backbone: str, st: Settings | None = None):
        self.P, self.ad, self.b = platform, adapter, backbone
        self.st = st or Settings()
        self.llm = LLM(backbone, self.st.temperature, self.st.max_out_tokens)

    def run_session(self, req: Request, config: str, draft: dict, tag: str) -> Session:
        st, P = self.st, self.P
        b, d, m = split_key(config)
        s = Session(f"{req.request_id}#{config}", req.request_id, req.unit, req.rtype, P.name, config,
                    draft["step"], has_ref=req.reference is not None, ref_kind=req.ref_kind)
        try:
            s.plan = self.ad.compile(req, draft["code"], m).to_dict()
        except Exception as e:                     # an unparsed draft is a flag, not a crash
            s.plan = {"ops": [], "flags": {"unparsed": 1, "dry_run_fails": 1}, "meta": {"error": str(e)[:200]}}
        msgs = [{"role": "system", "content": P.system_prompt(req, config)},
                {"role": "user", "content": f"Request: {req.text}\n\n{P.method_instruction(m)}\n\n"
                 f"A drafted plan for data option `{d}`:\n```{P.language}\n{draft['code']}\n```\n"
                 "Execute it with `run`, fix it from the feedback if needed, then call `answer`."}]
        pending, inspected, value = None, False, None
        for _ in range(st.max_turns):
            r = self.llm.chat(msgs, tools=TOOLS)
            if not r.tool_calls:
                self._explore(s, Step(turn("explore", r)))
                msgs += [{"role": "assistant", "content": r.text or ""},
                         {"role": "user", "content": "Use one of the tools: run, inspect, answer or give_up."}]
                continue
            tc = r.tool_calls[0]
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            msgs.append({"role": "assistant", "content": r.text or "", "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.function.arguments}}]})

            def reply(text):
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": text[:st.obs_chars]})

            if name == "answer":
                if pending is None:
                    reply("No successful run yet; run the code first.")
                    self._explore(s, Step(turn("explore", r)))
                    continue
                pending.final = turn("final", r)
                s.answer = P.parse_answer(req, str(args.get("value", "")))
                break
            if name == "give_up":
                if pending is not None:
                    pending.outcome, pending = "rejected", None
                s.giveup = turn("giveup", r)
                break
            if name == "inspect":
                res = P.inspect(req, args.get("code", ""), tag)
                obs = res.observation[:st.obs_chars]
                inspected = inspected or any(c.reads_data for c in res.calls)
                self._explore(s, Step(turn("explore", r, obs), res.calls))
                reply(obs)
                continue
            if name == "run":
                if pending is not None:
                    pending.outcome, pending = "rejected", None
                if len(s.attempts) >= st.n_max:
                    break
                res = P.run(req, args.get("code", ""), tag)
                obs = res.observation[:st.obs_chars]
                y = "success" if res.ok else (res.outcome or self.ad.classify(res.observation))
                a = Attempt(len(s.attempts) + 1, y, Step(turn("attempt", r, obs), res.calls), res.plan, inspected)
                s.attempts.append(a)
                if res.ok:
                    pending, value = a, res.value
                reply(obs)
                if not res.ok and len(s.attempts) >= st.n_max:
                    break
                continue
            reply(f"Unknown tool {name}.")
        if pending is not None and s.answer is None:
            pending.outcome = "rejected"
        s.extras["last_value"] = value
        return s

    @staticmethod
    def _explore(s: Session, step: Step):
        """Before attempt 1 a step is exploration; afterwards it is an inspection after the last attempt."""
        (s.attempts[-1].after if s.attempts else s.explore).append(step)
