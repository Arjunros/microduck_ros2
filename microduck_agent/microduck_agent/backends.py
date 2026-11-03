"""LLM backends for the agent. Each turns (system, messages, tools) into the next
assistant turn: some text and/or tool calls.

Neutral message format used by the agent:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [ToolCall]}
  {"role": "tool", "id": str, "name": str, "content": str, "image_jpeg": bytes|None}

  RuleBackend          offline keyword parser; no model, no network. Default.
  OpenAICompatBackend  any /v1/chat/completions server with tool calling:
                       Ollama (local, free), OpenAI, LM Studio, vLLM, Groq, ...
  AnthropicBackend     Claude, via the Messages API.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .tools import ToolCall


class BackendError(RuntimeError):
    pass


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list = field(default_factory=list)


def _post_json(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise BackendError(f"{url}: HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise BackendError(f"{url}: {e.reason}") from None


# --------------------------------------------------------------------------- rule
_NUM = r"(-?\d+(?:\.\d+)?)"


class RuleBackend:
    """Understands simple commands without any model, e.g.
    "go to the kitchen then sit", "turn left 45", "walk forward 0.5 and kick",
    "go to the ball and kick it", "where are you", "stop"."""

    name = "rule"

    def __init__(self, robot):
        self.robot = robot

    def complete(self, system, messages, tools):
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        results = [m for m in messages[last_user + 1:] if m["role"] == "tool"]
        if results:   # we already acted: summarise
            fails = [r for r in results if r["content"].startswith("FAILED")]
            parts = [r["content"].removeprefix("OK: ").removeprefix("FAILED: ") for r in results]
            text = "; ".join(parts)
            return AssistantTurn(("Stopped: " if fails else "Done: ") + text)
        calls = self.parse(messages[last_user]["content"])
        if not calls:
            return AssistantTurn(
                "Sorry, I didn't understand. Try: 'go to the kitchen', 'turn left 90', 'walk forward 0.5', "
                "'kick', 'roll', 'pick up', 'sit', 'stand', 'look left', 'where are you', 'stop'"
                + (", 'go to the ball'" if self.robot.has_ball else "") + ".")
        for i, c in enumerate(calls):
            c.id = f"rule_{i}"
        return AssistantTurn("", calls)

    def parse(self, text):
        t = text.lower().strip()
        clauses = re.split(r"\s*(?:,|;|\band then\b|\bthen\b|\band\b|\.)\s*", t)
        calls = []
        for c in filter(None, clauses):
            call = self._clause(c, calls)
            if call is None:
                continue
            # "go sit on the sofa": a move that also names a place -> walk there first
            if call.name in ("do_skill", "look", "see", "turn") and re.search(r"\b(go|walk|head|come|run)\b", c):
                place = self.robot.resolve_place(c)
                if place:
                    calls.append(ToolCall("go_to", {"place": place}))
            calls.append(call)
        return calls

    def _clause(self, c, previous):
        num = re.search(_NUM, c)
        n = float(num.group(1)) if num else None
        if re.search(r"\b(stop|halt|freeze|wait)\b", c):
            return ToolCall("stop")
        if re.search(r"\bwhere (am i|are you)\b|\bposition\b|\blocation\b", c):
            return ToolCall("where_am_i")
        if re.search(r"\b(what do you see|what can you see|look around|describe|picture|photo|camera)\b", c):
            return ToolCall("see")
        if "ball" in c and re.search(r"\b(go|walk|head|move|get|find|come)\b", c) and self.robot.has_ball:
            return ToolCall("go_to_ball")
        if re.search(r"\bkick\b", c):
            side = "left" if "left" in c else "right"
            return ToolCall("do_skill", {"skill": f"kick_{side}"})
        if re.search(r"\b(roll|roulade|somersault|tumble|flip)\b", c):
            return ToolCall("do_skill", {"skill": "roulade"})
        if re.search(r"\b(pick|grab|peck)\b", c):
            return ToolCall("do_skill", {"skill": "ground_pick"})
        if re.search(r"\b(sit)\b", c):
            return ToolCall("do_skill", {"skill": "sit"})
        if re.search(r"\b(stand|get up|rise)\b", c):
            return ToolCall("do_skill", {"skill": "stand"})
        if re.search(r"\blook\b", c):
            for d in ("left", "right", "up", "down"):
                if d in c:
                    return ToolCall("look", {"direction": d})
            return ToolCall("look", {"direction": "center"})
        if re.search(r"\bturn around\b|\bu-?turn\b", c):
            return ToolCall("turn", {"degrees": 180.0})
        if re.search(r"\b(turn|rotate|spin)\b", c):
            deg = abs(n) if n is not None else 90.0
            return ToolCall("turn", {"degrees": -deg if "right" in c else deg})
        # "go to the kitchen", "kitchen", "back home": a known place or alias anywhere in the clause
        place = self.robot.resolve_place(
            re.sub(r"\b(go|walk|head|move|navigate|come|run|get|to|the|back|into|in)\b", " ", c))
        if place and not re.search(r"\b(forward|backward|back up|ahead)\b", c):
            return ToolCall("go_to", {"place": place})
        if re.search(r"\b(forward|ahead|backward|backwards|back up|walk|step|move)\b", c):
            dist = abs(n) if n is not None else 0.5
            if re.search(r"\bcm\b|centimet", c):
                dist /= 100.0
            if re.search(r"\b(backward|backwards|back up|back)\b", c):
                dist = -dist
            return ToolCall("walk", {"distance_m": max(-2.0, min(2.0, dist))})
        return None


# ------------------------------------------------------------------ OpenAI-compatible
class OpenAICompatBackend:
    """POST {base_url}/chat/completions with tools. Works with Ollama's OpenAI endpoint
    (base_url http://localhost:11434/v1), OpenAI, LM Studio, vLLM, Groq, ..."""

    name = "openai"

    def __init__(self, model, base_url="https://api.openai.com/v1", api_key=None, timeout=120.0):
        if not model:
            raise BackendError("set model:=<name> (any tool-calling model, e.g. an Ollama model you have pulled)")
        self.model, self.base_url, self.api_key, self.timeout = model, base_url.rstrip("/"), api_key, timeout

    def to_wire(self, system, messages):
        out = [{"role": "system", "content": system}]
        pending_images = []
        for m in messages:
            if m["role"] != "tool" and pending_images:
                out.append({"role": "user", "content": pending_images})
                pending_images = []
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name, "arguments": json.dumps(c.args)}}
                                         for c in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["id"], "content": m["content"]})
                if m.get("image_jpeg"):
                    # Tool messages can't carry images in this API: attach it as a user message.
                    b64 = base64.b64encode(m["image_jpeg"]).decode()
                    pending_images = [{"type": "text", "text": "Camera image from `see`:"},
                                      {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
        if pending_images:
            out.append({"role": "user", "content": pending_images})
        return out

    def complete(self, system, messages, tools):
        body = {"model": self.model, "messages": self.to_wire(system, messages),
                "tools": [{"type": "function", "function": t} for t in tools], "tool_choice": "auto"}
        headers = {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = _post_json(f"{self.base_url}/chat/completions", body, headers, self.timeout)
        try:
            msg = resp["choices"][0]["message"]
        except (KeyError, IndexError):
            raise BackendError(f"unexpected response: {str(resp)[:300]}") from None
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(fn.get("name", ""), args, tc.get("id") or f"call_{i}"))
        return AssistantTurn(msg.get("content") or "", calls)


# ------------------------------------------------------------------------ Anthropic
class AnthropicBackend:
    """Claude via the Messages API (tool use, images in tool results)."""

    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model="claude-haiku-5-5", api_key=None, timeout=120.0, max_tokens=1024):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise BackendError("set ANTHROPIC_API_KEY")
        self.model, self.timeout, self.max_tokens = model or "claude-haiku-5-5", timeout, max_tokens

    def to_wire(self, messages):
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.args}
                           for c in m.get("tool_calls") or []]
                out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(done)"}]})
            elif m["role"] == "tool":
                content = [{"type": "text", "text": m["content"]}]
                if m.get("image_jpeg"):
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": base64.b64encode(m["image_jpeg"]).decode()}})
                block = {"type": "tool_result", "tool_use_id": m["id"], "content": content,
                         "is_error": m["content"].startswith("FAILED")}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)          # results of one turn go together
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, system, messages, tools):
        body = {"model": self.model, "max_tokens": self.max_tokens, "system": system,
                "messages": self.to_wire(messages),
                "tools": [{"name": t["name"], "description": t["description"],
                           "input_schema": t["parameters"]} for t in tools]}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        resp = _post_json(self.URL, body, headers, self.timeout)
        text, calls = [], []
        for b in resp.get("content", []):
            if b.get("type") == "text":
                text.append(b["text"])
            elif b.get("type") == "tool_use":
                calls.append(ToolCall(b["name"], b.get("input") or {}, b["id"]))
        if not text and not calls and "error" in resp:
            raise BackendError(str(resp["error"])[:300])
        return AssistantTurn("\n".join(text), calls)


def make_backend(kind, robot, model="", base_url="", api_key_env=""):
    kind = (kind or "rule").lower()
    if kind == "rule":
        return RuleBackend(robot)
    if kind == "ollama":
        return OpenAICompatBackend(model, base_url or "http://localhost:11434/v1", api_key=None)
    if kind == "openai":
        key = os.environ.get(api_key_env or "OPENAI_API_KEY")
        return OpenAICompatBackend(model, base_url or "https://api.openai.com/v1", key)
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-haiku-5-5", os.environ.get(api_key_env or "ANTHROPIC_API_KEY"))
    raise BackendError(f"unknown backend '{kind}' (rule | ollama | openai | anthropic)")
