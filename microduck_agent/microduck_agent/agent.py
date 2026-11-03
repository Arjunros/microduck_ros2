"""The agent loop: user text -> model -> tool calls on the robot -> results -> ... -> reply."""

import threading

from .tools import ToolResult, tool_specs

SYSTEM_PROMPT = """You control Microduck, a small (25 cm) two-legged robot duck, in a home.
You act by calling tools; each one blocks until the robot has finished. Plan multi-step
requests as a sequence of calls ("go to the kitchen and sit" = go_to, then do_skill sit).

Facts:
- Known places: {places}.{ball}
- Moves: kick_left / kick_right only work with a ball right in front of that foot
  (use go_to_ball first), roulade = forward roll, ground_pick = peck at the floor, sit, stand.
- The duck can't do moves while sitting: stand first.
- Walking is slow (~0.2 m/s). Prefer go_to for anything beyond a step or two.
- If a tool fails, say why in one sentence and don't retry more than once.

Reply in one or two short, friendly sentences once you're done, like a duck that's
proud of itself. Don't list tool calls."""


class Agent:
    def __init__(self, backend, robot, max_steps=12, history_turns=6, on_event=None):
        self.backend = backend
        self.robot = robot
        self.max_steps = max_steps
        self.history_turns = history_turns
        self.on_event = on_event or (lambda kind, text: None)
        self.history = []
        self.cancel = threading.Event()

    def system_prompt(self):
        places = ", ".join(sorted(self.robot.places)) or "none (navigation unavailable)"
        ball = " There is a ball somewhere; go_to_ball finds it." if self.robot.has_ball else ""
        return SYSTEM_PROMPT.format(places=places, ball=ball)

    def _trim(self):
        """Keep the last N user turns (and everything after the first kept one)."""
        users = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(users) > self.history_turns:
            self.history = self.history[users[-self.history_turns]:]
        # Images are large: keep only the most recent one.
        imgs = [m for m in self.history if m.get("image_jpeg")]
        for m in imgs[:-1]:
            m["image_jpeg"] = None

    def run(self, text):
        """Handle one user request. Returns the reply text."""
        self.cancel.clear()
        self.history.append({"role": "user", "content": text})
        tools = tool_specs(self.robot.places, self.robot.skills, self.robot.has_ball, self.robot.has_camera)
        system = self.system_prompt()
        reply = ""
        for _ in range(self.max_steps):
            if self.cancel.is_set():
                reply = "Stopped."
                break
            self.on_event("thinking", "")
            turn = self.backend.complete(system, self.history, tools)
            self.history.append({"role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls})
            if not turn.tool_calls:
                reply = turn.text.strip()
                break
            for call in turn.tool_calls:
                if self.cancel.is_set():
                    res = ToolResult(False, "cancelled")
                else:
                    self.on_event("tool", f"{call.name}({', '.join(f'{k}={v}' for k, v in call.args.items())})")
                    res = self.robot.execute(call)
                content = ("OK: " if res.ok else "FAILED: ") + res.text
                self.on_event("result", content)
                self.history.append({"role": "tool", "id": call.id, "name": call.name,
                                     "content": content, "image_jpeg": res.image_jpeg})
        else:
            reply = "I ran out of steps for that one."
        if not reply:
            reply = "Done."
        # Close the turn with plain assistant text (if it ended on tool results), so the
        # next request starts from a well-formed conversation for every backend.
        last = self.history[-1]
        if last["role"] != "assistant" or last.get("tool_calls"):
            self.history.append({"role": "assistant", "content": reply, "tool_calls": []})
        self._trim()
        return reply
