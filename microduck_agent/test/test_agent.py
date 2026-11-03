import json

import pytest

from microduck_agent import backends
from microduck_agent.agent import Agent
from microduck_agent.backends import (AnthropicBackend, AssistantTurn, OpenAICompatBackend, RuleBackend,
                                      make_backend)
from microduck_agent.places import load_places
from microduck_agent.tools import Robot, ToolCall, ToolResult, tool_specs

PLACES = {"kitchen": (-2.6, 1.9), "living room": (-2.5, -1.4), "bedroom": (2.1, 2.2), "start": (0.0, 0.0)}
ALIASES = {"sofa": "living room", "home": "start", "lounge": "living room"}


class FakeRobot(Robot):
    places, aliases, has_ball = PLACES, ALIASES, True

    def __init__(self):
        self.log = []

    def _ok(self, name, **kw):
        self.log.append((name, kw))
        return ToolResult(True, f"{name} {kw}".strip())

    def go_to(self, place):
        if self.resolve_place(place) is None:
            return ToolResult(False, f"unknown place {place}")
        return self._ok("go_to", place=self.resolve_place(place))

    def go_to_point(self, x, y): return self._ok("go_to_point", x=x, y=y)
    def go_to_ball(self): return self._ok("go_to_ball")
    def walk(self, distance_m): return self._ok("walk", distance_m=distance_m)
    def turn(self, degrees): return self._ok("turn", degrees=degrees)
    def do_skill(self, skill): return self._ok("do_skill", skill=skill)
    def look(self, direction): return self._ok("look", direction=direction)
    def where_am_i(self): return self._ok("where_am_i")
    def stop(self): return self._ok("stop")
    def see(self): return ToolResult(True, "image attached", b"\xff\xd8jpeg")


@pytest.mark.parametrize("text, expected", [
    ("go to the kitchen", [("go_to", {"place": "kitchen"})]),
    ("Go to the kitchen and then sit", [("go_to", {"place": "kitchen"}), ("do_skill", {"skill": "sit"})]),
    ("go sit on the sofa", [("go_to", {"place": "living room"}), ("do_skill", {"skill": "sit"})]),
    ("sit", [("do_skill", {"skill": "sit"})]),
    ("walk to the sofa", [("go_to", {"place": "living room"})]),
    ("go back home", [("go_to", {"place": "start"})]),
    ("turn left 45", [("turn", {"degrees": 45.0})]),
    ("turn right", [("turn", {"degrees": -90.0})]),
    ("turn around", [("turn", {"degrees": 180.0})]),
    ("walk forward 30 cm", [("walk", {"distance_m": 0.3})]),
    ("step back", [("walk", {"distance_m": -0.5})]),
    ("go to the ball and kick it with your left foot",
     [("go_to_ball", {}), ("do_skill", {"skill": "kick_left"})]),
    ("do a roll then pick it up", [("do_skill", {"skill": "roulade"}), ("do_skill", {"skill": "ground_pick"})]),
    ("look left", [("look", {"direction": "left"})]),
    ("where are you?", [("where_am_i", {})]),
    ("what do you see", [("see", {})]),
    ("stop", [("stop", {})]),
    ("get up", [("do_skill", {"skill": "stand"})]),
])
def test_rule_parser(text, expected):
    calls = RuleBackend(FakeRobot()).parse(text)
    assert [(c.name, c.args) for c in calls] == expected


def test_rule_parser_gives_help_on_nonsense():
    turn = RuleBackend(FakeRobot()).complete("", [{"role": "user", "content": "sing me a song"}], [])
    assert not turn.tool_calls and "didn't understand" in turn.text and "kitchen" in turn.text


def test_rule_parser_explains_missing_places():
    robot = FakeRobot()
    robot.places, robot.aliases = {}, {}
    turn = RuleBackend(robot).complete("", [{"role": "user", "content": "go to the kitchen"}], [])
    assert not turn.tool_calls and "make_map" in turn.text


def test_agent_runs_a_plan_with_the_rule_backend():
    robot = FakeRobot()
    agent = Agent(RuleBackend(robot), robot)
    reply = agent.run("go to the kitchen then sit")
    assert robot.log == [("go_to", {"place": "kitchen"}), ("do_skill", {"skill": "sit"})]
    assert reply.startswith("Done:")
    agent.run("stand up and turn around")                       # history keeps working
    assert robot.log[-2:] == [("do_skill", {"skill": "stand"}), ("turn", {"degrees": 180.0})]


class ScriptedBackend:
    """Plays back assistant turns, recording what it was shown."""
    name = "scripted"

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []

    def complete(self, system, messages, tools):
        self.seen.append([dict(m) for m in messages])
        return self.turns.pop(0)


def test_agent_feeds_results_back_and_stops_on_text():
    robot = FakeRobot()
    be = ScriptedBackend([
        AssistantTurn("", [ToolCall("go_to", {"place": "bedroom"}, "a"), ToolCall("see", {}, "b")]),
        AssistantTurn("I'm in the bedroom and I see a bed!"),
    ])
    agent = Agent(be, robot)
    assert agent.run("what's in the bedroom?") == "I'm in the bedroom and I see a bed!"
    second = be.seen[1]
    tool_msgs = [m for m in second if m["role"] == "tool"]
    assert [m["id"] for m in tool_msgs] == ["a", "b"]
    assert tool_msgs[0]["content"].startswith("OK: go_to")
    assert tool_msgs[1]["image_jpeg"] == b"\xff\xd8jpeg"


def test_agent_reports_failures_and_unknown_tools():
    robot = FakeRobot()
    be = ScriptedBackend([AssistantTurn("", [ToolCall("go_to", {"place": "garage"}, "a"),
                                             ToolCall("fly", {}, "b")]),
                          AssistantTurn("There's no garage here, and I can't fly.")])
    Agent(be, robot).run("fly to the garage")
    contents = [m["content"] for m in be.seen[1] if m["role"] == "tool"]
    assert contents[0].startswith("FAILED: unknown place") and contents[1].startswith("FAILED: unknown tool")


def test_cancel_closes_every_tool_call():
    robot = FakeRobot()
    agent = Agent(None, robot)

    class CancelAfterFirst(ScriptedBackend):
        pass

    be = CancelAfterFirst([AssistantTurn("", [ToolCall("walk", {"distance_m": 1}, "a"),
                                              ToolCall("turn", {"degrees": 90}, "b")])])
    agent.backend = be
    orig = robot.walk

    def walk_then_cancel(distance_m):
        agent.cancel.set()
        return orig(distance_m)

    robot.walk = walk_then_cancel
    assert agent.run("walk and turn") == "Stopped."
    tool_msgs = [m for m in agent.history if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs] == ["OK: walk {'distance_m': 1}", "FAILED: cancelled"]
    assert agent.history[-1]["role"] == "assistant" and not agent.history[-1]["tool_calls"]


def test_tool_specs_enumerate_places_and_skills():
    specs = {t["name"]: t for t in tool_specs(PLACES, has_ball=True)}
    assert specs["go_to"]["parameters"]["properties"]["place"]["enum"] == sorted(PLACES)
    assert "kick_left" in specs["do_skill"]["parameters"]["properties"]["skill"]["enum"]
    assert "go_to_ball" in specs and "see" in specs
    assert "go_to_ball" not in {t["name"] for t in tool_specs(PLACES, has_ball=False)}


HISTORY = [
    {"role": "user", "content": "look around"},
    {"role": "assistant", "content": "", "tool_calls": [ToolCall("see", {}, "t1")]},
    {"role": "tool", "id": "t1", "name": "see", "content": "OK: image", "image_jpeg": b"JPG"},
]


def test_openai_wire_format_and_parsing(monkeypatch):
    sent = {}

    def fake_post(url, body, headers, timeout):
        sent.update(url=url, body=body, headers=headers)
        return {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "go_to", "arguments": '{"place": "kitchen"}'}}]}}]}

    monkeypatch.setattr(backends, "_post_json", fake_post)
    be = OpenAICompatBackend("qwen3:8b", "http://localhost:11434/v1")
    turn = be.complete("SYS", HISTORY, tool_specs(PLACES))
    assert sent["url"] == "http://localhost:11434/v1/chat/completions" and "authorization" not in sent["headers"]
    msgs = sent["body"]["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "see", "arguments": "{}"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "OK: image"}
    assert msgs[4]["role"] == "user" and msgs[4]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert sent["body"]["tools"][0]["type"] == "function"
    assert [(c.name, c.args, c.id) for c in turn.tool_calls] == [("go_to", {"place": "kitchen"}, "c1")]


def test_anthropic_wire_format_and_parsing(monkeypatch):
    sent = {}

    def fake_post(url, body, headers, timeout):
        sent.update(url=url, body=body, headers=headers)
        return {"content": [{"type": "text", "text": "On my way!"},
                            {"type": "tool_use", "id": "tu1", "name": "do_skill", "input": {"skill": "sit"}}]}

    monkeypatch.setattr(backends, "_post_json", fake_post)
    be = AnthropicBackend("claude-haiku-5-5", api_key="k")
    turn = be.complete("SYS", HISTORY, tool_specs(PLACES))
    assert sent["headers"]["x-api-key"] == "k" and sent["body"]["system"] == "SYS"
    assert sent["body"]["tools"][0]["input_schema"]["required"] == ["place"]
    m = sent["body"]["messages"]
    assert m[1]["content"] == [{"type": "tool_use", "id": "t1", "name": "see", "input": {}}]
    result = m[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1" and not result["is_error"]
    assert result["content"][1]["source"]["media_type"] == "image/jpeg"
    assert turn.text == "On my way!" and turn.tool_calls[0].args == {"skill": "sit"}


def test_make_backend_requires_models_and_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_backend("rule", FakeRobot()).name == "rule"
    with pytest.raises(backends.BackendError):
        make_backend("ollama", FakeRobot())                    # no model
    with pytest.raises(backends.BackendError):
        make_backend("anthropic", FakeRobot())                 # no key
    with pytest.raises(backends.BackendError):
        make_backend("gpt9000", FakeRobot())


def test_load_places(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("places:\n  Kitchen:\n    x: 1\n    y: 2\n    aliases: [Fridge]\n")
    assert load_places(str(f)) == ({"kitchen": (1.0, 2.0)}, {"fridge": "kitchen"})
    assert load_places(str(tmp_path / "missing.yaml")) == ({}, {})
    json.dumps(load_places(str(f)))   # plain data


def test_openai_backend_over_real_http():
    """Full round trip through urllib against a local OpenAI-compatible server (as Ollama would be)."""
    import http.server
    import threading

    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            received.append((self.path, self.headers.get("authorization"), body))
            first = len(received) == 1
            msg = ({"role": "assistant", "content": None, "tool_calls": [
                       {"id": "x1", "type": "function",
                        "function": {"name": "go_to", "arguments": json.dumps({"place": "sofa"})}}]}
                   if first else {"role": "assistant", "content": "Comfy on the sofa now!"})
            data = json.dumps({"choices": [{"message": msg}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        robot = FakeRobot()
        be = OpenAICompatBackend("test-model", f"http://127.0.0.1:{srv.server_port}/v1", api_key="sk-test")
        reply = Agent(be, robot).run("go relax on the sofa")
    finally:
        srv.shutdown()
    assert reply == "Comfy on the sofa now!"
    assert robot.log == [("go_to", {"place": "living room"})]              # alias resolved by the robot
    path, auth, body = received[1]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test" and body["model"] == "test-model"
    assert body["messages"][-1] == {"role": "tool", "tool_call_id": "x1",
                                    "content": "OK: go_to {'place': 'living room'}"}


def test_backend_error_is_readable():
    be = OpenAICompatBackend("m", "http://127.0.0.1:9/v1")      # nothing listens on port 9
    with pytest.raises(backends.BackendError):
        be.complete("s", [{"role": "user", "content": "hi"}], [])
