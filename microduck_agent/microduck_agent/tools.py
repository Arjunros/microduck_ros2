"""What the agent can make the duck do: tool schemas (shared by every LLM backend)
and the robot interface the ROS node implements (and tests fake)."""

from dataclasses import dataclass, field

SKILL_NAMES = ("kick_left", "kick_right", "roulade", "ground_pick", "sit", "stand")


@dataclass
class ToolResult:
    ok: bool
    text: str
    image_jpeg: bytes = None   # set by `see`: shown to vision-capable models


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    id: str = ""


def tool_specs(places, skills=SKILL_NAMES, has_ball=False, has_camera=True):
    """JSON-schema tool definitions. Place and skill names are enumerated so the
    model can only pick ones that exist."""
    place_names = sorted(places)
    specs = [
        {"name": "go_to",
         "description": "Walk to a named place in the home, avoiding obstacles (Nav2). Blocks until arrived or failed.",
         "parameters": {"type": "object", "required": ["place"],
                        "properties": {"place": {"type": "string", "enum": place_names}}}},
        {"name": "go_to_point",
         "description": "Walk to a point (x, y) in metres in the map frame.",
         "parameters": {"type": "object", "required": ["x", "y"],
                        "properties": {"x": {"type": "number"}, "y": {"type": "number"}}}},
        {"name": "walk",
         "description": "Walk straight for a distance in metres (negative = backwards). For short moves only.",
         "parameters": {"type": "object", "required": ["distance_m"],
                        "properties": {"distance_m": {"type": "number", "minimum": -2, "maximum": 2}}}},
        {"name": "turn",
         "description": "Turn in place. Positive degrees = left (counter-clockwise), negative = right.",
         "parameters": {"type": "object", "required": ["degrees"],
                        "properties": {"degrees": {"type": "number", "minimum": -360, "maximum": 360}}}},
        {"name": "do_skill",
         "description": "Perform a trained move: kick_left / kick_right (kick a ball in front of that foot), "
                        "roulade (forward roll), ground_pick (pick something up from the floor), sit, stand.",
         "parameters": {"type": "object", "required": ["skill"],
                        "properties": {"skill": {"type": "string", "enum": list(skills)}}}},
        {"name": "look",
         "description": "Point the head: left, right, up, down or center.",
         "parameters": {"type": "object", "required": ["direction"],
                        "properties": {"direction": {"type": "string",
                                                     "enum": ["left", "right", "up", "down", "center"]}}}},
        {"name": "where_am_i",
         "description": "Current position and the nearest named place.",
         "parameters": {"type": "object", "properties": {}}},
        {"name": "stop",
         "description": "Stop moving immediately and cancel navigation.",
         "parameters": {"type": "object", "properties": {}}},
    ]
    if has_ball:
        specs.insert(2, {"name": "go_to_ball",
                         "description": "Walk up to the ball and stop just behind it, facing it, ready to kick.",
                         "parameters": {"type": "object", "properties": {}}})
    if has_camera:
        specs.append({"name": "see",
                      "description": "Take a picture with the head camera and look at it.",
                      "parameters": {"type": "object", "properties": {}}})
    return specs


class Robot:
    """Interface the agent drives. Every method returns a ToolResult."""

    places = {}          # name -> (x, y)
    aliases = {}         # alias -> place name
    has_ball = False
    has_camera = True
    skills = SKILL_NAMES

    def go_to(self, place): raise NotImplementedError
    def go_to_point(self, x, y): raise NotImplementedError
    def go_to_ball(self): raise NotImplementedError
    def walk(self, distance_m): raise NotImplementedError
    def turn(self, degrees): raise NotImplementedError
    def do_skill(self, skill): raise NotImplementedError
    def look(self, direction): raise NotImplementedError
    def where_am_i(self): raise NotImplementedError
    def stop(self): raise NotImplementedError
    def see(self): raise NotImplementedError

    def resolve_place(self, name):
        n = (name or "").strip().lower()
        if n in self.places:
            return n
        if n in self.aliases:
            return self.aliases[n]
        for p in self.places:               # "the kitchen", "kitchen please"
            if p in n:
                return p
        for a, p in self.aliases.items():
            if a in n:
                return p
        return None

    def execute(self, call):
        fn = getattr(self, call.name, None)
        if fn is None or call.name.startswith("_") or call.name in ("execute", "resolve_place"):
            return ToolResult(False, f"unknown tool '{call.name}'")
        try:
            return fn(**(call.args or {}))
        except TypeError as e:
            return ToolResult(False, f"bad arguments for {call.name}: {e}")
