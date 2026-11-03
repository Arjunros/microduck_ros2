"""Named places for the agent: the YAML written by `make_map` (or your own)."""

import os


def load_places(path):
    """Returns ({name: (x, y)}, {alias: name}). Missing file -> empty (go_to unavailable)."""
    path = os.path.expanduser(path or "")
    if not path or not os.path.isfile(path):
        return {}, {}
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    places, aliases = {}, {}
    for name, v in (data.get("places") or {}).items():
        name = str(name).lower()
        places[name] = (float(v["x"]), float(v["y"]))
        for a in v.get("aliases") or []:
            aliases[str(a).lower()] = name
    return places, aliases
