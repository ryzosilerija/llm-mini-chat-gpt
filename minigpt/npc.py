"""Prompt builders for game NPCs. Shared by the API server and the synthetic-data script,
so training data and inference prompts use the exact same format.

State is a plain dict:
{
  "map": "snow", "time_of_day": "night", "weather": "rain", "hp": 80, "max_hp": 100,
  "position": [x, y, z],
  "nearby": [{"id": "player_1", "kind": "player", "distance": 4.2, "direction": "north", "hostile": false}],
  "events": ["player_1 waved"]
}
"""

from __future__ import annotations

ACTION_PROMPT = "Decide your next action. Options: {options}. Answer with exactly one option."


def _entity(e: dict) -> str:
    s = e.get("id", "?")
    kind = e.get("kind")
    if kind and kind not in s:
        s += f" ({kind})"
    extra = []
    if e.get("distance") is not None:
        extra.append(f"{round(float(e['distance']))}m")
    if e.get("direction"):
        extra.append(str(e["direction"]))
    if e.get("hostile"):
        extra.append("hostile")
    return s + (f" [{', '.join(extra)}]" if extra else "")


def render_state(state: dict | None) -> str:
    if not state:
        return "Nothing notable around."
    parts = []
    if state.get("map"):
        parts.append(f"Location: {state['map']}.")
    tw = ", ".join(str(state[k]) for k in ("time_of_day", "weather") if state.get(k))
    if tw:
        parts.append(f"Conditions: {tw}.")
    if state.get("hp") is not None:
        parts.append(f"Health: {round(float(state['hp']))}/{round(float(state.get('max_hp') or 100))}.")
    nearby = state.get("nearby") or []
    if nearby:
        nearby = sorted(nearby, key=lambda e: e.get("distance") if e.get("distance") is not None else 1e9)
        parts.append("Nearby: " + "; ".join(_entity(e) for e in nearby[:8]) + ".")
    else:
        parts.append("Nobody nearby.")
    events = state.get("events") or []
    if events:
        parts.append("Recent: " + "; ".join(str(e) for e in events[-3:]) + ".")
    return " ".join(parts)


def build_system(persona: dict, state: dict | None) -> str:
    name = persona.get("name", "Stranger")
    desc = persona.get("persona", "a quiet wanderer")
    return (f"You are {name}, {desc}. You live inside a game world. Stay in character, "
            f"keep replies short, and only mention things you can see. World: {render_state(state)}")


def candidate_actions(actions: list[dict]) -> list[str]:
    """[{"type": "follow", "targets": ["player_1"]}, {"type": "idle"}] -> ["follow player_1", "idle"]"""
    out = []
    for a in actions:
        targets = a.get("targets") or []
        if targets:
            out += [f"{a['type']} {t}" for t in targets]
        else:
            out.append(a["type"])
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def build_action_messages(persona: dict, state: dict | None, history: list[dict],
                          candidates: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": build_system(persona, state)},
        *history[-4:],
        {"role": "user", "content": ACTION_PROMPT.format(options=", ".join(candidates))},
    ]


def parse_action(choice: str) -> dict:
    kind, _, target = choice.partition(" ")
    return {"type": kind, "target": target or None}
