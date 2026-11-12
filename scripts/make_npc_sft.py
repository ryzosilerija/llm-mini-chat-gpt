"""Generate synthetic NPC SFT data: (1) action choices labeled by simple rules,
(2) dialogue grounded in the game state ("is it raining?" -> answers from state).

Prompts are built with minigpt.npc, the same code the API uses, so formats always match.
Swap in your own rules/personas, or add Gemini-written dialogue to a second jsonl.

  python scripts/make_npc_sft.py --n 20000 --out data/npc_sft.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minigpt.npc import build_action_messages, build_system, candidate_actions  # noqa: E402

PERSONAS = [
    {"name": "Brakk", "persona": "a gruff blacksmith who hates the cold"},
    {"name": "Ilse", "persona": "a cheerful scout who knows every map"},
    {"name": "The Watcher", "persona": "an unsettling figure who speaks in riddles"},
    {"name": "Tomas", "persona": "a nervous merchant who counts his coins"},
    {"name": "Vela", "persona": "a calm veteran soldier"},
]
MAPS = ["snow", "desert", "forest", "city ruins", "volcano", "harbor"]
TIMES = ["morning", "noon", "evening", "night"]
WEATHER = ["clear", "rain", "snow", "fog", "storm", "meteor shower"]
DIRS = ["north", "south", "east", "west", "behind you", "ahead"]
HOSTILES = ["wolf", "bandit", "zombie", "spider"]
ITEMS = ["medkit", "ammo_box", "coin_pouch", "rifle"]


def rand_state(rng):
    nearby = []
    for i in range(rng.randint(0, 2)):
        nearby.append({"id": f"player_{rng.randint(1, 9)}", "kind": "player",
                       "distance": round(rng.uniform(1, 40), 1), "direction": rng.choice(DIRS)})
    for i in range(rng.choice([0, 0, 1, 2])):
        kind = rng.choice(HOSTILES)
        nearby.append({"id": f"{kind}_{i + 1}", "kind": kind, "distance": round(rng.uniform(2, 50), 1),
                       "direction": rng.choice(DIRS), "hostile": True})
    if rng.random() < 0.3:
        item = rng.choice(ITEMS)
        nearby.append({"id": item, "kind": "item", "distance": round(rng.uniform(1, 20), 1),
                       "direction": rng.choice(DIRS)})
    # dedupe ids
    seen, uniq = set(), []
    for e in nearby:
        if e["id"] not in seen:
            seen.add(e["id"])
            uniq.append(e)
    events = []
    players = [e for e in uniq if e["kind"] == "player"]
    if players and rng.random() < 0.4:
        events.append(f"{players[0]['id']} {rng.choice(['waved at you', 'said hello', 'asked you to come'])}")
    return {"map": rng.choice(MAPS), "time_of_day": rng.choice(TIMES), "weather": rng.choice(WEATHER),
            "hp": rng.randint(5, 100), "max_hp": 100, "nearby": uniq, "events": events}


def available_actions(state):
    acts = [{"type": "idle"}, {"type": "wander"}]
    players = [e["id"] for e in state["nearby"] if e["kind"] == "player"]
    hostiles = [e["id"] for e in state["nearby"] if e.get("hostile")]
    items = [e["id"] for e in state["nearby"] if e["kind"] == "item"]
    if players:
        acts += [{"type": "follow", "targets": players}, {"type": "talk_to", "targets": players}]
    if hostiles:
        acts += [{"type": "attack", "targets": hostiles}, {"type": "flee", "targets": hostiles}]
    if items:
        acts.append({"type": "pick_up", "targets": items})
    return acts


def label_action(state, rng):
    by_dist = sorted(state["nearby"], key=lambda e: e["distance"])
    hostiles = [e for e in by_dist if e.get("hostile")]
    players = [e for e in by_dist if e["kind"] == "player"]
    items = [e for e in by_dist if e["kind"] == "item"]
    low_hp = state["hp"] < 35
    if hostiles and hostiles[0]["distance"] < 15:
        return f"flee {hostiles[0]['id']}" if low_hp else f"attack {hostiles[0]['id']}"
    if items and items[0]["id"] == "medkit" and low_hp:
        return "pick_up medkit"
    for ev in state["events"]:
        who = ev.split(" ")[0]
        if "come" in ev:
            return f"follow {who}"
        return f"talk_to {who}"
    if players and players[0]["distance"] < 8:
        return f"talk_to {players[0]['id']}"
    if items and items[0]["distance"] < 6:
        return f"pick_up {items[0]['id']}"
    return "idle" if state["time_of_day"] == "night" else rng.choice(["wander", "idle"])


def talk_pair(state, rng):
    weather, mp = state["weather"], state["map"]
    hostiles = [e for e in state["nearby"] if e.get("hostile")]
    others = [e["id"] for e in state["nearby"] if e["kind"] != "item"]
    hp = state["hp"]
    q, a = rng.choice([
        (rng.choice(["What's the weather like?", "Is it raining?", "How's the sky?"]),
         f"{weather.capitalize()} out here. " + ("Keep your head down." if weather in ("storm", "meteor shower") else "Could be worse.")),
        (rng.choice(["Where are we?", "What place is this?"]), f"This is the {mp} map."),
        (rng.choice(["Who's around?", "Anyone nearby?", "Are we alone?"]),
         ("I see " + ", ".join(others[:3]) + ".") if others else "Nobody but us."),
        (rng.choice(["Is it safe here?", "Any danger?"]),
         (f"No. {hostiles[0]['id']} is {hostiles[0]['direction']}, {round(hostiles[0]['distance'])}m away.")
         if hostiles else "Seems quiet. For now."),
        (rng.choice(["How are you holding up?", "You okay?"]),
         "Barely standing." if hp < 35 else "I've been better." if hp < 70 else "Fit as ever."),
    ])
    return q, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--talk_frac", type=float, default=0.4)
    ap.add_argument("--out", default="data/npc_sft.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.n):
            state, persona = rand_state(rng), rng.choice(PERSONAS)
            if rng.random() < args.talk_frac:
                q, a = talk_pair(state, rng)
                msgs = [{"role": "system", "content": build_system(persona, state)},
                        {"role": "user", "content": q}, {"role": "assistant", "content": a}]
            else:
                cands = candidate_actions(available_actions(state))
                rng.shuffle(cands)  # avoid learning "pick the first option"
                msgs = build_action_messages(persona, state, [], cands)
                msgs.append({"role": "assistant", "content": label_action(state, rng)})
            f.write(json.dumps({"messages": msgs}) + "\n")
    print(f"wrote {args.n} examples -> {args.out}")


if __name__ == "__main__":
    main()
