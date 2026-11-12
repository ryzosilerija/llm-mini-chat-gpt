// Drop-in client for the three.js game. Keep the API key server-side in production:
// call this from your Node game server (Node 18+ has fetch), not from the browser.

export class NpcBrain {
  constructor({ baseUrl = "http://localhost:8000", npcId, apiKey } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.npcId = npcId;
    this.headers = { "Content-Type": "application/json", ...(apiKey ? { "X-API-Key": apiKey } : {}) };
  }

  async #post(path, body) {
    const res = await fetch(`${this.baseUrl}${path}`, {
      method: "POST", headers: this.headers, body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`${path} -> ${res.status}: ${await res.text()}`);
    return res;
  }

  /** Chat with the NPC. Pass onDelta to stream tokens into a speech bubble. */
  async talk(playerId, message, state, onDelta) {
    const stream = typeof onDelta === "function";
    const res = await this.#post(`/v1/npc/${this.npcId}/talk`, { player_id: playerId, message, state, stream });
    if (!stream) return (await res.json()).reply;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "", reply = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) !== -1) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if (!line.startsWith("data:")) continue;
        const evt = JSON.parse(line.slice(5));
        if (evt.delta) onDelta(evt.delta);
        if (evt.done) reply = evt.reply;
      }
    }
    return reply;
  }

  /** Ask what to do next. Returns { action: {type, target}, scores, say? }. */
  async act(state, actions, { playerId = null, bark = false } = {}) {
    const res = await this.#post(`/v1/npc/${this.npcId}/act`, { state, actions, player_id: playerId, bark });
    return res.json();
  }

  reset(playerId) {
    const q = playerId ? `?player_id=${encodeURIComponent(playerId)}` : "";
    return this.#post(`/v1/npc/${this.npcId}/reset${q}`, {});
  }
}

// ---------------------------------------------------------------------------
// Example wiring: the LLM picks the intent, your game code does the movement.

export function snapshotFor(npc, world) {
  const nearby = world.entities
    .filter((e) => e !== npc && e.position.distanceTo(npc.position) < 50)
    .map((e) => ({
      id: e.id,
      kind: e.kind,                                   // "player" | "wolf" | "item" ...
      distance: +e.position.distanceTo(npc.position).toFixed(1),
      direction: world.compassDirection(npc, e),      // "north", "behind you", ...
      hostile: !!e.hostile,
    }));
  return {
    map: world.mapName, time_of_day: world.timeOfDay, weather: world.weather,
    hp: npc.hp, max_hp: npc.maxHp, position: npc.position.toArray(),
    nearby, events: npc.recentEvents.slice(-3),
  };
}

export function allowedActions(state) {
  const ids = (pred) => state.nearby.filter(pred).map((e) => e.id);
  const acts = [{ type: "idle" }, { type: "wander" }];
  const players = ids((e) => e.kind === "player");
  const hostiles = ids((e) => e.hostile);
  const items = ids((e) => e.kind === "item");
  if (players.length) acts.push({ type: "follow", targets: players }, { type: "talk_to", targets: players });
  if (hostiles.length) acts.push({ type: "attack", targets: hostiles }, { type: "flee", targets: hostiles });
  if (items.length) acts.push({ type: "pick_up", targets: items });
  return acts;
}

// Call every ~2s per NPC (not every frame). Run actions with your own pathfinding/animation.
export async function thinkTick(brain, npc, world) {
  const state = snapshotFor(npc, world);
  const { action, say } = await brain.act(state, allowedActions(state), { bark: Math.random() < 0.1 });
  const target = action.target ? world.byId(action.target) : null;
  switch (action.type) {
    case "follow":  npc.moveTo(target.position, { keepDistance: 3 }); break;
    case "talk_to": npc.faceTowards(target.position); npc.wave(); break;
    case "attack":  npc.attack(target); break;
    case "flee":    npc.fleeFrom(target.position); break;
    case "pick_up": npc.moveTo(target.position).then(() => npc.pickUp(target)); break;
    case "wander":  npc.moveTo(world.randomPointNear(npc.position, 15)); break;
    default:        npc.idle();
  }
  if (say) npc.showSpeechBubble(say);
}
