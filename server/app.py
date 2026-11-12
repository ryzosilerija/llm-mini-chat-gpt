"""HTTP API for the game.

Run:
  MINIGPT_CKPT=checkpoints/sft/best.pt MINIGPT_TOKENIZER=checkpoints/sft/tokenizer.json \
  uvicorn server.app:create_app --factory --host 0.0.0.0 --port 8000

Env:
  MINIGPT_CKPT, MINIGPT_TOKENIZER   required
  MINIGPT_DEVICE                    auto | cpu | cuda
  MINIGPT_NPCS                      path to personas json (default server/npcs.json)
  MINIGPT_API_KEY                   if set, requests need header  X-API-Key: <key>
  MINIGPT_CORS_ORIGINS              comma-separated origins (default *)

Docs at /docs (Swagger) once running.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from minigpt.npc import build_action_messages, build_system, candidate_actions, parse_action

from .sessions import SessionStore

DEFAULT_PERSONA = {"name": "Stranger", "persona": "a quiet wanderer of few words"}


# ---------- schemas ----------

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=4000)


class SamplingParams(BaseModel):
    max_tokens: int = Field(120, ge=1, le=512)
    temperature: float = Field(0.6, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0, le=1000)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)

    def kwargs(self) -> dict:
        return self.model_dump(include={"max_tokens", "temperature", "top_k", "top_p", "repetition_penalty"})


class ChatRequest(SamplingParams):
    messages: list[Message] = Field(min_length=1, max_length=64)
    stream: bool = False


class Entity(BaseModel):
    id: str = Field(max_length=64)
    kind: str = Field("unknown", max_length=32)
    distance: float | None = None
    direction: str | None = Field(None, max_length=32)
    hostile: bool = False


class GameState(BaseModel):
    map: str | None = Field(None, max_length=64)
    time_of_day: str | None = Field(None, max_length=32)
    weather: str | None = Field(None, max_length=32)
    hp: float | None = None
    max_hp: float | None = None
    position: list[float] | None = None
    nearby: list[Entity] = Field(default_factory=list, max_length=32)
    events: list[str] = Field(default_factory=list, max_length=10)


class TalkRequest(SamplingParams):
    player_id: str = Field(max_length=64)
    message: str = Field(min_length=1, max_length=1000)
    state: GameState | None = None
    stream: bool = False


class ActionSpec(BaseModel):
    type: str = Field(pattern=r"^[a-z_]{1,32}$")
    targets: list[str] = Field(default_factory=list, max_length=16)


class ActRequest(BaseModel):
    state: GameState
    actions: list[ActionSpec] = Field(min_length=1, max_length=16)
    player_id: str | None = Field(None, max_length=64, description="include recent chat with this player")
    bark: bool = Field(False, description="also generate a short line to say")


# ---------- app ----------

def _load_personas(path: str | None) -> dict:
    p = Path(path) if path else Path(__file__).with_name("npcs.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def create_app(engine=None, personas: dict | None = None) -> FastAPI:
    if engine is None:
        from minigpt.engine import Engine

        ckpt, tokp = os.environ.get("MINIGPT_CKPT"), os.environ.get("MINIGPT_TOKENIZER")
        if not ckpt or not tokp:
            raise RuntimeError("set MINIGPT_CKPT and MINIGPT_TOKENIZER")
        engine = Engine(ckpt, tokp, os.environ.get("MINIGPT_DEVICE", "auto"))
    if personas is None:
        personas = _load_personas(os.environ.get("MINIGPT_NPCS"))

    sessions = SessionStore()
    api_key = os.environ.get("MINIGPT_API_KEY")

    def require_key(x_api_key: str | None = Header(default=None)):
        if api_key and not (x_api_key and hmac.compare_digest(x_api_key, api_key)):
            raise HTTPException(status_code=401, detail="invalid API key")

    app = FastAPI(title="mini-chatgpt NPC API", version="0.1.0", dependencies=[Depends(require_key)])
    origins = [o.strip() for o in os.environ.get("MINIGPT_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    def persona_for(npc_id: str) -> dict:
        return personas.get(npc_id) or personas.get("default") or DEFAULT_PERSONA

    def sse(gen: Iterator[str], on_done=None) -> StreamingResponse:
        def events():
            full = ""
            for delta in gen:
                full += delta
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            full = full.strip()
            if on_done:
                on_done(full)
            yield f"data: {json.dumps({'done': True, 'reply': full})}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/health")
    def health():
        return {"status": "ok", "model": engine.info, "npcs": sorted(personas)}

    @app.post("/v1/chat")
    def chat(req: ChatRequest):
        msgs = [m.model_dump() for m in req.messages]
        if req.stream:
            return sse(engine.stream_chat(msgs, **req.kwargs()))
        return {"reply": engine.chat(msgs, **req.kwargs())}

    @app.post("/v1/npc/{npc_id}/talk")
    def talk(npc_id: str, req: TalkRequest):
        state = req.state.model_dump() if req.state else None
        user_msg = {"role": "user", "content": req.message}
        msgs = [{"role": "system", "content": build_system(persona_for(npc_id), state)},
                *sessions.get(npc_id, req.player_id), user_msg]

        def remember(reply: str):
            sessions.append(npc_id, req.player_id, user_msg, {"role": "assistant", "content": reply})

        if req.stream:
            return sse(engine.stream_chat(msgs, **req.kwargs()), on_done=remember)
        reply = engine.chat(msgs, **req.kwargs())
        remember(reply)
        return {"npc_id": npc_id, "reply": reply}

    @app.post("/v1/npc/{npc_id}/act")
    def act(npc_id: str, req: ActRequest):
        state = req.state.model_dump()
        candidates = candidate_actions([a.model_dump() for a in req.actions])
        if len(candidates) > 64:
            raise HTTPException(status_code=422, detail="too many action candidates (max 64)")
        history = sessions.get(npc_id, req.player_id) if req.player_id else []
        persona = persona_for(npc_id)
        ranked = engine.choose(build_action_messages(persona, state, history, candidates), candidates)
        best = ranked[0][0]
        out = {"npc_id": npc_id, "action": parse_action(best),
               "scores": {c: round(s, 4) for c, s in ranked}}
        if req.bark:
            msgs = [{"role": "system", "content": build_system(persona, state)},
                    {"role": "user", "content": f"You decided to {best}. Say one short line out loud."}]
            out["say"] = engine.chat(msgs, max_tokens=40, temperature=0.9)
        return out

    @app.post("/v1/npc/{npc_id}/reset")
    def reset(npc_id: str, player_id: str | None = None):
        return {"cleared": sessions.reset(npc_id, player_id)}

    return app
