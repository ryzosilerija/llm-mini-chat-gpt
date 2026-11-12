# mini-chatgpt

A small ChatGPT-style model trained end to end (tokenizer → pretraining → chat fine-tuning) and served as an HTTP API that game NPCs use to **talk** and to **decide what to do** based on the live game state.

Built in PyTorch, trained on Kaggle GPUs. The architecture and the Muon/AdamW optimizer split come from my from-scratch [LLM study](https://github.com/ryzosilerija/LLM).

## What's in it

| Part | Details |
|---|---|
| Model | Decoder-only transformer: RMSNorm, RoPE, SwiGLU, tied embeddings, KV cache. Default ~30M params (8 layers, d=512, ctx 512) |
| Tokenizer | Byte-level BPE, 16k vocab, chat special tokens (`<\|system\|>`, `<\|user\|>`, `<\|assistant\|>`, `<\|end\|>`) |
| Optimizer | Muon on hidden matrices, AdamW on embeddings + norms. WSD or cosine schedule |
| Training | One script for pretraining and SFT. AMP (bf16, or fp16 + GradScaler on T4/P100), grad accumulation, resume, `metrics.jsonl` |
| SFT data | General chat (SmolTalk) + synthetic NPC data: state-grounded dialogue and rule-labeled action choices |
| API | FastAPI: chat, streaming (SSE), NPC talk with per-player memory, NPC action selection, API key, CORS |

### How NPC decisions work

Small models are bad at emitting valid JSON, so the API doesn't ask the model to write an action. The game sends the **allowed actions** (`follow player_1`, `flee wolf_1`, `idle`, …). The server scores each one as a possible assistant reply (mean log-probability) and returns the best. The result is **always a valid action**, comes with a confidence score per option, and costs one batched forward pass.

The model picks the intent. The game executes it: pathfinding, animation, combat.

```
game state ──► prompt (persona + rendered world) ──► score every allowed action ──► {"type":"flee","target":"wolf_1"}
```

## Repo layout

```
minigpt/            model, tokenizer, chat template, data loaders, Muon, sampling, inference engine, NPC prompts
scripts/            fetch_corpus, train_tokenizer, prepare_pretrain, make_npc_sft, prepare_sft, train, chat_cli, kaggle_pipeline.sh
server/             FastAPI app, session memory, NPC personas (npcs.json)
configs/            pretrain_30m, sft_30m, tiny_debug(+_sft)
examples/           game_client.js — client + example think loop for the three.js game
tests/              model/KV-cache/template/API tests
```

## Quickstart (local, CPU, ~1 min)

```bash
pip install -r requirements-dev.txt
pytest

# tiny end-to-end run on any .txt files
python scripts/train_tokenizer.py --input "data/debug/raw/*.txt" --vocab_size 512 --out checkpoints/debug_tokenizer.json
python scripts/prepare_pretrain.py --input "data/debug/raw/*.txt" --tokenizer checkpoints/debug_tokenizer.json --out data/debug/pretrain --val_frac 0.05
python scripts/train.py --config configs/tiny_debug.json
python scripts/make_npc_sft.py --n 3000 --out data/debug/npc_sft.jsonl
python scripts/prepare_sft.py --jsonl data/debug/npc_sft.jsonl --out data/debug/sft
python scripts/train.py --config configs/tiny_debug_sft.json
```

## Training on Kaggle

New notebook → GPU (T4 x2 or P100) → Internet on:

```bash
!git clone https://github.com/ryzosilerija/mini-chatgpt
%cd mini-chatgpt
!bash scripts/kaggle_pipeline.sh
```

Steps: FineWeb-Edu sample (add your Gutenberg `.txt` to `data/raw/gutenberg/`) → tokenizer → `train.bin`/`val.bin` → pretrain (`configs/pretrain_30m.json`: 10k steps × 65k tokens ≈ 650M tokens) → NPC + SmolTalk SFT data → SFT.

Sessions are time-capped. If pretraining gets cut off, save `checkpoints/` as a Kaggle dataset, attach it to the next session, copy it back, and run with `RESUME=1`.

Download `checkpoints/sft/best.pt` + `checkpoints/sft/tokenizer.json` when done.

Train config keys can be overridden from the CLI: `--set max_steps=2000 lr_muon=0.01`.

## Using an existing pretrained model (NorMuon repo)

A run from [NorMuon-Swiglu-LayerNorm-RoPE-at-scale](https://github.com/ryzosilerija/NorMuon-Swiglu-LayerNorm-RoPE-at-scale) imports exactly: the logits are identical. Its LayerNorm, fused QKV, untied head, FFN width and from-scratch BPE tokenizer are all supported.

```bash
python scripts/import_pretrained.py --run path/to/runs/pretrain --out checkpoints/imported \
    --ref_repo path/to/NorMuon-Swiglu-LayerNorm-RoPE-at-scale   # optional: verifies logits match
python scripts/train.py --config configs/sft_imported.json      # SFT on top of it
```

`--run` needs `model.pt` + `tokenizer.json`. If training stopped on the time limit before writing `model.pt`, `ckpt.pt` + `config.json` works too.

On Kaggle: add the pretraining notebook's output as an input, then run `RUN=/kaggle/input/<notebook>/runs/pretrain bash scripts/kaggle_sft_imported.sh`.

## Running the API

```bash
MINIGPT_CKPT=checkpoints/sft/best.pt MINIGPT_TOKENIZER=checkpoints/sft/tokenizer.json \
MINIGPT_API_KEY=change-me \
uvicorn server.app:create_app --factory --host 0.0.0.0 --port 8000
```

Swagger docs: `http://localhost:8000/docs`. Optional env: `MINIGPT_DEVICE`, `MINIGPT_NPCS` (persona file), `MINIGPT_CORS_ORIGINS`.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Model info, loaded personas |
| POST | `/v1/chat` | Plain chat. `{"messages":[...], "stream": false}` |
| POST | `/v1/npc/{npc_id}/talk` | Player talks to NPC. Remembers history per (NPC, player); world state goes into the prompt |
| POST | `/v1/npc/{npc_id}/act` | Pick next action from allowed list. Optional `bark: true` adds a short spoken line |
| POST | `/v1/npc/{npc_id}/reset` | Clear memory (all players, or `?player_id=`) |

All requests need `X-API-Key` when `MINIGPT_API_KEY` is set.

**Act**

```bash
curl localhost:8000/v1/npc/brakk/act -H "X-API-Key: change-me" -H "Content-Type: application/json" -d '{
  "state": {"map":"snow","time_of_day":"night","weather":"storm","hp":20,"max_hp":100,
            "nearby":[{"id":"wolf_1","kind":"wolf","distance":6,"direction":"north","hostile":true}]},
  "actions": [{"type":"idle"},{"type":"attack","targets":["wolf_1"]},{"type":"flee","targets":["wolf_1"]}],
  "bark": true
}'
# -> {"action":{"type":"flee","target":"wolf_1"},"scores":{...},"say":"..."}
```

**Talk (streaming)** returns SSE events: `data: {"delta": "..."}`, then `data: {"done": true, "reply": "..."}`.

### Game integration

See `examples/game_client.js`:

- `NpcBrain.talk()` streams tokens into a speech bubble.
- `NpcBrain.act()` returns an action.
- `thinkTick()` maps actions to movement.

Call `act` every ~2 s per NPC, not every frame. Call the API from the Node game server, not the browser, so the key stays private.

New actions only need a new `type` in the request plus training examples for it (`scripts/make_npc_sft.py`).

## Limitations

- A ~30M model holds short conversations but hallucinates, and can mention things that aren't in the state. Keep replies short (`max_tokens` ~60–120), use low temperature for factual NPC lines, and treat it as flavor.
- Action quality depends on the SFT data. The included rules are a starting point. Better: log real game situations and label them, or have a larger model write varied dialogue into another jsonl and pass it via `--jsonl`.
- One forward pass at a time (global lock). Fine for a handful of NPCs on CPU. For many NPCs, batch requests or run on GPU.
