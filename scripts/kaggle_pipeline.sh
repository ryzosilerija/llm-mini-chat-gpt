#!/usr/bin/env bash
# Full pipeline for a Kaggle GPU notebook (Settings: GPU T4 x2 or P100, Internet ON).
#   !git clone https://github.com/ryzosilerija/mini-chatgpt && cd mini-chatgpt && bash scripts/kaggle_pipeline.sh
# Kaggle sessions are capped (~12h): if pretraining is cut off, save /kaggle/working/mini-chatgpt/checkpoints
# as a dataset, attach it to the next session, copy it back and rerun with RESUME=1.
set -euo pipefail

pip install -q -r requirements.txt
RESUME_FLAG=""
[[ "${RESUME:-0}" == "1" ]] && RESUME_FLAG="--resume"

# 1. Corpus (~2B chars ≈ 450M tokens). Drop your Gutenberg .txt files into data/raw/gutenberg/ too.
if [[ ! -d data/raw/fineweb ]]; then
  python scripts/fetch_corpus.py --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \
    --out data/raw/fineweb --max_chars 2000000000
fi

# 2. Tokenizer + token files
[[ -f checkpoints/tokenizer.json ]] || \
  python scripts/train_tokenizer.py --input "data/raw/**/*.txt" --vocab_size 16384 --out checkpoints/tokenizer.json
[[ -f data/pretrain/train.bin ]] || \
  python scripts/prepare_pretrain.py --input "data/raw/**/*.txt" --tokenizer checkpoints/tokenizer.json --out data/pretrain

# 3. Pretrain
python scripts/train.py --config configs/pretrain_30m.json $RESUME_FLAG

# 4. SFT data: general chat + synthetic NPC data (+ your own jsonl if you have it)
python scripts/make_npc_sft.py --n 20000 --out data/npc_sft.jsonl
python scripts/prepare_sft.py \
  --hf HuggingFaceTB/smoltalk:everyday-conversations HuggingFaceTB/smoltalk:systemchats-30k \
  --jsonl data/npc_sft.jsonl --out data/sft

# 5. SFT
python scripts/train.py --config configs/sft_30m.json

echo "Download checkpoints/sft/best.pt and checkpoints/sft/tokenizer.json from the Output tab."
