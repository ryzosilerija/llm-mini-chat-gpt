#!/usr/bin/env bash
# SFT on top of an already-pretrained NorMuon run (no pretraining here).
# In the Kaggle notebook: Add Input -> your pretraining notebook's output, then:
#   !git clone https://github.com/ryzosilerija/mini-chatgpt && cd mini-chatgpt && \
#    RUN=/kaggle/input/<your-pretrain-notebook>/runs/pretrain bash scripts/kaggle_sft_imported.sh
set -euo pipefail
: "${RUN:?set RUN to the folder with model.pt (or ckpt.pt + config.json) and tokenizer.json}"

pip install -q -r requirements.txt
python scripts/import_pretrained.py --run "$RUN" --out checkpoints/imported

python scripts/make_npc_sft.py --n 12000 --out data/npc_sft.jsonl
python scripts/prepare_sft.py \
  --hf "HuggingFaceTB/smoltalk:everyday-conversations*4" "allenai/soda@40000" \
  --jsonl data/npc_sft.jsonl --out data/sft

python scripts/train.py --config configs/sft_imported.json
echo "Download checkpoints/sft/best.pt and checkpoints/sft/tokenizer.json from the Output tab."
