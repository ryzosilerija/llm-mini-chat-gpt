"""Chat with a checkpoint in the terminal.

  python scripts/chat_cli.py --ckpt checkpoints/sft/best.pt --tokenizer checkpoints/sft/tokenizer.json
Commands: /reset clears history, /quit exits.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minigpt.engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_tokens", type=int, default=200)
    args = ap.parse_args()

    eng = Engine(args.ckpt, args.tokenizer)
    print(f"loaded {eng.info}")
    history = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user == "/quit":
            break
        if user == "/reset":
            history = []
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        print("bot> ", end="", flush=True)
        reply = ""
        for delta in eng.stream_chat([{"role": "system", "content": args.system}, *history],
                                     max_tokens=args.max_tokens, temperature=args.temperature):
            reply += delta
            print(delta, end="", flush=True)
        history.append({"role": "assistant", "content": reply.strip()})
        print()


if __name__ == "__main__":
    main()
