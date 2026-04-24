from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .probe import ProbeConfig, build_atlas


def main() -> None:
    p = argparse.ArgumentParser(description="Sub-Zero CLI")
    p.add_argument("--model", required=True, help="HF model id/path")
    p.add_argument("--corpora-dir", required=True)
    p.add_argument("--out", default="./atlas.pt")
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required for CLI usage") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=None,
        dtype=dtype,
        low_cpu_mem_usage=False,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    cfg = ProbeConfig(
        corpora_dir=args.corpora_dir,
        max_prompts_per_class=args.max_prompts,
        max_length=args.max_length,
    )
    atlas = build_atlas(model, tokenizer, cfg, task_batches=None, cache_path=args.out)
    print(f"Built atlas: layers={len(atlas.layers)} sacred={atlas.sacred_layers} -> {args.out}")


if __name__ == "__main__":
    main()
