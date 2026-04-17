#!/usr/bin/env python3
"""Confirm Exp15 checkpoint results against the original evaluation surface.

This runner does three things the original script does not do cleanly in this
environment:
1. Restores the MemEval dependency from /tmp/MemEval.
2. Loads LoRA adapters with AutoPeftModelForCausalLM.
3. Uses the existing QA cache in strict mode so reproduced scores are
   confirmation runs, not fresh OpenAI calls.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]


def _resolve_better_memory_root() -> Path:
    env_path = os.environ.get("BETTER_MEMORY_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = [ROOT.parent / "better_memory"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


BETTER_MEMORY = _resolve_better_memory_root()
MEMEVAL_ROOT = Path(os.environ.get("MEMEVAL_ROOT", "/tmp/MemEval"))
MEMEVAL_SRC = Path(os.environ.get("MEMEVAL_SRC", str(MEMEVAL_ROOT / "src")))
MEMEVAL_LOCOMO = Path(os.environ.get("MEMEVAL_LOCOMO", str(MEMEVAL_ROOT / "data/locomo10.json")))
EXPECTED_LOCOMO = Path(os.environ.get("PRISM_LOCOMO_PATH", "/tmp/locomo10.json"))
QA_CACHE_PATH = BETTER_MEMORY / "openai_qa_cache.json"
OUT_DIR = ROOT / "results"

CHECKPOINTS = {
    "sft4": BETTER_MEMORY / "exp15_sft_qwen7b_4ep",
    "inferential_from_temporal_heavy": BETTER_MEMORY / "exp15_sft_qwen7b_inferential_from_temporal_heavy",
    "temporal_heavy": BETTER_MEMORY / "exp15_sft_qwen7b_temporal_heavy",
    "wave2_clean_plus_temporal": BETTER_MEMORY / "exp15_sft_qwen7b_wave2_clean_plus_temporal",
}

LOGGED_RESULTS = {
    "sft4": {
        "locomo": 0.498,
        "lme": 0.477,
        "locomo_categories": {1: 0.334, 2: 0.498, 3: 0.261, 4: 0.514, 5: 0.884},
    },
    "inferential_from_temporal_heavy": {
        "locomo": 0.498,
        "lme": 0.469,
        "locomo_categories": {1: 0.328, 2: 0.510, 3: 0.282, 4: 0.508, 5: 0.860},
    },
    "temporal_heavy": {
        "locomo": 0.493,
        "lme": 0.469,
        "locomo_categories": {1: 0.330, 2: 0.540, 3: 0.213, 4: 0.524, 5: 0.860},
    },
    "wave2_clean_plus_temporal": {
        "locomo": 0.476,
        "lme": 0.471,
        "locomo_categories": {1: 0.291, 2: 0.472, 3: 0.225, 4: 0.533, 5: 0.860},
    },
}


class CacheMissError(RuntimeError):
    """Raised when a strict cache-only evaluation encounters a missing QA key."""


def ensure_paths() -> None:
    if str(BETTER_MEMORY) not in sys.path:
        sys.path.insert(0, str(BETTER_MEMORY))
    if str(MEMEVAL_SRC) not in sys.path:
        sys.path.insert(0, str(MEMEVAL_SRC))


def ensure_datasets() -> None:
    ensure_paths()
    from agents_memory.benchmarks.longmemeval import _download_split
    from agents_memory.locomo import download_locomo

    download_locomo()
    _download_split("oracle")
    if MEMEVAL_LOCOMO.exists() and not EXPECTED_LOCOMO.exists():
        shutil.copyfile(MEMEVAL_LOCOMO, EXPECTED_LOCOMO)


def patch_learned_extractor() -> None:
    ensure_paths()
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig
    import torch

    import experiment15_learned_extraction as e15

    def _patched_init(self, model_path: str | Path, device: str = e15.DEVICE, load_in_4bit: bool = False):
        del device  # device_map handles placement.
        model_path = str(model_path)
        print(f"Loading extractor from {model_path}...")

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        common_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_cfg,
                **common_kwargs,
            )
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                **common_kwargs,
            )

        model.eval()
        self.model = model
        self.tok = tok
        print("  Extractor loaded.")

    e15.LearnedExtractor.__init__ = _patched_init


def build_cache_only_qa(model_name: str = "gpt-4.1", strict: bool = True):
    qa_cache = json.loads(QA_CACHE_PATH.read_text()) if QA_CACHE_PATH.exists() else {}
    stats = {
        "model": model_name,
        "cache_size": len(qa_cache),
        "hits": 0,
        "misses": 0,
        "missing_examples": [],
    }

    def qa_fn(question: str, ctx: str) -> str:
        key = hashlib.sha256(f"{model_name}||{question}||{ctx}".encode()).hexdigest()
        if key in qa_cache:
            stats["hits"] += 1
            return qa_cache[key]
        stats["misses"] += 1
        if len(stats["missing_examples"]) < 5:
            stats["missing_examples"].append(
                {
                    "question": question,
                    "ctx_head": ctx[:240],
                }
            )
        if strict:
            raise CacheMissError(
                f"Missing QA cache entry for question={question!r}. "
                "Strict cache-only mode cannot continue."
            )
        return "None"

    return qa_fn, stats


def category_means(results: dict, cats: list) -> dict:
    out = {}
    for cat in cats:
        out[str(cat)] = float(np.mean(results.get(cat, [0.0])))
    return out


def eval_checkpoint(alias: str, args: argparse.Namespace) -> dict:
    ensure_paths()
    import experiment15_learned_extraction as e15
    from experiment9_learned_encoder import LME_CATEGORY_NAMES, LOCOMO_CATEGORY_NAMES
    import torch

    ckpt = CHECKPOINTS[alias]
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. "
            "Set BETTER_MEMORY_ROOT to the original better_memory checkout."
        )

    if args.use_temporal_prompt:
        e15.SYSTEM_PROMPT = e15.SYSTEM_PROMPT_TEMPORAL

    qa_fn, qa_stats = build_cache_only_qa(strict=args.strict_cache)
    t0 = time.time()
    extractor = e15.LearnedExtractor(ckpt, load_in_4bit=True)

    print(f"Starting LoCoMo eval for {alias}...", flush=True)
    loco_results = e15.evaluate_on_locomo(
        extractor,
        qa_fn,
        context_window=args.context_window,
        temporal_k=args.locomo_temporal_k,
        adversarial_k=args.locomo_adversarial_k,
    )
    print(f"Finished LoCoMo eval for {alias}. Starting LME...", flush=True)
    lme_results = e15.evaluate_on_lme(
        extractor,
        qa_fn,
        n_per_cat=args.n_lme,
        context_window=args.context_window,
        multisess_k=args.lme_multisess_k,
    )
    print(f"Finished LME eval for {alias}.", flush=True)

    locomo_cats = sorted(LOCOMO_CATEGORY_NAMES.keys())
    lme_cats = sorted(LME_CATEGORY_NAMES.keys())
    locomo_cat_means = category_means(loco_results, locomo_cats)
    lme_cat_means = category_means(lme_results, lme_cats)
    locomo_mean = float(np.mean([locomo_cat_means[str(c)] for c in locomo_cats]))
    lme_mean = float(np.mean([lme_cat_means[str(c)] for c in lme_cats]))
    elapsed_min = (time.time() - t0) / 60.0

    del extractor
    gc.collect()
    torch.cuda.empty_cache()

    logged = LOGGED_RESULTS.get(alias, {})
    comparison = {}
    if logged:
        comparison = {
            "logged_locomo_mean": logged.get("locomo"),
            "logged_lme_mean": logged.get("lme"),
            "locomo_delta": None if logged.get("locomo") is None else locomo_mean - float(logged["locomo"]),
            "lme_delta": None if logged.get("lme") is None else lme_mean - float(logged["lme"]),
        }

    return {
        "alias": alias,
        "checkpoint": ckpt.name,
        "elapsed_min": round(elapsed_min, 2),
        "args": {
            "n_lme": args.n_lme,
            "context_window": args.context_window,
            "locomo_temporal_k": args.locomo_temporal_k,
            "locomo_adversarial_k": args.locomo_adversarial_k,
            "lme_multisess_k": args.lme_multisess_k,
            "use_temporal_prompt": args.use_temporal_prompt,
            "strict_cache": args.strict_cache,
        },
        "qa_cache": qa_stats,
        "locomo": {
            "categories": locomo_cat_means,
            "mean": locomo_mean,
        },
        "lme": {
            "categories": lme_cat_means,
            "mean": lme_mean,
        },
        "logged_comparison": comparison,
    }


def print_result(result: dict) -> None:
    alias = result["alias"]
    print(f"\n{'=' * 80}")
    print(f"{alias}")
    print(f"{'=' * 80}")
    print(
        f"LoCoMo={result['locomo']['mean']:.3f}  "
        f"LME={result['lme']['mean']:.3f}  "
        f"cache_hits={result['qa_cache']['hits']}  "
        f"cache_misses={result['qa_cache']['misses']}  "
        f"time={result['elapsed_min']:.1f}m"
    )
    if result["logged_comparison"]:
        print(
            f"vs logged: "
            f"LoCoMo {result['logged_comparison']['locomo_delta']:+.3f}, "
            f"LME {result['logged_comparison']['lme_delta']:+.3f}"
        )


def write_incremental_outputs(out_dir: Path, all_results: list[dict], failures: list[dict]) -> None:
    summary = {
        "generated_at_unix": int(time.time()),
        "results": all_results,
        "failures": failures,
    }
    summary_path = out_dir / "confirmed_exp15_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    for result in all_results:
        model_path = out_dir / f"{result['alias']}.json"
        model_path.write_text(json.dumps(result, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="sft4",
        help="Comma-separated checkpoint aliases.",
    )
    parser.add_argument("--n-lme", type=int, default=10)
    parser.add_argument("--context-window", type=int, default=0)
    parser.add_argument("--locomo-temporal-k", type=int, default=8)
    parser.add_argument("--locomo-adversarial-k", type=int, default=5)
    parser.add_argument("--lme-multisess-k", type=int, default=20)
    parser.add_argument("--use-temporal-prompt", action="store_true")
    parser.add_argument("--strict-cache", action="store_true", default=True)
    parser.add_argument("--allow-cache-misses", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    if args.allow_cache_misses:
        args.strict_cache = False

    aliases = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in aliases if m not in CHECKPOINTS]
    if unknown:
        raise SystemExit(f"Unknown model alias(es): {unknown}")

    ensure_datasets()
    patch_learned_extractor()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    failures = []
    for alias in aliases:
        try:
            result = eval_checkpoint(alias, args)
            all_results.append(result)
            print_result(result)
            write_incremental_outputs(out_dir, all_results, failures)
        except CacheMissError as exc:
            failures.append({"alias": alias, "error": str(exc)})
            write_incremental_outputs(out_dir, all_results, failures)
            print(f"\n[cache-miss] {alias}: {exc}")
        except Exception as exc:  # pragma: no cover - debug aid for long runs
            failures.append({"alias": alias, "error": f"{type(exc).__name__}: {exc}"})
            write_incremental_outputs(out_dir, all_results, failures)
            print(f"\n[failed] {alias}: {type(exc).__name__}: {exc}")

    summary_path = out_dir / "confirmed_exp15_summary.json"
    if not summary_path.exists():
        write_incremental_outputs(out_dir, all_results, failures)
    print(f"\nWrote {summary_path}")
    if all_results:
        ranked = sorted(
            all_results,
            key=lambda r: (r["locomo"]["mean"], r["lme"]["mean"]),
            reverse=True,
        )
        print("\nLeaderboard:")
        for idx, result in enumerate(ranked, start=1):
            print(
                f"  {idx}. {result['alias']:<34} "
                f"LoCoMo={result['locomo']['mean']:.3f} "
                f"LME={result['lme']['mean']:.3f}"
            )

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
