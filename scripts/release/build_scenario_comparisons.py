#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path

from confirm_exp15_results import (
    CHECKPOINTS,
    build_cache_only_qa,
    ensure_datasets,
    ensure_paths,
    patch_learned_extractor,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SHORTLIST_PATH = SCRIPT_DIR / "scenario_shortlist.json"
OUT_PATH = ROOT / "results" / "scenario_comparisons.json"
LOCOMO_PATH = Path(os.environ.get("PRISM_LOCOMO_PATH", "/tmp/locomo10.json"))


def load_shortlist() -> list[dict]:
    return json.loads(SHORTLIST_PATH.read_text())["scenarios"]


def load_locomo() -> dict[str, dict]:
    data = json.loads(LOCOMO_PATH.read_text())
    return {conv["sample_id"]: conv for conv in data}


def build_single_conv_system(conv: dict, extractor):
    ensure_paths()
    import experiment15_learned_extraction as e15
    from experiment3 import batch_embed
    from experiment7_locomo_props import is_filler
    from experiment9_learned_encoder import PRISMv3Rerank

    conversation = conv["conversation"]
    speaker_a = conversation.get("speaker_a", "A")
    speaker_b = conversation.get("speaker_b", "B")
    speakers = [speaker_a, speaker_b]

    session_keys = sorted(
        [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )
    turns = []
    session_dates = []
    for sk in session_keys:
        date_key = f"{sk}_date_time"
        date = conversation.get(date_key, "")
        if date:
            session_dates.append(e15.normalize_session_date(date))
        for t in conversation[sk]:
            tx = t.get("text", "").strip()
            if tx:
                turns.append({**t, "session_date": date, "_emb": None, "idx": len(turns), "_surprise_norm": 0.0})

    timeline_header = ""
    rich_timeline_header = ""
    if session_dates:
        session_summaries_dict = conv.get("session_summary", {})
        simple_lines = [f"  - Session {i+1}: {d}" for i, d in enumerate(session_dates)]
        timeline_header = "Conversation sessions:\n" + "\n".join(simple_lines) + "\n\n"
        rich_lines = []
        for i, (sk, d) in enumerate(zip(session_keys, session_dates)):
            summary = session_summaries_dict.get(f"{sk}_summary", "")
            if summary:
                rich_lines.append(f"  - Session {i+1} ({d}): {summary}")
            else:
                rich_lines.append(f"  - Session {i+1}: {d}")
        rich_timeline_header = "Conversation sessions:\n" + "\n".join(rich_lines) + "\n\n"

    texts = [
        f"[{t['session_date']}] {t['speaker']}: {t['text']}" if t.get("session_date") else f"{t['speaker']}: {t['text']}"
        for t in turns
    ]
    embs_orig = batch_embed(texts)
    for i, t in enumerate(turns):
        t["_emb"] = embs_orig[i]

    fact_turns = [t for t in turns if not is_filler(t["text"])]
    if fact_turns:
        sp_list = [t["speaker"] for t in fact_turns]
        tx_list = [t["text"] for t in fact_turns]
        dt_list = [t.get("session_date", "") for t in fact_turns]
        ctx_list = e15.build_turn_contexts(turns, fact_turns, 0)
        props_list = extractor.extract_batch(sp_list, tx_list, dt_list, contexts=ctx_list, bs=8)
    else:
        props_list = []

    sys_c = PRISMv3Rerank(speakers, beta=12.0, rerank_k=40)
    for t in turns:
        sys_c.ingest_raw(t)
    if fact_turns:
        dated_props = []
        for t, props in zip(fact_turns, props_list):
            raw_date = t.get("session_date", "")
            norm_date = e15.normalize_session_date(raw_date) if raw_date else ""
            processed = e15.postprocess_props(props, t["speaker"], raw_date)
            dated_props.append([f"[{norm_date}] {p}" if norm_date else p for p in processed])
        sys_c.ingest_props(fact_turns, dated_props)

    session_summaries_dict = conv.get("session_summary", {})
    if session_summaries_dict:
        summary_turns = []
        summary_props = []
        for sk, d in zip(session_keys, session_dates):
            summary_text = session_summaries_dict.get(f"{sk}_summary", "")
            if not summary_text:
                continue
            raw_date = conversation.get(f"{sk}_date_time", "")
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_text) if s.strip()]
            for sent in sentences:
                fake_turn = {
                    "speaker": speaker_a,
                    "text": sent,
                    "session_date": raw_date,
                    "idx": len(turns) + len(summary_turns),
                    "_emb": None,
                    "_surprise_norm": 0.0,
                }
                summary_turns.append(fake_turn)
                prop_str = f"[{d}] {sent}" if d else sent
                summary_props.append([prop_str])
        if summary_turns:
            sum_texts = [
                f"[{t['session_date']}] {t['text']}" if t.get("session_date") else t["text"]
                for t in summary_turns
            ]
            sum_embs = batch_embed(sum_texts)
            for i, t in enumerate(summary_turns):
                t["_emb"] = sum_embs[i]
            sys_c.ingest_props(summary_turns, summary_props)

    return {
        "system": sys_c,
        "timeline_header": timeline_header,
        "rich_timeline_header": rich_timeline_header,
    }


def answer_question(prepared: dict, question: str, category: int, qa_fn):
    system = prepared["system"]
    timeline_header = prepared["timeline_header"]
    rich_timeline_header = prepared["rich_timeline_header"]

    is_temporal_cat = category == 2
    is_multihop_cat = category == 4
    is_adversarial_cat = category == 5
    use_rich = (is_temporal_cat or is_multihop_cat) and bool(rich_timeline_header)
    header = rich_timeline_header if use_rich else timeline_header
    if is_multihop_cat:
        k_val = 10
    elif is_adversarial_cat:
        k_val = 5
    elif is_temporal_cat:
        k_val = 8
    else:
        k_val = 8

    retrieved = system.retrieve(question, k=k_val)
    ctx_parts = []
    top_retrieval = []
    for item in retrieved:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = str(item[0])
        ctx_parts.append(text)
        top_retrieval.append(text)
    ctx = header + ("\n".join(ctx_parts) or "No memory.")
    pred = qa_fn(question, ctx)
    return pred, top_retrieval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="sft4")
    args = parser.parse_args()
    import torch

    ensure_datasets()
    patch_learned_extractor()
    ensure_paths()

    import experiment15_learned_extraction as e15

    locomo = load_locomo()
    shortlist = load_shortlist()
    qa_fn, qa_stats = build_cache_only_qa(strict=True)
    aliases = [m.strip() for m in args.models.split(",") if m.strip()]

    scenarios_out = [{**scenario, "systems": []} for scenario in shortlist]
    scenario_by_conv: dict[str, list[dict]] = {}
    for scenario in scenarios_out:
        scenario_by_conv.setdefault(scenario["source_id"], []).append(scenario)

    for alias in aliases:
        extractor = e15.LearnedExtractor(CHECKPOINTS[alias], load_in_4bit=True)
        prepared_by_conv = {}
        for conv_id, conv_scenarios in scenario_by_conv.items():
            conv = locomo[conv_id]
            prepared = build_single_conv_system(conv, extractor)
            prepared_by_conv[conv_id] = prepared
            for scenario in conv_scenarios:
                prediction, top_retrieval = answer_question(
                    prepared,
                    scenario["question"],
                    int(scenario["category"]),
                    qa_fn,
                )
                scenario["systems"].append(
                    {
                        "name": alias,
                        "prediction": prediction,
                        "top_retrieval": top_retrieval,
                    }
                )

        del prepared_by_conv
        del extractor
        gc.collect()
        torch.cuda.empty_cache()

    out = {
        "qa_cache": qa_stats,
        "scenarios": scenarios_out,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
