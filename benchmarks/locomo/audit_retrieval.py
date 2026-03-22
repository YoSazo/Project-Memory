from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from benchmarks.locomo.run_memla_benchmark import (
    LOCOMO_CATEGORY_NAMES,
    ROOT,
    MemlaBenchmarkRunner,
    _coalesce_turns,
    _load_build_conv_module,
    _locomo_evidence_text,
    _new_runtime,
)
from memory_system.memory.episode_log import Chunk


GENERIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "last",
    "me",
    "my",
    "next",
    "of",
    "on",
    "or",
    "our",
    "since",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "up",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yesterday",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _tokens(text: str, *, ignored_tokens: Iterable[str]) -> set[str]:
    ignored = {t.lower() for t in ignored_tokens}
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {
        tok
        for tok in toks
        if len(tok) >= 3 and tok not in GENERIC_STOPWORDS and tok not in ignored
    }


def _evidence_targets(evidence: str) -> list[str]:
    lines = [line.strip() for line in re.split(r"[\r\n]+", evidence) if line.strip()]
    return lines or ([evidence.strip()] if evidence.strip() else [])


def _score_match(target_text: str, chunk_text: str, *, ignored_tokens: Iterable[str]) -> dict[str, Any]:
    target_norm = _normalize(target_text)
    chunk_norm = _normalize(chunk_text)
    exact_substring = bool(target_norm and target_norm in chunk_norm)

    target_tokens = _tokens(target_text, ignored_tokens=ignored_tokens)
    chunk_tokens = _tokens(chunk_text, ignored_tokens=ignored_tokens)
    overlap = sorted(target_tokens & chunk_tokens)
    recall = float(len(overlap)) / float(len(target_tokens)) if target_tokens else 0.0

    return {
        "exact_substring": exact_substring,
        "target_token_count": len(target_tokens),
        "chunk_token_count": len(chunk_tokens),
        "token_overlap": overlap,
        "token_recall": recall,
    }


def _best_match(
    targets: list[str],
    chunks: list[Chunk],
    *,
    ignored_tokens: Iterable[str],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for target in targets:
        for chunk in chunks:
            match = _score_match(target, chunk.text, ignored_tokens=ignored_tokens)
            score = float(match["exact_substring"]) * 10.0 + float(match["token_recall"])
            record = {
                "score": score,
                "target": target,
                "chunk_id": chunk.id,
                "chunk_type": chunk.chunk_type,
                "chunk_text": chunk.text,
                **match,
            }
            if best is None or score > float(best["score"]):
                best = record
    return best or {
        "score": 0.0,
        "target": "",
        "chunk_id": None,
        "chunk_type": None,
        "chunk_text": "",
        "exact_substring": False,
        "target_token_count": 0,
        "chunk_token_count": 0,
        "token_overlap": [],
        "token_recall": 0.0,
    }


def _present(match: dict[str, Any]) -> bool:
    return bool(match["exact_substring"] or float(match["token_recall"]) >= 0.6)


def _ignored_names(*names: str) -> set[str]:
    out = {"assistant", "user"}
    for name in names:
        for tok in re.findall(r"[a-z0-9]+", str(name).lower()):
            if tok:
                out.add(tok)
    return out


def _audit_record(
    *,
    dataset: str,
    sample_id: str,
    category: str,
    question_input: str,
    evidence: str,
    ground_truth: str,
    stored_chunks: list[Chunk],
    retrieved_chunks: list[Chunk],
    ignored_tokens: Iterable[str],
) -> dict[str, Any]:
    evidence_targets = _evidence_targets(evidence)
    evidence_in_storage = _best_match(evidence_targets, stored_chunks, ignored_tokens=ignored_tokens)
    evidence_in_retrieval = _best_match(evidence_targets, retrieved_chunks, ignored_tokens=ignored_tokens)

    answer_targets = [str(ground_truth).strip()] if str(ground_truth).strip() else []
    answer_in_storage = _best_match(answer_targets, stored_chunks, ignored_tokens=ignored_tokens) if answer_targets else None
    answer_in_retrieval = _best_match(answer_targets, retrieved_chunks, ignored_tokens=ignored_tokens) if answer_targets else None

    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "category": category,
        "question_input": question_input,
        "evidence": evidence,
        "ground_truth": ground_truth,
        "stored_chunk_count": len(stored_chunks),
        "retrieved_chunk_count": len(retrieved_chunks),
        "evidence_present_in_storage": _present(evidence_in_storage),
        "evidence_present_in_retrieval": _present(evidence_in_retrieval),
        "evidence_best_storage_match": evidence_in_storage,
        "evidence_best_retrieval_match": evidence_in_retrieval,
        "answer_present_in_storage": _present(answer_in_storage) if answer_in_storage is not None else None,
        "answer_present_in_retrieval": _present(answer_in_retrieval) if answer_in_retrieval is not None else None,
        "answer_best_storage_match": answer_in_storage,
        "answer_best_retrieval_match": answer_in_retrieval,
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "total_records": len(records),
        "evidence_present_in_storage": 0,
        "evidence_present_in_retrieval": 0,
        "answer_present_in_storage": 0,
        "answer_present_in_retrieval": 0,
        "by_dataset": {},
        "by_category": {},
    }

    dataset_counters: dict[str, Counter[str]] = defaultdict(Counter)
    category_counters: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        if record["evidence_present_in_storage"]:
            out["evidence_present_in_storage"] += 1
        if record["evidence_present_in_retrieval"]:
            out["evidence_present_in_retrieval"] += 1
        if record["answer_present_in_storage"]:
            out["answer_present_in_storage"] += 1
        if record["answer_present_in_retrieval"]:
            out["answer_present_in_retrieval"] += 1

        for bucket, key in ((dataset_counters, record["dataset"]), (category_counters, record["category"])):
            bucket[key]["count"] += 1
            bucket[key]["evidence_present_in_storage"] += int(bool(record["evidence_present_in_storage"]))
            bucket[key]["evidence_present_in_retrieval"] += int(bool(record["evidence_present_in_retrieval"]))
            bucket[key]["answer_present_in_storage"] += int(bool(record["answer_present_in_storage"]))
            bucket[key]["answer_present_in_retrieval"] += int(bool(record["answer_present_in_retrieval"]))

    out["by_dataset"] = {k: dict(v) for k, v in sorted(dataset_counters.items())}
    out["by_category"] = {k: dict(v) for k, v in sorted(category_counters.items())}
    return out


def parse_args() -> argparse.Namespace:
    default_locomo = ROOT.parent / "external" / "locomo" / "data" / "locomo10.json"
    default_locomo_plus_repo = ROOT.parent / "external" / "Locomo-Plus"

    parser = argparse.ArgumentParser(description="Audit Memla retrieval coverage on LoCoMo/Locomo-Plus.")
    parser.add_argument("--dataset", choices=["locomo", "locomo_plus", "both"], default="both")
    parser.add_argument("--locomo-file", type=Path, default=default_locomo)
    parser.add_argument("--locomo-plus-repo", type=Path, default=default_locomo_plus_repo)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=ROOT / "benchmarks" / "locomo" / "results" / "memla_retrieval_audit.json",
    )
    parser.add_argument("--model", type=str, default="qwen3.5:4b")
    parser.add_argument("--extractor", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--max-conversations", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--max-plus-samples", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--reset-between-conversations", action="store_true")
    return parser.parse_args()


def run_locomo_audit(
    *,
    runner: MemlaBenchmarkRunner,
    locomo_file: Path,
    max_conversations: int,
    max_questions: int,
    reset_between_conversations: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    samples = json.loads(locomo_file.read_text(encoding="utf-8"))[:max_conversations]

    shared_runtime = None
    if not reset_between_conversations:
        shared_runtime = _new_runtime(
            user_id="memla_locomo_audit",
            extractor=runner.extractor,
            model=runner.model,
            num_ctx=runner.num_ctx,
        )

    try:
        for conv_index, sample in enumerate(samples):
            runtime = shared_runtime
            if runtime is None:
                runtime = _new_runtime(
                    user_id=f"memla_locomo_audit_{conv_index}",
                    extractor=runner.extractor,
                    model=runner.model,
                    num_ctx=runner.num_ctx,
                )

            try:
                runner.ingest_locomo_conversation(runtime=runtime, sample=sample)
                stored_chunks = runtime.log.fetch_top_level_chunks(user_id=runtime.user_id, limit=9999)
                conversation = sample["conversation"]
                ignored_tokens = _ignored_names(conversation.get("speaker_a", ""), conversation.get("speaker_b", ""))
                questions = list(sample.get("qa") or [])[:max_questions]

                for qa in questions:
                    question = str(qa.get("question") or "").strip()
                    category = LOCOMO_CATEGORY_NAMES.get(int(qa.get("category") or 0), f"category_{qa.get('category')}")
                    retrieved = runtime.chunks.retrieve(user_id=runtime.user_id, query_text=question, k=runner.top_k)
                    records.append(
                        _audit_record(
                            dataset="locomo",
                            sample_id=str(sample.get("sample_id") or conv_index),
                            category=category,
                            question_input=question,
                            evidence=_locomo_evidence_text(conversation, qa.get("evidence") or []),
                            ground_truth=str(qa.get("answer") or ""),
                            stored_chunks=stored_chunks,
                            retrieved_chunks=retrieved,
                            ignored_tokens=ignored_tokens,
                        )
                    )
            finally:
                if reset_between_conversations:
                    runtime.close()
    finally:
        if shared_runtime is not None:
            shared_runtime.close()

    return records


def run_locomo_plus_audit(
    *,
    runner: MemlaBenchmarkRunner,
    locomo_file: Path,
    locomo_plus_repo: Path,
    max_plus_samples: int,
    reset_between_conversations: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    builder = _load_build_conv_module(locomo_plus_repo)
    locomo_samples = json.loads(locomo_file.read_text(encoding="utf-8"))
    plus_samples = json.loads((locomo_plus_repo / "data" / "locomo_plus.json").read_text(encoding="utf-8"))[:max_plus_samples]

    shared_runtime = None
    if not reset_between_conversations:
        shared_runtime = _new_runtime(
            user_id="memla_locomo_plus_audit",
            extractor=runner.extractor,
            model=runner.model,
            num_ctx=runner.num_ctx,
        )

    try:
        for sample_index, plus_sample in enumerate(plus_samples):
            runtime = shared_runtime
            if runtime is None:
                runtime = _new_runtime(
                    user_id=f"memla_locomo_plus_audit_{sample_index}",
                    extractor=runner.extractor,
                    model=runner.model,
                    num_ctx=runner.num_ctx,
                )

            try:
                locomo_item = locomo_samples[sample_index % len(locomo_samples)]
                context = builder.build_context(plus_sample, locomo_item)
                query_turns = list(context.get("query_turns") or [])
                dialogue = list(context.get("dialogue") or [])
                history_turns = dialogue[:-len(query_turns)] if query_turns else dialogue
                runner.ingest_turn_blocks(
                    runtime=runtime,
                    session_id=f"locomo_plus_{sample_index}",
                    turn_blocks=_coalesce_turns(history_turns),
                    user_speaker=str(context.get("speaker_a") or ""),
                )

                stored_chunks = runtime.log.fetch_top_level_chunks(user_id=runtime.user_id, limit=9999)
                query_text = "\n".join(
                    f"{turn.get('speaker', 'Unknown')}: {str(turn.get('text') or '').strip()}"
                    for turn in query_turns
                    if str(turn.get("text") or "").strip()
                )
                evidence_text = "\n".join(
                    f"{turn.get('speaker', 'Unknown')}: {str(turn.get('text') or '').strip()}"
                    for turn in (context.get("cue_turns") or [])
                    if str(turn.get("text") or "").strip()
                )
                retrieved = runtime.chunks.retrieve(user_id=runtime.user_id, query_text=query_text, k=runner.top_k)
                ignored_tokens = _ignored_names(context.get("speaker_a", ""), context.get("speaker_b", ""))

                records.append(
                    _audit_record(
                        dataset="locomo_plus",
                        sample_id=f"locomo_plus_{sample_index}",
                        category="Cognitive",
                        question_input=query_text,
                        evidence=evidence_text,
                        ground_truth="",
                        stored_chunks=stored_chunks,
                        retrieved_chunks=retrieved,
                        ignored_tokens=ignored_tokens,
                    )
                )
            finally:
                if reset_between_conversations:
                    runtime.close()
    finally:
        if shared_runtime is not None:
            shared_runtime.close()

    return records


def main() -> int:
    args = parse_args()

    runner = MemlaBenchmarkRunner(
        model=args.model,
        backend="mock",
        top_k=args.top_k,
        temperature=0.0,
        num_ctx=args.num_ctx,
        extractor=args.extractor,
        train_online=False,
        max_turns=args.max_turns,
        train_steps=1,
    )

    records: list[dict[str, Any]] = []
    if args.dataset in {"locomo", "both"}:
        records.extend(
            run_locomo_audit(
                runner=runner,
                locomo_file=args.locomo_file,
                max_conversations=int(args.max_conversations),
                max_questions=int(args.max_questions),
                reset_between_conversations=args.reset_between_conversations,
            )
        )
    if args.dataset in {"locomo_plus", "both"}:
        records.extend(
            run_locomo_plus_audit(
                runner=runner,
                locomo_file=args.locomo_file,
                locomo_plus_repo=args.locomo_plus_repo,
                max_plus_samples=int(args.max_plus_samples),
                reset_between_conversations=args.reset_between_conversations,
            )
        )

    summary = _summarize(records)
    payload = {"summary": summary, "records": records}
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote retrieval audit for {len(records)} records to {args.output_file}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
