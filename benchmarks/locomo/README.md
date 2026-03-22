# Memla LoCoMo Bench

This directory contains a Memla-native runner for the official **LoCoMo** dataset and the official **Locomo-Plus** repository.

It does three things:

1. Ingests the benchmark conversations into Memla with isolated benchmark-only SQLite/adapters.
2. Produces judge-compatible prediction JSON for LoCoMo QA and Locomo-Plus Cognitive.
3. Lets you smoke-test the pipeline with no external LLM by using `--backend mock`.

## Repo setup

The runner expects:

- `C:/Users/samat/Project Memory/external/locomo`
- `C:/Users/samat/Project Memory/external/Locomo-Plus`

Those are the official repos cloned during setup.

## Smoke test

```powershell
py -3 -m benchmarks.locomo.run_memla_benchmark --smoke --dataset both
```

This runs:

- 1 LoCoMo conversation
- 3 LoCoMo questions
- 3 Locomo-Plus cognitive samples
- `--backend mock`
- fast ingest mode (no online LoRA updates)

and writes:

- `benchmarks/locomo/results/memla_predictions.json`

## Real run

Set your usual Memla generation backend first.

### Ollama

```powershell
$env:LLM_PROVIDER="ollama"
$env:LLM_BASE_URL="http://127.0.0.1:11434"
py -3 -m benchmarks.locomo.run_memla_benchmark --dataset both --model qwen3.5:4b
```

### OpenAI-compatible

```powershell
$env:LLM_PROVIDER="openai"
$env:LLM_BASE_URL="https://api.openai.com"
$env:LLM_API_KEY="YOUR_KEY"
py -3 -m benchmarks.locomo.run_memla_benchmark --dataset both --model gpt-4o-mini
```

Useful switches:

- `--reset-between-conversations`
  Use this for a cold-start baseline.
- `--train-online`
  Uses Memla's heavier turn-by-turn online training path during ingest. This is the continual-learning experiment, and on CPU it is currently very slow.
- `--train-steps N`
  Shrinks online update size. Useful if you want to probe the adaptive path on limited hardware.
- `--extractor llm`
  Uses the Memla LLM extractor instead of heuristic extraction.
- `--max-conversations N`
- `--max-questions N`
- `--max-plus-samples N`
- `--max-turns N`
  Caps ingested turn blocks per conversation/sample. Useful for quick online-training checks.

## Official judging

The output JSON matches the structure expected by the official Locomo-Plus judge.

```powershell
py -3 .\external\Locomo-Plus\evaluation_framework\task_eval\llm_as_judge.py `
  --input-file .\Project-Memory\benchmarks\locomo\results\memla_predictions.json `
  --out-file .\Project-Memory\benchmarks\locomo\results\memla_judged.json `
  --summary-file .\Project-Memory\benchmarks\locomo\results\memla_judge_summary.json `
  --model gpt-4o-mini `
  --backend call_llm `
  --concurrency 4
```

## Notes

- By default the runner keeps memory/adapters continuous within each dataset split, which is the learning-curve setting.
- `--reset-between-conversations` gives you the static baseline.
- The runner uses isolated temporary SQLite databases and adapter directories, so it does not contaminate your normal Memla workspace.
- The fast default ingest path is what is practical on CPU right now. The online-learning path is still best treated as a long-running or GPU-backed experiment.
