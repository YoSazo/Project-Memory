# Project Memory

**The AI that never forgets you — and lets you show it what matters.**

Every LLM session starts from zero. You re-explain your stack, your context, your decisions — every single time. This fixes that. Persistently. Locally.

But memory alone isn't enough. Current chat interfaces have one input: text. The system guesses what context matters. Project Memory introduces a **spatial prompt interface** — you select memories on a live knowledge graph, draw connections between them, and *then* type your question. The model receives both your text and the relational structure you explicitly chose.

Works with **Ollama** (local), **Anthropic** (Claude), or **OpenAI-compatible APIs**. Your memory is **SQLite on your machine**. Your retrieval model trains locally via LoRA.

## What makes this different

1. **You prompt with a graph, not just text.** Click memory nodes to pin them as context. The model answers through the lens you selected — not the lens it guessed.

2. **Drawing connections trains retrieval.** When you link two memories, the system fires a contrastive training signal that pulls their embeddings closer. Your act of organizing knowledge teaches the retriever how to retrieve.

3. **The learning loop is closed properly.** Training signal comes from whether the LLM *actually used* a retrieved memory in its response — not from the retriever scoring itself. This prevents the system from calcifying on its own biases.

4. **Generation is untouched.** Only the retrieval model (MiniLM) gets fine-tuned via LoRA. The generation LLM (Ollama/Claude/GPT) stays as a black box. No risk of degrading output quality.

## Quick start

**Prerequisites:** Python 3.11+, Ollama running locally

```bash
git clone https://github.com/YoSazo/Project-Memory.git
cd Project-Memory
pip install -r requirements.txt
```

### Web UI (recommended)

```bash
ollama serve
python app.py --model qwen3.5:4b
```

Opens `http://localhost:8765` in your browser. Pick any local Ollama model from the dropdown.

**Flags:**
- `--port 8765` — server port
- `--model qwen3.5:4b` — default model (any Ollama model)
- `--db ./memory.sqlite` — database path
- `--user_id default` — user identity for multi-user setups

### CLI mode

```bash
python -m memory_system.main --model qwen3.5:4b --db ./memory.sqlite --user_id default
```

### Anthropic / OpenAI

```bash
export LLM_PROVIDER=anthropic
export LLM_API_KEY="sk-ant-..."
python -m memory_system.main --model claude-sonnet-4-6 --db ./memory.sqlite
```

```bash
export LLM_PROVIDER=openai
export LLM_API_KEY="YOUR_KEY"
export LLM_BASE_URL="https://api.openai.com"
python -m memory_system.main --model gpt-4o-mini --db ./memory.sqlite
```

## The interface

```
┌─────────────────────────────────────────────────────────┐
│  Project Memory          [model picker]    [New Session] │
├───────────────────┬─────────────────────────────────────┤
│                   │  Pinned: [fact: Byron] [note: ROAS] │
│   Memory Graph    │                                     │
│   (D3 force)      │  You: what should I do differently  │
│                   │       next month?                    │
│   ● facts         │                                     │
│   ● decisions     │  Assistant: Based on the Byron      │
│   ● entities      │  Creative and campaign fatigue data  │
│   ● notes         │  you highlighted...                  │
│   ━━ your links   │                                     │
│                   ├─────────────────────────────────────┤
│  [search memories]│  [message input...          ] [Send] │
│                   │  [👍 Good] [👎 Bad] [Recall] [Clear] │
└───────────────────┴─────────────────────────────────────┘
```

**Graph interactions:**
- **Click** a node → pin it as context (purple glow, appears as chip above input)
- **Shift+Click** two nodes → connect/disconnect them (trains retrieval)
- **Shift+Drag** from one node to another → draw a connection
- **Drag** → move nodes
- **Scroll** → zoom
- **Search box** → filter/highlight by keyword
- **Esc** → clear all pins

## How it works

### The 5-step memory pipeline

1. **Episode + chunk logging** — Every message is logged to SQLite. Key facts, decisions, entities, and notes are extracted and stored as retrievable chunks.

2. **Hybrid retrieval + LoRA reranking** — Candidates are scored by semantic similarity (MiniLM embeddings) + keyword overlap + recency + frequency. A LoRA adapter on MiniLM reranks the top candidates.

3. **Closed-loop training** — After the LLM responds, the system measures which retrieved chunks were actually *used* in the response (token overlap). Used chunks = positive signal. Ignored chunks = negative signal. User corrections flip the signal. This trains the LoRA to surface what actually helps, not what scores highest.

4. **EWC weight protection** — Elastic Weight Consolidation uses Fisher information to protect important retrieval weights from catastrophic forgetting. Frequently-used retrieval pathways are "bolded" — harder to overwrite.

5. **Multi-user merge + safe subspace** — PCA extracts shared retrieval directions across users. Updates are projected into an agreement subspace so one user's adapter can't degrade another's.

### The spatial prompt layer (new)

On top of the pipeline, the web UI adds:

- **Pinned context injection** — Selected nodes bypass retrieval and get injected directly into the system prompt as highest-priority context.
- **User-drawn connections** — Persisted in SQLite (`user_links` table). Each link fires `micro_gradient_pass` bidirectionally — pulling the two chunks' embeddings closer in the LoRA's representation space.
- **Graph as training signal** — Every manual connection is a high-confidence contrastive pair. Over time, the retriever converges toward the user's mental model of how their knowledge is structured.

### Training signals (4 sources)

| Signal | Source | Confidence | Weight |
|--------|--------|------------|--------|
| Chunk usage | LLM referenced chunk in response | Medium | 1x |
| Correction | User's next message contradicts response | Medium | 0.6-0.8x |
| Explicit | `/good` or `/bad` command | High | 1x |
| Spatial | User drew a connection on the graph | Highest | 1x (bidirectional) |

## Commands (CLI mode)

- `/new_session` — new session (memory persists)
- `/recall` — show retrieved chunks from last turn
- `/good` — positive feedback on last response
- `/bad` — negative feedback on last response
- `/merge_adapters` — multi-user merge (Steps 4-5)
- `/exit` — quit

## Project structure

```
Project-Memory/
├── app.py                          # Web UI server (FastAPI + SSE streaming)
├── static/index.html               # Frontend (D3 graph + chat)
├── memory_system/
│   ├── main.py                     # CLI entry point
│   ├── ollama_client.py            # Universal LLM client (Ollama/OpenAI/Anthropic)
│   ├── memory/
│   │   ├── episode_log.py          # SQLite persistence (episodes, chunks, user_links)
│   │   ├── chunk_manager.py        # Hybrid retrieval (semantic + keyword + recency)
│   │   └── llm_extractor.py        # Chunk extraction from messages
│   ├── middleware/
│   │   ├── ttt_layer.py            # Turn-by-turn orchestration + learning loop
│   │   ├── context_builder.py      # System prompt assembly + deferred training
│   │   └── quality.py              # Chunk usage scoring + correction detection
│   ├── adapters/
│   │   ├── lora_manager.py         # MiniLM + LoRA loading/saving/embedding
│   │   ├── gradient_pass.py        # Micro gradient steps on retrieval LoRA
│   │   ├── ewc.py                  # Elastic Weight Consolidation
│   │   └── merge.py                # Multi-user PCA merge
│   └── projection/
│       └── gradient_filter.py      # Safe subspace projection
├── tests/                          # 13 tests covering all pipeline steps
├── requirements.txt
└── memory.sqlite                   # Your memory (local, yours)
```

## Requirements

```
requests>=2.31,<3
torch>=2.1
transformers>=4.40
peft>=0.7
safetensors>=0.4
sentence-transformers>=2.2
fastapi>=0.100
uvicorn>=0.20
```
