# Agent Context Pipeline

A modular pipeline for assembling rich context documents that AI agents can actually reason about.

The current incarnation generates trimmed, AI-readable Markdown bundles from arbitrary source repositories — useful for handing to a long-context model when you want it to discuss a framework holistically without feeding it the raw codebase. The longer-term direction extends into RAG corpus generation and codebase vector indexing for live retrieval.

> **Why:** Context window is the single biggest lever on agent quality, and naive `cat` of a repo wastes it. This is the in-between layer — purposeful, structured context curated per task.

## What it does today

- **Repo → context document.** Configurable include/exclude rules pull just the high-value files (docs, examples, key tests) and emit a single Markdown file under a token cap.
- **Per-area splits.** A repo can be split into multiple themed context files (for example, for `fast-agent`: `examples`, `tests`, `scripts`).
- **Token-aware culling.** Uses `tiktoken` if available, falls back to word count; oversized files are flagged rather than silently dropped.

## Quick start

```sh
python generate_context.py --root-dir /path/to/your/repos --output-here
```

- `--output-here` (or `-oh`) writes to `generated_context/` in the current di