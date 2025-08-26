# Multiagents

A collection of small, focused AI agents built with **LangChain**, **LangGraph**, and **DSPy**.  
Examples include a **Wiki RAG agent** for grounded question-answering and a **Fantasy Premier League (FPL) agent** for squad planning and constraint checking.  
This repository serves as a lightweight playground to explore orchestration patterns and agent design choices across frameworks.

---

## Features

- **Wiki RAG Agent**  
  Hybrid retrieval over Wikipedia (BM25 + embeddings) with prompt-aware context windows and citation-style answers.

- **FPL Agent**  
  Tool-using agent that fetches player data, enforces constraints (budget, positions, max-players-per-club), proposes and validates squads, and explains trade-offs.

- **Framework Mix & Match**  
  - *LangChain* for quick tool wiring and chains  
  - *LangGraph* for graph-shaped control flow (branch/repair/validate loops)  
  - *DSPy* for trainable prompting and module signatures  

---

## Repo Structure

```
.
├─ backend/                 # Agent backends, services, and utilities
├─ pyproject.toml           # Package metadata / build system
└─ requirements.txt         # Python dependencies
```

---

## Quickstart

1. **Create environment & install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set environment variables**
   ```bash
   export OPENAI_API_KEY=...       # or your chosen LLM provider key
   export TAVILY_API_KEY=...       # if using web search tools
   export LANGCHAIN_TRACING_V2="false"
   ```

3. **Run an example agent (local script)**
   ```bash
   python backend/examples/wiki_rag_agent.py --q "How do transformers use self-attention?"
   ```

4. **Start a simple API (if provided)**
   ```bash
   uvicorn backend.main:app --reload --port 8001
   ```

---

## Agents (Examples)

- **Wiki RAG Agent**
  - Uses a retriever (BM25 + embeddings) over Wikipedia snapshots or a web search tool.
  - Ranks, deduplicates, and stuffs/summarizes into the LLM with source attributions.

- **FPL Agent**
  - Tools: player-pool loader, constraint validator, cost calculator, explanation generator.
  - *LangGraph* loop: `propose → validate → (repair?) → explain → END`.

> Some agents may be implemented multiple ways to compare orchestration styles (pure LangChain vs LangGraph state graph; DSPy modules for signatures/reasoning).

---

## Design Notes

- **RAG Defaults**
  - Chunking with overlap; hybrid retrieval (keyword + vector).
  - Strict context window budgeting; anti-hallucination guardrails (schema enforcement, cite-before-claim).

- **Graphs over Chains**
  - For agents that must *repair* and *re-validate* (e.g., FPL squads), a graph with explicit edges is clearer and safer than a linear chain.

- **DSPy Modules**
  - Encapsulate signatures for “extract → reason → write” patterns, optionally trainable with small preference datasets.

---

## Development

- Format & lint:
  ```bash
  ruff check . && ruff format .
  ```

- Tests (if present):
  ```bash
  pytest -q
  ```

---

## Roadmap

- Add evaluation harness (task-specific metrics for RAG faithfulness and FPL constraint satisfaction).  
- Experiment with local LLMs and retrieval stores.  
- Multi-agent comparisons (planner / worker / critic) across frameworks.  

---

## License

MIT License (see `LICENSE`).
