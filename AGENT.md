# ENDEAVOR_LOCAL_AGENT_TH — Agent Overview

Quick operating map for an agent working directly in this public repository.

Read order:

1. [`CLAUDE.md`](CLAUDE.md) — hard constraints.
2. **This file** — architecture and quick workflow.
3. [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md) — full procedure.
4. [`README.md`](README.md) — authoritative public product/runtime documentation.

## Core architecture

The project is a local Thai-capable agent built around a local MLX model and LangGraph/ReAct orchestration, with multiple front ends sharing runtime infrastructure.

Important components documented by the project include:

- `react.py` / planner logic — model/tool reasoning and planning;
- `graph.py` — LangGraph state/routing/retry behavior;
- `llm.py` — local OpenAI-compatible MLX client;
- `runtime_common.py` — shared runtime behavior used by CLI/Web paths;
- `agent_server.py` — authenticated WebSocket/REST server;
- `endeavor_agent.py` — CLI path;
- `awake_engine.py` — standing-trigger execution;
- `workspace/`, logs/history/memory — runtime state.

Some public-tree components are compiled `.so` files. Treat them as opaque when their source is absent.

## Start every task

Before editing:

1. inspect Git status;
2. identify which runtime/front end is affected;
3. read the relevant README architecture/security section;
4. identify shared code paths before duplicating behavior;
5. state the exact invariant and realistic verification available in the public tree.

## Hard boundaries

Preserve:

- local model/API use by default;
- authenticated Web/WS access;
- workspace-only writes and protected sensitive paths;
- process-level sandboxing and bounded execution;
- symlink/traversal defenses;
- shared runtime semantics across front ends;
- persistence/history/memory formats unless intentionally migrated;
- standing-trigger safety and lifecycle;
- model-specific tool/prompt/generation tuning.

## Authentication

`.agent_token` is local secret runtime state and must never be committed or echoed. Do not weaken auth because it is inconvenient in development. `AGENT_AUTH_DISABLED=1` is a temporary dev-only escape hatch and must not become the normal launch path.

## Model verification

The harness is tuned for its documented production model. A different model may be useful for experimentation, but its behavior is not production verification.

When changing prompts, tools, planner/routing, context trimming, or generation parameters, explicitly check whether the production model path is available. If it is not, report that limit rather than silently substituting another model.

## Compiled artifact rule

If source for a `.so` is absent:

- do not claim source audit of its internals;
- do not patch/reverse-engineer it by default;
- reason from interfaces and observed behavior only;
- do not promise a source fix that cannot be made in this repository.

## Testing

This public tree does not currently document one universal deterministic root test command comparable to the other public repos. Before testing, inspect the currently available public source/test/developer artifacts and use only supported entry points you can actually verify.

Do not invent a command or describe compiled developer artifacts as source tests. Separate static/source checks, local runtime checks, and live production-model checks in the final report.

## Git/release hygiene

Do not commit `.agent_token`, `.env`, logs, history DBs, memory files, workspaces, credentials, private documents, or personal absolute paths.

Full workflow: [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md).

**Mental model:** one local agent runtime, multiple front ends, shared infrastructure, explicit auth/sandbox boundaries, model-specific behavior.
