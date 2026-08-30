# ENDEAVOR_LOCAL_AGENT_TH — Agent Procedure

Complete repository-agent workflow for this standalone public project.

Document roles:

- `AGENTS.md` — discovery entry point.
- `CLAUDE.md` — hard constraints.
- `AGENT.md` — quick architecture/workflow map.
- **This file** — full execution procedure.
- `README.md` — authoritative public product/runtime documentation.

## 1. Start-of-task procedure

Before changing code:

1. Read `CLAUDE.md` and `AGENT.md`.
2. Inspect Git status and preserve unrelated work.
3. Classify the change: orchestration/planner, shared runtime, CLI, Web/API, tool, persistence, standing trigger, model/generation, sandbox/security, packaging/docs.
4. Read the relevant README section.
5. Identify whether the affected logic is public Python/HTML source or an opaque compiled `.so`.
6. State the exact invariant and the strongest verification available from this public repository.

## 2. Architecture ownership

Do not duplicate shared lifecycle/state logic casually across front ends.

The documented architecture uses multiple entry points over shared infrastructure. Before changing one UI/runtime path, inspect whether the same behavior belongs in `runtime_common.py` or another shared layer.

Preserve consistent semantics across CLI, Web server, and desktop/runtime consumers where they intentionally share behavior.

A front-end-specific fix must not silently diverge from another supported path unless the difference is deliberate and documented.

## 3. Local-model boundary

The project is local-first.

- The normal LLM path is the configured local MLX/OpenAI-compatible server.
- Do not silently send conversation history, local file content, memory, or tool output to a cloud LLM.
- Do not substitute a different model and label the result production verification.
- Model/prompt/tool-routing behavior is tuned for the documented production model and may change materially on another model.

For prompt/planner/tool/generation changes, distinguish:

1. source/static verification;
2. deterministic helper verification;
3. local runtime verification;
4. live production-model behavior.

Report exactly which levels ran.

## 4. Authentication boundary

`agent_server.py` protects REST/WebSocket access with a local token.

Preserve these rules:

- `.agent_token` is secret runtime state;
- token creation/permissions must remain safe;
- requests without valid auth fail closed;
- do not log or expose the token;
- do not commit the token;
- `AGENT_AUTH_DISABLED=1` is development-only and must not become default behavior;
- disabling auth while a general browser or untrusted local client can reach the server is unsafe.

Any auth change requires explicit negative tests/verification for missing and invalid credentials where public source/testing permits it.

## 5. Filesystem and process sandbox

For tool/path/process changes, preserve:

- writes confined to the documented workspace unless an explicit dev-only mode is intentionally used;
- protected credential/system paths remain blocked;
- path checks use canonicalized paths and resist symlink/`..` traversal;
- process execution remains sandboxed according to the documented policy;
- timeouts/bounded execution remain enforced;
- errors/logs do not disclose sensitive local content unnecessarily.

Do not create an alternate tool path that bypasses the shared guardrail merely because the primary path refuses an action.

## 6. Tool changes

When changing a tool:

1. Read its registration/description, implementation, validators, and consumers.
2. Preserve the documented write/read/network boundary.
3. Keep outputs bounded and useful to the model.
4. Reject invalid input before side effects.
5. Keep model-facing descriptions aligned with runtime behavior.
6. Check whether the same tool semantics are assumed by CLI/Web/desktop paths.
7. Use synthetic test data rather than real private documents or credentials.

## 7. Planner/routing/orchestration changes

For `react`, planner, graph, prompt, or routing changes:

- define the expected classification/routing behavior before editing;
- avoid broad prompt changes to solve one narrow failure when a deterministic helper can solve it;
- check interactions with `create_plan`, tool selection, retries, synthesis, and context trimming;
- preserve single-source/shared behavior where documented;
- test with the intended production model when claiming production routing quality.

Do not conclude a limitation is a model ceiling until deterministic/prompt/routing causes have been examined.

## 8. Persistence/history/memory

Conversation history, persistent memory, web cache, and runtime state have different lifecycles.

When changing persistence:

- identify which store owns the data;
- preserve existing format/lifecycle or provide an explicit migration;
- do not mix process-memory caches with persistent user memory silently;
- bound loaded history/context;
- do not commit runtime databases, logs, or user memory;
- handle corruption/missing state fail-safely.

## 9. Standing triggers / awake engine

Standing triggers can cause work without a new interactive prompt, so changes require extra care.

Preserve:

- explicit trigger creation semantics;
- bounded/reasonable scheduling behavior;
- clear distinction between one-shot and recurring triggers;
- safe shutdown/restart lifecycle;
- no accidental duplicate firing from multiple runtime paths;
- the same auth/tool/sandbox restrictions as interactive work.

Do not make background execution a bypass around user-facing safety boundaries.

## 10. Web/API/UI changes

For `agent_server.py`, WebSocket/REST, `chat.html`, or desktop integration:

- preserve authentication;
- preserve localhost/local-machine assumptions unless deliberately redesigned;
- preserve append/stream/state semantics expected by clients;
- keep shared business logic out of duplicated UI-only branches when a shared runtime layer exists;
- validate malformed requests and disconnected clients safely;
- do not expose workspace file contents to third-party web resources by default.

## 11. Compiled `.so` artifacts

Some public-tree behavior is represented only by compiled extension artifacts.

If corresponding source is not present:

1. Treat the binary as opaque.
2. Do not claim line-level/source-level inspection.
3. Do not reverse-engineer or binary-patch it as the normal workflow.
4. Test only observable interfaces/behavior you can invoke safely.
5. If a requested fix requires missing source, state that limitation explicitly.
6. Do not fabricate source paths or infer exact internals from symbol/file names.

## 12. Testing procedure

Unlike several other public projects, this repository currently does not document one canonical root deterministic regression command and contains compiled developer artifacts rather than a clear public Python test suite.

Therefore:

1. Inspect the current public tree for actual test/source entry points before running anything.
2. Run syntax/static/import checks only where source and dependencies make them meaningful.
3. Run the smallest local runtime acceptance check for the changed path when safe.
4. Use the documented production model for production-behavior claims.
5. Never invent a test command or call a compiled artifact a source regression suite.
6. Report verification limits explicitly.

If a proper deterministic public test suite is added later, update `CLAUDE.md`, `AGENT.md`, this section, and README together.

## 13. Documentation changes

Keep roles distinct:

- `AGENT.md` = quick map;
- this file = detailed agent procedure;
- `CLAUDE.md` = hard constraints;
- README = product/user documentation.

Public agent docs must remain self-contained and must not rely on private parent-repository workflow.

## 14. Git/release hygiene

Before commit/push:

- inspect status/diff;
- stage only intended files;
- ensure `.agent_token`, `.env`, logs, history DBs, memory, workspace output, credentials, screenshots/private docs, and machine-specific absolute paths are not staged;
- distinguish source changes from regenerated/compiled artifacts;
- do not commit a regenerated binary unless the project release process intentionally requires it and its provenance is understood;
- never force-push unless explicitly requested and appropriate.

## 15. Completion criteria

A task is complete when applicable items hold:

- requested change is implemented in source actually available here;
- shared runtime/front-end semantics remain consistent;
- auth/sandbox/local-model boundaries remain intact;
- verification claims match what was actually tested;
- missing-source/compiled-artifact limitations are disclosed;
- final diff contains only intended changes;
- no runtime secret/private state is staged.

**Decision rule:** preserve shared local-runtime/security invariants; never compensate for missing evidence by guessing about opaque binaries.
