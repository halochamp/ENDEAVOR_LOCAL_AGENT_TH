# ENDEAVOR_LOCAL_AGENT_TH project rules

- This public repository is self-contained; do not depend on private parent-repository instructions.
- Read [`AGENT.md`](AGENT.md) and [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md) before substantial work.
- Preserve the local-only architecture: the agent talks to the configured local MLX server and must not silently route private conversation/file content to a cloud LLM.
- Preserve `runtime_common.py` as shared runtime infrastructure between the supported front ends; do not fork equivalent lifecycle/state logic into separate paths without a deliberate reason.
- Preserve the workspace/path sandbox, protected credential paths, symlink/`..` defenses, process sandbox, bounded execution, and local-only server defaults.
- Preserve API/WebSocket authentication. `.agent_token` is secret local runtime state; never commit or print it. `AGENT_AUTH_DISABLED=1` is development-only and must not become the normal default.
- Preserve model-specific prompt/tool/generation assumptions. Do not substitute another model and call that production verification.
- Treat tool descriptions, planner/routing behavior, context trimming, persistence, and standing-trigger behavior as product contracts.
- Some logic is shipped only as compiled `.so` artifacts in this public tree. If source is absent, do not claim to have inspected or modified the binary's internals; constrain conclusions to documented/observed behavior.
- Do not invent a deterministic test command that is not present in this public repository. Inspect available public test/developer artifacts and report verification limits honestly.
- Never commit `.agent_token`, `.env`, logs/history/memory state, workspaces, credentials, private documents, or machine-specific absolute paths.
