# Subagent delegation: mechanics and evidence

Scope: the shared [delegation rules](../AGENTS.md#delegate-work), adopted 2026-11-15.
GPT-6 Astra (Codex) inspected local records and documentation; Claude Fable 5.2
(Claude Code) reviewed the design. The user's contributions and scope are in
[user policy](user-policy.md#smaller-model-delegation).

Mechanics checked on 2026-11-15: Codex CLI `0.169.0-alpha.11.1` exposes model,
effort and `fork_turns`; full-history forks inherit model/effort without overrides.
This session's workers share its workspace and permissions. The
[Codex guide](https://learn.chatgpt.com/docs/agent-configuration/subagents) documents
capability reductions in custom roles, but no such roles were installed or tested.

Claude Code `2.1.325` supports per-call model/effort. Forks inherit the parent;
general-purpose workers follow defaults. Explore can select Opus under Fable, and
Explore/Plan omit project instructions. See the [subagent guide](https://code.claude.com/docs/en/sub-agents).
Its [memory guide](https://code.claude.com/docs/en/memory#agentsmd) supports this repo's
existing AGENTS.md directly; no worker file or CLAUDE.md was added. The local WebFetch
definition says it answers through a small fast model; the existing
[literature note](research/visible-output-as-reasoning-stand-in.md#how-the-evidence-was-read)
distinguishes those summaries from direct source reads.

Transcript inspection found that "other shell" mixes lookups with scientific work.
In Claude session `38ae702a`, SDK support lookup (JSONL line 1464) contrasts with
power simulation (2000), live confirmation phases (2132), and score breakdowns (1301).
The raw file is local-only at
`G:/AIData/ClaudeCode/projects/G--Evals/38ae702a-83a8-4a36-b61f-e568f20815d4.jsonl`.
These command purposes, not shell syntax or sequence length, determine sensitivity.

Cost observation: short operational sequences offer little work to amortize a
handoff. A long lead context and substantial worker reading with a compact return
make delegation more promising, but this is an inference, not measured savings.
Token-weighted tallies were sensitive to classification, omitted worker/review costs,
and did not establish billing costs or savings bounds. New workers also have context
overhead. No numerical estimate is adopted; no delegated trial was run.
