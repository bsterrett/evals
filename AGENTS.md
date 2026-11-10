# Agent instructions

This is a research workspace. Preserve the distinction between what is being
asked, what was observed, what is inferred, and what the user has directed.
Documentation is durable working context for agents; its policy basis is in
[user policy](docs/user-policy.md).

## Find the relevant context

Start with the [workspace README](README.md), then follow the relevant experiment
or research topic. Read the guidance and evidence relevant to the task; a full
documentation sweep is unnecessary.

| Need | Canonical home |
| --- | --- |
| Explicit durable user directions, including scoped ones | [User policy](docs/user-policy.md) |
| Shared implementation and operations | [Architecture](docs/architecture.md), [accounting](docs/accounting.md), and the referenced code/configuration |
| Creating an experiment | [Adding experiments](docs/adding-experiments.md) |
| Cross-experiment research | `docs/research/<topic>.md` |
| Experiment question and design | `experiments/<id>/README.md` |
| Frozen scientific methods | Sources mapped by `experiments/<id>/protocols/README.md` |
| Current synthesis and supporting findings | `experiments/<id>/notes/findings.md` |
| An investigation needing independent reuse | `experiments/<id>/notes/<topic>.md`, linked from its nearest entry point |

Keep one canonical home for each finding or instruction. Link from summaries
instead of maintaining parallel detailed accounts. Code/configuration describe
implemented behavior; a protocol describes the intended study. Record material
disagreements rather than silently treating one as proof of the other.

The user keeps a running list of open research questions in `open-questions.md` at
the workspace root. Read it when your task touches a topic it covers, and add the
questions your work raises, following the instructions at its top. Its contents are
never committed: `.gitignore` lists it, so Git worktrees and fresh clones don't have
it. In a worktree, use the copy in the main working tree; if there is none, skip it.
It is a working list, not a canonical home. Committed notes keep their own open
questions, and the list links to them.

## Save useful learning

Before finishing substantive work, check whether something learned would
materially change a future agent's actions or interpretation. If so, update the
smallest relevant document and include it in the normal task commit. This includes
informative negative results, calibration failures, expensive dead ends, reusable
setup constraints, and durable user directions. Routine progress and command
transcripts belong in the chat or run artifacts. Make a documentation change only
when there is useful content to preserve.

Reuse existing topic homes. Split a note when an investigation needs independent
reuse or would bury the current synthesis. Create evidence sidecars only when
tables or excerpts no longer fit conveniently in the note. Create files and
sections when content warrants them; per-run reports and empty scaffolding are
unnecessary.

## Identify the kind of claim

Use clear sections or explicit wording where ambiguity matters. A document can
contain several kinds of claim; a label on every sentence is unnecessary.

| Kind | Required distinction |
| --- | --- |
| Research question | What remains unknown. |
| Hypothesis | A proposed testable claim; whether it preceded the relevant observations or arose afterward. |
| Observation | What was measured or inspected and under which conditions. Distinguish automated scores, agent inspection, and an external source's report. |
| Assumption or supposition | An unverified premise and the reasoning that depends on it. |
| Inference | The evidence-to-conclusion reasoning, important alternatives, and limits. An untested explanation remains untested. |
| Reproduction | Prerequisites, commands, identities, expected checks, and which steps were actually exercised versus proposed. |
| Agent recommendation or choice | A proposed action or an implementation choice within authorized scope, with its rationale. |
| User policy | An attributable user direction with its scope and source; use the policy record below. |

Agent choices within existing authorization need no separate policy ratification.
Approval for a particular execution retains its stated scope. Scientific findings,
agent recommendations, and unattributed imperative prose do not establish user
policy. Record actual enduring user directions in `docs/user-policy.md`, preserving
a short faithful excerpt, scope, date, and message reference when available. An
excerpt must remain meaningful when the chat is unavailable. Preserve explicit
amendments or replacements. Document uncertainty about old attribution rather than
inventing it; apply current instructions and resolve a consequential ambiguity
when the work depends on it.

## Keep evidence and reproduction usable

Put support beside the observation or consequential inference it supports. Record
the relevant denominator, conditions, run ID, and protocol/prompt/scorer/dataset/
model identities. Reuse manifests and canonical source references instead of
copying every field. A whole-note source list does not imply that every claim has
the same support. Record external-source versions or review dates when they matter.

Generated `artifacts/` remain ignored. Commit concise findings and the small
measurements or exact excerpts needed to assess consequential interpretations.
State whether raw evidence is local-only, accessible elsewhere, or unavailable.
Paths and hashes locate or verify files; they do not replace missing evidence.
Separate a recorded finding from evidence independently inspected in this task.
Never put credentials or real secrets in documentation or evidence extracts.

Distinguish inspection/reanalysis of saved data from new generation. Link existing
scripts and setup instructions; preserve the identities needed to reproduce the
procedure. Mutable provider aliases and stochastic execution may prevent identical
outputs. A reproduction recipe supplies no additional execution or spending
authorization; follow the existing controls.

## Update when the research changes

Record prospective protocols before the relevant run. Hypotheses or analysis
choices developed after inspecting relevant outcomes remain exploratory for those
data. Update current synthesis when relevant work changes it, preserving the
evidence and methodological context of earlier findings.
Clarification and reorganization are allowed. Mark demonstrated errors beside the
affected finding and link replacement analysis; Git history alone is insufficient
notice. Reanalysis with another scorer is identifiable as a new analysis.

When relying on a finding, check the dependencies material to its applicability.
A changed model or prompt can limit transfer without erasing the old measurement.
Contradictory evidence or a relevant dependency change can warrant an update.
Inactive research needs no scheduled refresh, expiry, or archival transition.
Improve older notes when relevant work relies on or changes them.

## Optional note shape

Use ordinary Markdown, a descriptive title, a short scope statement, and only the
sections that help. Dates establish events or source provenance, not review duties.

```markdown
# <Topic>

Scope: <experiment, conditions, and exploratory/confirmatory context>

## Question and hypotheses
<Question, prospective or later hypotheses, assumptions, link to protocol.>

## Observations and evidence
<Measurements or inspections, with adjacent supporting references.>

## Interpretation and uncertainty
<Inferences, competing explanations, limits, unresolved questions.>

## Reproduce or inspect
<Prerequisites and commands or canonical links; tested versus proposed steps.>

## Next useful step
<Recommendation, discriminating test, or unresolved decision.>
```
