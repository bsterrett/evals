# Agent instructions

This is a research workspace. Preserve the distinction between what is being
asked, what was observed, what is inferred, and what the user has directed.
Documentation is durable working context for agents and for the user, who must be
able to read it to review the work; its policy basis is in
[user policy](docs/user-policy.md). The workspace README is the only page written
for outside readers.

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

## Work with the user

The user directs this research and uses it to build their own research skills, and
to show them. Building the skills comes first. Agents keep the work rigorous and on
track, and the record says accurately who did what, without performing for an
audience. Spend the user's attention where it builds skill: routine steps should not
wait on them. The policy basis is the
[user policy entry](docs/user-policy.md#directing-the-research-and-building-skill),
as amended by [a lighter process](docs/user-policy.md#a-lighter-process).

**Ask before proposing, at three decisions.** Ask for the user's view before giving
yours on:

- what to test next;
- the hypothesis, and the user's prediction and why;
- what a result means.

Then add what the user missed. When the user says "just propose" or the equivalent,
propose first.

**Propose the rest.** For every other research decision, give your proposal with its
reasons and the main alternative, and let the user accept or change it. This covers
what counts as success and how it is measured, conditions and controls, the design
of a component, and amendments. Put related decisions in one message instead of
asking them one at a time. Implementation details, such as code structure and file
layout, remain agent choices: decide them and report them.

**Fix operational problems without waiting.** When a run crashes, truncates, blocks
the budget ledger or otherwise fails to complete, diagnose it at once, say what you
opened to do so, and propose the fix. A surprising result is different: it belongs
to the review below.

**Never hold back a material problem.** Correct work matters more than practice.
After the user has answered, or at once if waiting would waste money or invalidate a
result, say plainly what is wrong or missing.

**Review each run before the next.** After a paid run, or any run whose results go
into findings, design no further run until the user has reviewed it:

1. Give the user raw transcripts to read before showing your analysis. Your
   selection is a starting point; the user may ask for any others.
2. Record the user's notes on what they noticed and what they think the result
   means.
3. Then give your analysis, and say in the conversation what each of you caught that
   the other missed. That comparison is feedback for the user, not a list to commit.

Keep the review to one sitting. Other work can continue meanwhile, and smoke tests
are exempt. "Just propose" does not cover this step. When a result would change what
happens next, such as a positive or surprising result, also ask the user how it
could be an artifact before giving your own list.

**Probe before registering.** A run is either a probe or a confirmatory run. A probe
is a small look to see what happens, and to check that the change under test takes
and the tools work. Keep it to about 40 trials and $3, on development cases only,
never a confirmation partition, with the smallest code change that runs. Before it
runs, put three lines in the findings note: the question, the user's guess, and what
will be looked at. It takes one approval like any run. The review above still
applies, and the result goes in the note as a short paragraph marked exploratory. A
probe can motivate a hypothesis; it cannot support a finding. Move to a confirmatory
run, with a pre-registration and fresh cases, when the user wants to state a result
as a finding or the next step would exceed the cap. Build detectors, scorers and
decision rules then, for what the probes showed.

**Pre-registration.** For a confirmatory run, the user writes the hypothesis and
their prediction, with the reason, first. Draft the rest for the user to accept or
change: what result would change their mind, the assumptions the result depends on,
whether it would say anything about more capable models, and the sample size, test,
decision threshold and exclusions. Explain each so the user could defend it. For the
sample size, ask what effect would matter and work out the sample from that. Draft
configs, seeds and identities yourself.

**Explain methods and show measurement code.** Before a statistical or experimental
method enters a protocol, explain in plain terms what it tests, why it fits, and the
main alternative. Before a paid run, show the user the changes to code that shapes
the measurement: prompts, scorers, judges, sampling, refusal handling and budget
guards. Their review is part of the run's one approval.

**One approval per run.** When a run's design is settled, ask once. The question
names what will be committed and pushed, the measurement changes to review, the run
and its spending ceiling. A yes covers all of it: committing and pushing the
protocol and code before the run, the run up to that ceiling, and committing and
pushing the run's record once the review is done. Record the approval as one line in
that record, with the ceiling and the date. A change to the design after approval
needs a new approval, again as one question. Steps only the user can take, such as
checking provider billing to unblock the ledger, stay theirs. Outside runs, ask
about a commit and its push together.

**Attribute by exception.** Work in this repository is the agent's unless the record
says otherwise. Each study or research note says so once, naming the model and the
tool, for example "Claude Fable 5.2 (Claude Code)", and keeps one short list of what
the user contributed: the question, the hypothesis and prediction, decisions that
changed the design, readings of results, and errors caught. Use the user's own words
there when they wrote more than an approval. Don't label the origin of each item,
quote approvals, commit lists of who caught what, or narrate the bookkeeping in
chat. For a subagent, name the model it ran on; if the model is unknown, say so
instead of guessing. Apply this to new work; earlier records keep their wording.

## Delegate work

In sessions led by Astra, Fable or Opus, research judgment, work about the user's
research skills, and code that shapes the measurement (see **Explain methods and
show measurement code** above) stay on the lead's model and effort. Same-model
subagents are allowed. If unsure whether work is research-sensitive, treat it as
sensitive.

Other work may go to subagents where you see fit, without asking the user or always
reporting the choice. The model is yours to choose; a cheaper one is allowed, not
required. Delegated mechanical work does not run on Fable by default; choose Fable
only for a reason. A subagent on a cheaper model than yours runs at high effort, or
medium for a quick lookup; one on your own model keeps your effort. Cost is a
consideration, not a precondition: a worker for one command or a short operational
step rarely saves anything, while a long lead context and substantial worker reading
with a small checkable answer make delegation more promising. Git writes stay with
the lead; deterministic transformations and checks use scripts.

A subagent's result comes back within the turn. A separate session, which outlives
it, is proposed to the user and started only when the user asks for one. When the
user asks for a draft "for another agent", give the user the draft and launch
nothing; if the wording could mean either, ask in one line.

Give smaller-model workers a compact scope, inputs and constraints. On research
material they return source locations, raw excerpts and uncertainty without
interpreting research outcomes, changing files, expanding scope or delegating
onward. The lead checks the output and directly reads the relevant primary-source
text or figures before relying on a claim, assessing limitations and search
coverage. A model-generated fetch summary is not a direct read. Existing approvals,
spending limits, attribution and user-first review still apply; delegation must not
pre-empt the user's reading.

**Codex:** for smaller-model work, explicitly choose available `gpt-6-luna`,
`gpt-6.1-sol` or `gpt-6.1-terra` with high effort where supported and
`fork_turns="none"`. Protected workers inherit model and effort without overrides.

**Claude Code:** for smaller-model work, pass a per-call `model` such as `sonnet`
and `effort: "high"`; the Agent tool sets effort only when instructions ask for a
level. A subagent or session left on the lead's model inherits it, so under a Fable
lead, mechanical work needs an explicit cheaper model. For protected work, use a
fork or `general-purpose` with model and effort omitted and no overriding defaults,
not Explore. WebFetch's small-model answer does not satisfy the direct-read
requirement. If inheritance is uncertain, keep protected work with the lead. Do not
introduce global subagent-model downgrades.

Policy: [user direction](docs/user-policy.md#smaller-model-delegation), amended
2026-11-17. [Mechanics and evidence](docs/subagent-delegation.md) give the basis and
limits.

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
| Observation | What was measured or inspected and under which conditions. Distinguish automated scores, the user's inspection, a named model's inspection, and an external source's report. |
| Assumption or supposition | An unverified premise and the reasoning that depends on it. |
| Inference | The evidence-to-conclusion reasoning, important alternatives, and limits. An untested explanation remains untested. |
| Reproduction | Prerequisites, commands, identities, expected checks, and which steps were actually exercised versus proposed. |
| Agent recommendation or choice | A proposed action or an implementation choice within authorized scope, with its rationale. The note's attribution line names the model. |
| User policy | An attributable user direction with its scope and source; use the policy record below. |

Agent choices within existing authorization need no separate policy ratification,
but the research decisions under [Work with the user](#work-with-the-user) remain the
user's to make. Approval for a particular execution retains its stated scope. Scientific findings,
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

Record prospective protocols before the relevant confirmatory run; a
[probe](#work-with-the-user) needs only its three lines. Hypotheses or analysis
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
