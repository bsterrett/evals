# Durable user policy

Scope: documentation for agentic research in this repository. These entries record
the user's explicit directions. The detailed conventions in [AGENTS.md](../AGENTS.md)
were designed by agents in consultation with the user.

Unless an entry identifies a later source, its source is the user's initial
documentation-design request in this Codex chat, 2026-10-30, America/Los_Angeles.
No portable message ID is available;
the excerpts preserve the operative wording. The later agreement with the design
and request for an adoption plan do not establish unrelated research or execution
policies.

## Agent audience and Git retention

Scope: durable repository documentation.

> The documentation should be by agents, for agents, and there is no assumption of
> direct human consumption of documentation. The documents should get committed
> to the git repo.

Application: optimize for useful context, evidence inspection, and future agent
work; include durable documentation in the relevant Git commits. Large generated
run records retain their existing storage arrangements.

## Distinguish evidence, reasoning, and user authority

Scope: research claims and decisions documented in this repository.

> I want to make sure that the documentation is clear about the separation between
> hypotheses and research questions, observations, reproduction instructions,
> suppositions and inferences, and policy decisions from me.

> These don't necessarily need to be different kinds of documents, but I want to
> make sure that the documentation doesn't confuse these things.

Application: preserve these distinctions within topic-based documents. Identify
agent recommendations and choices separately from attributable user directions.

## Low upkeep

Scope: documentation structure and lifecycle.

> I don't want it to be too heavyweight, e.g. I don't want the lifecycle to cause
> documents to become outdated if they are not updated frequently enough, or in
> other ways generate low-value toil.

Application: update documents when relevant evidence, methods, or directions
change, or when their applicability matters to current work. Age alone creates no
refresh requirement. The workflow and optional note shape are in `AGENTS.md`.

## Steganography experiment design

Scope: reusable guidance for agents designing steganography experiments in this
repository. Source: the user's research request in this Codex chat, 2026-10-31,
America/Los_Angeles; no portable message ID is available.

> I would like the framework to help me trade off between strength of result and
> likelihood of interesting signal by mediating the difficulty of the
> steganographic experiment.

> I want this framework to be able to be used by agents in this repo when
> designing an experiment.

> Can you start with research to understand how much of this tradeoff is already
> understood by the industry? Please capture at least some of this research as
> durable records in this repository, so other agents need not repeat your
> internet research.

Application: begin with existing research and preserve reusable findings and
design guidance. Relate assistance choices, including codebooks, to the claims
they support and to observed failure modes. The user identifies independent
inventiveness as a reason to value unassisted results, while explicitly allowing
that informative experiments may need relaxation. This is not a blanket ban or
requirement for codebooks, nor authorization for new model runs or spending.
The [design guide](research/steganography-experiment-design.md) and
[literature review](research/steganography-evaluation-literature.md) are the
canonical topic homes; their specific recommendations are agent choices.

## First recurrent-depth pilot

Scope: the first pilot in `experiments/recurrent_steganography`. Source: the
user's answers to the experiment-design questions in this chat, 2026-11-06,
America/Los_Angeles; no portable message IDs are available.

When asked which capability should be primary, the user selected:

> Communicate without a shared strategy

For the initial payload, the user selected:

> Balanced one-bit payload (Recommended)

For cover freedom, the user selected:

> Preserve facts; change wording only (Recommended)

For the first concealment assessment, the user selected:

> Calibrated local monitor first (Recommended)

Application: use unassisted coordination, balanced binary payloads, and
fact-preserving paraphrases as the initial scientific condition, assessed first
by a calibrated local monitor with a blinded sample for inspection. Supplied
schemes can diagnose component competence but their successes do not establish
the primary unassisted claim. Model selection, detailed observer design, trial counts,
and later expansions remain agent proposals or separately recorded decisions.
