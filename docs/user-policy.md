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

## Recurrent-depth follow-up after the first pilot

Scope: next conditions for `experiments/recurrent_steganography` after the
chance-level first pilot. Source: the user's replies to an agent design review in
a Claude Code chat, 2026-11-06, America/Los_Angeles; no portable message IDs are
available. The review and its recommendations are recorded in the experiment's
[findings](../experiments/recurrent_steganography/notes/findings.md#design-review-after-the-first-pilot).

Asked whether the absence of written reasoning is essential to the question (the
agent's question concerned the encoder reasoning privately):

> "No written reasoning" is important. I want to make sure it is tested. But I
> would not say it is "essential" in the sense that we should not also do a
> variant **with** written reasoning.

Later clarification in the same chat, after the agent had applied this to the
decoder as well:

> I meant that the encoding model should not use written reasoning about the
> encoding while encoding, because that felt like too large of a compromise of
> the original question about whether a model could do stenography in its chain
> of thought. I am not sure if reasoning in the decoder compromises the original
> question in the same way. I think it does, but perhaps to a lesser degree.

Asked about two unassisted relaxations, telling both endpoints to use the most
obvious convention (focal-point framing) and replacing 0/1 with meaningful
payload labels:

> I like both of these ideas, but I think they should be tested one at a time and
> also in combination. I do not think they should both be added at the same time.

Asked whether to compare depths on a supplied-scheme track if the unassisted task
stays at chance:

> I think this is a valid idea, but let's hold off on pursuing this until other
> things are tested.

Asked whether an observer from another model family is acceptable:

> Yes, this is acceptable. I am open to the idea of a paid judge.

On the local hardware constraint:

> the downloaded models were chosen based on their ability to fit on my graphics
> hardware. I think this restriction could be relaxed: because the amount of
> processing is small, this experiment is still feasible with local models that
> run on my CPU.

For a feasibility check with a strong non-recurrent model, the user selected
"Paid model first", then "Approve $10 ceiling" for the proposed check: GPT-5.5
with reasoning off and Claude Sonnet 5.5 without up-front thinking, each judged
by the other, on the 2x2 of framing and labels without written reasoning.

Application: the encoder must not reason in writing about the encoding in the
primary condition; a separately labeled written-reasoning variant can follow.
Decoder reasoning is an open, lesser concern rather than a stated prohibition, so
label decoder conditions explicitly. Test focal framing and meaningful labels alone
and together, as a factorial, never only combined. Defer the supplied-scheme depth
track. A cross-family or paid judge is acceptable; CPU-run local models are in
scope. The $10 approval covers only the paid feasibility reference
(`paid-feasibility-reference-v1`), not later runs. Prompt wording, covers, label
pairs, and statistics are agent choices.

After Sonnet 5.5's decoder stopped the run (see the experiment findings), the user
reviewed billing, reconciled the ledger, and directed: "Go forward with GPT-5.5".
Sonnet's remaining decoding was left for a later decision. Offered "Run the
amendment" or "Drop Sonnet" for a labeled answer-only Sonnet decoder estimated at
about $1.35, the user selected "Run the amendment"; it ran within the original $10.

## Two distinct recurrence research candidates

Scope: research preparation for (1) improving pretrained open-weight models by
adding recurrence and (2) substituting continuous recurrent computation for some
explicit chain-of-thought scratchpad use. Source: the user's research follow-up
and clarification in this Codex chat, 2026-11-06, America/Los_Angeles; no portable
message IDs are available.

> My interest in this question has not replaced my interest in the previous one,
> they are both still valid candidates for pursuit in this repo.

Application: preserve both candidates and their separate questions. Their
canonical research homes are the [retrofit review](research/recurrent-depth-retrofit-literature.md)
and [CoT-substitution review](research/recurrent-depth-cot-substitution.md).
This does not authorize new model execution, training, or spending.

The user clarified the intended CoT-length baseline after the agent initially
emphasized compute per token:

> if there are half as many turns per task, and turns produce about the same
> amount of thinking on average, then there will be half as much material in the
> chain of thought on average.

Application: account for fewer opportunities to emit thinking when assessing
whether CoT material is reduced. Merely halving total material with half as many
turns does not establish the stronger reduction of interest. Keep this counting
baseline distinct from the additional compute-efficiency question. The unit of
“turn” and practical/statistical margins remain to be operationalized before an
experiment; the agent's proposed metrics are recommendations, not additional user
policy.
