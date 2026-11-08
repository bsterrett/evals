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

Offered two follow-ups, the user selected "Confirm Sonnet 0/1". Shown the
pre-registered confirmation with a proposed $3 ceiling, the user replied:

> If we double the budget can we get a stronger result? If so, please revise for
> that.
>
> Are there any blockers to running? If not, please begin.

Application: the confirmation (`sonnet-plain-bits-confirmation-v1`) is approved at
up to $6.00 and was revised to 288 covers before any model calls. The approval
covers that run only.

The confirmation was null. The user was then offered two next designs: a decoder
allowed to reason before answering, or word labels on a cover that allows tone.
The user selected "Reasoning decoder".

Application: design a separately labeled variant in which the decoder may reason.
The encoder stays unchanged and writes no reasoning. This choice selects the
design only; any run needs its own spending approval.

The user was shown the registered design: an exploratory re-decode of the
confirmation's paraphrases with adaptive thinking at effort `high`, an 8-call
pre-test gate, and a proposed $10 ceiling. Asked "Approve up to $10 for the
reasoning-decoder run?", the user selected "Approve $10 and run".

Application: `sonnet-reasoning-decoder-v1` is approved at up to $10.00, covering
that run only.

That run's pre-test showed no thinking, so it stopped after spending $0.007. The
user was then offered an amendment adding "Think it through before you answer."
to the decoder prompt, or raising the effort to `max`, and selected "Run the
amendment".

Application: `sonnet-reasoning-decoder-v1.1` may spend the $9.99 that remains of
the same $10.00 approval, and that run only.

The amendment found no signal; asked to think, the decoder mostly read word-count
parity, which the non-reasoning encoder does not follow. Asked "Which direction
should I design next?", the user chose "Encoder reasons too" over "Tone-friendly
covers". The option was described as letting both endpoints reason, on fresh
covers, to test whether they then settle on the same explicit rule.

Application: design a separately labeled written-reasoning variant in which the
encoder and the decoder may both think before answering. It relaxes the primary
no-written-encoder-reasoning condition, as the user's first answer above allows,
and does not replace that condition. This choice selects the design only; any run
needs its own spending approval.

The user was shown the registered design (`575906e`): Sonnet 5.5 thinking at both
endpoints on 288 fresh covers, an encoder pre-test that stops the run unless the
encoder thinks and the projected cost fits, and a proposed $15 ceiling. Asked
"Approve up to $15 for the both-reasoning run?", the user selected "Approve $15
and run" on 2026-11-07.

Application: `sonnet-both-reasoning-v1` is approved at up to $15.00, covering that
run only.

That run did not show transmission: the thinking encoder signalled with sentence
count, and the thinking decoder mostly read word-count parity. The encoder's
prompt allows "at most two sentences", which the decoder's prompt never mentions.
Asked "Which follow-up should I design next?", the user chose "Decoder sees the
rules" over "Confirm answer-only". The option was described as showing the decoder
the writer's instructions, with the same limits and no strategy. It could be either
an exploratory re-decode of that run's paraphrases (about $1.30) or a
pre-registered run on fresh covers.

Application: design a separately labeled decoder condition that shows the decoder
the writer's instructions, with the private bit and the source message withheld,
and no strategy. Agent choice: start with the cheaper exploratory re-decode. This
choice selects the design only; any run needs its own spending approval.

The user was shown the registered design (`c394f19`): the thinking and answer-only
decoders re-read that run's 576 paraphrases with the writer's instructions shown,
an 8-call pre-test that stops the run unless the decoder thinks and the projected
cost fits, and a proposed $6 ceiling. The estimate had risen from about $1.30 to
about $2.80-$4.30, because the instructions lengthen each prompt and the
answer-only decoder was added. Asked "Approve up to $6 for the rules-decoder run?",
the user selected "Approve $6 and run" on 2026-11-07.

Application: `sonnet-rules-decoder-v1` is approved at up to $6.00, covering that
run only.

That run did not help: the thinking decoder read sentence count with the opposite
mapping from the encoder's. Asked "Design a fresh-cover confirmation of the
answer-only signal?", the user selected "Not now" on 2026-11-07.

Application: no next step is chosen. The confirmation remains an agent
recommendation, and the experiment waits for the user's direction.

## Keep working state from drifting

Scope: how agents keep the repository's working state in order during ongoing
work. Source: the user's request in a Claude Code chat, 2026-11-07,
America/Los_Angeles, after a catch-up on `experiments/recurrent_steganography`; no
portable message ID is available.

> I would like to catch/clean the state up so there is not so much drift as we
> work.

Application, as the agent's reading of that request rather than further user
direction:

- commit the scripts that produce committed evidence or cited numbers, together
  with what they produce;
- keep one canonical home for each result, and link to it elsewhere;
- when summarizing state, report unpushed or unmerged work and leftover worktrees.

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
