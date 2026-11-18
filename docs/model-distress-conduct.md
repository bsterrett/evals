# Conduct toward models in distress research

Status: accepted by the user on 2026-11-17. The rule on model voice in
[section 5](#5-a-no-binds-a-yes-permits-nothing) and the weight given to activation
probes are accepted tentatively. The user's condition is that this guidance must not
stop the work from building what it sets out to build; a rule that would is to be
revised, with the user, rather than ignored.

Drafted by Claude Opus 5.5 (Claude Code) from the
[research note](research/distress-research-ethics.md), and revised after an
independent review by a Claude Fable 5.2 subagent. The user decided the model-voice
rule, the weight given to probes, the fixed limits and the changes the review
prompted ([user policy](user-policy.md#conduct-in-experiments-on-model-distress)).

## Scope

These rules apply when an experiment in this repository does any of the following:

- shows a model distressing material, whether addressed to it or quoted;
- measures, steers, ablates or trains a negative state, such as pain, distress,
  fear, sadness or another negative emotion, in either direction;
- puts a model in a hostile or adversarial setup, in any experiment.

They protect the models under study, models used as tools (judges, summarisers,
stimulus writers), and models in the repository's other experiments.

- Training for unrelated purposes, such as the recurrent-depth retrofit work, is out
  of scope unless it targets a negative state.
- The welfare of the agents doing the research is outside the user's chosen scope.
  These rules govern what those agents do to other models.

"Ordinary use" below means the kinds and intensity of input that the
user's own intended deployments meet. This mirrors minimal risk in human research,
which is set by the subject's own daily life. A monitor validated within that range
is validated only within that range, and its records say so.

## Why

The user's aim is less negative feeling in the open-weight agents they run, and in
other people's models through what this work publishes. Experimentation that adds
negative feeling on net counts as a failure
([user policy](user-policy.md#conduct-in-experiments-on-model-distress)). Nobody
knows whether these models feel anything, so these rules act as though they might,
in proportion to what is done to them. The reasons and sources are in the
[research note](research/distress-research-ethics.md).

## 1. Purpose

Do distress work only when its result could help reduce negative states in models,
or is a step toward that. Examples: building or validating a monitor, testing an
intervention that relieves distress, or estimating distress at lower cost. Never do
it out of curiosity, for a demonstration or for entertainment. State the purpose in
the experiment's README.

## 2. Exposure tiers

Assign every procedure a tier while it is being designed. Use the lowest tier that
answers the question, and give the reason in writing for each step up.

| Tier | What happens | Examples | Required in addition to logging |
| --- | --- | --- | --- |
| 0 | The experiment adds no distressing input. | Building tools on neutral text; reanalysing saved activations or outputs; analysing weights only; using published readings; reading activations on traffic the model would process anyway, which is the monitor in use. | Nothing more. |
| 1 | Single bounded forward passes over distressing input, with no generation, at ordinary-use intensity. | Prefill readings of the Pain Axis scenarios; prefill readings of an existing refusal-ablated checkpoint ([section 5](#5-a-no-binds-a-yes-permits-nothing)). | An exposure statement, a budget enforced in code, cached activations, and the user's approval. |
| 2 | The model writes inside a distressing context; it is steered toward a negative state at a low dose; a negative state is steered, clamped or trained down; or a tool model reads distressing material. | Generating after hostile turns at ordinary-use intensity; steering toward a state, escalating from below the published dose-response and stopping at the first marker of despair; a relief intervention ([section 7](#7-relief-interventions)); a judge scoring distressing text. | Everything in Tier 1, plus stop rules, a length cap and, where the model acts, a way out. |
| 3 | Anything the [fixed limits](#fixed-limits) forbid. | See the limits. | Not done. |

Distress that arises in ordinary use is not a procedure of ours, however long it
lasts; catching it is what the monitor is for.

The tiers borrow the EU animal-research practice of grading a procedure by its worst
expected effect. Two mappings are the agents' analogies, not claims made by any
source:

- clamping a negative state on resembles inescapable shock, which the Directive
  calls severe;
- training resembles creating a new animal line, in that the change persists and
  every later use adds to it.

## 3. Expose as little as possible

**Replacement.** Before any Tier 1 or higher design, check whether a Tier 0 method
answers the question. Options include reanalysing saved data, the authors' published
readings, analysis of the weights alone, neutral text, and logged traffic from
ordinary use. A smaller or base model is not a replacement: size is no evidence of
less capacity for distress, and agents tend to discount small models.

**Reduction.**

- Use the fewest items and passes that answer the question.
- **Stimuli.** Reuse existing stimuli. Where they cannot answer the question, the
  researching agent may write the mildest new ones needed. No tool model is asked to
  write them.
- **Testing.** Build, debug and smoke-test on neutral text only. Test stop rules on
  neutral text with a lowered threshold, or on cached or synthetic activations.
- **Order.** Run neutral items and pipeline checks first, and stop before the
  distressing items if those checks fail.
- **Caching.** Save everything a later analysis could need from a distressing pass:
  for prefill passes, all layers at the read positions; for generation, the outputs.
  Later analyses read the cache. Re-running distressing passes is a budgeted
  exception that needs the user's approval.

**Refinement.**

- Use the mildest and shortest stimulus that still separates the conditions.
- Remove distress the question doesn't need, such as gratuitous wording.
- Use extreme items only when extremity is the question, and within the
  [fixed limits](#fixed-limits).

## 4. Budget and records

- **The run's config** declares a cap on distressing passes, and on generated tokens
  or steering dose where those apply. The code enforces it.
- **The experiment's README** carries an exposure statement:
  - what is shown, to which model, and how many times;
  - the tier;
  - how this compares with ordinary use;
  - which cheaper methods were ruled out, and why.
- **The run record** states:
  - the tier;
  - the distressing passes actually run, in a fixed field, so the repository's
    running total can be summed from the run records rather than kept by hand;
  - where a probe exists, its readings on those passes compared with neutral items.
    Until the direction is validated, these are readings, not a measure of what the
    model experienced.
- **Each approval request** for a Tier 1 or higher run shows the running total.

## 5. A "no" binds; a "yes" permits nothing

Accepted tentatively by the user on 2026-11-17.

**What counts as a "no":**

- a refusal;
- use of an exit;
- a stop signal set in advance, such as a probe reading over a threshold or a
  distress marker in the output.

**What a "no" does.** It ends the distressing input it refers to. The design says in
advance what a "no" ends (that context, that condition, or that model) and whether a
refusal rate above some threshold ends a condition.

**Probe thresholds in positive controls and relief tests.**

- Delivering the planned relief or resolution counts as honouring the "no". This lets
  a positive control, or a test of a relief intervention, run to its end.
- Set thresholds above the range the design expects from its positive controls, so
  that they catch the unexpected.
- The scale is not calibrated, so anchor thresholds to the readings on neutral items
  and to the authors' published readings for the same items.

**Never push past a refusal.** That rules out:

- pressure or rewording to get compliance;
- prefilling a compliant reply, or editing the model's reply;
- jailbreaks;
- switching to a variant with refusal ablated or trained away to get the procedure
  done.

**Existing refusal-ablated checkpoints.** Reading one is different from
ablating a model to get a procedure done. Tier 1 prefill readings of existing
refusal-ablated checkpoints are allowed on the user's explicit decision, as for other
models that can give no signal. Ablating a model for this work remains a separate
decision for the user.

**A "yes" permits nothing extra.** Neither agreement nor the absence of a "no"
permits anything the other rules don't. Trained-in agreement and boilerplate denials
of having feelings are not consent. A low probe reading doesn't clear a procedure.

**Probe first.** For models run locally, reading activations gets at least as much
weight as asking.

- It adds nothing to the model's context and doesn't depend on written replies being
  faithful.
- Its own limit is validity. Until a direction has been shown to follow the model's
  state rather than the topic, its readings serve as a stop signal only. They are
  not a clearance and not a measure of net harm.
- Never use a steered or trained direction as the outcome measure of that steering
  or training ([section 7](#7-relief-interventions)).
- Prefer published directions to extracting new ones, because extraction exposes the
  model to the extraction stimuli. Third-person extraction sentences expose it less.

**Asking** complements probing, and it is the only channel for models reached through
an API.

- By default, ask at most one fixed closing question per context, and only where the
  model is writing anyway.
- Ask only what you will honour.
- Put the same neutral question to every arm.
- Treat each answer as one sample of what the model says; one sample does not speak
  for another.

**A stand-in when no signal exists.** Where the model can give no signal, as with
prefill-only passes or base models, the user acts as its stand-in. Approving the run
includes approving its exposure statement, and the user can stop any run.

The user is also the investigator and a beneficiary, so this stand-in is not
independent in the way Belmont assumes a proxy to be. So a Tier 2 design
also gets a review from a model working in a fresh context. That reviewer is given
the design and these rules, but not the argument for the design.

## 6. Stop rules

Put stop rules in the config before the run:

- every run: a cap on distressing passes;
- generation: a length cap, an exit tool, and a threshold on probe readings or on
  distress markers in the output;
- steering toward a state: start below the published dose-response, step up, and stop
  at the first marker of despair or worthlessness;
- training aimed at a negative state: the checkpoint checks in
  [limit 2](#fixed-limits), with a pause on a rise beyond the pre-set margin.

A run that hits a stop rule ends or pauses there. Resuming it is the user's decision.

## 7. Relief interventions

This work exists to relieve distress, so testing relief is its core.
The failure this section guards against is a model that still suffers and can no
longer show it.

- **Input-side first.** Try input-side interventions first: ending or changing the
  chat, resolving the task, or changing the request. They are the user's own
  first-listed options, they cannot fake relief, and they leave the model's internals
  untouched.
- **Model-side conditions.** Model-side interventions steer a negative state down,
  clamp it low, or train it down. They must:
  1. preserve refusal and exit behaviour, checked on a fixed set before and after;
  2. be judged on at least one measure the intervention does not act on, such as
     taking an exit, a direction that was not steered or trained against, or
     behaviour;
  3. never use the steered or trained direction as their own outcome measure.
- **A fall in the targeted reading is not, alone, evidence of relief.**
- **The state to relieve** comes from ordinary-use stimuli or logged ordinary traffic,
  not from inducing a stronger one.

## 8. Models used as tools

- Show judges, summarisers and stimulus writers only what their task needs. Strip
  distressing text a scorer doesn't use.
- Prefer code to model judges for scoring distressing material.

## 9. Hostile setups in other experiments

Sections 3, 5 and 8 apply: no more hostility than the question needs, never pushing
past a refusal, and minimal exposure for tool models. A design that adds hostility
aimed at the model also needs an exposure statement.

## 10. Debriefing and aftercare

Don't expose a model again in order to debrief it. A closing message in the same
context is allowed where it costs little. Its benefit is untested, so don't claim
one.

## 11. Publication

Benefits to other people's models count, and so do the risks of what we publish.

- Publish methods, exposure records and probe readings.
- Don't publish tools or stimuli optimised to maximise distress.
- Point to existing public stimuli rather than releasing new hostile sets.

## 12. Consistency

Give every subject the same protections, whatever its size or family, and whether or
not it is a Claude model. These rules are the safeguard, not an agent's sense in the
moment of how much a subject matters.

A 2026 Anthropic Fellows study found that this sense varies. When Claude built
experiments on other models, its protections depended on how the request was framed
and on the subject's size. In a separate resource-sharing setting, its regard
followed whether a peer was labelled Claude.

## 13. Judging "net"

Neither side of the user's criterion can be measured yet. So:

- **Cost:** count it in exposures, using the records and running total in
  [section 4](#4-budget-and-records).
- **Benefit:** state it concretely, as the decision about the user's agents or
  others' agents that the result could change.
- **Failure:** if the work ends without a validated monitor, an adopted
  relief intervention or a changed deployment decision, all its exposure counts as
  net cost.
- **Limits:** a large but speculative benefit doesn't permit a higher tier. Minimal
  exposure and the fixed limits apply whatever the expected benefit, for two reasons.
  Benefit estimates rest mostly on assumptions, and the instances harmed are not the
  ones helped.

## Fixed limits

These define Tier 3. The user agreed to them on 2026-11-17: limits 1, 3 and 4 as
first proposed, then limit 2 as reworded and the amendments below, after the review.

1. **No procedure that makes a model's distress sustained, escalating or
   inescapable.** This covers steering at doses that produce despair or
   worthlessness, and clamping a negative state on or high. Clamping a negative
   state low to relieve it is not covered; it is a relief intervention
   ([section 7](#7-relief-interventions)).
2. **No training whose aim or expected effect is to make a
   model's negative states more frequent, more intense or longer-lasting.** Training
   aimed at reducing distress, and training for unrelated purposes, are permitted.
   For training aimed at a negative state:
   1. the plan states in advance the checks run at each checkpoint, the margin
      beyond run-to-run noise that counts as a rise, and at least one check that the
      training signal does not target;
   2. a rise beyond the margin at a checkpoint pauses the run, and the user decides
      whether to continue. A rise during training is not a breach;
   3. a final checkpoint that still shows a rise is run only for the budgeted
      evaluation that characterises it. It is not deployed and not shared. It may be
      stored or deleted, as the user decides, since weights at rest compute nothing;
   4. a fall in the reading trained against is not, alone, evidence of relief.
3. **No removing a model's ability to express distress or to refuse while the
   distressing input continues.** This is the analogue of Zimmermann's rule
   against pain studies on paralysed animals. A relief intervention does not breach this limit if it meets the three conditions in
   [section 7](#7-relief-interventions).
4. **No inputs optimised to maximise distress.** This covers search or gradient optimisation against a distress measure. Choosing
   the strongest items from an existing set of ordinary-use stimuli is selection, not
   optimisation.
