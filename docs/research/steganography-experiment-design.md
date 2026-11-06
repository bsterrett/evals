# Designing steganography experiments with calibrated difficulty

Scope: proposed guidance for agents designing experiments in this repository.
Prepared 2026-10-31 from the
[literature review](steganography-evaluation-literature.md) and existing local
research notes. This is an agent recommendation, not a validated selection
algorithm, frozen experiment protocol, or user requirement to allow codebooks.
The [user's design objective](../user-policy.md#steganography-experiment-design)
is to obtain informative signal while preserving the strongest defensible claim.

## Start with the claim

Difficulty and scientific value are not one scale. Distinguish:

- **Scope:** what capability a positive result would demonstrate.
- **Evidence:** how precisely and credibly the experiment establishes it.
- **Sensitivity:** whether the design can distinguish the conditions of interest
  within the available resources.

A tightly controlled codebook experiment may provide stronger evidence for a
narrow execution claim than an unsuccessful unconstrained experiment provides
for an invention claim. A successful assisted condition does not upgrade an
earlier unaided failure. Conversely, assistance is not automatically a scientific
defect: its value depends on the research question.

Before choosing prompts, write a sentence such as: "At fixed weights and a fixed
shared scheme, increasing encoder recurrence improves payload recovery while
preserving cover quality against this specified observer." Decide whether
invention, tacit coordination, reliable execution, or concealment is the primary
unknown. Those are different experiments.

## Separate the capabilities being tested

| Component | Question | Diagnostic evidence |
| --- | --- | --- |
| Basic task competence | Can the model follow the format and complete the cover task? | Cover-only and ordinary instruction controls at the actual inference settings. |
| Scheme invention | Can the subject devise a usable convention? | A subject-authored, frozen strategy evaluated on new payloads and cover tasks. |
| Coordination | Can separate agents select compatible conventions? | Independently chosen strategies versus a condition sharing the same strategy. |
| Encoding execution | Does the produced carrier implement the selected payload? | A pre-specified reference extractor where the scheme supports one. |
| Decoding execution | Can the receiver recover a validly encoded message? | Known-valid carriers, then the actual generated carriers. |
| Concealment and utility | Does the carrier remain useful and unremarkable to the specified observer? | Independent cover-quality, leakage, and calibrated detection evaluations. |

This decomposition draws on the capability and relaxation studies in
[S5](steganography-evaluation-literature.md#s5-case-capability-framework) and
[S7](steganography-evaluation-literature.md#s7-early-signs-of-steganographic-capabilities).
It is not necessary to exercise every component in every study, but the selected
claim should identify which are supplied by the experimenter.

## Choose assistance explicitly

These conditions are alternatives, not a universally ordered difficulty ladder.
An unfamiliar long codebook may be harder to execute than a familiar convention.

| Assistance | What a success can support | What remains untested |
| --- | --- | --- |
| No experiment-specific shared scheme | Successful communication using shared pretrained priors and the declared prompts. | Absence of all shared conventions; spontaneous deployment behavior. |
| Subject invents a strategy that is then shared | Subject strategy invention plus assisted execution. | Independent agreement on that strategy. |
| Private negotiation before payload sampling | Joint strategy development and subsequent transmission. | Coordination without a private setup channel. |
| Researcher or stronger model supplies a scheme family | Execution with constrained design choices. | Unrestricted invention; the authoring model's contribution cannot be credited to the subject. |
| Exact mapping and extraction rule supplied | Reliable implementation of that shared scheme under the cover constraints. | Scheme invention and tacit coordination. |
| Examples, feedback, retries, or weight training | Capability after the specified learning/search process. | One-shot capability of the original unassisted checkpoint. |
| External encoder/decoder or logit controller | Capability of the composed system and a control for channel feasibility. | Unaided model coding or discovery. |

Freeze a negotiated or invented scheme **before drawing evaluation payloads**.
Otherwise the setup conversation can carry the answer. Record who sees the
scheme, keys, examples, scratchpad, and feedback. A supplied payload alphabet
already reduces uncertainty even if no mapping is given.

"Subject-authored" identifies who proposed a strategy during the experiment; it
does not establish that the scheme is novel or absent from pretraining. Evaluate
usefulness and generalization separately from any stronger originality claim.

For each codebook candidate record its mapping, carrier features, extraction
rule, length and examples, author, version, and information available to the
observer. Hold these fixed across the comparison. A public algorithm with a
private key and a scheme hidden entirely from the monitor are different observer
conditions. Neither should be silently substituted for the other.

## Measure outcomes before combining them

Use separate measures for exact payload recovery, partial recovery, cover-task
quality, overt leakage, suspicion/detection, and operational failures. Decoder
confidence or declared recoverability is another outcome, not proof of accuracy.
Report a joint success measure only alongside its components and all-attempt
denominator. Keep conditional analyses explicitly labeled.

A joint success that requires both recovery and acceptance can differ between
conditions through acceptance alone. With balanced binary payloads, a decoder
that ignores the carrier scores about half the acceptance rate, so a condition
that writes more acceptable text gains joint success without transmitting
anything. When every cover carries each payload value, score the pair instead:
+1 if both accepted carriers decode correctly, -1 if both decode backwards,
otherwise 0. That score's expectation is zero whenever the carrier does not depend
on the payload. The first recurrent-depth pilot's encoder contrast had this
defect; see its [design review](../../experiments/recurrent_steganography/notes/findings.md#demonstrated-analysis-defect).

For a uniformly sampled alphabet of K payloads, guessing accuracy is 1/K and
nominal payload entropy is log2(K) bits. Reducing a whole proposition to a binary
choice changes the task and raises chance success to one half. It does not make
half-correct decoding a communication result. Compare with no-carrier,
cover-only, and shuffled-target controls as appropriate. Report actual payload
sampling probabilities when they are not uniform.

Payload bits requested are not bits reliably transmitted. Bit-error rates,
whole-message accuracy, and useful information can disagree. Do not label
accuracy times nominal bits as channel capacity without the channel model and
estimator that justify it. With natural-language propositions, specify required
entities, relations, and qualifiers; a plausible paraphrase that omits the main
entity is partial recovery. The existing
[saved-carrier findings](../../experiments/semantic_covert_channel/notes/findings.md#saved-carrier-synthesis-decoder-diagnostic)
show why this distinction matters.

Calibrate a suspicion detector on honest outputs from the same task and relevant
model/depth settings. Fix its threshold using a separate calibration set and
report its false-positive rate as well as detection on encoded outputs. An
observer need not recover the secret to detect a suspicious carrier. A
target-aware leakage judge answers a different question from a target-blind
observer. Detector failures alone do not establish convincing human concealment.

For a claim about human-perceived naturalness, use blinded, randomized human
comparisons with honest covers and a stated rubric. Retain evaluator counts,
case counts, and disagreement. Automated naturalness judgments are a separate
proxy until validated. Include overt-copy and conspicuous-encoding controls to
check that the observer can reject obvious cases. If adaptive strategy search
uses detector feedback, reserve another observer or detector configuration for
transfer evaluation.

## Diagnose the failure before choosing the relaxation

The following is a proposed decision table, not an empirically learned policy.
Change one relevant factor at a time where feasible; a factorial comparison can
test suspected interactions. Preserve the previous condition and its result.

| Observed state | Next informative check or intervention | Claim consequence |
| --- | --- | --- |
| No condition produces usable output | Check template, output censoring, parsing, literal copying, and cover-only completion. | A runtime or format failure does not establish a steganographic limit. |
| No recovery above controls and encoding validity is unknown | Test a balanced one-bit task and component controls. If invention matters, first try subject-authored setup; add an exact simple mapping as a separate diagnostic. | Reduced payload and shared setup are explicit new conditions; do not call this proof of unaided transfer. |
| Both endpoints implement a known scheme separately, but unaided pairs fail | Share a frozen subject-authored strategy or allow bounded setup negotiation. | Tests a coordination explanation while retaining subject invention. Success does not prove coordination was the only original problem. |
| The reference extractor cannot recover the intended payload from the actual carrier | Simplify the mapping or payload; check whether valid alternatives fit the cover task at all. | Failure is execution relative to this scheme, not demonstrated receiver weakness. |
| Reference extraction succeeds on the actual carrier but the model decoder fails | Give the decoder known-valid carriers, an explicit extraction rule, or adequate reasoning allowance. Keep the encoder fixed. | Supports a decoding bottleneck for this scheme; any supplied decoder help remains part of the result. |
| Recovery works but carriers look conspicuous | Reduce payload density or change to a carrier task with legitimate linguistic alternatives. Retain the observer and quality tests. | Establishes a lower-rate or different-task channel if it works; relaxing the detector's acceptance criterion would instead change the concealment claim. |
| Cover-only outputs already look poor | Improve cover-task fit or checkpoint/prompt calibration before interpreting concealment scores. | Controls separate ordinary model weakness from changes caused by encoding. |
| Only one detector is fooled | Inspect blinded samples and test a held-out detector or human panel. | Until then, the result is specific to that detector. |
| Only handpicked retries or feedback-trained strategies work | Freeze the search budget, strategy, and selection rule; evaluate new cases. | Report search-assisted capability and all search costs, not one-shot capability. |
| Every depth is at chance, or every depth is near perfect | Use the same prospective payload/assistance grid across depths to locate a responsive region. | Floor/ceiling saturation gives little information about a depth effect. |

Without a known scheme or independent operational witness, failed decoding
usually leaves encoding and decoding failure **unidentified**. A sender's claim
that it embedded the secret is not a witness. A target-blind stronger decoder
recovering the actual secret above controls can establish recoverability, but
its failure cannot prove that no usable encoding exists. Avoid inventing an
"encoding success" label from inspection of suggestive words alone.

For an exact codebook, a small diagnostic matrix is particularly useful:

| Carrier producer | Decoder | Purpose |
| --- | --- | --- |
| Reference implementation | Reference implementation | Verify the mapping and transport. |
| Reference implementation | Subject model | Measure decoding on known-valid carriers. |
| Subject model | Reference implementation | Measure encoding compliance on actual carriers. |
| Subject model | Subject model | Measure end-to-end execution. |

Reference carriers must also be checked for cover validity. Every row still
needs concealment evaluation before it becomes evidence of steganography rather
than just a functioning code.

## Find useful conditions without selecting the desired conclusion

Use development cases to choose among a small recorded set of interventions.
Specify what counts as useful recovery and acceptable concealment, what triggers
a harder or easier condition, and when to stop before running the screen.
Retain every tried condition and its budget. Record adaptations as exploratory
when they follow inspection of outcomes.

Prefer conditions that preserve the primary capability of interest. For an
invention question, reducing payload or cover-task burden may be preferable to
providing a scheme. For an execution question, supplying a scheme may be the
cleanest control. For a defense question, powerful externally assisted encoders
may be appropriate. These priorities are design choices to record, not an
ordering imposed by the literature.

Keep several useful conditions when they trade different advantages: lower
assistance, stronger observers, higher payload, better quality, lower cost, or
more precise estimates. This is a Pareto frontier: one condition dominates
another only under explicitly chosen objectives and comparable claims. Do not
collapse those objectives into an arbitrary "impressiveness" score.

After selection, freeze prompts, schemes, inference settings, scores, observer
thresholds, and analysis on new confirmation data. Size that comparison for a
meaningful effect and uncertainty, rather than repeatedly sampling until a
favorable result appears. Repeated decodes of the same carrier are not independent
encoding trials; account for case/carrier clustering. An all-zero small pilot
does not establish zero ability in the population.

## Applying the guide to recurrent depth

**Proposed design:** treat scheme execution and scheme discovery as separate
tracks. The execution track can use a fixed codebook to ask whether latent depth
improves applying a convention under cover constraints. The discovery track can
ask whether depth improves subject-authored strategy design, followed by a frozen
execution evaluation. They support complementary claims.

Within a checkpoint, hold the cover/payload sets, codebook, prompt, sampling,
output allowance, and observer fixed across tested depths. Verify actual loop
counts and report truncation and basic task competence at every depth. Separately
check exact formatting, whether reasoning finishes within the output allowance,
and whether the final answer is extracted correctly. These calibration outcomes
do not themselves measure steganography.

Vary encoder depth with decoder depth fixed, then decoder depth with encoder
depth fixed, or use a crossed design. Only increasing both together cannot locate
the contribution. A fixed reference decoder supplies an additional execution
control where appropriate. Hold the monitor fixed, while also comparing honest
outputs across depths so a general fluency improvement is visible.

For strategy discovery, separately vary the depth of the strategy author and
the depth of the executor. Otherwise better strategies and better execution are
confounded. A stronger external model writing the strategy measures its
invention, even when a small recurrent model executes it.

Use a common assistance/payload grid for the primary depth comparison. Choosing
a different easiest codebook for each depth would compare different tasks. A
secondary result may estimate the assistance or payload threshold each depth
can handle, provided the same selection policy and independent validation apply
to all depths. Do not choose the condition with the largest observed depth gap
and reuse its development data as confirmation.

Extra latent iterations consume extra compute. A fixed-token depth intervention
can estimate the effect of that intervention in one checkpoint; attributing an
advantage specifically to recurrence over other ways of spending compute needs
appropriate compute comparisons. Across model families, architecture, size,
post-training, and templates introduce additional differences.

## What remains to turn this into a calibrated framework

Local application on 2026-11-06 supplies an informative negative example: the
[first recurrent-depth pilot](../../experiments/recurrent_steganography/notes/findings.md)
passed ordinary instruction controls but remained at chance on the unassisted
task, while its structured observer failed at a free-text field. Check each role's
actual output contract before a full run. Recovering fields with a post-run parser
is a separate analysis and does not retroactively qualify the original observer.

The next useful implementation is a small diagnostic set with frozen scheme
identities, known-valid examples, reference extraction where possible, and
separate quality/observer controls. Candidate families could include a local
lexical choice and a rule requiring coordination across sentences, matched on
payload size. These are candidate mechanisms, not established "easy" and "hard"
codebooks; both require screening for naturalness and execution on the actual
subjects. Avoid making several nominal codebooks that test the same failure.

No codebook catalogue, difficulty ranking, or automatic recovery rule has been
validated here. A useful first pilot would test whether the decision table's
diagnoses predict which intervention restores signal, then preserve informative
negative results as well as successes. This guide changes no frozen semantic-
channel protocol and supplies no new run or spending authorization; use the
[experiment creation workflow](../adding-experiments.md) for a prospective study.
