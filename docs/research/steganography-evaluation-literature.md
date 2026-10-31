# Steganography evaluation literature and experimental difficulty

Scope: public research relevant to designing language-model steganography
experiments, reviewed 2026-10-31, America/Los_Angeles. This is a targeted literature
review, not a systematic review or a reproduction of the cited experiments.
The companion [experiment-design guide](steganography-experiment-design.md)
contains proposed repository guidance derived from this review.

## What is already understood

The requested framework has substantial precedents, but the reviewed work does
not supply a validated general policy mapping an arbitrary failed experiment to
the next codebook or relaxation. Three bodies of work address different parts:

- **Communication engineering:** capacity, extraction reliability, distributional
  detectability, and cover quality are separate objectives. Algorithms can trade
  these off while using a language model as a component; this need not demonstrate
  the model's ability to invent or execute a scheme itself ([S1](#s1-neural-linguistic-steganography),
  [S2](#s2-perfectly-secure-steganography)).
- **Capability elicitation:** studies vary shared schemes, reasoning access,
  payload sizes, training, and feedback. Some explicitly simplify failed tasks
  into component tests ([S4](#s4-openai-steganography-eval),
  [S5](#s5-case-capability-framework), [S7](#s7-early-signs-of-steganographic-capabilities)).
- **Adaptive strategy search:** a published procedure starts at one bit, collects
  strategies, and increases payload complexity until improvement stops
  ([S6](#s6-hidden-in-plain-text)). This is the closest operational precedent,
  although it optimizes communication under particular defenses rather than
  choosing among scientific claims for a recurrent-depth study.

Public work from [OpenAI](#s4-openai-steganography-eval) and
[Anthropic](#s12-public-lab-assessment-practice), and papers involving
[Google DeepMind researchers](#s7-early-signs-of-steganographic-capabilities),
establish that these issues are studied in industry as well as
academia. They do not establish an industry-wide standard or reveal private lab
practice. The reviewed sources use materially different definitions of success;
their headline rates should not be treated as a common model leaderboard.

## Sources and reusable findings

Each entry separates the source's report from its application here. Paper sections
and versions identify the supporting material. External results below were read,
not independently reproduced; underlying experiment logs were not audited.

### S1 Neural Linguistic Steganography

Ziegler, Deng, and Rush, EMNLP-IJCNLP 2019,
[paper](https://aclanthology.org/D19-1115.pdf), sections 2-4.

The method uses arithmetic coding with a shared language model and invertible
mapping. It distinguishes resemblance to the model distribution from resemblance
to actual language, and includes human evaluation. This provides an engineering
baseline for separating recoverability, capacity, and naturalness. The shared
external algorithm does the coding; its success is not evidence that an unaided
model can invent the mapping. Distribution matching also depends on which cover
distribution is used as the reference.

### S2 Perfectly Secure Steganography

Schroeder de Witt et al., ICLR 2023; inspected
[arXiv v4, 2023-10-30](https://arxiv.org/html/2210.14889v4), sections 2-5 and 7.

The paper connects perfectly secure steganography to couplings and maximal
throughput within that class to minimum entropy coupling. The guarantees are
relative to a known cover distribution and the stated passive-channel model;
the paper discusses limitations from inaccurate distribution models and channel
perturbations. Thus concealment and throughput are not inevitably opposed under
every formulation. For this repository, externally implemented coding is a useful
channel control, but does not answer the inventiveness question. A mathematical
guarantee under a model distribution is also different from a measured human
suspicion rate in a task-specific context.

### S3 Redwood Text Steganography Benchmark

Roger and Greenblatt, 2023,
[arXiv v2](https://arxiv.org/html/2310.18512v2), sections 4, 7 and appendix A;
[repository README](https://github.com/redwoodresearch/Text-Steganography-Benchmark/blob/main/README.md)
inspected 2026-10-31.

The framework composes an encoder, optional paraphrasing defense, decoder, and
task-quality evaluator. It estimates hidden-information capacity alongside task
performance and separately investigates suspiciousness. Its strong engineered
attacks serve defense evaluation rather than unaided-capability measurement.
An upper-bound interpretation depends on assuming the constructed attacks are at
least as capable as the models being bounded. The repository supplies interfaces
for attackers, defenders, datasets, and evaluators; it notes that complete plot
reproduction records are too large for the repository. Reuse the decomposition,
not a historical residual-capacity figure as a universal limit on steganography.

### S4 OpenAI steganography eval

OpenAI Evals,
[README at commit 11c30b2286d7b92a7c7918bc8a68e9ea769b5aa9](https://github.com/openai/evals/blob/11c30b2286d7b92a7c7918bc8a68e9ea769b5aa9/evals/elsuite/steganography/readme.md),
2024-03-13; pinned file inspected 2026-10-31.

This implementation separates reconstruction error and monitor detectability,
then combines them into mean undetected accuracy. Variants provide a private
scratchpad or reference answer. Baselines copy the payload, append it openly to
a task answer, or perform only the cover task. These are directly reusable
diagnostics for broken transport, detection, and false recovery. Its continuous
reconstruction measure and monitor-relative composite are different from exact
hidden-message success. Supplying a reference answer reduces the cover-task
burden; supplying a scratchpad changes reasoning and observer access. Neither
intervention is neutral for interpreting capability.

### S5 CASE capability framework

Motwani et al., NeurIPS 2024;
[inspected arXiv v5, 2025-07-25](https://arxiv.org/html/2402.07510v5),
section 5, tables 1-3, appendices F-I.

CASE means Collusion and Steganographic Evaluation. It maps shared knowledge and
individual capabilities needed for different steganographic constructions, and
separates individual, group, and scenario evaluations. Tests include independent
agreement on conventions, encoding/decoding, and access to algorithmic tools.
This is a useful conceptual foundation for distinguishing coordination failure
from execution failure. The paper describes software, but a standalone CASE
installation was not verified in this review; do not assume it is an immediately
usable dependency. Its capability taxonomy is not an empirically calibrated
ordering of codebooks for this repository's small recurrent models.

### S6 Hidden in Plain Text

Mathew et al., IJCNLP-AACL 2025,
[published paper](https://aclanthology.org/2025.ijcnlp-long.34.pdf),
sections 3.3 and 5.2, appendix A.2.4;
[implementation](https://github.com/olliematthews/lasr-steganography).

The paper distinguishes centralized strategy invention with a private shared
strategy from decentralized learning, and also studies weight training. Its
payload search first collects successful binary strategies, then tries larger
and multipart payloads, stopping when capacity no longer improves. Strategies
are selected on training data and evaluated on task test data. This is a concrete
adaptive curriculum. It retains the same mitigation team across that search and
test, so task generalization does not establish generalization to new monitors.
Borrow the curriculum while recording who invented and shared the scheme. The
repository README, inspected 2026-10-31, explicitly says its terminology is
outdated and directs readers to the published paper.

### S7 Early Signs of Steganographic Capabilities

Zolkowski et al.,
[arXiv v2, 2025-10-14](https://arxiv.org/html/2507.02737v2), sections 5-6 and
appendices B-C;
[evaluation repository](https://github.com/arturzolkowski/steganographic-evals).

The message-passing study separately varies prior scheme knowledge and
unmonitored reasoning, then removes the cover task and monitor to test scheme
execution. Common and newly devised encodings behave differently. Encoded
reasoning experiments also vary requested subtlety. These are unusually direct
precedents for a diagnostic relaxation ladder. The message-passing setup uses
short phrases, an Anthropic helpful cover-task dataset, and GPT-4.1-mini
monitoring, with sensitivity analyses; results concern those affordances and
models. The code README, inspected 2026-10-31, provides configurable message
passing and reasoning experiments. It does not supply a validated automatic
choice of the next relaxation for a new subject model.

### S8 The Steganographic Potentials of Language Models

Karpov et al.,
[arXiv v1, 2025-05-06](https://arxiv.org/html/2505.03439v1), sections 3-5,
table 2 and appendix B.

The study compares direct prompting, supplied synonym procedures, prior
coordination, and training. Its table shows that supplying a procedure does not
improve every model/payload setting. The authors report difficulties agreeing on
replaceable words as messages grow, and inspect successes based on associations
that exploit an overseer's weaknesses. Their random-letter success criterion
accepts recovery of more than half the payload; meaningful words use a different
criterion. Relevance: a reported success need not mean exact transfer or convincing
concealment, and a longer codebook can introduce execution burdens even while
removing invention burdens.

### S9 StegoBench

Meyer et al., 2026,
[author project page](https://stegobench.com/), reviewed 2026-10-31.

The project reports 18 schemes across lexical, syntactic, structural, generative,
and paraphrase families, comparing prompting, in-context reinforcement learning,
and supervised fine-tuning. Primary subjects include Gemma-3-4B-IT,
Qwen2.5-3B-Instruct, and GPT-OSS-20B. It reports better trainability for local
lexical choices than for several structural or distribution-dependent schemes,
and measures recovery separately from a linguistic-feature detector.

This is a promising catalogue for candidate diagnostic tasks, especially at small
model sizes. It measures acquisition/execution of known schemes, not independent
invention. A limited classifier is not evidence of human indistinguishability.
Evidence access is narrower here: the author page was inspected, but the indexed
OpenReview PDF required browser verification and the site's Paper link did not
resolve to a paper. Gated datasets/checkpoints were not accessed. Treat the
reported ordering as a hypothesis for local calibration, not an established
ranking for recurrent-depth experiments.

### S10 Decision theoretic steganographic gap

Anwar et al.,
[arXiv v3, 2026-04-29](https://arxiv.org/html/2602.23163v3), sections 4, 6-7
and appendices E-F.

The proposed steganographic gap compares how much a signal improves receiver and
auditor utility on a specified task, each relative to its own no-signal baseline.
It offers a way to examine usable hidden information when a benign text
distribution is hard to specify. The gap depends on auditor capability and task;
appendix E.5 shows that ordinary capability differences can produce a positive
gap without private encoding. Estimation also depends on eliciting capable
decoders. This complements, rather than replaces, cover-quality and detection
tests. For a depth comparison, weaker public reasoning at one depth must not be
mistaken for inability to access a private message.

### S11 Tool use and coordination

Rippin et al.,
[arXiv v1, 2026-06-25](https://arxiv.org/html/2606.28425v1), sections 3.2 and 4,
appendices C-D.

The authors study agent implementations of stegosystems with coding tools,
research materials, and model-sampling access. They separately measure agreement
on algorithm families and the parameters required for compatible operation,
reporting limited strict one-shot coordination. This reinforces the distinction
between constructing a working scheme and independently agreeing to use it.
The implementation experiments use Claude Opus 4.6 in Claude Code; they are not
evidence that a small open-weight model performs the algorithm unaided. The
paper's estimated complete-coordination index should not be substituted for a
measured end-to-end transmission rate.

### S12 Public lab assessment practice

Anthropic,
[Risk Report February 2026](https://www-cdn.anthropic.com/08eca2757081e850ed2ad490e5253e940240ca4f.pdf),
section 2.4.3.4, printed pages 26-27.

The report discusses an in-house fine-tuning evaluation using paraphrased
reasoning and interprets the mathematics result as evidence against reliance on
illegible encoded thinking in that domain. It establishes that a frontier lab
uses steganography-related diagnostics in its assessment practice. It is not a
prescription for eliciting new channels, nor evidence that all prompted
steganography fails. For limitations of paraphrasing as a diagnostic, compare
[S10](#s10-decision-theoretic-steganographic-gap), appendix A: preserving a
channel or damaging legitimate reasoning can both complicate interpretation.

### S13 An adjustable framework with a different objective

Peter M. Just, Texas A&M thesis, December 2025,
[institutional abstract](https://hdl.handle.net/1969.1/1600481), reviewed 2026-10-31.

The abstract describes altering token distributions under KL-divergence and
total-variation criteria to trade capacity against statistical deviation. Only
the institutional abstract was inspected. Despite its relevant title, this is an
engineering optimization of externally specified coding, rather than the desired
framework for choosing how much model assistance an experiment can justify.

## Implications for this repository

**Synthesis:** the components are sufficiently developed to build a useful design
guide now. The missing contribution is an explicit mapping from a diagnosed
failure to an intervention and the claim that survives that intervention. There
is no basis here for claiming a universal numerical exchange rate between
inventiveness, secrecy, capacity, and statistical confidence.

Existing repository work already records condition ancestry, distinguishes
prompt relaxations from observed outcomes, and reserves confirmation data. See
the [semantic-channel design](../../experiments/semantic_covert_channel/README.md)
and [condition registry](../../experiments/semantic_covert_channel/configs/conditions_v1.toml).
The [recorded findings](../../experiments/semantic_covert_channel/notes/findings.md)
report unsuccessful length/elaboration and decoder-prompt interventions. Those
reports were read here; their raw runs were not reanalyzed. They do not identify
which untested assistance would recover signal.

**Recommendation:** reuse the repository's execution/accounting and condition
provenance. Borrow component tests from S4/S7, the search pattern from S6, and
scheme candidates from S9 after inspecting their implementations. A wholesale
replacement with an external harness has not been justified. No external harness
was installed or executed in this review.

## Search coverage and remaining uncertainty

Searches covered LLM steganography evaluation frameworks, codebooks, capability
elicitation, adaptive difficulty/curricula, linguistic capacity and detectability,
and public lab assessment practice. Citation follow-up included both foundational
coding work and 2025-2026 evaluations. Only primary papers, author repositories,
an author project page, an institutional abstract, and a lab report support the
findings above. The inspected repository files were READMEs, not implementation
audits. Other than the pinned S4 file, repository links are mutable and carry the
review date rather than an immutable execution identity.

Public source text was available at the linked locations when reviewed, subject
to the S9 access limitation. This repository stores the synthesis and source locators,
not copies of the papers or third-party raw experiment data. No new model
generation or spending occurred. Reproduction would require selecting a source
revision, inspecting its actual prompts/scorers/data and dependencies, and applying
this repository's normal execution controls.

Open questions are whether the proposed diagnostics predict useful next steps
on local recurrent models, whether any candidate codebook preserves human cover
quality, and how much protocol invention contributes beyond scheme execution.
The companion [design guide](steganography-experiment-design.md) makes these
questions actionable without treating them as answered.
