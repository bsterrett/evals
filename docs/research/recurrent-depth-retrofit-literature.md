# Adding recurrent depth to pretrained open-weight models

Scope: literature and public implementation review, 2026-11-06,
America/Los_Angeles. The user is considering a new experiment on improving an
already-trained model by adding recurrence, and explicitly requested coverage of
both frozen-weight methods and methods requiring additional training. This is
research preparation, not an experiment protocol or authorization to run training.

## Answer and evidence status

**Yes: this is an existing research direction, including reusable implementations.**
It is usually called *retrofitting recurrence*, *recurrent-depth conversion*,
*inner looping*, or *test-time depth adaptation*. The papers below provide direct
examples. Evidence supports improvements in particular model/task/budget settings;
it does not establish a universal upgrade for arbitrary checkpoints.

Two questions must remain separate: whether a retrofit improves the starting
checkpoint, and whether it beats the best alternative use of the same training
and inference resources. The recent positive and negative studies below address
different versions of these questions.

Observations in this note are external authors' reported measurements or this
agent's inspection of public text/code. No weights were downloaded, models run,
or reported results independently reproduced during this review. Existing local
recurrent-model calibration is recorded [elsewhere](local-recurrent-model-setup.md)
and is not evidence that a retrofit works.

The companion [methods and terminology](recurrent-depth-methods-and-terms.md)
provides the taxonomy. Existing [native recurrent model descriptions](astra-recurrent-depth.md#published-recurrence-designs-we-can-inspect)
remain their canonical home.

The separate candidate of [replacing some written CoT scratchpad use with
continuous recurrent computation](recurrent-depth-cot-substitution.md) has its own
review and measurement criteria. It remains distinct from the general retrofit
performance question, including when both use the same implementations.

## Direct literature: frozen base models

### Inner Loop Inference — Lys et al., February 2026

[Paper, v1](https://arxiv.org/html/2602.14759v1), especially sections II-IV and
Table I; [authors' code](https://github.com/jonathanlys01/looped-transformer).
Repeats a middle layer range and regularizes the activations by interpolation
with the ordinary forward-pass state or previous iterations. Studies Gemma 2
2B/9B and Llama 3 8B. This is an inference intervention without weight updates.

Added 2026-11-15, from an automated extraction of the v1 HTML (WebFetch) requested by
Claude Fable 5.2, not a direct reading: the loop keeps every state `h(t)` at the loop
boundary, with `h(0)` the ordinary forward-pass state, and feeds the next pass a
weighted mix. "Uniform" is the mean of all states so far (equation 9), "moving
average" is `eta h(0) + (1 - eta) h(t)` (equation 10, `eta` not stated), and
"auto-alignment" weights states by a softmax of their inner products with `h(0)`.
Three repetitions; Gemma 2 2B loops layers 10 to 13. Multiple-choice tasks are scored
by length-normalized likelihood. Cache handling during decoding is not described.

The paper's text does not say whether `h(0)` is the state entering the loop or
leaving it after the ordinary pass, nor which state the later layers receive. A
second automated reading, of `src/looped_llama.py` in the authors' repository on
2026-11-15, says the code differs from the equations in two ways: it stores and
mixes the output of every looped layer across iterations, not only the loop's
output, and its first iteration is the ordinary pass, so one repetition is the
unmodified model. It also reported a moving-average weight of 0.5, each pass given
its own cache slot, and two apparent bugs in that file. The reader would not quote
the code, so none of this is verified. The toolkit offers both arrangements.

The gains are small and uneven: Gemma 2 2B WinoGrande, five-shot, changes from
68.75 +/- 1.30 to 69.53 +/- 1.29 with uniform interpolation, whereas ARC-Easy
changes from 80.30 to 80.01. These are the paper's displayed uncertainties, not
a paired significance test. All naive-loop placements in its Gemma 2 2B
WinoGrande sweep underperform baseline. Treat the proposed distribution-shift
explanation as an interpretation of those outcomes, not a demonstrated universal
mechanism. Repository access and its minimal setup README were verified; the
implementation was not audited or executed here.

### Training-Free Looped Transformers — Chen et al., May 2026

[Paper, v1](https://arxiv.org/html/2605.23872v1), sections 2-3 and appendices
G/J/P. Replaces a full-strength repeated update with damped refinement, optionally
anchored to the original block output; distinguishes block-wise and layer-wise
execution. Reports seven model families and gains including **+2.64 percentage
points on MMLU-Pro for Qwen3-4B-Instruct**. Naive looping, wide windows, and some
small-model configurations fail. Most evidence concerns multiple-choice scoring;
transfer to open-ended generation requires a separate check.

**Details that shape a replication (added 2026-11-15).** These come from two
automated extractions of the v1 HTML (WebFetch, which answers through a small model),
requested by Claude Fable 5.2 while building the
[retrofit toolkit](../retrofit-toolkit.md). They were not read directly, and the
extraction itself flagged internal inconsistencies in the paper. Check each against
the paper before relying on it.

- The update is damped Euler with step `1/K`: `x <- (1 - 1/K) x + (1/K) g(x)`, for
  `K` sub-steps (equation 10). The anchored form mixes in the unmodified block
  output: `beta g(x0) + (1 - beta) F^K(x0)` (equation 16), with `beta = 0.5` in
  appendix E. `K` is 2 or 3.
- Windows are written `[a-b]` with both ends included. Qwen3-4B-Instruct mostly uses
  layers 15 to 18 with `K = 3`; some cells use 11 to 14 or 17 to 19.
- **Selection used the reported benchmarks.** Appendix P searches about 720
  configurations per cell and keeps the best, so the published gains are not
  held-out estimates. Appendix D's "no per-cell tuning" conflicts with this.
- **The attention cache is shared across passes.** One stored key/value set per
  looped layer serves every pass during decoding, written from the state before or
  after the loop. This approximates the looped computation; the toolkit's runner
  instead keeps one set per pass, which equals recomputing the whole prefix. The two
  agree for likelihood scoring, where no decoding happens, and can differ for
  generated text.
- The MMLU-Pro cell is labelled "5sh CoT", which suggests generated answers, not
  option likelihoods. The scoring method is not stated. lm-eval 0.4.11's own
  `mmlu_pro` task, read directly from its source, is five-shot with written reasoning
  and takes the answer from the generated text.

A third extraction the same day, of the appendices, added the settings below. The
same caution applies.

- **Per-cell settings for Qwen3-4B-Instruct (Table 6).** Windows are 0-based with
  both ends included. Seven of nine cells use layers 15 to 18, block mode, `K = 3`
  and a strategy named "Euler", with the cache written from the first pass and loops
  on every decoding step. MedMCQA and OpenBookQA use layers 11 to 14. ARC-Challenge
  uses "damped Euler" with a coefficient of 0.5 and a halting threshold the paper
  does not define. Reported scores, baseline to looped: MMLU-Pro five-shot 57.14 to
  59.79, GPQA-Main 33.71 to 35.71, SciQ 93.30 to 95.10, MMLU 68.15 to 69.45,
  CommonsenseQA seven-shot 78.87 to 79.93, MedMCQA 53.40 to 54.13, ARC-Challenge
  25-shot 62.54 to 63.14, C-Eval 72.21 to 72.73, OpenBookQA 40.20 to 40.60.
- **Whether the "Euler" cells use the anchor is not stated.** The table gives no
  anchor value for them.
- **Harness settings.** lm-eval 0.4.11, BF16 weights, SDPA attention, "deterministic
  seeds" with no values, "default lm-eval-harness batch sizes", and nothing on a chat
  template or task names. Appendix F.1 names Transformers 4.46.
- **The cache choice changes the generated-text result.** Table 20 gives the MMLU-Pro
  gain as +2.64 points writing the cache from the first pass, +1.00 from the last and
  -8.86 with none. Appendix N.1 says every number uses the last pass unless stated,
  while Table 6 lists the first for this model.
- Other tables give different numbers for the same model: MMLU at layers 17 to 19
  with `K = 2` is 69.14 in Table 14, against 69.45 in Table 6.

The toolkit's attempt to reproduce the CommonsenseQA cell is recorded in the
[toolkit note](../retrofit-toolkit.md#reproducing-a-published-number).

**Availability observation:** the [advertised repository](https://github.com/L-z-Chen/Training-Free-Looped-Transformer)
and its README returned 404 through direct GitHub access during this review.
Search-indexed copies still describe a Qwen3/Qwen3-MoE runtime patch. That indexed
description is not a verified current release. A separately accessible
[independent implementation](https://github.com/JegZheng/tflt-dev) offers a possible
starting point, not a validated replacement. Its README distinguishes the
upstream two-model recipe from a paper-oriented recipe and defaults to bypassing
loops during decoding. These settings need checking before a replication.

### CoLa and PoLar — Li, Li, and Zhou, 2025-2026

[CoLa preliminary study, July 2025](https://arxiv.org/abs/2507.07996) and
[PoLar, reviewed v2](https://arxiv.org/html/2606.06574v2), sections 3-4 and
Tables 2-3; [official code](https://github.com/tianyi-lab/PoLar).
Treat layers as reusable functions and predict paths that skip, retain, or repeat
short segments. Base LLM weights stay frozen, but the practical method **trains a
controller** using paths discovered by offline search. It is not a zero-training
end-to-end procedure.

For Llama-3.2-3B-Instruct on DART-Math difficulty 1, Table 2 reports pass@1 of
54.6% versus 51.1% for greedy baseline. Larger pass@5 gains involve five candidate
programs and must not be presented as one-answer accuracy. Reviewed v2 specifies
question deduplication before difficulty-wise splitting. Its public README and
root directory now include `mcts_discovery/`, with a September 2026 release note;
older accounts claiming search code is missing do not describe this inspected
state. The documented support is four named Llama/Qwen checkpoints, not arbitrary
architectures. The full training pipeline was not exercised.

### MACRO — Batorski et al., August 2026

[Paper, v1](https://arxiv.org/html/2608.05872v1), method and experimental details;
[official code](https://github.com/Batorskq/MACRO).
Learns a task-specific route distribution over a frozen model using training
feedback, then selects one route using validation data. Supports repeating,
skipping, and adding saved residual states. Unlike an input-specific router, the
deployed route is fixed for a task. This is base-weight-free adaptation, with
search and labeled-data costs.

The paper reports a Qwen3-1.7B GSM8K example improving from 43.4% to 69.5% using
33 layer calls instead of 28. This illustrates possibility, not a predicted
effect size for another task. The inspected README describes train/validation/test
separation and six model configurations, including quantized configurations.
Those precision and selection conditions belong in any attempted comparison.

### RYS and model relayering — practical primary sources

[RYS repository](https://github.com/dnhkng/RYS) provides a reusable scanner,
multi-block search, runtime wrappers, and checkpoint export. Its `(i,j)` recipe
executes layers `0..j-1`, then `i..N-1`, repeating the intervening interval.
Inspected runtime code creates shallow module copies sharing parameter tensors
but assigns distinct attention-cache indices. The exporter instead materializes
duplicated tensors in the saved checkpoint. These are distinct storage choices.

The published probes and search tools are useful engineering evidence; private
historical runs are explicitly excluded from the release. Selecting the best
path on small Math/EQ probes needs independent held-out evaluation before any
claim of general improvement.

[RYS-Llama-3.1-8B-Instruct](https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct)
and [RYS-XLarge](https://huggingface.co/dnhkng/RYS-XLarge) are plausible examples
of what the user remembers, not an identification. Naming varies: related cards
also use “Repeat Your Steps,” while other sources expand RYS differently. Prefer
the actual layer path over the acronym. This review does not infer a matched
baseline comparison from a model card's leaderboard score.

## Direct literature: conversion through further training

### Relaxed Recursive Transformers — Bae et al., 2024 / ICLR 2025

[Reviewed arXiv v3](https://arxiv.org/html/2410.20672v3), sections 2-3;
[conference publication](https://proceedings.iclr.cc/paper_files/paper/2025/file/54d6a55225cebbdc16fbb0e45c5bdf2b-Paper-Conference.pdf).
Converts pretrained LLMs into smaller repeated-block models using weight
selection/averaging and recovery training. Its relaxed version adds distinct
low-rank corrections at different logical depths. This is partial parameter
sharing rather than one identical transformation everywhere.

The central result is compression with performance retention: for example,
recursive Gemma near 1B competes with other 1B models and recovers much of the
original 2B model's quality. It does not show that the compressed model universally
outperforms its original parent. The reported continuous depth-wise batching
speedups are theoretical/hypothetical estimates in this reviewed version, not
measured deployment improvements.

### Teaching Pretrained Language Models to Think Deeper — McLeish et al., November 2025

[Paper, v1](https://arxiv.org/html/2511.07384v1), sections 3-4, Table 1, and
appendix C; [official conversion/training code](https://github.com/mcleish7/retrofitting-recurrence);
[checkpoint collection](https://huggingface.co/collections/tomg-group-umd/retrofitting-recurrence).
Directly studies converting TinyLlama-1.1B, OLMo-2-0425-1B, and Llama-3.2-1B.
Keeps prelude/core/coda layers, removes others, adds an input-reinjection adapter,
then uses continued pretraining with a recurrence curriculum and a recovery phase.

Table 1's two-phase TinyLlama condition, trained at recurrence 4 and evaluated at
32, scores GSM8K 51.2% versus 45.2% for the non-recurrent two-phase condition;
MATH is 14.2% versus 12.8%. This is not a zero-cost intervention: the mixed-data
experiment uses 26B recovery tokens followed by 26B mixed-data tokens. Inference
at 32 recurrences also spends additional compute. The separate compute-curve
experiments compare against continued training of non-recurrent parents. This is
strongly relevant methodological precedent, not proof of a cheap local recipe.

### LoopUS — Park et al., May 2026

[Paper, v1](https://arxiv.org/html/2605.11011v1), sections 3-4 and appendix A;
[official code](https://github.com/Thrillcrazyer/LoopUS);
[released Qwen3-1.7B checkpoint](https://huggingface.co/Thrillcrazyer/Qwen3_1.7B_LoopUS).
Explicitly presented as a post-training framework for recasting pretrained LLMs.
Uses an encoder, repeated middle block, and decoder; an input-dependent update
gate; sampled intermediate supervision; and a separate confidence head for
adaptive stopping. These are different interventions and warrant separate
ablations. The paper reports reasoning-oriented benchmark improvements; this
review makes no independent effectiveness claim.

The inspected README exposes a configurable `--model-name`, encoder/decoder layer
ranges, training-token budget, and recurrence count. Its example is continued
training on FineWeb-Edu, not a frozen-weight patch. A configurable interface does
not establish support for every model family. This candidate was already linked
in the repository's earlier model survey; the new relevance is its conversion
procedure rather than merely its released checkpoint.

### UpLoop — Chen et al., October 2026

[Paper, v1](https://arxiv.org/html/2610.25004v1), sections 3-4 and Table 1.
Loops each selected layer locally, rather than repeatedly traversing a whole
middle block. Adds per-layer/per-loop attention and MLP residual gates, plus
intermediate-depth and improvement losses. The default SFT updates all model
parameters, including gates; freezing base weights is not its default recipe.

For Qwen3-4B Base after SFT, the paper reports a 6.17-point average advantage
over its non-looped condition across 20 benchmarks at three loops. Evaluations
use sampling and mean pass@1, with at least four samples per example and 64 on
AIME. Its first-page note says code and models will be released; a working release
was not verified here. Treat this recent preprint as a useful design hypothesis,
not the most accessible implementation for a first replication.

### A parameter-efficient installation study — Shapiro, 2026

[Paper, reviewed v1](https://arxiv.org/html/2608.11233v1), sections 6-11;
[adapter model card](https://huggingface.co/mshapiro123/recurrent-qwen2.5-0.5b-r16-adapter).
Converts Qwen2.5-0.5B-Instruct using a re-entry bridge and either about 6M trained
adapter/bridge parameters over frozen base weights or about 180M full-block
parameters. The 1,792-row symbolic evaluation reports 83.76% versus 84.04%.
This is evidence for installing a particular iterative mechanism at a small
trainable-parameter budget, not broad instruction-following improvement.

Important limitations are minimal zero-shot transfer to verbal renderings and
interference when learning the inverse task. The reported 7.6x latency advantage
compares one-token recurrent answers with a 96-token scratchpad at batch one on
an A100; it is a system comparison with different output lengths, not a general
per-FLOP or batched-throughput speedup. Here again, frozen base weights do not mean
that nothing was trained.

## Contrary findings and retention

[There Is No Free Loop, Sukthanker and Cameron, v1, October 2026](https://arxiv.org/html/2610.42030v1),
sections 3-5, Table 2, and appendices D/E, compares looped models with equally
trained non-looped twins and alternative uses of inference compute. Its short
fine-tune improves Qwen3.5-0.8B macro accuracy on GSM8K/MATH-500 from 44.2% to
48.3% across four training seeds. Corresponding 4B results are 82.7% to 82.3%
across two seeds, without a significant gain. Multiple larger-model recipes also
fail to show an advantage. These tested regimes do not establish impossibility
for larger models or refute every operator above.

Appendix E identifies a useful failure mode: a lower perplexity can reflect
changed output calibration rather than better answers. Agent recommendation:
include downstream accuracy and a temperature-calibration control when selecting
placements by perplexity. This negative study and the positive studies should
both inform the experiment; their operators, budgets, tasks, and decoding differ.

[One Model, Two Paths, Shen et al., v1, October 2026](https://arxiv.org/html/2610.33517v1),
sections 3-6, studies retaining a model's original single-pass path while training
its repeated-layer path. Joint loss weighting and guidance from the frozen
original improve the retention/gain tradeoff. The authors distinguish retention
relative to the starting model from retention relative to ordinary continued
training. This is an additional objective and control strategy, not a universal
performance guarantee.

## Related approaches that change the question

| Approach and primary source | Relationship to the proposed experiment |
| --- | --- |
| [Recirculation, Mozer et al., v1, August 2026](https://arxiv.org/html/2608.17981v1), sections 2 and 4.6 | Feeds a norm-matched deep representation back toward a shallow layer as tokens are processed. Preserves state across token steps and requires serial prefill; it is not ordinary same-token depth looping. Fixed coefficients require no gradient training; the adaptive variant learns coefficients with base weights frozen. Table 2 explicitly flags a condition tuned on MMLU test data; keep that condition separate from held-out evidence. |
| [T2MLR, Cai et al., v1, July 2026](https://arxiv.org/html/2607.15178v1), sections 2 and 4.4 | Temporal middle-layer feedback from the previous token into the current token; includes retrofitting a pretrained 1.7B transformer through tuning. A different information path from repeatedly applying a block before emitting one token. |
| [Coconut, Hao et al., reviewed v4](https://arxiv.org/abs/2412.06769v4) | Feeds the final hidden state back as a subsequent input embedding. Continuous latent reasoning advances along extra sequence positions; it is not identical to reusing a middle block at the same position. |
| [LoOPSD, Fan et al., v1, October 2026](https://arxiv.org/html/2610.18162v1), section 3 | Searches a frozen repeated-layer teacher, then distills into a student using the ordinary layer order. An alternative if the aim is improved final performance without serving a recurrent student. Search and student training still cost compute. |
| [SOLAR depth up-scaling, reviewed v3](https://arxiv.org/abs/2312.15166v3) | Builds a larger model from copied layers and continues pretraining. Untied copies can diverge; parameter growth changes the comparison. |
| [Mergekit passthrough](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md#passthrough-passthrough) | Reusable tooling for assembling unchanged layer ranges, including repetition through overlapping slices. Checkpoint construction alone does not supply recurrence training, stable activations, or evidence of improved performance. |

### Recirculation details that shape an implementation

These come from three automated extractions of the v1 HTML on 2026-11-16, made to
build the [toolkit's recirculation design](../retrofit-toolkit.md#feedback-across-token-steps).
A small model read the page each time; no person or top-tier model read the paper
directly, and Appendix A was not in the extracted text.

- **The update.** Equation 1 mixes a token's source-layer state into its own
  destination-layer state, one unrolling step later, with weights `alpha` and `beta`.
  Equation 2 first rescales the source to the destination's L2 norm. The states are
  residual-stream outputs "after" a layer, indexed by unrolling step, input step and
  layer.
- **Passes.** The text says the stack runs twice per input step, that only the
  previous step's stack is recirculated so each is recirculated exactly once, and
  that the read-out follows the first pass. Appendix A is reported to describe `k`
  recirculations with `k + 1` stacks per step.
- **Serial prefill.** The abstract and section 5.2 say the prompt must be processed
  token by token.
- **Coefficients.** Sweeps used `alpha` of 0.04, 0.07, 0.10 and 0.16. Appendix C is
  reported to use 0.07, with `beta = 1 - alpha` on Gemma 3 1B and `beta = 1` on 4B
  and 12B, where the text says a non-convex mixture is required. One extraction also
  reported 0.15 for the section 4.2 evaluation; the two were not reconciled.
- **Placements, as {source, destination}.** Gemma 3 PT: 1B {11, 4}, 4B {18, 9}, 12B
  {35, 16}. Gemma 3 IT: 4B {18, 8}, 12B {29, 16}. The MMLU condition on 4B PT used
  {16, 5} with `alpha` 0.09, selected on MMLU development problems.
- **Variants not in the toolkit.** Table B.2 lists eight ways to normalize and mix.
  `alpha` was ramped over the first 10 tokens on the 1B model only. The learned
  variant trains a small network, with the base model frozen, to produce
  token-specific mixing vectors, and is reported to cut perplexity by 23.0% on
  average against 8.5% for fixed coefficients.
- **Not stated in the extracted text:** whether layer indices start at zero, where in
  the residual stream the mixture is written, how the attention cache is handled,
  whether the second pass continues above the source layer, and any code link.

### LoopUS details that shape an implementation

Read on 2026-11-16 to load the released checkpoint in the
[toolkit](../retrofit-toolkit.md#a-checkpoint-trained-to-loop). Claude Fable 5.2 read
the sources below as page text in a browser, not through a summarizing model:

- the authors' code at `Thrillcrazyer/LoopUS` commit
  `003ad7a5a5967d059b4639751777edd94d5640a4` (2026-10-13): all of
  `models/modeling_lds.py`, `models/configuration_lds.py`, `utils/inference.py`,
  `utils/lm_eval_harness.py`, `evaluate.py`, `script/eval_model.sh` and the README,
  and the training loop in `training_runtime.py`;
- the paper's v1 HTML: the gate in section 3.1, section 4.1, Table 1 and appendix
  A.1 to A.5. The rest of the paper was not read;
- the checkpoint `Thrillcrazyer/Qwen3_1.7B_LoopUS` at commit
  `5d66726aa98197c1b5876e26339cf393d82b436c`: its configuration, model card and the
  list of tensors in its weight file.

None of the code was run.

- **Three blocks.** The first layers run once, the middle layers loop, and the last
  layers, final norm and output head run once. For Qwen3-1.7B the paper's appendix
  A.1 and the checkpoint agree: layers 0 and 1, then 2 to 26, then 27.
- **The gate.** The code and the paper's equations 3 to 5 agree. With `M(h)` the
  looped block's output for state `h`: `delta = softplus(W (M(h) - h) + b)`,
  `alpha = exp(delta * A)` with `A = -exp(A_log)`, and the next state is
  `alpha * M(h) + (1 - alpha) * h`. `W` is a product of a 2048-to-128 and a
  128-to-2048 matrix. The gate follows every pass, the first included. The gate
  class's own description states the mixture with the two states exchanged; the code
  below it and the paper do not.
- **The confidence head** is a layer norm and a linear layer. It reads the change
  before the gate, `M(h) - h`, at the last valid token of each sequence. Under the
  default rule a sequence stops when the sigmoid of the head exceeds a threshold. The
  code has two other rules, on the size of the state change and on a cumulative
  probability.
- **Stopping works differently with and without a cache.** Without one, each row of a
  batch stops separately and keeps its state while the others continue. With one, the
  loop ends only when every row has passed the threshold, each row uses the state
  from the first pass on which it did, and the caches of the passes that did not run
  are left out of the returned cache and start empty on the next call. So after a
  token stops early, later tokens at the deeper passes attend only to the tokens
  since then.
- **Caches.** The looped block keeps one attention cache per pass.
- **Evaluation.** The authors score likelihoods through their own lm-eval adapter,
  not the harness's stock one: left-padded batches with an attention mask, no cache,
  and the context and continuation tokenized separately without special tokens. The
  released template sets at most 8 passes, a threshold of 0.6, batch size 8 and seed
  2026. Appendix A.3 gives the same 8 and 0.6 as standard inference, zero-shot.
- **Training.** The paper reports FineWeb-Edu, 3 billion tokens, context 1,024 and 20
  loop steps of which 5 random ones are supervised per batch; the README's example
  command uses 1.5 billion tokens. In the training loop each step is one pass of the
  block and its gate, and the first layers run without gradients.
- **The released configuration** records 20 passes, a threshold of 0.9, the threshold
  rule, and the head read on every pass. It was written by Transformers 5.3.0. The
  pinned 4.57.1 cannot load it directly: the rotary base is under `rope_parameters`,
  and the tokenizer settings give `extra_special_tokens` as a list.
- **Which parent.** The model card names `Qwen/Qwen3-1.7B-Base`, and the configuration
  names `Qwen/Qwen3-1.7B`. The weights settle it. The checkpoint's embedding, final
  norm and all 22 tensors of layers 0 and 1 are bit-identical to `Qwen/Qwen3-1.7B` at
  `70d244cc`, and over the whole decoder it is 0.019 away from that model in relative
  distance against 0.060 from the Base model. The record is in the
  [acceptance summary](../../local_models/gated-check-20261116.json).
- **Published numbers for Qwen3-1.7B (Table 1), without and with LoopUS:** WikiText
  perplexity 21 and 16.9, LAMBADA perplexity 12.21 and 7.43, and accuracy on MMLU
  55.4 and 56.6, HellaSwag 46.2 and 46.3, ARC-Easy 72.5 and 74.9, ARC-Challenge 40.2
  and 43.1, PIQA 72.2 and 73.3, WinoGrande 61.3 and 63.0, OpenBookQA 28 and 29.6.
  The paper does not say which Qwen3-1.7B the first row is.

## Reusability and inspected evidence

There is no single verified “apply to any model” framework in the reviewed set.
There are reusable research implementations with architecture and runtime limits.
In particular, a stock exported deeper checkpoint and a runtime weight-sharing
loop need different memory and cache accounting.

GitHub sources were accessed through the connected GitHub tool on 2026-11-06.
The following small identities make consequential code observations inspectable;
they identify files, not an installed or tested environment.

| Repository/source | Actually inspected | Consequence |
| --- | --- | --- |
| `dnhkng/RYS`, `main` | README and `src/core/layer_duplicator.py`, file blob `dfaefd01ed14b613dd1312339c65f82b5ab65f96` | Runtime layer copies share weights and receive new cache indices; exporter storage differs. |
| `mcleish7/retrofitting-recurrence`, `master` | README, conversion directory, and `convert_llama.py`, blob `78ad359abe04f4be30e00b1243178e9a0059ab68` | Actual weight-mapping/conversion functions exist. README pins conversion to `transformers==4.51.0` because of cache changes and requires model-file replacement/comparison steps. This is research code requiring adaptation. |
| `tianyi-lab/PoLar`, `main` | Root directory and README, blob `eab2cbf36ef8415cf5e31f4ebd0d48563b48874e` | Offline MCTS directory is present; README documents four supported models and controller supervision inputs. |
| `Thrillcrazyer/LoopUS`, `main` | README on 2026-11-06. On 2026-11-16, at commit `003ad7a5a5967d059b4639751777edd94d5640a4`, the model class, configuration, evaluation code and training loop, as [listed above](#loopus-details-that-shape-an-implementation) | Conversion/training CLI and released-checkpoint links are available. The released Qwen3-1.7B checkpoint loads in the [toolkit](../retrofit-toolkit.md#a-checkpoint-trained-to-loop) through a forward pass rewritten from this code. The authors' code itself was not run, so their output was not compared with the toolkit's directly. |
| `Batorskq/MACRO`, `main` | README | Search, validation selection, and model-specific configurations are documented; underlying executor was not audited. |
| `jonathanlys01/looped-transformer`, `main` | Repository metadata and README | Accessible setup source; substantive implementation unverified. |
| Training-Free Looped Transformers | Direct repository/README 404; indexed README; accessible `JegZheng/tflt-dev` README | Current official code availability unresolved. Independent code's documented defaults require scrutiny. |

Raw paper/repository content remains available at the linked public sources;
no upstream source snapshot or experimental run artifacts were created here.
Versioned papers and the concise excerpts of numerical results above are the
durable evidence retained in this repository. Mutable code branches must be
pinned to a full repository revision before an actual run.

## Agent recommendation for a discriminating experiment

The broad possibility question is already answered affirmatively under some
conditions. A useful new question is **which retrofit improves which pretrained
checkpoint, with how much additional training, at a fixed resource budget?**

1. Start with one ordinary non-recurrent checkpoint whose unmodified runtime is
   verified. A 1.7B-4B Qwen checkpoint is a practical candidate given the published
   implementations; this is a selection suggestion, not a local training-memory
   estimate. Ouro/Dory are useful native-recurrence comparators, but do not supply
   the missing unmodified parent for a causal retrofit comparison.
2. Separate a frozen-weight arm (ordinary pass, naive repetition, stabilized
   repetition) from a trained-conversion arm. Choose placements and coefficients
   on development data; freeze them before held-out evaluation. Charge search and
   controller fitting separately from inference.
3. For learned conversion, compare with the same parent continued on the same
   data without recurrence. Report both matched-token and matched-training-FLOP
   comparisons when feasible; these are different budgets. If compression removes
   layers, include the parent and a comparable compressed control.
4. Keep prompts, visible reasoning allowance, precision, and scoring aligned.
   Report accuracy against actual FLOPs or latency, as well as at fixed output
   tokens. Multiple samples/programs and pass@k must be charged explicitly.
5. Verify executed layers, cache indices, RoPE positions, prefill/decode policy,
   and cached-versus-uncached outputs. Distinct virtual depths normally need
   distinct KV states unless deliberate sharing is part of the method. Inspect
   precision, peak memory, and general-capability retention alongside task scores.
6. Report the full preregistered depth sweep, including degradation and stopping
   failures. Improvement from one to two passes does not imply monotonic gains or
   useful extrapolation to arbitrarily many passes.

A first frozen-weight replication could establish whether the effect survives
held-out selection and open-ended scoring in this workspace before investing in
conversion training. The choice among RYS, interpolation, damped refinement, and
learned routing remains an experiment-design decision. Record a prospective
protocol before generation; no new experiment directory is needed for this review.
