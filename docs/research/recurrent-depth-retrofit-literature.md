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

## Direct literature: frozen base models

### Inner Loop Inference — Lys et al., February 2026

[Paper, v1](https://arxiv.org/html/2602.14759v1), especially sections II-IV and
Table I; [authors' code](https://github.com/jonathanlys01/looped-transformer).
Repeats a middle layer range and regularizes the activations by interpolation
with the ordinary forward-pass state or previous iterations. Studies Gemma 2
2B/9B and Llama 3 8B. This is an inference intervention without weight updates.

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
| `Thrillcrazyer/LoopUS`, `main` | README | Conversion/training CLI and released-checkpoint links are available; compatibility and numerical correctness were not tested. |
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
