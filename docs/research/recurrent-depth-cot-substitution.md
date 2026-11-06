# Can recurrent depth substitute for a written chain-of-thought scratchpad?

Scope: literature review and experiment-design analysis, 2026-11-06,
America/Los_Angeles. This is a separate candidate from [improving pretrained
models by retrofitting recurrence](recurrent-depth-retrofit-literature.md).
Both remain active research interests; the user's directions and clarification
are recorded in [user policy](../user-policy.md#two-distinct-recurrence-research-candidates).
No new model generation, training, downloads, or independent reproduction was
performed for this review.

## Question and current assessment

The user's hypothesis is that continuous feedback from later activations to
earlier computation might replace some use of explicit CoT as a scratchpad,
preserving or improving task success. The user specifically distinguishes a
reduction beyond what follows automatically from fewer opportunities to produce
CoT from the separate question of compute efficiency.

The user proposed this hypothesis before this follow-up literature review. It is
prospective for repository experiments; no confirmatory protocol has been fixed.

**This direction has been explored directly.** Coconut studies hidden-state
feedback across successive latent positions; LOTUS combines recurrent depth with
parallel latent scratchpad positions; TaH2 studies adaptive recurrent computation
and explicit reasoning under matched decoding compute. These provide relevant
evidence, but answer different parts of the question. Details and qualifications
are recorded beside each source below.

Agent assessment: there is evidence for moving reasoning from emitted text into
continuous computation after suitable training, and for some inference-only
alternatives. The reviewed work does not establish a generally successful
frozen-weight conversion of an existing thinking checkpoint under the user's
stronger turn-normalized criterion. This is a bounded literature-search result,
not proof that no such result exists.

## What a thinking model is, and what actually crosses the boundary

“Thinking model,” “reasoning model,” and “large reasoning model” are common labels,
not one architecture definition. For this candidate, a useful operational
definition is a checkpoint trained to emit an intermediate reasoning region
before a final answer. Training can involve supervised traces, reinforcement
learning, or both. [DeepSeek-R1, reviewed v2](https://arxiv.org/abs/2501.12948v2)
includes reasoning behavior learned through RL without human-written reasoning
trajectories. It would therefore be too narrow to require supervised CoT traces.

[Qwen's official Qwen3 description](https://qwenlm.github.io/blog/qwen3/), reviewed
2026-11-06, documents thinking/non-thinking modes and parses `</think>` as the
boundary before the answer. Such a boundary is an output convention; it does not
by itself establish a separate recurrent architecture. Within an ordinary causal
decoder response, generated reasoning tokens become context for later generated
tokens. This need not involve a second user message or a separate API request.

Schematically, the selected next-input path is:

```text
ordinary decoding: h -> vocabulary scores -> select token -> embedding -> next step
soft decoding:     h -> vocabulary probabilities -> weighted embedding -> next step
continuous bridge: h -> optional projection/normalization -> next latent step/pass
```

The user's information-bottleneck argument is plausible, with qualifications:

- Selecting one token limits the selected embedding to a discrete vocabulary.
  This applies to greedy argmax as well as stochastic sampling: temperature is
  not required for that restriction. A continuous bridge can carry a richer
  signal, but its useful information and the recipient's ability to use it remain
  empirical questions.
- Tokens are **not the only continuous-state path** in an ordinary transformer.
  Later tokens attend to keys and values derived from earlier activations.
  [Transformers cache documentation](https://huggingface.co/docs/transformers/main/cache_explanation),
  `main` reviewed 2026-11-06, explains this retained attention state. Disabling
  caching normally recomputes that information; it does not remove the attention
  path. The proposed bridge adds a route, rather than creating the first possible
  continuous route between decoding steps.
- A final-layer residual vector and a first-layer input embedding can have the
  same dimension but different distributions and functional roles. A raw identity
  connection, a learned bridge, a soft embedding, and a normalized blend are
  different interventions. Information retention alone does not make any one
  compatible with frozen pretrained weights.

These are architectural distinctions and agent reasoning, not new measurements.
Two useful targets remain separate: recurrence at the **same sequence positions**,
and continuous feedback into **new latent positions** that extend the workspace.
The [methods glossary](recurrent-depth-methods-and-terms.md) supplies the broader
taxonomy.

## Primary empirical evidence

All numbers in this section are authors' reported results. They were checked in
the cited versions, not independently reproduced here.

### Coconut: the closest literal hidden-state feedback precedent

[Hao et al., reviewed v4](https://arxiv.org/html/2412.06769v4), sections 3 and 5,
Table 1. Uses the final hidden state directly as a subsequent input embedding,
with curriculum training to replace explicit reasoning steps. This is sequential
latent-position generation, not a same-position block loop.

On GPT-2/GSM8K's 1,319 test problems, Table 1 reports explicit CoT at
42.9 +/- 0.2% and Coconut at 34.1 +/- 1.5%; the displayed token counts are 25.0
and 8.2. On ProsQA, accuracy improves from 77.5 to 97.0% while token count falls
from 49.4 to 14.2. Thus the literal feedback idea has task-dependent outcomes;
the original math result does not preserve CoT accuracy. Continuous slots still
require forward passes and attention state. Fewer emitted words is not an account
of their total computation.

### CODI: distilling explicit reasoning into continuous thoughts

[Shen et al., reviewed v3](https://arxiv.org/html/2502.21074v3), sections 3-4 and
Figure 4. Jointly trains explicit teacher and implicit student tasks with shared
weights and hidden-state alignment. Reports matching explicit-CoT GSM8K
performance at GPT-2 scale with approximately 3.1x CoT compression. Its six
continuous thoughts plus two delimiters are counted as eight reasoning positions.
The reported compact-trace latency improvement is about 2.7x at batch one on an
A100. These are trained latent-sequence results, not a no-training recurrent-depth
patch or a claim of parity at all model sizes.

### LOTUS: recurrent depth plus a latent scratchpad

[Fan, Svete, and Lee, reviewed v2](https://arxiv.org/html/2606.31779v2), sections
3-4, Tables 1-3 and appendix D. Starts from CoT-tuned checkpoints, then fully
fine-tunes a loop over padded latent positions with CoT-aligned supervision.
The Llama-3.2-3B setting uses 150 latent positions and six reported recurrent
iterations. On natural-language GSM8K, accuracy is 68.13 +/- 0.77% versus
68.41 +/- 0.59% for explicit CoT; thought latency is 140.8 versus 963.6 ms on one
H100 at batch one. These are close means, not a preregistered noninferiority
demonstration. Main compact-trace accuracy is 70.0 versus 71.5%.

This closely addresses substitution, while changing workspace width, supervision,
and training. Parallel latent slots still cost memory/FLOPs; thought-phase latency
is not total latency. The latent-block budget covers 99% of the training-style
step counts, with autoregressive fallback otherwise.

### TaH2: the most directly relevant compute-normalized comparison

[You et al., v1](https://arxiv.org/html/2609.35748v1), sections 3-5 and appendix A.
Post-trains Qwen3-Base variants on the same data with single-pass, fixed-loop, and
adaptive-loop architectures. A normalized learned updater maps a final-layer
state plus the token embedding into the next pass; a trained decider allocates
depth per token. The main 1.7B training uses 3.4B tokens.

Fixed-loop baselines have steeper accuracy-compute slopes yet underperform
single-pass at matched decoding compute. TaH2 reports about a 3.4-point AIME
advantage at matched decoding FLOPs. Math evaluation averages 32 samples with
temperature 0.6; it is not pass@32 answer selection. This is relevant to resource
efficiency, but it retains explicit CoT and does not establish the user's separate
per-turn reduction criterion. These architectures are post-trained comparisons
from common base weights, not frozen alterations of the released Ouro/Huginn
checkpoints or of an already post-trained thinking model.

### Soft Thinking: an inference-only test of the token-selection idea

[Zhang et al., v1](https://arxiv.org/html/2505.15778v1), section 3 and Table 1.
Uses a probability-weighted mixture of token embeddings during thinking, retaining
ordinary token decoding for the answer. On DeepSeek-R1-Distill-Qwen-32B, the
math-task aggregate improves from 81.32 to 83.03% pass@1 while generation length
falls from 4,995 to 3,875 positions (22.4%). This directly concerns an already
trained reasoning model, but bypasses hard token selection through a soft
embedding rather than raw last-to-first feedback or a same-position depth loop.

Its entropy-based Cold Stop can insert the end-of-thinking marker. Therefore
shorter traces can partly reflect changed stopping, which needs an ablation.
Latent positions are still computation, even when no token is sampled.

### Counterevidence to an automatic information-preservation benefit

[Wu et al., reviewed v3](https://arxiv.org/html/2508.03440v3), sections 3-5;
[ICLR 2026 publication](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0895de150660b6c2aef9b0f098cb1dcd-Abstract-Conference.html).
Their probes find that vanilla soft-input reasoning often follows its dominant
token component, rather than maintaining independent reasoning branches. Adding
stochasticity improves results in their evaluated settings. This challenges a
particular mechanism story, not every continuous or recurrent design. Eliminating
token sampling need not eliminate harmful commitment or improve search.

[A Model with No Head and Many Thoughts, Koriagin et al., v1](https://arxiv.org/html/2608.31069v1),
sections 4-5 and Table 1, learns a smaller latent projector for feedback, with
optional LoRA. Its DeepSeek-R1-Distill-Qwen-1.5B row reports aggregate single-answer
accuracy of 57.32% versus 58.09% for the unchanged discrete-CoT model, while
pass@32 improves from 83.23 to 86.22%. This qualifies the abstract's broad
performance summary: improved multi-sample coverage does not demonstrate the
user's required single-answer success preservation. It also retains stochastic
continuous selection rather than removing randomness altogether.

## Other relevant evidence and controls

| Primary source and inspection depth | Relevance and limit |
| --- | --- |
| [From Explicit CoT to Implicit CoT, Deng et al., v1](https://arxiv.org/abs/2405.14838), abstract | Gradually removes explicit steps while fine-tuning. A non-recurrent internalization control: shorter CoT need not be caused by adding loops. The abstract's Mistral-7B result is over 50% GSM8K without intermediate text; it does not establish parity with every explicit-CoT condition. |
| [ThinkRouter, Xu et al., v1](https://arxiv.org/html/2602.11683v1), sections 4-5 and Table 1 | Switches between discrete and soft embedding steps using confidence, without model-weight training. For Qwen3-1.7B, its STEM aggregate is 56.42% versus 36.71%, with length 14,422 versus 15,394; different benchmarks and decoding prevent treating this as a replication of Soft Thinking. A hybrid-feedback control, not depth recurrence. |
| [Reasoning with Latent Thoughts, Saunshi et al., v1 / ICLR 2025](https://arxiv.org/html/2502.17416v1), section 5.4 | Provides a theoretical construction simulating CoT steps with loops. An expressivity result is not a procedure for converting arbitrary pretrained weights. |
| [Internalizing Chain-of-thought into Looped Transformers, Zhou and Lu, v1](https://arxiv.org/html/2610.25998v1), introduction and theorem setup | Constructs a finite-precision family trading a factor of loop count for generated tokens and context under stated conditions. This illustrates resource substitution in a constructed model, not empirical success or a reduction beyond the user's mechanical turn-count baseline. |

## Preserve the user's turn-count null separately from resource matching

The user corrected an earlier agent interpretation: the key counting concern is
**fewer opportunities to emit thinking**, rather than merely greater work per
token. Use an abstract emission-turn unit until it is operationally defined.
Do not silently equate a turn with a sampled token, an entire assistant response,
or a chat round. The choice matters before implementing an experiment.

Let `N` be emission turns per task and `M` be the total material in the explicit
reasoning region, measured consistently (for example, tokens under one tokenizer).
Then `M = N * (M/N)`. If the recurrent condition has half as many turns and the
same mean material per turn, total material halves automatically.

A useful descriptive statistic, for positive counts, is:

```text
opportunity-normalized ratio = (M_loop / M_base) / (N_loop / N_base)
                            = (M_loop / N_loop) / (M_base / N_base)
```

At half the turns, a total-material ratio of 0.50 gives a normalized ratio of
1.0; a material ratio of 0.30 gives 0.60. The second is less material per turn,
not only fewer turns. These are illustrative arithmetic examples, not results.
If turns are defined as single reasoning-token emissions, material per turn is
fixed by definition and this particular normalization becomes tautological;
the experiment would need another operational unit or endpoint.

Agent recommendation: prespecify the unit, a practically meaningful reduction
margin beyond the counting null, and an accuracy noninferiority margin. A
confidence interval excluding zero accuracy difference is not the criterion for
noninferiority; assess whether its lower bound excludes the allowed loss.
Failure to detect a difference does not establish equal success rates. Pair tasks,
bootstrap at the task level when sampling multiple trajectories, include failures
and timeouts, and report zero-turn cases separately. Success-only length averages
can select different problem subsets in the two conditions.

Resource efficiency is an **additional** endpoint. For `L` layers and an `m`-layer
block traversed `K` times, the approximate layer-call multiplier is
`[L + (K-1)*m]/L`. Full-stack two-pass execution gives 2; repeating half the layers
twice gives 1.5. This does not define the user's turn-count null. Report measured
prefill/thought/answer FLOPs, actual passes, latent positions, cache memory, and
latency separately. Include initialization and answer passes, not just a loop
counter. A single-passage ratio cannot account for growing attention context,
parallel latent workspaces, adaptive depth, or hardware utilization.

## What would establish reduced dependence rather than shorter output?

Agent hypotheses for a prospective experiment, developed after this literature
review and not tested here:

1. **Behavioral substitution:** at a specified success tolerance, the recurrent
   variant needs less emitted reasoning beyond the emission-opportunity null.
2. **Causal substitution:** removing latent feedback substantially harms the
   recurrent variant, while removing explicit scratchpad access harms it less
   than a matched ordinary model. Shorter visible output alone does not prove
   this. A latent state can also contain information that decodes into a long CoT.
3. **Resource advantage:** the desired accuracy is reached with less total
   inference work or latency. This can hold or fail independently of the first
   two hypotheses.
4. **Feedback-mechanism advantage:** continuous feedback improves the outcome
   relative to a matched discrete selection/re-embedding interface with the same
   workspace and execution schedule. This is stronger than showing that a larger
   effective graph helps.

A discriminating design would vary latent depth/workspace and explicit-CoT
allowance independently, comparing accuracy-material and accuracy-compute curves.
Include the unchanged thinking checkpoint, a matched continued-training control,
non-recurrent trace shortening/internalization, and identical stopping controls.
Assess actual final answers; do not count a model that merely stops early or
fails to finish as having learned to need less scratchpad.

For causal interventions, distinguish changing logged text from changing its
actual model inputs and attention state. If reasoning tokens are edited, their
associated KV state must be recomputed or deliberately manipulated; displaying
altered text while retaining original cached activations does not remove the
scratchpad. Ablations that change positions, context length, or distribution need
controls, and a negative ablation alone does not identify a unique mechanism.

## Implementation observations and reuse

Access checked 2026-11-06 through the connected GitHub tool. No listed command
was executed and no implementation was validated numerically in this task.

| Source | Actually inspected | Practical implication |
| --- | --- | --- |
| [Coconut](https://github.com/facebookresearch/coconut) | README and `coconut.py`, file blob `894f887d4bcb4c3fb290339c56b74b7f9ae6348a` | Code substitutes the preceding hidden vector into latent input positions. README explicitly supports initialization from a CoT checkpoint and documents a training curriculum. |
| [LOTUS](https://github.com/yingfan-bot/lotus) | README and `scripts/lotus.py`, blob `5710e01ec0393e20704c00116316db13d854d3f0` | Released CoT/LOTUS checkpoints and wrapper support a closer paired study. Default injection code adds the original embedding and hidden state, consistent with paper equation 2; an adjacent docstring instead describes averaging. Follow executed code, not that stale comment. Initialization and final forwards need separate accounting. |
| [CODI](https://github.com/zhenyi4/codi) | README | Exposes projection/LayerNorm and latent-count options and links GPT-2/1B checkpoints. Useful continuous-feedback baseline; no runtime audit. |
| [Soft Thinking](https://github.com/UCSB-AI/Soft-Thinking) | README | Uses a modified SGLang 0.4.6.post1 stack. Authors warn of weaker small-model results and device/precision reproducibility problems. These are author reports, not observed failures here; do not assume the 32B findings transfer to local small models. |
| [TaH/TaH2](https://github.com/thu-nics/TaH) | README | October 2026 release includes TaH2 code, data, an inference engine, and `TaH2-1.7B-max2` plus a single-iteration Standard checkpoint. TaH and TaH2 require separate environments. The executor and model cards were not audited here. |

The central next choice is whether to study **same-position depth replacing some
explicit thinking**, **latent-position replacement of a scratchpad**, or both.
LOTUS is a close empirical precedent for the second plus recurrence; TaH2 is a
useful adaptive-depth and compute-accounting precedent for the first, with explicit
CoT retained. Neither makes arbitrary raw feedback into an existing thinking
checkpoint a proven drop-in improvement.

Raw evidence remains at the public papers and repositories; no upstream source
snapshot was retained locally. Exact inspected versions, small measurements, and
code file identities above are the durable evidence. Pin complete repositories,
checkpoints, prompts, data, scorer, precision, and execution semantics before a
prospective run. This note is research preparation, not a frozen protocol or
execution/spending authorization.
