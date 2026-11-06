# Recurrent depth: methods and terminology

Scope: a working taxonomy for the [pretrained-model retrofit review](recurrent-depth-retrofit-literature.md),
2026-11-06. These categories organize the inspected literature; they are not an
exhaustive list of all recurrent neural architectures or universal terminology.
Performance findings and implementation-availability checks live in that review.

## Describe a method along independent axes

“Recurrent depth” alone underspecifies the computation. Record at least:

| Axis | Important alternatives | Why it matters |
| --- | --- | --- |
| Training origin | Native recurrent pretraining; conversion from a pretrained ordinary model | A successful native model does not establish that an existing checkpoint can be improved cheaply. |
| Adaptation | No gradients; learned controller/coefficients over a frozen base; adapters; full-model updates | Frozen base weights and a training-free procedure are different claims. Architecture search also has a cost. |
| Repeated unit | Entire stack; middle block; individual layer; attention/MLP branch | Equal loop counts can represent very different execution graphs. |
| Weight sharing | Exact tying; shared backbone plus loop-specific corrections; physically duplicated untied copies | Identical initial values do not imply sharing after training or equal memory use. |
| State update | Plain composition; input reinjection; gated mixing; damped residual refinement | These change what repeated passes can do, not just how many occur. |
| Scheduling | Fixed depth; input-level route; token-level halting; convergence test | Maximum allowed depth may differ from actual executed depth. |
| Recurrence axis | Same-position depth; across-token state; extra latent sequence positions | These expose different information paths and causal controls. |
| Attention state | Separate KV per virtual depth; intentionally shared KV; selective cache writes | Sharing weights does not automatically make KV states interchangeable. |
| Serving policy | Loop during prefill, decoding, or both | A likelihood-scoring result may exercise no autoregressive decode loop. |

Sources and examples for these distinctions are the
[retrofit review](recurrent-depth-retrofit-literature.md),
[Universal Transformers](https://arxiv.org/abs/1807.03819v3),
[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524), and
[Coconut](https://arxiv.org/abs/2412.06769v4). This table is an agent synthesis, not
a standardized classification taken from one paper.

## Implementation families

Let `P` be an input-side prelude, `G` a reusable middle block, and `Q` an output-side
coda. All formulas below are schematics; they omit causal attention and caches.

| Family | Computation | Representative primary sources |
| --- | --- | --- |
| Whole-stack recurrence | Apply the shared transformer stack `K` times before readout. | [Ouro](https://arxiv.org/abs/2510.25741); native-model details are in the [existing survey](astra-recurrent-depth.md#ouro). |
| Middle-block recurrence | `e=P(x); z[0]=e; z[k+1]=G(z[k]); y=Q(z[K])`. | [RYS runtime](https://github.com/dnhkng/RYS/blob/main/src/core/layer_duplicator.py); [CoLa](https://arxiv.org/abs/2507.07996). |
| Input-reinjected recurrence | `e=P(x); z[k+1]=G(Adapter(e,z[k]))`. A fixed prompt representation enters every iteration. | [Huginn](https://arxiv.org/html/2502.05171v2); [pretrained conversion](https://github.com/mcleish7/retrofitting-recurrence). |
| Layer-local recurrence | Finish repeated applications of one layer before advancing to the next layer. | [UpLoop, section 3.1](https://arxiv.org/html/2610.25004v1). |
| Gated recurrence | Mix a proposed update with the previous state; a gate can depend on state, depth, or both. | [LoopUS, section 3.1](https://arxiv.org/html/2605.11011v1). Update gating and halting are separate mechanisms. |
| Regularized or damped looping | Interpolate hidden states, or use `z <- z + alpha*(G(z)-z)` with small steps and optionally an original-output anchor. | [Inner Loop Inference, section II](https://arxiv.org/html/2602.14759v1); [Training-Free Looped Transformers, section 2](https://arxiv.org/html/2605.23872v1). Different stabilization rules are different operators. |
| Relaxed parameter sharing | `W[k]=W_shared+A[k]B[k]`: loop/depth-specific low-rank corrections supplement shared weights. | [Relaxed Recursive Transformers, section 2.3](https://arxiv.org/html/2410.20672v3). |
| Searched/learned layer programs | Execute a selected list of layer indices with skips/repeats; a route may vary by input or remain fixed per task. | [PoLar](https://arxiv.org/html/2606.06574v2); [MACRO](https://arxiv.org/html/2608.05872v1). These combine recurrence with routing. |
| Equilibrium computation | Solve `z*=G(z*,x)` with a root finder; train through implicit differentiation. | [Deep Equilibrium Models](https://proceedings.neurips.cc/paper/8358-deep-equilibrium-models.pdf). A related implicit-depth family, not proof that an arbitrary pretrained block has a useful stable fixed point. |

Loop order matters. For two layers `A,B`, repeating the block twice executes
`A,B,A,B`; repeating each layer locally twice executes `A,A,B,B`. Both reuse the
same weights and make four calls, but generally produce different states.

Input reinjection, gating, learned halting, and cache sharing can be combined;
they are not mutually exclusive architecture names.

## Adjacent methods to keep separate

| Term/family | Distinction |
| --- | --- |
| Temporal recurrence / cross-token latent feedback | Carries hidden state from previous tokens into the current token, as in [T2MLR](https://arxiv.org/html/2607.15178v1). It can augment a pretrained model without being a same-position depth loop. |
| Recirculation | The [Mozer et al. method](https://arxiv.org/html/2608.17981v1) leaks deep activations toward shallow representations while processing tokens serially. Preserve its scheduling and state semantics rather than treating it as another name for repeating a block. |
| Continuous or latent thought tokens | [Coconut](https://arxiv.org/abs/2412.06769v4) feeds a hidden vector into a later input position. Extra latent positions can consume cache/context even without visible words. |
| Depth up-scaling / layer stacking / Frankenmerging | [SOLAR](https://arxiv.org/abs/2312.15166v3) and [mergekit passthrough](https://github.com/arcee-ai/mergekit/blob/main/docs/merge_methods.md#passthrough-passthrough) reuse pretrained layer values to build a deeper checkpoint. Untied copies add parameters and can diverge during further training. |
| Looped-teacher distillation | [LoOPSD](https://arxiv.org/html/2610.18162v1) uses a recurrent execution to supervise an ordinary student. A better student does not demonstrate recurrence in that student's inference graph. |
| Early exit / layer skipping / mixture of depths | Change how much of a graph executes. They add recurrent depth only when some transformations are actually re-applied. [MoR](https://arxiv.org/abs/2507.10524) explicitly combines sharing and adaptive token-level recurrence. |
| Autoregressive chain of thought / self-consistency | Adds written tokens or multiple candidate answers. Useful compute baselines; not latent layer recurrence by themselves. |
| Sequence-recurrent architectures and memories | RNNs, state-space models, and recurrent attention memories primarily organize state along the sequence axis. “Recurrent” in a name does not establish variable depth per output token. |

## Search glossary

Terms are grouped by usefulness for retrieval, not asserted to be interchangeable.

| Search terms | Working meaning and qualification |
| --- | --- |
| **depth recurrence, recurrent depth, depth-recurrent transformer** | Repeated hidden computation along a depth/iteration axis before readout; a broad family. |
| **looped transformer, looped language model, looped LLM** | Common umbrella terms for reapplying a transformer unit or stack. Check the repeated unit and training origin. |
| **recursive transformer, layer tying, cross-layer parameter sharing, weight-tied depth** | Emphasize parameter reuse. A fixed number of tied layers need not support useful arbitrary test-time depth. |
| **Universal Transformer (UT), adaptive computation time (ACT)** | Foundational shared-depth architecture and a halting mechanism associated with adaptive computation. UT includes position-wise stopping; ACT is a mechanism, not a retrofit package. [Primary source](https://arxiv.org/abs/1807.03819v3). |
| **retrofitting recurrence, recurrent conversion, post-training upcycling** | Introduce recurrence into a pretrained ordinary model; check whether layers are removed and which weights are updated. |
| **inner looping, inference-time looping, test-time depth adaptation, relayering** | Especially productive terms for frozen-checkpoint interventions and execution-path changes. |
| **block-wise, middle-loop, sandwich, prelude/core/coda, encoder/reasoner/decoder** | Descriptions of loop placement. “Encoder” here can mean early causal decoder layers, not a bidirectional encoder. |
| **layer-wise or layer-local looping** | Repeats each selected layer locally; differs from repeating their composition. |
| **input injection, input reinjection, re-entry bridge, latent adapter** | Restores a fixed input representation or maps a later state into a reusable block. These are related motifs, not identical modules. |
| **selective gate, residual gate, damped update, interpolation, anchoring** | Mechanisms controlling state updates and drift. A gate need not decide when to stop. |
| **adaptive depth, adaptive halting, confidence head, exit gate** | Controls actual compute allocation. Report forced-depth curves separately from learned stopping. |
| **deep supervision, intermediate supervision, recurrence curriculum, healing/recovery training** | Training techniques for readable intermediate states, increasing depth, or recovering after model surgery. |
| **depth extrapolation, test-time compute scaling, inference-time scaling** | Running beyond trained depth, or spending additional inference compute more broadly. The latter includes many non-recurrent approaches. |
| **latent reasoning, latent deliberation, continuous thought** | Broader behavioral/method terms; do not uniquely identify a graph or establish that an iteration performs a logical reasoning step. |
| **RYS, CoLa, PoLar, MACRO, LoopUS, UpLoop** | Named methods/projects in the [literature review](recurrent-depth-retrofit-literature.md); search with authors/model names to avoid acronym collisions. |
| **DEQ, deep equilibrium, implicit depth, fixed-point iteration** | Equilibrium/root-solving terminology. Convergence and useful computation are separate questions. |

## Counting depth and resources

For an unmodified `L`-layer stack with an `m`-layer middle block traversed `K`
times, the elementary layer-call count is `L + (K-1)*m`. Equivalently, a
prelude/core/coda model executes `p + K*m + q` layer calls. Here `K=1` includes
the original traversal; some implementations count only extra repetitions.
State the convention explicitly.

Layer calls are not FLOPs across different widths, attention layouts, or MoE
routing. Parameter storage, KV-cache memory, training activations, FLOPs,
sequential latency, search cost, and output-token count are separate measurements.
A shared parameter tensor can be used at many logical depths while requiring
different KV states at each. [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)
explicitly studies recursion-wise caching and a deliberate shared-KV variant.

“More compute improves performance,” “same parameter count improves performance,”
“same FLOPs improves performance,” and “better deployment latency” are different
hypotheses. The companion review records the measured support and limitations;
this glossary establishes no new empirical result.
