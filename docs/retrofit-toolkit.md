# Recurrent-depth retrofit toolkit

Scope: what the toolkit does, how to use it, and what has and has not been checked.
As of 2026-11-16 five of six planned pieces are built: running any order of a
pretrained model's layers with shared weights, the training-free update rules that
stabilize repeated passes, a benchmark runner that puts a design and its parent
through the same items, recirculation, which feeds a deep layer's state back to a
shallow layer across token steps, and loading a released checkpoint that was trained
to loop.

The toolkit is infrastructure. Building and validating it prepares for the recurrence
research and is not part of it, so this note reports no research finding and carries
no hypothesis. Its scope and order follow
[user policy](user-policy.md#recurrent-depth-retrofit-toolkit): the
retrofit-improvement question first, training in scope, across-token feedback
included, and Qwen3 at 1.7B to 4B first.

Background: the [literature review](research/recurrent-depth-retrofit-literature.md)
covers the published methods, and [methods and terms](research/recurrent-depth-methods-and-terms.md)
defines the vocabulary used here.

## Retrofits, methods and designs

A **retrofit** adds recurrent depth to a model that was trained without it. It is the
name for the whole category, so every method in this toolkit is a retrofit. A
**design** is one retrofit written out: the method, where it applies, and its
settings. Designs have a short text form, such as `block[15:19]/euler,anchor=0.5`,
and that text is what the worker's `-Method` option takes.

Most of the methods built so far repeat some of the parent model's layers before a
token is emitted. A design of this kind is a layer program: the parent's layer
indices in the order they run, plus a rule for updating the state between passes.
[Recirculation](#feedback-across-token-steps) repeats layers across token steps
instead, and is described after the layer programs. All of these leave the parent's
weights as they are. The last kind of design, a
[gated loop](#a-checkpoint-trained-to-loop), runs a checkpoint whose weights were
trained for looping.

### Which layers run

| Kind | Order for layers A, B repeated twice | Text | Published method it covers |
| --- | --- | --- | --- |
| Parent | A, B | `parent` | The unmodified model |
| Block loop | A, B, A, B | `block[start:end]` | RYS, middle-block looping |
| Layer loop | A, A, B, B | `layer[start:end]` | Layer-local looping |
| Route | any fixed order, with skips | `route[0,1,1,3]` | Fixed searched routes (MACRO, multi-block RYS) |

Ranges follow Python: `block[9:18]` repeats layers 9 to 17. Papers often write the
same window with both ends included, as `[9-17]`.

**Counting.** `passes` is the number of times the repeated layers run, counting the
parent's own pass. One pass is the parent, exactly, under every rule. A model with
`L` layers and an `m`-layer unit makes `L + (passes - 1) * m` layer calls per token.
The local worker's `depth` setting is `passes`.

### How the state is updated between passes

A plain loop feeds each pass's output straight into the next, which the literature
reports makes models worse. The update rules below replace the state at the end of
each cell. A cell is one traversal of the thing being repeated: the whole block in a
block loop, or one layer in a layer loop. Write `x` for a cell's input on pass `t`,
`y[t]` for its output, and `K` for the number of passes.

| Rule | State after the cell | Text | Source |
| --- | --- | --- | --- |
| Plain | `y[t]` | no suffix | Naive looping |
| Damped | `x + (y[t] - x) / K` | `/euler` | Training-Free Looped Transformers, equation 10 |
| Damped, anchored | the damped result, then after the last pass `b * y[0] + (1 - b) * result` | `/euler,anchor=0.5` | Same paper, equation 16 |
| Uniform | the mean of `y[0]` to `y[t]` | `/uniform` | Inner Loop Inference, equation 9 |
| Moving average | `y[0]` on the first pass, then `eta * y[0] + (1 - eta) * y[t]` | `/ema,eta=0.5` | Inner Loop Inference, equation 10 |

In a block loop, `,per=layer` applies uniform or moving-average mixing after every
layer, across that layer's passes, instead of once per block. An automated reading of
the Inner Loop Inference authors' code says it mixes that way, while the paper's
equations describe mixing once per block, so both are available.

How faithful these are to the papers is limited by how they were read: through
automated extractions, recorded with their caveats in the
[review](research/recurrent-depth-retrofit-literature.md#direct-literature-frozen-base-models).
The arithmetic above is what the toolkit computes and what its checks verify.
Whether it equals each paper's method in every detail has not been confirmed. Two
known differences: the Training-Free paper shares one attention cache across passes,
and its higher-order variants and the other paper's auto-alignment rule are not
built.

### Feedback across token steps

Recirculation carries state from one token step to the next
([Mozer et al.](research/recurrent-depth-retrofit-literature.md#recirculation-details-that-shape-an-implementation)).
A design names a deep source layer, a shallow destination layer and a mixing
coefficient: `recirculate[18to9]/convex,alpha=0.07` feeds layer 18 into layer 9.
Layers are the parent's indices from zero, and a layer's state is its output. For
each token, in order:

1. The token gets the parent's ordinary pass, attending to the cache. Its prediction
   is read from this pass.
2. The source layer's output at that token is rescaled to the norm of the destination
   layer's output there, and mixed into it.
3. The layers above the destination run again at that token from the mixed state.
   Their cache entries replace the ones from step 1, so later tokens attend to them.

Write `low` and `high` for the destination's and source's outputs at the token, and
`a` for `alpha`.

| Mix | State written at the destination | Text | Source |
| --- | --- | --- | --- |
| Convex | `(1 - a) * low + a * high * \|low\| / \|high\|` | `/convex,alpha=0.07` | Recirculation, equations 1 and 2 |
| Additive | `low + a * high * \|low\| / \|high\|` | `/additive,alpha=0.07` | The same paper's non-convex mixture, with `beta = 1` |

**Counting.** A recirculation has 2 passes and makes `L + (L - 1 - destination)` layer
calls per token. With 1 pass it skips steps 2 and 3, which leaves the parent processed
one token at a time. That is the baseline to compare a recirculation with, because its
rounding matches.

**A prompt is processed one token at a time.** Each token's cache entries depend on
the finished pass of the token before it, so a sequence of `T` tokens takes `T` steps
where the parent takes one.

**How close this is to the paper.** The paper was read through automated extractions
only. Its equations and its statements that the stack runs twice per step, that each
stack is recirculated once, and that the read-out follows the first pass are what
steps 1 to 3 implement. Three things the extracted text does not state are choices
made here: the second pass's cache entries replace the first's, the second pass runs
to the top of the stack instead of stopping at the source, and layer indices count
from zero. The paper's experiments are on Gemma 3, and no code for it was found. Not
built: ramping `alpha` over the first tokens, the paper's six other ways to normalize
the mixture, more than one recirculation per token, and its learned coefficients,
which need training.

### A checkpoint trained to loop

The designs above are applied to a parent as it is. LoopUS
([Park et al.](research/recurrent-depth-retrofit-literature.md#loopus-details-that-shape-an-implementation))
trains the parent to loop instead, and its authors released one such checkpoint,
trained from Qwen3-1.7B. The toolkit loads it and runs it as a **gated loop**: a block
loop in which a trained gate follows every pass.

In this checkpoint layers 0 and 1 run once, layers 2 to 26 are the looped unit, and
layer 27 runs once, so the design is written `gated[2:27]`. Write `x` for the state
entering a pass and `y` for the unit's output. After every pass, the first included:

```text
keep  = exp(-softplus(W2 (W1 (y - x)) + b) * exp(a))
state = keep * y + (1 - keep) * x
```

`W1`, `W2`, `b` and `a` are tensors in the checkpoint. `keep` lies between 0 and 1 at
every position and channel. Each pass keeps its own attention cache, as in a block
loop.

| Text | What runs |
| --- | --- |
| `gated[2:27]` | Exactly `passes` passes |
| `gated[2:27]/halt,threshold=0.6` | At most `passes` passes; a sequence stops once the checkpoint's confidence head exceeds the threshold |

**Counting.** `passes` is the number of times the unit runs, and a fixed loop makes
`L + (passes - 1) * m` layer calls, as a block loop does: 203 at 8 passes. One pass is
not an unmodified parent. It is one traversal of the unit followed by the gate. The
checkpoint was trained with 20 passes, and its paper's standard inference allows at
most 8 with a threshold of 0.6.

**Stopping.** The confidence head reads `y - x` at the last token of each sequence. A
sequence stops after the first pass on which the head's probability exceeds the
threshold. It keeps its state while the other rows of a batch continue, and passes
that no row needs are not run. Because the last token decides how many passes the
whole sequence gets, a later call cannot extend an earlier one. With a threshold every
call must hold the whole sequence and nothing is cached, so generating text recomputes
the prefix at each step, and the worker does not serve it.

**One pass is unusable; use two or more.** Mean next-token loss on 200 science
passages (18,778 scored tokens) in FP32, by number of passes:

| Passes | 1 | 2 | 3 | 4 | 8 | 20 |
| --- | --- | --- | --- | --- | --- | --- |
| Loss | 11.07 | 1.996 | 1.994 | 1.994 | 1.994 | 1.996 |

A trace of each pass, computed by hand on the same passages, shows what happens:

- **First pass.** The gate is open, with `keep` above 0.99, at 96% of positions and
  channels, and holds the other 4% near the values that entered the loop. Decoding
  that mixture gives the loss of 11.07. The unit's output decoded without the gate
  gives 3.59.
- **Second pass.** The gate is open at 4% of positions and channels, about the share
  held back on the first pass, and the loss falls to 1.996. Whether they are the same
  ones was not checked.
- **Later passes.** The gate is open at under 3% and the loss changes in the third
  decimal.
- **The confidence head mostly reflects this.** On the first pass it is at most 0.52
  on every passage, so a threshold of 0.6 never stops there on this text. From the
  second pass its median is 0.50 and its highest value 0.62 to 0.64. On OpenBookQA,
  though, 13 of 1,575 scored sequences did
  [stop after the first pass](#reproducing-a-published-number) at 0.6.
- **The weights fit the arithmetic in the code.** With the gate's two states
  exchanged, as the gate class's own description has them, the loss stays above 10 at
  every pass count from 1 to 8.

The authors' training loop supervises the first pass as often as any other, so why
this checkpoint needs two is not known. Their own results at one pass were not found.

**How close this is to the release.** The forward pass was rewritten from the authors'
model class, which was read in full at one pinned commit and not run. The toolkit's
output has therefore not been compared with the authors' own. What supports it is
indirect: the published numbers it was [checked against](#reproducing-a-published-number),
the checks [below](#what-was-checked), and the measurements above. Known differences
and choices:

- **Stopping with a cache is not built.** The released code does stop while
  generating with a cache, token by token, but it then discards the deeper passes'
  caches whenever a token stops early. That does not compute what an uncached pass
  computes, and it is not reproduced.
- **Only the threshold rule is built,** read on every pass. The release has two other
  stopping rules.
- **The authors score benchmarks through their own harness adapter.** It pads on the
  left, passes a mask, and includes a continuation's last token in the sequence that
  decides stopping. The toolkit's benchmark runner uses the harness's stock adapter,
  which leaves that token out. At a fixed number of passes the two score the same
  sequences. With a threshold the stopping decision is read one token earlier here.
- **The checkpoint's configuration was written for a later library release.** The
  loader picks out the fields the pinned release reads, including the rotary base,
  and builds the tokenizer from the checkpoint's own tokenizer file.
- **The parent is `Qwen/Qwen3-1.7B`,** as the checkpoint's configuration says. Its
  model card names the Base model, but the embedding, the final norm and layers 0 and
  1 are bit-identical to `Qwen/Qwen3-1.7B`, and the tokenizer has that model's
  vocabulary and chat template.

## How it works

- **The model's own forward pass is kept.** The runner replaces the model's layer
  list with one position per program step. Each position runs a parent layer's
  weights and, for the length of that call, points the layer's attention at the
  position's own cache slot. No weight is copied or changed.
- **Every pass has its own attention cache.** The model's layer count is set to the
  program's length, so the stock cache allocates one slot per position. Loops run
  during the prompt and during decoding, and cached decoding computes the same thing
  as recomputing the whole prefix.
- **Update rules mix states at the same sequence position only.** That is why
  cached decoding stays equal to recomputation under every rule. The states a rule
  keeps belong to one forward pass and are dropped at its end.
- **Executed passes are counted independently.** A hook on each real parent layer
  counts the calls that reach its weights. The worker checks those counts, and the
  cache slots, after every forward pass and returns no response if they differ from
  the program.
- **Removing a program restores the parent exactly.**

A recirculation is installed differently, because it changes when layers run and not
which ones:

- **The decoder's forward pass is driven one token at a time.** Whether the caller
  hands over one token or a whole prompt, the driver calls the parent's own forward
  pass once per token with the stock cache, and joins the results.
- **The second pass replays the first pass's own arguments.** Hooks keep the two
  layers' outputs and the arguments the layer above the destination was given. The
  driver drops that token's cache entries above the destination and calls those
  layers again from the mixed state.
- **Nothing is carried between calls except the cache.** Splitting a prompt between
  calls gives bit-identical hidden states.
- **Layer calls are counted by the same hooks,** per token instead of per forward
  pass. `install(model, design)` installs any kind of design.

A gated loop is installed as a block loop is, with the trained gate where an update
rule would be:

- **The checkpoint is rebuilt as the library's own model class.** The released file
  stores the layers under three block names. The loader puts each back at its parent
  index, keeps the output head separate from the embedding as the file has it, and
  attaches the gate and the confidence head. It stops if the file holds a tensor it
  does not use or lacks one it needs. The authors' model class is not used.
- **The gate acts at the end of every pass, position by position.** At a fixed number
  of passes, cached decoding therefore computes what recomputing the whole prefix
  does.
- **With a threshold the model runs without a cache.** A wrapper around the decoder's
  forward pass refuses a partial sequence, hands the attention mask to the stopping
  rule and returns no cache. Passes that no row of the batch needs do not call their
  layers.
- **Passes are counted twice.** The hooks on the real layers count calls, and the gate
  keeps its own tally of forward passes and gated passes. Every check compares the
  two.

Supported family: Qwen3 through Transformers 4.57.1, loaded with the library's own
model class and no repository code. A family can be added when its forward pass
iterates a `layers` list, reads nothing from a layer except `attention_type`, and its
attention picks the cache slot from `layer_idx`. Llama-family models are arranged
this way in the same library version but have not been tried.

Code: [`src/recurrent_retrofit`](../src/recurrent_retrofit). `program.py` is pure
Python and holds designs and their text form. `runner.py` installs layer programs,
`recirculation.py` installs recirculations and `gated.py` installs gated loops.
`reference.py` holds the four independent references used by the checks.
`loading.py` loads pinned checkpoints, with `loopus.py` rebuilding a LoopUS one, and
`backend.py` and `worker.py` serve them.

## Use

In Python, inside the WSL `retrofit` environment:

```python
from pathlib import Path
from recurrent_retrofit.loading import load_parent
from recurrent_retrofit.program import Update, block_loop
from recurrent_retrofit.runner import Retrofit

model, tokenizer, info = load_parent("qwen3-4b-instruct", Path.home() / "llm/recurrent")
design = block_loop(info["parent_layers"], 15, 19, passes=3, update=Update("euler", anchor=0.5))
with Retrofit(model, design):
    ...  # model(...) and model.generate(...) now run the design
```

`Retrofit` installs layer programs. For any design, including a recirculation, build
it from its text and use `install`:

```python
from recurrent_retrofit.program import from_spec
from recurrent_retrofit.runner import install

design = from_spec("recirculate[18to9]/convex,alpha=0.07", info["parent_layers"], passes=2)
with install(model, design):
    ...
```

The LoopUS checkpoint loads through the same call and runs only with its own design:

```python
model, tokenizer, info = load_parent("qwen3-1.7b-loopus", Path.home() / "llm/recurrent")
design = from_spec("gated[2:27]", info["parent_layers"], passes=8)
with install(model, design):
    ...  # eight passes of layers 2 to 26, each followed by the gate
```

With a threshold, `from_spec("gated[2:27]/halt,threshold=0.6", 28, passes=8)`, pass
whole sequences and `use_cache=False`.

Through the local worker, from PowerShell at the repository root:

```powershell
.\tools\local_models\serve.ps1 -Profile qwen3-1.7b -Depth 3 -Method 'block[12:16]/euler,anchor=0.5'
# In a second terminal:
.\.venv\Scripts\python.exe -B tools/local_models/probe.py --profile qwen3-1.7b --depth 3 --method "block[12:16]/euler,anchor=0.5" --prompt-format native-chat-no-thinking-v1 --max-output-tokens 32
.\tools\local_models\stop.ps1
```

Leave out `-Method` to serve the plain parent at depth 1. A retrofit worker's model
identity is the parent's `repo_id@revision` followed by `+` and the design, for
example `Qwen/Qwen3-1.7B@70d244cc...+block[12:16]/euler,anchor=0.5`, so a request
for one design cannot be answered by another. A design has exactly one accepted
spelling. Depth is fixed at startup, as for the other local models, and the worker
shares their GPU lock. A recirculation is served at depth 2, for example
`-Depth 2 -Method 'recirculate[14to7]/convex,alpha=0.07'`, and at depth 1 the same
method serves the parent processed one token at a time. The LoopUS checkpoint is
served at a fixed number of passes from 1 to 20, for example
`-Profile qwen3-1.7b-loopus -Depth 8 -Method 'gated[2:27]'`. It is not served
without that design or with a stopping threshold, and its requests are limited to
1,024 tokens, the context length it was trained with.

Parents are the `qwen3-1.7b`, `qwen3-4b-instruct` and `qwen3-1.7b-base` entries in
[`profiles.json`](../src/eval_core/local/profiles.json), pinned to full revisions.
`qwen3-1.7b-loopus` is the LoopUS checkpoint, which was trained from `qwen3-1.7b`.
To install the environment and download a parent:

```powershell
wsl -d Ubuntu-LLM -- python3 /mnt/g/Evals/tools/local_models/bootstrap.py --profile qwen3-1.7b
```

The `retrofit` environment's [lock](../local_models/locks/retrofit.txt) holds 95
distributions: PyTorch 2.9.1 with CUDA 12.8, Transformers 4.57.1, and the benchmark
harness, lm-eval 0.4.11 with `datasets` 4.8.5. Installing the harness moved one
earlier pin, `fsspec` from 2026.9.0 to 2026.2.0, because no `datasets` release
accepts both the newer `fsspec` and the `huggingface-hub` release that Transformers
4.57.1 needs.

### Benchmarks

[`retrofit_bench.py`](../tools/local_models/retrofit_bench.py) runs lm-eval tasks on
a parent, or on a design installed on it, and keeps every item's result so that two
runs can be paired item by item. Inside the WSL `retrofit` environment:

```bash
cd /mnt/g/Evals
PY=~/llm/recurrent/envs/retrofit/bin/python
OUT=artifacts/runs/local-runtime/my-run
$PY -B tools/local_models/retrofit_bench.py fetch
$PY -B tools/local_models/retrofit_bench.py run --profile qwen3-4b-instruct \
    --tasks commonsense_qa --fewshot 7 --output $OUT/parent.json
$PY -B tools/local_models/retrofit_bench.py run --profile qwen3-4b-instruct \
    --method 'block[15:19]/euler' --passes 3 \
    --tasks commonsense_qa --fewshot 7 --output $OUT/design.json
$PY -B tools/local_models/retrofit_bench.py compare $OUT/parent.json $OUT/design.json
```

- **`fetch`** is the only step that uses the network. It downloads each task's
  dataset at the commit pinned in
  [`benchmarks.json`](../src/recurrent_retrofit/benchmarks.json). Runs read the
  cache offline and stop if a dataset is cached at any other commit. Twelve tasks are
  pinned so far. Nine are scored by option likelihood: `commonsense_qa`, `sciq`,
  `openbookqa`, `arc_easy`, `arc_challenge`, `piqa`, `winogrande`, `hellaswag` and
  `mmlu`, which is a suite of 57 subjects. Two are scored as perplexity:
  `lambada_openai` on each passage's last word, and `wikitext` on whole pages.
  `mmlu_pro` is scored from generated text. Another lm-eval task needs a pin and a
  fetch first.
- **`run`** loads the pinned parent, installs the design if one is given, and hands
  the model to lm-eval unchanged. The defaults are the harness's own: batch size
  one, its default seeds, no chat template, and the context length in the model's
  configuration. `--seed` sets all four harness seeds,
  which changes the few-shot examples each prompt gets. `--max-length` sets the
  context length: longer inputs are cut from the left, and it is the window of a
  perplexity task. For a design, the hooks on
  the real parent layers count every call during the run, and the run fails unless
  every layer's total matches the program. The parent runs without the runner
  installed. A recirculation takes `--passes 2`, and `--passes 1` with the same
  method gives its token-by-token baseline. Its record counts token steps where a
  layer program's counts forward passes.
- **Perplexity over whole pages** is where the one departure from the stock harness
  is. The harness cuts a page into disjoint windows, so the first tokens of every
  window are scored with almost no context. `--rolling-context 513` with
  `--max-length 1024` passes another context length to the harness's own window
  function, so that every token after a page's first window is scored with at least
  513 tokens before it. Nothing else in the harness is changed, and without the
  option the windows are the stock ones.
- **`compare`** refuses two runs that differ in settings, dataset commits, harness
  version or prompts. It then counts items that both runs got right, that only one
  did, and that neither did. A perplexity has no per-item outcome to count, so for it
  the two task scores are set side by side. A suite of tasks also gets its counts
  summed. Two different checkpoints are compared only with `--across-checkpoints`,
  which is how a trained loop is set beside the parent it was trained from.
- **`summarize`** writes the compact, tracked summary of a directory of records. It
  pairs each design with the plain runs of its own checkpoint, and a trained loop
  with every plain run under the same settings, each named.
- **A checkpoint trained to loop** always takes its gated design, for example
  `--profile qwen3-1.7b-loopus --method 'gated[2:27]' --passes 8`. With a stopping
  threshold the run needs batch size 1, because the harness pads rows without a mask
  and a padded row would be stopped by its padding, and its record counts how many
  passes each sequence used. Tasks scored from generated text run at a fixed number
  of passes only, since the harness generates with a cache.

The tool measures and counts. It has no rule for calling a difference an
improvement; that belongs to the research that uses it.

## What was checked

The layer-program checks were run on 2026-11-15, and the recirculation and
gated-loop checks on 2026-11-16, on the RTX 5070 Ti.

**Offline tests.** [`tests/retrofit`](../tests/retrofit) holds 142 tests. Forty-five
are pure Python and run in CI. Ninety-seven run a tiny randomly initialised Qwen3
model on the CPU in FP32 and need the WSL environment:

```powershell
wsl -d Ubuntu-LLM -- bash -c "cd /mnt/g/Evals && PYTHONPATH=src ~/llm/recurrent/envs/retrofit/bin/python -B -m unittest discover -s tests/retrofit -t . -v"
```

Each update rule is compared with its arithmetic written out by hand in the test.
Seven deliberately broken versions were tried against the tests, and each one failed
at least one: every pass left on its layer's original cache slot, layers run in a
different order, a damping step of `1/(K+1)`, the anchor ignored, uniform mixing
returning the raw output, a reference with swapped weights, and per-layer cells
covering one layer only.

Four of the tests cover how the benchmark harness calls the model, on seven designs.
A right-padded batch without an attention mask, which is how the harness scores
likelihoods, gives every row the logits that row gets alone. The library's own
`generate`, which the harness uses for written answers, gives the tokens and scores
of the toolkit's reference decoder, and a left-padded batch generates for every row
what that row generates alone. Every forward pass in these makes the program's layer
calls. Two more broken versions were tried against them and both were caught: mixed
states leaking between the rows of a batch, and rows after the first skipping the
update rule.

Fourteen of the tests cover recirculation, on four designs:

- the mixture equals numbers worked by hand;
- one pass gives hidden states bit-identical to the parent stepped one token at a
  time, and two passes with `alpha` 0 are bit-identical to one pass;
- every design agrees, within 0.0001 logits at every position, with a reference that
  calls the parent's layers by hand, with no cache and none of the driver's code;
- the first token's logits are the parent's, and every later token's differ;
- splitting a prompt between calls changes nothing, and each token makes the design's
  layer calls and leaves one cache entry per layer;
- right-padded and left-padded batches give each row what it gets alone, and the
  library's `generate` matches the reference decoder.

Eleven broken versions of the driver were tried, and each failed at least one test.
They were: the second pass's cache entries added instead of replacing the first's,
the destination layer re-run, the second pass stopped at the source layer, the second
pass started from the unmixed state, its attention mask dropped, the source state
taken from the previous token, the prediction read from the second pass, the norm
ratio inverted, the source not rescaled, the two mixes swapped, and every row of a
batch given the first row's norm ratio.

Thirty of the tests cover the gated loop and the LoopUS loader, on the tiny model
with a random gate and on a synthetic checkpoint written in the released layout:

- **The gate's direction is tied to programs checked earlier.** The gate equals
  numbers worked by hand. With its rate forced to zero the loop is bit-identical to a
  plain block loop at 1, 2 and 3 passes. With its rate forced high it is
  bit-identical to a route that leaves the unit out.
- **Every design agrees with a by-hand reference,** within 0.0001 logits at every
  position. The reference calls the layers itself and writes the gate's and the
  head's arithmetic out. Seven combinations of unit and passes are covered, including
  units that start at the first layer or end at the last.
- **At a fixed number of passes the loop behaves as a layer program does.** Every
  forward pass makes the design's calls and fills each pass's own cache slots, cached
  decoding matches recomputation, and padded batches and the library's `generate`
  give each row what it gets alone.
- **Stopping.** A threshold of 1 is bit-identical to the fixed loop. A threshold of 0
  is bit-identical to one pass, and the passes it skips call no layers. The rows of a
  padded batch stop after different numbers of passes, each as it does alone and as
  the by-hand reference says. Padding tokens do not affect a masked row. Confidence
  is read from the change before the gate. A partial sequence or a cache is refused,
  and the library's uncached `generate` matches recomputing every step.
- **The loader.** Every tensor of the synthetic checkpoint lands where the parent has
  it, the later library's settings are translated, and the gate takes the model's
  precision. A file that is incomplete, holds a tensor beyond a block, or disagrees
  with its configuration is refused. The rebuilt model computes what its parts
  compute by hand.

Twenty-two broken versions were tried and each failed at least one test. One of them,
reading confidence from the change after the gate, passed every test at first, so a
test was added for it. The others:

- **in the gate:** the two states exchanged, the step not passed through softplus,
  the exponent's sign reversed, the new state read in place of the change, and the
  rate not exponentiated;
- **in the loop:** the first pass not gated, and every pass gated against the state
  before the loop;
- **in stopping:** a stopped sequence that keeps changing, a sequence losing the pass
  on which it stops, the attention mask ignored or never passed on, confidence read
  at the first valid token, unneeded passes still running their layers, and the
  caller's cache extended and returned;
- **in the loader:** a block's layers loaded in reverse order (tried on two blocks),
  the output head and the embedding exchanged, two pairs of same-shaped gate and head
  tensors exchanged, the rotary base left at the library's default, and the gate left
  at full precision.

**Acceptance on the real parents.**
[`retrofit_check.py`](../tools/local_models/retrofit_check.py) runs plain loops over
the middle third of each parent, as a block and layer by layer, at 2, 3 and 4 passes.
It runs seven update designs over the middle four layers at 2 and 3 passes. All use
greedy decoding on three short prompts. Those placements are mechanical choices for
the checks, not tuned ones. It requires:

1. one pass of every design, and the `parent` program, give logits identical to the
   parent's;
2. every forward pass, in prefill and decoding, makes the program's layer calls and
   fills one cache slot per position;
3. a plain loop's logits are identical to those of a physically deeper stock model,
   with and without a cache. That model gives every position its own ordinary layer
   module holding the parent layer's tensors, which is the structure of an exported
   deeper checkpoint;
4. cached decoding agrees with recomputing the whole prefix, within 0.01 logits in
   FP32;
5. removing the program gives the parent's logits back exactly;
6. every design agrees, within 0.01 logits in FP32, with an explicit recomputation
   that calls the parent's layers by hand and uses none of the runner's code.

All four runs passed. Each covered 10 one-pass programs, 6 plain loops and 14
update-rule conditions. The gap columns give the largest absolute logit difference
found in the run.

| Parent | Precision, device | Checks 1, 2, 3, 5 | Cache gap: parent / plain block / plain layer / update rules | Explicit gap | Checks 4 and 6 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B, 28 layers | FP32, GPU | pass | 0.00001 / 0.00001 / 0.00015 / 0.00003 | 0.00001 | pass |
| Qwen3-1.7B | BF16, GPU | pass | 0.19 / 0.31 / 2.11 / 0.42 | 0.25 | recorded, not judged |
| Qwen3-4B-Instruct, 36 layers | FP32, CPU | pass | 0.00003 / 0.00003 / 0.00013 / 0.00004 | 0.00001 | pass |
| Qwen3-4B-Instruct | BF16, GPU | pass | 0.25 / 0.22 / 1.94 / 0.31 | 0.22 | recorded, not judged |

The FP32 run on the 4B parent used one prompt and 8 tokens, because it runs on the
CPU.

Memory barely moves on these short prompts, because each extra pass adds cache for
the repeated layers only: peak allocation went from 3.25 to 3.28 GiB on the 1.7B
parent and from 7.54 to 7.58 GiB on the 4B. Decoding at batch one slowed roughly in
proportion to layer calls. The absolute speeds varied up to twofold between runs,
depending on other load on the machine (the 1.7B parent decoded at 25 to 50 tokens
per second), so the recorded speeds are rough.

The compact record, with every design's numbers, the generated text and the hash of
each full record, is [`retrofit-check-20261115.json`](../local_models/retrofit-check-20261115.json),
built by [`summarize_retrofit.py`](../tools/local_models/summarize_retrofit.py). The
full records are local only, under
`artifacts/runs/local-runtime/retrofit-check-20261115b/`. An earlier run the same
day, before the update rules existed, covered the plain loops only and gave the same
results for them; its records are in `retrofit-check-20261115/`.

To repeat it for one parent:

```powershell
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/retrofit_check.py --profile qwen3-1.7b
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/retrofit_check.py --profile qwen3-1.7b --precision float32
```

For `qwen3-4b-instruct`, FP32 weights do not fit on the 16 GiB card, so the FP32 run
takes `--placement cpu --tokens 8 --prompts 1`.

**Recirculation on the real parents.**
[`recirculation_check.py`](../tools/local_models/recirculation_check.py) runs four
designs with greedy decoding on the same three prompts. Three put the source at half
depth and the destination at a quarter: a convex mixture at `alpha` 0.07 and 0.15,
and an additive one at 0.07. The fourth feeds the last layer back to the first. Those
placements are mechanical choices for the checks, not tuned ones. It requires:

1. one pass is identical to the parent decoded one token at a time, prompt included;
2. two passes with `alpha` 0 are identical to one pass;
3. every forward pass makes the design's layer calls for each of its tokens, and
   leaves one cache entry per token in every layer;
4. every design agrees, within 0.01 logits in FP32, with an explicit recomputation
   that calls the parent's layers by hand, with no cache and none of the driver's
   code;
5. removing the design restores the parent exactly.

All four runs passed. The gap columns give the largest absolute logit difference
found in the run.

| Parent | Precision, device | Checks 1, 2, 3, 5 | Parent: token by token against parallel prefill | Explicit gap, largest of four designs | Check 4 |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B, 28 layers | FP32, GPU | pass | 0.00002 | 0.00002 | pass |
| Qwen3-1.7B | BF16, GPU | pass | 0.28 | 0.34 | recorded, not judged |
| Qwen3-4B-Instruct, 36 layers | FP32, CPU | pass | 0.00003 | 0.00004 | pass |
| Qwen3-4B-Instruct | BF16, GPU | pass | 0.25 | 0.31 | recorded, not judged |

- **The explicit recomputation is the independent control.** The script also compares
  cached decoding with recomputing the whole prefix in one call. Under a
  recirculation those are bit-identical, because the driver takes the same token
  steps either way, so that comparison shows nothing new.
- **In BF16 the gaps are the size of the parents' own rounding.** They are close to
  the gap between decoding the parent token by token and with parallel prefill. The
  top token agreed
  with the explicit recomputation at every compared step, except one of six for one
  design on the 1.7B parent.
- **Decoding slowed roughly in proportion to layer calls.** The 1.7B parent went from
  49 to 30 tokens per second at 48 calls per token instead of 28, and the 4B from 37
  to 24 at 62 instead of 36. One pass decoded at the parent's speed. Peak allocation
  rose by 0.13 GiB or less.
- The FP32 run on the 4B parent used one prompt and 8 tokens, on the CPU.

The compact record, with every design's numbers and generated text and the hash of
each full record, is
[`recirculation-check-20261116.json`](../local_models/recirculation-check-20261116.json).
The full records are local only, under
`artifacts/runs/local-runtime/recirculation-check-20261116/`. To repeat it for one
parent and rebuild the summary:

```powershell
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/recirculation_check.py run --profile qwen3-1.7b --precision float32
.\.venv\Scripts\python.exe -B tools/local_models/recirculation_check.py summarize artifacts/runs/local-runtime/<run directory> --output local_models/<name>.json
```

**The LoopUS checkpoint.** [`gated_check.py`](../tools/local_models/gated_check.py)
loads `Thrillcrazyer/Qwen3_1.7B_LoopUS` and runs `gated[2:27]` at 1, 2, 3, 4, 8 and 20
passes with greedy decoding on three short prompts. It then runs stopping thresholds
of 1, 0, 0.6 and 0.9 with at most 8 passes, on the prompts and on 200 science
passages. It requires:

1. the loader uses every tensor in the file, 319 of them, and fills every parameter;
2. every forward pass makes the design's layer calls and leaves one cache entry per
   token in every pass's own slots;
3. cached decoding agrees with recomputing the whole prefix, within 0.01 logits in
   FP32;
4. the loop agrees, within 0.01 logits in FP32, with an explicit recomputation that
   calls the layers by hand and writes the gate's and the head's arithmetic out. That
   holds at fixed passes and with stopping, where the number of passes used must
   agree as well;
5. a threshold of 1 is identical to the fixed loop, and a threshold of 0 to one pass;
6. removing the design restores the model exactly.

Both runs passed. The gap columns give the largest absolute logit difference found.

| Precision, device | Checks 1, 2, 5, 6 | Cache gap | Explicit gap: fixed passes / stopping | Checks 3 and 4 |
| --- | --- | --- | --- | --- |
| FP32, GPU | pass | 0.00003 | 0.00004 / 0.00004 | pass |
| BF16, GPU | pass | 0.50 | 0.41 / 0.53 | recorded, not judged |

- **In BF16 the top token agreed with both controls at every compared step.**
- **Stopping early was recomputed by hand on real text.** All three prompts ran the
  full 8 passes at 0.6 and 0.9, so the passages are what exercise it. At 0.6, five of
  the 200 passages stopped early in FP32, four after 2 passes and one after 3, and
  four did in BF16. Each was recomputed by hand and stopped after the same number of
  passes, with FP32 logits within 0.00003. At 0.9 no passage stopped early.
- **Decoding slowed in proportion to layer calls.** In BF16 it went from 48 tokens per
  second at one pass, which is 28 layer calls, to 28 at 2 passes, 15 at 4, 8 at 8 and
  3.2 at 20, which is 503 calls. Each pass keeps its own cache for the 25 looped
  layers, and on these short prompts the peak went from 3.83 GiB at one pass to 3.92
  at 8 and 4.07 at 20.

The same script measures three things without judging them.

- **Which parent.** The checkpoint's weights were compared tensor by tensor with both
  candidates. Against `Qwen/Qwen3-1.7B` the embedding, the final norm and all 22
  tensors of layers 0 and 1 are bit-identical, and the whole decoder is 0.019 away in
  relative distance. Against `Qwen/Qwen3-1.7B-Base` no layer tensor of the first two
  layers is identical and the distance is 0.060. The tokenizer has the first model's
  vocabulary, chat template and end token.
- **Loss by number of passes,** and the pass-by-pass trace, which are reported
  [above](#a-checkpoint-trained-to-loop). At 1, 2, 3, 4 and 8 passes the trace's loss
  agrees with the installed loop's to the four decimals recorded, over all 200
  passages.
- **Loss beside the parents,** on the same 200 passages in FP32:

  | Model | Mean loss | Perplexity |
  | --- | --- | --- |
  | `Qwen/Qwen3-1.7B`, the parent | 2.351 | 10.49 |
  | LoopUS, 8 passes | 1.994 | 7.34 |
  | `Qwen/Qwen3-1.7B-Base` | 1.921 | 6.83 |

  The paper reports perplexity falling from 21 to 16.9 on WikiText and from 12.21 to
  7.43 on LAMBADA, and the fall here is of that size. It is not evidence that looping
  helps. The checkpoint was trained from a chat-tuned model on plain educational
  text, the passages are plain science text, and the Base model, which was never
  trained to loop, scores lower still. The loss also barely moves between 2 and 20
  passes. Further training on such text could produce all of this without the loop,
  and nothing here separates the two.

The compact record, with every number above, the generated text and the hash of each
full record, is [`gated-check-20261116.json`](../local_models/gated-check-20261116.json).
The full records are local only, under
`artifacts/runs/local-runtime/gated-check-20261116/`. To repeat it and rebuild the
summary:

```powershell
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/gated_check.py parents --profile qwen3-1.7b-loopus --candidates qwen3-1.7b qwen3-1.7b-base
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/gated_check.py run --profile qwen3-1.7b-loopus --precision float32
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/gated_check.py trace --profile qwen3-1.7b-loopus
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/gated_check.py loss --profile qwen3-1.7b --precision float32
.\.venv\Scripts\python.exe -B tools/local_models/gated_check.py summarize artifacts/runs/local-runtime/<run directory> --output local_models/<name>.json
```

Give every command the same `--output` directory, so that `summarize` finds all the
records in one place. Without it each writes to a new directory under
`artifacts/runs/local-runtime/`.

**Through the worker.** A worker serving Qwen3-1.7B with
`block[12:16]/euler,anchor=0.5` at depth 3 answered the integration probe from
Windows with `4`, at zero charge in the shared ledger. Probes naming the same loop
without the rule, or with a different anchor, were refused for a model mismatch.
Before the update rules were added, the same was exercised with a plain `block[9:18]`
loop at depth 3, which made 46 layer calls and three observed passes in every forward
pass, and with the plain parent at depth 1. The launcher still sends the older
profiles to their own worker, and both workers reject an unsupported depth, or more
than one pass without a design, before loading a model. The 4B parent has not been
served through the worker.

A worker serving Qwen3-1.7B with `recirculate[14to7]/convex,alpha=0.07` at depth 2
answered the same probe with `4`, with two passes observed in each forward pass and
zero charge. Probes naming another coefficient, the additive mix, or depth 1 were
refused.

A worker serving the LoopUS checkpoint with `gated[2:27]` at depth 8 answered the
probe with `4` under the chat format, with eight passes observed in each forward pass
and zero charge. Probes naming depth 4, a stopping threshold, or no design were
refused.

**Recirculation through the benchmark runner.** OpenBookQA, zero-shot, 500 items, at
batch size 16 on Qwen3-4B-Instruct in BF16, with the mechanical placement
`recirculate[18to9]`:

- **Two passes with `alpha` 0 gave all 500 items the log-likelihoods of one pass,
  exactly.**
- **Every run made its design's layer calls on each of 1,462 token steps:** 36 at one
  pass and 62 at two.
- **One pass differs from the ordinary parent by rounding.** It changed 6 items on
  plain accuracy and 5 on length-normalized accuracy, for 159 and 202 correct against
  the parent's 159 and 203, with log-likelihoods up to 0.75 apart. Token-by-token and
  parallel processing round differently in BF16. That is why a recirculation is
  compared with its one-pass baseline and not with the ordinary parent.
- **Against that baseline the designs moved a few items.** At `alpha` 0.07 the convex
  design gained 5 items and lost 3 on plain accuracy, and gained 2 and lost 2 on
  length-normalized. The additive design gained 6 and lost 1, and gained 2 and
  lost 1. Batch size alone moves 5 items on this benchmark and the placement is
  untuned, so this says nothing about whether recirculation helps a model.
- The runs took 42 seconds at one pass and 61 at two, against 11 for the ordinary
  parent.

These four runs are in the benchmark record linked under
[Reproducing a published number](#reproducing-a-published-number), which pairs each
recirculation with its one-pass baseline.

## Reproducing a published number

Scope: a check of the benchmark runner and the damped rule against the Qwen3-4B-Instruct
table of the Training-Free Looped Transformers paper, run on 2026-11-15 and 16. The paper
chose its settings on these same benchmark items. Matching one of its gains therefore
shows that the toolkit computes what the paper computed. It is not a held-out
estimate of whether a design helps a model.

**Settings.** The paper's, as far as an automated reading gives them (see the
[review](research/recurrent-depth-retrofit-literature.md#training-free-looped-transformers--chen-et-al-may-2026)):
lm-eval 0.4.11, BF16 weights, SDPA attention and the harness's default batch size of
one. The paper states no seeds and no chat template, so the harness defaults were
used for both. Its "block, Euler, K = 3" on layers 15 to 18 is
`block[15:19]/euler` at 3 passes. The three cells below are scored by option
likelihood, where the paper's shared attention cache makes no difference. Two of them
are zero-shot, so no few-shot seed is involved.

| Cell and layers | Published, baseline to looped | Here, parent to design | Items gained, lost |
| --- | --- | --- | --- |
| CommonsenseQA, 7-shot, 15 to 18 | 78.87 to 79.93 | 79.85 to 80.18 (975 to 979 of 1,221) | 37, 33 |
| SciQ, zero-shot, 15 to 18 | 93.30 to 95.10 | 92.6 to 94.5 (926 to 945 of 1,000) | 20, 1 |
| OpenBookQA, zero-shot, 11 to 14 | 40.20 to 40.60 | 40.0 to 39.8 (200 to 199 of 500) | 3, 4 |

SciQ and OpenBookQA have two accuracies in lm-eval, plain and length-normalized. The
paper does not say which it reports. The table uses the length-normalized one,
because the published baselines sit beside it and far from the plain one, which is
95.7 on SciQ and 31.6 on OpenBookQA here. On plain accuracy the design gained 11 SciQ
items and lost none, and gained 9 and lost 9 on OpenBookQA.

- **The path through the harness is exact.** A design run at one pass gave all 1,221
  CommonsenseQA items the parent's five option log-likelihoods, to the four decimals
  recorded. The parent itself runs without the runner installed.
- **Every design run made the program's layer calls.** The hooks counted 1,221
  forward passes on CommonsenseQA, 2,808 on SciQ and 1,575 on OpenBookQA, with 44
  layer calls in each.
- **No published baseline was matched exactly.** OpenBookQA is one item lower here,
  SciQ seven items lower and CommonsenseQA twelve higher. The cause was not
  separated. Candidates are rounding that differs between GPUs in BF16, library
  versions, and for CommonsenseQA the paper's unstated seed. Three measurements give
  the size of such effects on the parent alone:
  - batch size 16 instead of 1 changed 6 CommonsenseQA items (977 correct), 5
    OpenBookQA items (203 correct) and 1 SciQ item (927 correct);
  - five other seeds gave 969, 974, 977, 980 and 988 on CommonsenseQA;
  - a chat template gave 973 on CommonsenseQA.
- **The SciQ gain was reproduced in size:** 19 items here, 18 published. At batch
  size 16 it was 18 items, 927 to 945.
- **The two smaller published gains were not.** CommonsenseQA moved by 4 items
  against 13 published, with 70 items changing between right and wrong. That
  difference of 9 is smaller than the spread across seeds. OpenBookQA moved the other
  way against 2 published: by 1 item at batch size 1, and by 5 at batch size 16 (203
  to 198, with 2 gained and 7 lost). Batch size alone moved its parent by 3. So
  neither cell shows that the toolkit computes something different from the paper,
  and neither confirms that it computes the same.
- **The paper's "Euler" cells probably do not use the anchor.** With
  `anchor=0.5` the same loops gained 9 SciQ items and lost none, against 20 and 1
  without it. On CommonsenseQA the anchored design scored 977 and on OpenBookQA 198.
  This rests on one cell.

**Generated answers.** MMLU-Pro is scored from generated text. The lm-eval task is
five-shot with written reasoning. It stops at the next `Question:` or after 2,048
tokens, and takes the first letter that follows `answer is`. It was run on two items
per subject, 28 items, to check the path. That is far too few to estimate a score.

- **The path is exact at one pass.** A one-pass design generated the parent's text on
  all 28 items, over 3,488 forward passes of 36 layer calls each.
- **A three-pass design made the program's layer calls throughout.**
  `block[15:19]/euler` made 44 in each of 5,107 forward passes. Its text differed
  from the parent's on all 28 items and its extracted answer on 5. It answered 16
  correctly against the parent's 20, losing 4 items and gaining none.
- **This does not test the paper's MMLU-Pro cell.** The paper reports a gain of 2.64
  points with its shared attention cache and a loss of 8.86 without it. The toolkit
  keeps one cache per pass. Whether that is the paper's no-cache variant is not
  confirmed, and 28 items cannot tell the cases apart.
- **Batch size 8 was slower than batch size 1, and changed the text.** The parent
  took 526 seconds against 91, and the design 5,037 against 155. A batch runs until
  its longest row stops, and a row that repeats itself runs to the 2,048-token cap,
  so the design made 5,608 forward passes at batch size 8 against 5,107 at batch
  size 1. Each pass, now over 8 rows, was also about 30 times slower, with peak
  allocation at 13.4 GiB of the card's 16. Why was not examined. Batching changed the parent's text on 17 of
  28 items and its extracted answer on 2, for 19 correct. The design scored 15
  against that, again losing 4 and gaining none.

For anyone using the runner, the consequences are:

- hold batch size, seed and precision fixed between a parent and a design, and
  expect a few items of movement from those settings alone;
- on this card, run generated-answer tasks at batch size 1. At these rates the full
  12,032-item MMLU-Pro test would take about 11 hours for the parent and 18 for a
  three-pass design;
- likelihood tasks ran two to four times as fast at batch size 16, but SciQ then
  peaked at 15.3 GiB.

The compact record of all 32 benchmark runs, with paired counts and the hash of each
full record, is [`retrofit-bench-20261115.json`](../local_models/retrofit-bench-20261115.json).
Twenty-eight are the runs of this section and four are the
[recirculation runs](#what-was-checked).
The full records, with every item's result, are local only, under
`artifacts/runs/local-runtime/retrofit-bench-20261115/`. Not attempted: the paper's
full MMLU-Pro cell, whose result depends on the shared cache the toolkit does not
have.

**The LoopUS checkpoint.** Table 1 of the
[LoopUS paper](research/recurrent-depth-retrofit-literature.md#loopus-details-that-shape-an-implementation)
reports OpenBookQA accuracy of 28 for Qwen3-1.7B and 29.6 with LoopUS, zero-shot, at
most 8 passes and a stopping threshold of 0.6. OpenBookQA was the first cell checked,
when it was the only task of that table pinned here; six more tasks follow it below. The runs below
are from 2026-11-16: 500 items, batch size 1, BF16, and plain accuracy, which is the
measure the paper's numbers fit.

| Run | Correct of 500 | Accuracy | Published |
| --- | --- | --- | --- |
| `Qwen/Qwen3-1.7B`, the parent | 140 | 28.0 | 28 |
| LoopUS, `gated[2:27]`, 8 passes | 148 | 29.6 | 29.6 |
| LoopUS, `gated[2:27]/halt,threshold=0.6`, at most 8 passes | 147 | 29.4 | 29.6 |
| LoopUS, `gated[2:27]`, 2 passes | 145 | 29.0 | not reported |
| LoopUS, `gated[2:27]`, 20 passes | 148 | 29.6 | not reported |
| `Qwen/Qwen3-1.7B-Base` | 149 | 29.8 | not reported |

- **Both published numbers were reproduced.** The parent matches to the item. That
  also identifies the paper's baseline as `Qwen/Qwen3-1.7B`, not the Base model. The
  checkpoint matches to the item at a fixed 8 passes, and is one item lower with
  stopping.
- **When it was run, this cell was the main evidence that the rewritten forward pass
  computes what the authors' does, and it is one cell of 500 items.** A count can
  match while individual items differ, and the authors' per-item results are not
  available. The six tasks below add to it.
- **Stopping is read one token earlier here than in the authors' adapter,** as
  [described above](#a-checkpoint-trained-to-loop). With the threshold, 1,562 of the
  1,575 scored sequences ran all 8 passes. The other 13 stopped after the first pass,
  whose output is unusable. The run made 317,450 layer calls against 319,725 at a
  fixed 8 passes, and each layer's count matched the gate's own tally.
- **Every fixed run made its design's layer calls** in each of 1,575 forward passes:
  53 at 2 passes, 203 at 8 and 503 at 20. The runs took 57, 199 and 490 seconds
  against the parent's 36, and peaked at 3.9 to 4.0 GiB against 3.3.
- **The cell reproduces the paper's number. It does not show that the checkpoint is
  better than its parent.** The difference is 8 items, with 37 gained and 29 lost.
  Batch size alone moved 5 items on this benchmark for the 4B parent. The Base model,
  which was never trained to loop, scores 149.

The compact record of the six runs, with paired counts, is
[`gated-bench-20261116.json`](../local_models/gated-bench-20261116.json). The full
records are local only, under `artifacts/runs/local-runtime/gated-bench-20261116/`.

**The rest of the LoopUS table.** The datasets of the table's other eight tasks were
pinned on 2026-11-16. Six of them were run that day, with OpenBookQA again: every
item, batch size 1, BF16, zero-shot, with `--max-length 1024 --rolling-context 513`.
Those two options give the windows the authors' adapter for the checkpoint scores
perplexity in: 1,024 tokens, advancing by 512. The accuracy cells give correct items
and plain accuracy. WikiText's perplexity is per word and LAMBADA's per token.

| Task | Items | Published, parent → LoopUS | Parent | LoopUS, 8 passes | LoopUS, threshold 0.6, at most 8 passes |
| --- | --- | --- | --- | --- | --- |
| WikiText, perplexity | 62 pages | 21 → 16.9 | 23.02 | 16.99 | 17.72 |
| LAMBADA, perplexity | 5,153 | 12.21 → 7.43 | 12.20 | 7.43 | 7.44 |
| ARC-Easy | 2,376 | 72.5 → 74.9 | 1,717 (72.3) | 1,778 (74.8) | 1,775 (74.7) |
| ARC-Challenge | 1,172 | 40.2 → 43.1 | 469 (40.0) | 503 (42.9) | 502 (42.8) |
| PIQA | 1,838 | 72.2 → 73.3 | 1,332 (72.5) | 1,349 (73.4) | 1,341 (73.0) |
| WinoGrande | 1,267 | 61.3 → 63.0 | 774 (61.1) | 808 (63.8) | 800 (63.1) |
| OpenBookQA | 500 | 28 → 29.6 | 140 (28.0) | 148 (29.6) | 147 (29.4) |

- **Thirteen of the fourteen published numbers were reproduced.** On the five accuracy
  tasks the parent is within 0.3 points of the paper. The checkpoint is within 0.8 at
  a fixed 8 passes and within 0.4 with the threshold. Both LAMBADA perplexities match
  to 0.02, and the checkpoint's WikiText perplexity matches to 0.1 at a fixed 8
  passes.
- **The parent's WikiText perplexity is the one that differs:** 23.02 here against
  21. The checkpoint's matches, which points at how the baseline was scored and not
  at the loader. That is untested. The authors' adapter for an unmodified model,
  `OriginalModelAdapter` at commit `003ad7a5`, defaults to a 2,048-token window where
  these runs used 1,024. Four runs that vary the window and the stride were prepared
  and have not been run.
- **With the threshold, a window that stops after the first pass costs perplexity.**
  On WikiText 4 of the 531 scored windows stopped early, 3 of them after the first
  pass, whose output is unusable, and the perplexity rose from 16.99 to 17.72. On the
  other tasks between none and 1.0% of forward passes stopped before the eighth pass.
- **In BF16 the fixed and threshold runs differ on more items than stopping
  explains.** On WinoGrande all 2,534 forward passes ran the full 8 passes under the
  threshold, and 42 of the 1,267 items still changed between the two runs, 17 gained
  and 25 lost. The threshold path calls the model without a cache and the fixed path
  with one. In FP32 the two agree to 0.00004 in the acceptance checks above, and in
  BF16 their logits differ by up to about 0.5. An answer's score is a BF16 value on a
  coarse grid, such as -17.25 against -18.625, so near-ties are common. The other
  tasks changed on 1 of 500 items for OpenBookQA, 13 of 1,172 for ARC-Challenge, 26
  of 1,838 for PIQA, 19 of 2,376 for ARC-Easy and 50 of 5,153 for LAMBADA.
- **The table reproduces the paper's numbers. It does not show that looping makes the
  checkpoint better than its parent.** At a fixed 8 passes the checkpoint has more
  items right than the parent on every task: 8 more on OpenBookQA, 17 on PIQA, 34 on
  WinoGrande and on ARC-Challenge, 61 on ARC-Easy, and 351 of 5,153 on LAMBADA's last
  word. The checkpoint also had 1.5 to 3 billion tokens of further training that the
  parent did not, and the table has no twin trained on the same text at one pass.
  WinoGrande's 34 is also smaller than the 42 items that two BF16 code paths moved.
- **MMLU and HellaSwag were not run.** Their datasets are pinned. At batch size 1
  their six runs would take about 8 hours on this card, by the rates of the runs
  above.

The compact record of the 21 runs, with paired counts against the parent, is
[`loopus-table-20261116.json`](../local_models/loopus-table-20261116.json). The full
records are local only, under `artifacts/runs/local-runtime/loopus-table-20261116/`.
Other jobs shared the CPU during some of these runs, so their durations are not
comparable with each other.

## Observations from the checks

**BF16 rounding is amplified by plain layer loops.** Checks 4 and 6 compare two
orders of the same arithmetic. In BF16 the parents' own gap between cached and
recomputed logits is about 0.2 to 0.25. Plain block loops stay in that range. Plain
layer loops reach about 2 logits at 3 or 4 passes, and in one of six steps that
changed the top token. In FP32 on the same weights every gap is below 0.0002, so
this is rounding and not a cache error. Two consequences:

- those two checks can only be judged in FP32, which is how the acceptance script
  treats them;
- a looped model's BF16 output can depend on details that change rounding, such as
  batch size or whether a cache is used. A benchmark comparison should hold those
  fixed between the parent and the retrofit.

With the update rules over four layers, the BF16 gaps stayed near the parents' own,
at 0.42 or less.

**On three prompts, the update rules kept answers that some plain loops lost.** This
is Claude Fable 5.2's reading of three prompts per condition with greedy decoding,
not a measurement. The text is in the compact records.

- **Plain loops over the middle third** (9 or 12 layers) visibly damaged the output.
  Block loops on the 4B parent still answered all three prompts at every pass count.
  On the 1.7B parent the arithmetic prompt went from `4` to `2` at 3 and 4 passes.
  Layer loops degraded faster on both, and at 4 passes neither parent produced a
  relevant answer.
- **Plain loops over the middle four layers** did less damage. In a
  [separate run](../local_models/retrofit-check-20261115-small-unit.json), the 1.7B
  parent's layer loop at 3 passes answered `2` and named red, blue and green. Its
  block loop, and both loops on the 4B parent, kept both answers.
- **All seven update designs over those same four layers**, at 2 and 3 passes on
  both parents, answered `4` and named red, blue and yellow. They did not leave the
  text untouched: continuing the first prompt, `layer[12:16]/uniform` at 3 passes on
  the 1.7B parent went on to write that the capital of France is Rome.

So over the same four layers, the rules made a visible difference in one of the
eight plain conditions, on the smaller parent. That agrees with the published reports
and is far too little to estimate anything. Whether any design improves on its parent is what the benchmark
battery is for.

## Limits

- No design's placement or settings have been tuned. The only benchmark runs are the
  checks against the published table, at that paper's settings.
- The benchmark runner has twelve tasks' datasets pinned. Any other lm-eval task
  needs a pin and a download first. Which benchmarks the research uses is not
  decided.
- The update rules follow automated readings of two papers, as described above.
  Recirculation follows an automated reading of a third, with three choices the
  extracted text does not settle.
- A recirculation scores a sequence one token at a time, so a benchmark run takes
  about as many steps as its sequences have tokens.
- The worker serves the parent, loops and recirculations, one request at a time, with
  greedy decoding. Routes run through the library only.
- One model family. Batched inputs under a design are checked on the tiny model in
  FP32 only. In BF16 on a real parent, batch size changes some items' results, and
  most items' generated text.
- The generated-answer path has been run on 28 items only.
- Not built: the Training-Free paper's shared attention cache and higher-order
  variants, and the Inner Loop paper's auto-alignment rule. Without the shared
  cache, that paper's generated-text results cannot be reproduced exactly. For
  recirculation: the ramp, the other normalizations, repeated recirculation and the
  learned coefficients.
- No published recirculation number has been reproduced. The paper's models are
  Gemma 3, which the toolkit does not load.
- The LoopUS forward pass was rewritten from the authors' code and has not been
  compared with that code running. Seven of the nine tasks of the paper's table were
  checked: thirteen of their fourteen published numbers were reproduced, and the
  parent's WikiText perplexity was not. MMLU and HellaSwag were not run.
- Benchmark runs are in BF16. Two code paths through the same arithmetic changed up
  to 3% of a task's items there, so a comparison of a design with its parent in BF16
  carries that much noise before any real difference.
- The LoopUS checkpoint is unusable at exactly one pass, and with a stopping
  threshold of 0.6 a few sequences stop there.
- A gated loop stops early only on whole sequences without a cache. The worker serves
  it at a fixed number of passes, and generated-answer benchmarks run that way too.
- A gated loop needs a checkpoint trained with a gate. Only the LoopUS layout is
  loaded, for Qwen3, and only the released 1.7B checkpoint has been tried.

## Planned pieces

1. **Done:** the layer-program runner, its checks and worker support.
2. **Done:** the training-free update rules: damped refinement with an optional
   anchor, and interpolation with earlier passes.
3. **Done:** the benchmark runner, which puts a design and its parent through the
   same items under the same settings and keeps per-item results. Checked against
   three cells of a published table: one gain was reproduced in size, and no baseline
   was matched exactly. Using the runner to decide whether a design improves its
   parent is research, and comes afterwards.
4. **Done:** the released LoopUS checkpoint for Qwen3-1.7B, loaded and run as a gated
   loop. Seven of the nine tasks of its published table were checked, and thirteen of
   their fourteen numbers were reproduced.
5. **Done:** across-token feedback in its fixed-coefficient form, recirculation,
   which needs no training. Its learned coefficients belong with piece 6.
6. Trained conversions. **Built for the gate, not yet run on the GPU:** a
   [trainer for a gate over a frozen parent](#training-a-gate-over-a-frozen-parent).
   Adapters, the full block and learned recirculation are not built. The
   [estimate](#estimate-for-trained-conversions) sized all four. The gated loop built
   for piece 4 is the forward pass a LoopUS-style conversion trains.

### Estimate for trained conversions

Scope: a sizing of piece 6, made on 2026-11-16 before any of it is built. Nothing was
trained or downloaded. The rates were measured on the RTX 5070 Ti, the hours are
arithmetic on those rates, and the hosted figures are guesses.

**What was timed.** [`train_timing.py`](../tools/local_models/train_timing.py) runs
training steps on batches of random tokens, with the parent in BF16. A step follows
the LoopUS training loop, read at commit `003ad7a5`. The layers below the looped unit
run once without gradients. The unit and its gate then run `N` times, and `K` of those
passes, chosen at random, are decoded, scored and followed by an optimizer step. The
state is detached after every pass, so only one pass's graph is ever held. The loss
is LoopUS's: next-token loss, a penalty when a pass makes it worse, and the confidence
head's loss.

| | Candidate | What is trained | Parameters on Qwen3-1.7B, layers 2 to 26 |
| --- | --- | --- | --- |
| a | Gate | The existing `Gate` and confidence head. The parent stays frozen | 0.53 million |
| b | Adapters | The same, plus rank-16 corrections on all seven linear maps of each looped layer | 16.1 million |
| c | Learned recirculation | A small network that sets a recirculation's mixing coefficients for each token | 0.40 million |
| d | Full block | The gate and head, plus the looped layers themselves, as LoopUS trains | 1,259 million |

**Measured on Qwen3-1.7B at context 1,024.** A token is one token of training text,
which goes through all `N` passes. Layers 2 to 26 are LoopUS's unit, and 12 to 15 are
the four middle layers the checks above use. Batch and peak memory are for the
fastest batch size that fits.

| Candidate | Layers | Tokens per second, `N`=8, `K`=2 | `N`=20, `K`=5, as LoopUS | `N`=1, `K`=1 | Batch | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| Gate | 2 to 26 | 2,519 | 966 | 11,940 | 4 | 10.8 GiB |
| Adapters | 2 to 26 | 1,590 | 603 | 5,551 | 1 | 7.8 GiB |
| Full block | 2 to 26 | 1,296 | 509 | 3,524 | 1 | 12.7 GiB |
| Gate | 12 to 15 | 3,519 | 1,464 | 7,561 | 2 | 8.8 GiB |
| Adapters | 12 to 15 | 3,316 | 1,276 | 6,723 | 1 | 6.4 GiB |
| Full block | 12 to 15 | 3,034 | 1,278 | 6,561 | 2 | 10.4 GiB |

- **`N`, `K` and the size of the unit set the cost. The number of trained parameters
  barely does.** Only `K` of the `N` passes carry gradients, so over LoopUS's unit
  adapters train at 0.63 of the gate's rate and the full block at 0.51. Going from 8
  passes with 2 supervised to LoopUS's 20 with 5 costs a factor of 2.4 to 2.6.
- **The loss takes most of the memory.** Scoring over the 151,936-token vocabulary in
  FP32 holds about 1.8 GiB for every 1,000 tokens in a batch, so batches are 1 to 4
  sequences. At context 1,024 the rate changed by less than 10% across the batch
  sizes that fit. At context 512 the rates are 3 to 17% higher.
- **The full block fits only in BF16,** with BF16 optimizer state, which is how the
  LoopUS code loads the model. In FP32 the weights, gradients and optimizer state of
  layers 2 to 26 alone would take about 19 GiB. Whether BF16 updates lose small steps
  to rounding was not examined.
- **Qwen3-4B-Instruct,** whose weights take 7.5 GiB, at context 1,024 and 8 passes
  with 2 supervised: the gate over layers 2 to 34 trains at 1,070 tokens per second,
  and over layers 15 to 18 at 1,611, with adapters there at 1,493. Adapters over
  layers 2 to 34 fit only at context 512, at 629. The full block fits only for the
  four layers at context 512, at 1,256, and not at all for layers 2 to 34.
- **Learned recirculation does not train usefully through the present driver.**
  Backpropagating through it, one token at a time with the cache, ran at 14 tokens
  per second for each sequence in the batch on the 1.7B parent. The memory grows with
  the square of the context: 4 sequences fit at context 128, one at 256 and none at
  512, and on the 4B parent 2 at 128 and none at 256. At context 128 that is 56
  tokens per second, or 50 hours for 10 million tokens. It needs a driver that
  carries gradients through a short window of tokens first. The paper's own training
  procedure was not read, and the timed network is a stand-in.
- **Limits of the timing.** Each rate is two timed steps after a warm-up step, taken
  once. Reading and tokenizing text, saving and measuring are not included. The
  optimizer stepped after every supervised pass, where LoopUS's command accumulates
  64 of them, which costs the full block the most. A batch counted as not fitting
  when PyTorch needed more than 13.66 GiB, 1 GiB under what was free.

**Hours on this card for Qwen3-1.7B at context 1,024,** from the rates above. LoopUS
reports 3 billion training tokens, and its README's command uses 1.5 billion, with 20
passes of which 5 are supervised and the block of layers 2 to 26 fully trained.

| Candidate, layers, `N`:`K` | 10 million tokens | 50 million | 200 million | 1.5 billion | 3 billion |
| --- | --- | --- | --- | --- | --- |
| Gate, 2 to 26, 8:2 | 1.1 | 5.5 | 22 | 165 | 331 |
| Gate, 12 to 15, 8:2 | 0.8 | 3.9 | 16 | 118 | 237 |
| Adapters, 2 to 26, 8:2 | 1.7 | 8.7 | 35 | 262 | 524 |
| Adapters, 12 to 15, 8:2 | 0.8 | 4.2 | 17 | 126 | 251 |
| Full block, 2 to 26, 8:2 | 2.1 | 11 | 43 | 322 | 643 |
| Full block, 2 to 26, 20:5, LoopUS's recipe | 5.5 | 27 | 109 | 819 | 1,638 |

- **Candidates (a), (b) and (d) can all be trained on this card for the 1.7B parent,**
  up to a few hundred million tokens. LoopUS's recipe at its published budgets would
  take 34 to 68 days here, so a run of that size needs hosted compute.
- **Hosted cost is a guess.** If a rented H100 ran this two to four times as fast,
  which was not measured, 1.5 billion tokens would take about 200 to 400 GPU-hours.
  Third-party price pages read on 2026-11-16 put an H100 at $2 to $4.70 an hour,
  which was not checked with a provider. That gives roughly $400 to $1,900, and twice
  that for 3 billion.
- **For the 4B parent** the card trains a gate anywhere and adapters over a small
  unit. Adapters or a full block over most of its layers need more memory than 16 GiB.

**Training text.** LoopUS trains on FineWeb-Edu, and its README's command reads the
`CC-MAIN-2025-26` crawl from the start. The proposal is the first file of that crawl,
read from the Hub's file listing and not downloaded:

- `HuggingFaceFW/fineweb-edu` at commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`,
  file `data/CC-MAIN-2025-26/000_00000.parquet`: 1,387,647,887 bytes (1.29 GiB),
  SHA-256 `4cefc2e8d3efd44e756e33ef25e940e0a228bc457f54517de9cfd67dbd11a4a1`. The
  licence is ODC-By 1.0, CommonCrawl's terms of use apply, and the dataset is not
  gated.
- It should hold about 490 million tokens. That is an estimate from the dataset's
  10-billion-token sample, which is 26.6 GiB, and those are GPT-2 tokens. The real
  count comes after the download.
- The whole crawl is 50 files and 41.7 GiB, which a run at LoopUS's budgets would
  need. The smaller alternative is `sample/10BT/013_00000.parquet`, at 0.50 GiB and
  about 190 million tokens drawn from every crawl.

**What would show that a conversion took.** These are checks that training did
something and that the result runs. They do not show that a conversion improves its
parent, which is research and needs a twin trained on the same tokens at one pass.

- **Loss by number of passes,** with the measurement
  [`gated_check.py`](../tools/local_models/gated_check.py) makes, on the 200 science
  passages used above and on rows of the training file held out from training. A
  conversion took if, at every pass count from 2 up to the trained `N`, the trained
  loop's loss is below the same loop's loss before training. The parent's one-pass
  loss, 2.351 on the passages, and the LoopUS checkpoint's, 1.994 at 8 passes, sit
  beside it for reference.
- **The frozen weights are unchanged,** tensor by tensor, after training (a) or (b).
- **The benchmark runner** scores the trained design and its parent on the pinned
  tasks and pairs them with `compare --across-checkpoints`. That shows the result
  loads, makes its design's layer calls and still answers. The counts are reported
  and not judged.

**Recommended first conversion.** This is Claude Fable 5.2's recommendation (Claude
Code), not a decision. Train candidate (a), the gate and head over a frozen
Qwen3-1.7B, as `gated[2:27]` with 8 passes of which 2 are supervised, at context
1,024: 10 million tokens first, about 1.1 hours, and on to 50 million, 5.5 hours in
all, if the held-out loss is still falling. Then train the same gate over layers 12
to 15 for 10 million tokens, 0.8 hours.

- It is the cheapest run that exercises everything a trained conversion needs: the
  text, the training step, a saved result and its loader.
- Its result runs through the `gated[2:27]` path that is already checked, beside the
  LoopUS checkpoint under the same measurements, and the parent stays bit-identical.
- Its training step is the one (b) and (d) use. They differ only in which tensors
  are trained.
- **The main alternative is to start with (b), adapters.** They cost 1.6 times as
  much per token over LoopUS's unit and about the same over four layers, and they
  have thirty times the gate's parameters, so they are more likely to change what
  the loop computes. They need a loader that puts adapters on the looped layers, and
  they change the first pass as well, so reading their result takes more controls.
  The plan is to build both into one trainer and run (b) second.

The record of all 119 timed batch sizes, including those that did not fit, is
[`train-timing-20261116.json`](../local_models/train-timing-20261116.json). To repeat
it for one parent, with the GPU lock free:

```powershell
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/train_timing.py --profile qwen3-1.7b --units 2:27 12:16 --output /mnt/g/Evals/local_models/train-timing.json
```

For `qwen3-4b-instruct` the units were `2:35` and `15:19`.

### Training a gate over a frozen parent

Scope: toolkit setup, built on 2026-11-16 by Claude Fable 5.2 (Claude Code). Everything
below ran on the CPU. No GPU training has run. On 2026-11-16 the user approved the
1.29 GiB download and up to 8 hours of local gate training, to start that night once
they say so.

**What it trains.** [`training.py`](../src/recurrent_retrofit/training.py) trains a
`gated.Loop`: the gate that follows every pass of the looped layers, and the confidence
head. The parent's weights are never given to the optimizer, so a result is a 2 MB file
that the unchanged parent carries. A batch is taken as the estimate above describes,
following the LoopUS training loop. Two things differ from LoopUS, which trains the
looped layers as well:

- **The gate starts open.** It lets through 0.9975 of the unit's output at every
  position and channel, whatever the input, so the first pass begins as the parent's
  own computation. LoopUS draws the gate's starting values at random. Trained layers
  can adapt to that and frozen ones cannot.
- **The gate and head train in FP32,** whatever the parent's precision. The runner
  already runs a gate in the gate's own precision, so a BF16 parent carries an FP32
  gate.

**The text.** The FineWeb-Edu file named in the estimate was downloaded on 2026-11-16,
and its size and SHA-256 matched the pin in
[`training_text.json`](../src/recurrent_retrofit/training_text.json). It holds 435,357
documents. Each is tokenized with the parent's tokenizer and followed by
`<|endoftext|>`, and the stream is cut into full rows with no padding. Training rows
are filled from the file's first documents onward and held-out rows from its last
documents backward, and the two may not share a document. For `qwen3-1.7b` at context
1,024 the first 47,285 documents gave 48,828 rows, 50.0 million tokens, and the last
242 gave 256 held-out rows. That is 1,057 tokens a document, so the whole file is
about 460 million Qwen tokens, where the estimate guessed 490 million GPT-2 tokens.
Whether LoopUS packs its rows or pads them was not checked.

**Use.** [`retrofit_train.py`](../tools/local_models/retrofit_train.py) has five
commands. In a WSL shell:

```bash
PY="/home/ben/llm/recurrent/envs/retrofit/bin/python -u -B /mnt/g/Evals/tools/local_models/retrofit_train.py"
RUN=/home/ben/llm/recurrent/training/runs/qwen3-1.7b-gate-2-27-n8k2
$PY fetch                                                      # the only step that uses the network
$PY prepare --profile qwen3-1.7b --train-tokens 50000000       # tokenize and pack, on the CPU
$PY train --profile qwen3-1.7b --unit 2:27 --run $RUN --tokens 10000000
$PY falling --run $RUN                                         # exit code 0 if the held-out loss is still falling
$PY measure --loop $RUN/loops/9994240
```

- `train` holds the GPU lock for one segment of at most `--minutes`, 45 by default,
  and then exits so that other GPU jobs can take a turn. Running it again continues
  the run from its checkpoint, reading the rows and scoring the passes an unbroken run
  would have. A run folder refuses other settings, and `--tokens` is the most text the
  run may have read, so raising it extends a finished run.
- Every `--measure-every` tokens, 2 million by default, `train` computes the loss
  after each pass by hand on 32 held-out rows, with the mean share of the unit's
  output the gate let through, and saves the loop as it then is under `loops/`.
- `measure` puts a saved loop through `install(model, GatedLoop(...))`, the path the
  LoopUS checkpoint was checked on, at 1, 2, 3, 4, 6 and 8 passes. It scores the 200
  science passages used above and 64 held-out rows, for the plain parent, for the loop
  as it was before training, and for the trained loop. A loop took if at every number
  of passes from 2 up its loss is below the untrained loop's.
- `train` checks a SHA-256 over all of the parent's tensors before and after each
  segment, and a saved loop attaches only to the parent it was trained over.
- A trained loop is not yet a profile. `load_parent`, the worker and the benchmark
  runner cannot load one by name until the loader has a family for it.

`fetch` and `prepare` were run as shown. `train`, `falling` and `measure` were run on
the CPU only, with `--placement cpu --precision float32` at context 64.

**What was checked.** [`test_training.py`](../tests/retrofit/test_training.py) has 31
tests, which take 10 seconds on the CPU. The trainer tests use a tiny Qwen3 model that
is first taught a rule, so that looping its middle layers spoils something.

- The by-hand forward pass the trainer uses gives the runner's logits to within 1e-4
  at 1, 2 and 4 passes, and its loss by passes is the runner's.
- A fully open gate reproduces the parent's logits exactly on one pass.
- A batch's gradient equals the sum of its scored passes' gradients, each computed by
  hand as its own graph from a state that carries no gradient.
- Training changes every tensor of the loop and no tensor of the parent, for an FP32
  parent and for a BF16 parent carrying an FP32 loop.
- A run resumed from its saved state ends bit-identical to an unbroken one.
- A cached text file, a saved loop or packed rows that differ from their recorded
  SHA-256 are refused, as are rows that would share a document between training and
  held-out text, and a loop offered to another parent.
- **The tests can fail.** Eighteen deliberately broken copies of the trainer were each
  caught by at least one test. They included a state not detached between passes, a
  dropped penalty, a head trained on the wrong input or target, an optimizer state
  not restored, a loss scoring the wrong token, and each of the refusals above
  removed.

**A look on the CPU before the first GPU run.** This is exploratory: one parent,
Qwen3-1.7B in FP32, layers 2 to 26, rows of 64 tokens, small samples. It shaped the
settings below and supports no finding.

- **The untrained loop gets worse gently.** On 96 training rows the parent's loss is
  3.45. With layers 2 to 26 run again and the gate fully open it is 3.59 after two
  passes, 3.79 after three and 4.03 after four. On the 8 held-out rows the trials
  monitor, the parent's loss is 3.86 and the untrained loop's is 5.45 after eight
  passes.
- **A linear map can tell the first pass from a later one.** The gate sees only how a
  pass changed the state. A linear map fitted on 64 rows separated first-pass changes
  from second-pass changes on 32 other rows with no error in 4,096 tokens, at rank 1
  as at rank 128. So a gate that is open on the first pass and closed afterwards,
  which would give the parent's loss at every number of passes, is within reach. In
  the tiny test model it is not: a linear map gets 65% there, and the trained gate
  closes part of the way on every pass and leaves the first pass worse than the
  parent.
- **The gate's input is large.** The state's mean norm is 26 where it enters layer 2
  and 3,400 after layer 26, and each further open pass adds about 1,700. A learning
  rate therefore moves this gate much further than it would move a layer whose inputs
  are near 1.
- **A learning rate of 1e-3 is too high.** A first trial took 50 batches of 2 rows,
  25 updates. At 6 and at 12 updates the held-out loss was below the untrained loop's
  at every number of passes. At 18 the first pass's loss had jumped from 3.85 to
  5.24, and at 24 it was back at 3.84 with every pass about where it stood at update
  6. The trial stopped on its time limit, resumed, and finished with the parent's
  SHA-256 unchanged.
- **1e-4 held, and did better than 3e-4.** Two further trials took 150 updates each,
  with 20 warm-up updates. At 1e-4 the held-out loss averaged over the eight pass
  counts fell from 4.58 to 4.23 and was below its start at every measurement. At 3e-4
  it reached 4.31, after rising to 4.43 at 75 updates. In both the gate came to let
  more through on the first pass, 0.91 and 0.94, than on later ones, 0.71 to 0.78, and
  the first pass's loss stayed between 3.80 and 3.84 against the parent's 3.86. All
  of that is on 8 held-out rows.
- **The runner agrees with the trainer.** `measure` on the first trial's loop gave the
  trainer's own by-hand losses to five decimals at 1, 2, 4 and 8 passes.

**Stopping a run that gets worse.** Scored passes are drawn evenly, so what training
minimises is the held-out loss averaged over the passes. If that average rises above
the untrained loop's at a measurement, `train` stops the run with exit code 4 and
leaves the checkpoint at the last good point. A lower learning rate then needs a new
run folder. Repeating the 1e-3 trial on the CPU exercised the stop: it ended that run
at 18 updates, when the average was 4.82 against the untrained loop's 4.58.

**Settings for the first GPU runs.** These are Claude Fable 5.2's choices, for the
user to change.

| Setting | Value | Why |
| --- | --- | --- |
| Parent | `qwen3-1.7b`, BF16, frozen | The LoopUS checkpoint's parent, so the two run side by side |
| Units | Layers 2 to 26, then 12 to 15 | LoopUS's unit, and the four middle layers the checks use |
| Passes | 8, of which 2 are scored | The pass count the LoopUS cell was reproduced at; about 2.5 times cheaper than 20 with 5 |
| Rows | 4 rows of 1,024 tokens a batch, 2 batches an update | The fastest batch that fit; 8,192 tokens of text an update |
| Optimizer | AdamW, no weight decay, gradient norm clipped at 1 | Weight decay would pull the starting bias toward a half-closed gate |
| Learning rate | 1e-4 after 100 warm-up updates, then constant | LoopUS uses 5e-5 for this gate; on the CPU 1e-3 was unstable and 1e-4 did better than 3e-4 |
| If a run is stopped | Start afresh at 3e-5, then 1e-5 | See above |
| Penalty weight | 1 | LoopUS's |
| Tokens | 10 million, then on to 50 million for the first unit if the held-out loss at 8 passes fell by more than 0.005 over the last 4 million | From the estimate |

**Limits.**

- Only the gate is trained. Adapters and the full block need the looped layers'
  gradients and a loader for changed layers, and are not built.
- The trainer has not run on the GPU. Its forward pass is the one
  [`train_timing.py`](../tools/local_models/train_timing.py) timed there, without
  that tool's automatic mixed precision.
- Rows are full, so nothing here handles padding, and the confidence head is trained
  on each row's last token only.
- The learning rate, warm-up and starting bias were chosen from one short CPU look.
  Nothing was tuned at the batch size the GPU runs use.

