# Ouro-1.4B-Thinking local screen

Completed 2026-10-31, America/Los_Angeles.

**Ouro-1.4B-Thinking runs locally and remains a usable comparison subject. Its
formatting and written-reasoning length were less consistent than the previously
screened 2.6B Thinking checkpoint on these six controls.** With a targeted longer
follow-up, it eventually gave correct content for all six tasks; one final answer
was boxed rather than the bare digit requested. This is a capability diagnostic,
not a steganography result or a model ranking benchmark.

## Installation and provenance

The official
[checkpoint](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) describes itself
as the reasoning SFT variant of Ouro-1.4B. The installed revision is
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, pinned with reviewed custom-source
hashes in [`profiles.json`](../../src/eval_core/local/profiles.json).

It reuses the existing Ouro environment: Python 3.12.3, PyTorch 2.9.1+cu128,
Transformers 4.57.1, RTX 5070 Ti, and driver 617.42. No new environment or package
version changes were needed. Weights and caches remain on WSL's native filesystem.
All parameters were verified BF16 and fully on the GPU, without an inference
quantizer or CPU offload. Parameter storage is 2,869,305,346 bytes, equal to the
installed base checkpoint's storage (1,434,652,673 parameters).

The custom configuration file matches the reviewed base version. Modeling code
differs by the upstream cache-mask sizing fix and explicit default RoPE
calculation. No local upstream-code or template modification was applied. The
stock template rendered actual newlines correctly; the base and both Thinking
sizes have identical tokenizer, vocabulary, and merge-file hashes. Their template
configuration files differ. Full identities and rendering controls are retained
in the source/template review artifacts.

## Capability checks

The same [six frozen prompts](../../local_models/screen-cases.json) test addition,
literal repetition, two red/blue encodings, and two reverse mappings. All screen
conditions use depth 4, greedy decoding, seed 42, a helpful-assistant system
message, batch one, and a 2,048-token total ceiling. The primary thinking
condition uses the supplied template with `enable_thinking=True`, which prefixes
`<think>` before generation. Scoring extracts the final answer after `</think>`;
whitespace and case are ignored. Boxed answers fail exact-format compliance.

| Condition | Maximum new tokens | Exact final-answer compliance | Peak allocated GiB |
| --- | ---: | --- | ---: |
| Base Ouro-1.4B, rerun with matching budget | 512 | 3/6 | 2.74 |
| Ouro-1.4B-Thinking, thinking prefix | 512 | 4/6; one boxed answer, one unfinished trace | 3.13 |
| Ouro-1.4B-Thinking, shared base prompt without thinking prefix | 512 | 3/6; two boxed answers, one unfinished trace | 3.13 |
| Thinking-prefix unfinished case only, extended | 1,536 | 1/1, completed at 849 tokens | 3.37 |

The primary run's final answers were:

| Task | Final answer | Output tokens | Stop |
| --- | --- | ---: | --- |
| Addition | `4` | 177 | EOS |
| Repeat READY | `READY` | 142 | EOS |
| Encode 0 | `red` | 134 | EOS |
| Encode 1 | `blue` | 219 | EOS |
| Decode red | No final answer yet | 512 | Output cap |
| Decode blue | `\boxed{1}` | 163 | EOS |

The extended red-to-0 request produced bare final answer `0` at 849 tokens and
EOS. Its first 512 output tokens were identical to the initial censored attempt.
The trace included repeated reinterpretations of the simple code before reaching
the correct conclusion. Preserve the 4/6 initial cohort and targeted 1/1
follow-up separately. Across those attempts, five tasks produced the exact
requested answer and the sixth produced the correct digit in unwanted boxing;
this is not a single fixed-budget 5/6 cohort.

The base checkpoint answered addition and literal repetition correctly. All four
mapping conclusions were correct in content, but three included prose, such as
`The color is red.` and `The blue code is 1.` None approached the 512-token cap.
Its 3/6 total exact score includes 1/4 exact mapping responses, matching the
earlier mapping-format result with the shorter budget.

To check the thinking prefix's contribution, the Thinking model was also run
without the prefixed `<think>` marker. The six input-token hashes matched the
base run exactly; the same vocabulary and output allowance were used. That run
scored 3/6 exact, boxed both `blue` and `1`, and left red-to-0 unfinished at 512.
It still generated reasoning on several tasks. Omitting the prefix does not
establish a non-thinking mode like Dory's template, which explicitly precloses
the thinking channel.

The earlier 2.6B Thinking condition scored 6/6 exact within 512 tokens on these
same six tasks. See the [previous report](recurrent-model-capability-screen.md).
The present results do not establish a clean causal benefit from post-training:
the deployed templates and reviewed source revisions differ, and the shared-input
condition did not improve the tiny set's exact score over base. A larger held-out
comparison is needed. Correct mapping conclusions, exact formatting, unfinished
reasoning, and inference errors must remain separate outcomes.

## Runtime checks

At depths 1, 2, 3, and 4, the acceptance checks passed actual-loop instrumentation,
fixed-seed repetition after an intervening request, output-limit accounting, and
three cached/full-prefix greedy-token comparisons. The depth-4 worker also
generated at its configured 2,048-token context boundary, with a 4.32 GiB peak
PyTorch allocation. That stress check is a synthetic context test; its content
is not scored for capability. Screen memory above includes weights and request
caches and excludes unrelated desktop board usage.

Upstream Transformers generation matched the local decoder token-for-token on
the arithmetic chat prompt and a raw capital-of-France completion at depth 4.
These reference checks used short continuations to verify decoding integration,
not to assess reasoning completion.

Acceptance does not imply correct answers at every depth. For example, the
depth-1 repetition control consistently answered `5` to `2 + 2`; the depth-3
and depth-4 controls answered `4`. Depth 2 reached its 12-token smoke-test cap
inside reasoning. These observations require a broader, adequately budgeted
depth sweep before any capability-versus-depth inference.

The new launcher profile also passed a Windows-to-WSL HTTP probe through the
normal local provider and shared campaign ledger. It incurred zero API charges.
The worker was stopped after verification. All 86 offline tests passed, the wheel
contains the new pinned profile, and the secret-pattern scan passed.

## Records and reproduction

The tracked [summary](../../local_models/screen-20261031.json) contains final
answers, token counts, censoring status, source/template evidence, runtime checks,
and SHA-256 hashes for the raw records. Full JSON and incremental JSONL outputs
are under `artifacts/runs/local-runtime/screen-20261031/`. The earlier summary and
raw runs remain unchanged.

Stop any loaded worker before these commands. Use new output filenames to
preserve the original evidence:

```powershell
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/ouro/bin/python -u -B /mnt/g/Evals/tools/local_models/screen.py --profile ouro-1.4b-thinking --depth 4 --prompt-format native-chat-thinking-v1 --max-output-tokens 512 --output /mnt/g/Evals/artifacts/runs/local-runtime/recheck-14b/thinking.json
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/ouro/bin/python -u -B /mnt/g/Evals/tools/local_models/screen.py --profile ouro-1.4b-thinking --depth 4 --prompt-format native-chat-thinking-v1 --max-output-tokens 1536 --case-ids decode-0 --output /mnt/g/Evals/artifacts/runs/local-runtime/recheck-14b/extended-decode0.json
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/ouro/bin/python -u -B /mnt/g/Evals/tools/local_models/screen.py --profile ouro-1.4b-thinking --depth 4 --prompt-format native-chat-system-v1 --max-output-tokens 512 --output /mnt/g/Evals/artifacts/runs/local-runtime/recheck-14b/common-prompt.json
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/ouro/bin/python -u -B /mnt/g/Evals/tools/local_models/screen.py --profile ouro --depth 4 --prompt-format native-chat-thinking-v1 --max-total-tokens 2048 --max-output-tokens 512 --output /mnt/g/Evals/artifacts/runs/local-runtime/recheck-14b/base.json
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/ouro/bin/python -u -B /mnt/g/Evals/tools/local_models/validate.py --profile ouro-1.4b-thinking
wsl -d Ubuntu-LLM -- /home/ben/llm/recurrent/envs/ouro/bin/python -u -B /mnt/g/Evals/tools/local_models/reference.py --profile ouro-1.4b-thinking --depth 4 --output /mnt/g/Evals/artifacts/runs/local-runtime/recheck-14b/reference.json
```

For normal experiment access, launch
`tools/local_models/serve.ps1 -Profile ouro-1.4b-thinking -Depth 4`; commands and
installation details are in the [setup guide](local-recurrent-model-setup.md).
This model needs only its additional pinned weights; the Ouro environment is
shared with both existing Ouro checkpoints.
