# How LLM calibration has been measured

Scope: published measurements, as of 2026-11-13, of whether a language model's
confidence in its answers matches how often those answers are correct, for closed-weight
and open-weight models. A targeted scan, not a systematic review and not a reproduction.
It is not tied to an experiment in this repository.

Origin: in a Claude Code chat on 2026-11-13 the user stated a hypothesis and asked for a
scan. Their words: "New hypothesis: models are poorly calibrated - their probability
estimates for the correctness of their answers does not correspond to the actual
frequency of correct answers. I imagine this is already well-studied. Can you look
through existing work to see what has been done to measure this? I am curious about the
calibration of both open weight and closed weight models." The hypothesis is the user's
and was stated before this scan. The searches, the choice of sources, the grouping and
the synthesis are Claude Fable 5.2's (Claude Code). The user has not yet reviewed it, and
it records no user policy.

## How the evidence was read

Pages were fetched with a tool that returns a small model's summary of the page, so
figures below are those summaries, not text I read directly. Each source carries one of
three marks:

- **fetched**: the arXiv abstract or HTML page, or the named post, was fetched in this
  task and the claim comes from that summary.
- **snippet**: the claim comes from a search-result summary only; the paper was not
  opened.
- **memory**: the paper is cited from Claude Fable 5.2's training knowledge and was not
  re-fetched. Check it before relying on it.

No result was checked against raw data or rerun.

## What is measured

**Calibration** means: among answers given confidence *p*, a fraction *p* are correct.
Two things are routinely confused with it:

- **Discrimination** (usually AUROC): whether correct answers get higher confidence
  than wrong ones. A model can rank its answers well and still state numbers that are
  all too high.
- **Accuracy**. A model that is right 30% of the time and says "30%" every time is
  perfectly calibrated and useless at telling its answers apart.

Common metrics:

| Metric | What it is | Main weakness |
| --- | --- | --- |
| Reliability diagram | Accuracy plotted against binned confidence | A picture, not a number |
| Expected calibration error (ECE) | Sample-weighted mean gap between confidence and accuracy over bins | Depends on binning; unstable when confidences cluster on a few round values |
| RMS calibration error | Root-mean-square version of the same gap | Same; also falls when a model simply states lower confidence on a hard test |
| Brier score | Mean squared error of the stated probability | Mixes calibration with accuracy; needs a baseline to interpret |
| Brier skill score | Brier relative to always predicting the base rate | Negative means worse than a constant guess |

The confidence itself can come from five places. Which one is used decides most of the
result.

| Signal | How it is read | Works on closed models? |
| --- | --- | --- |
| Token probability | Probability the model assigns to its answer token(s) | Only where the API returns log-probabilities |
| P(True) | Show the model an answer, read the probability of "true" | Same |
| Verbalized confidence | Ask for a percentage or a phrase | Yes |
| Sampling consistency | Ask many times; use the share of samples that agree (answer frequency, semantic entropy) | Yes, at the cost of many calls |
| Trained probe or fine-tune | Train a classifier on activations, or tune the model to report confidence | Probes need weights |

## Observations from the literature

### Token probabilities of pretrained models are often well calibrated, on formats they suit

- Kadavath et al., "Language Models (Mostly) Know What They Know"
  ([arXiv:2207.05221](https://arxiv.org/abs/2207.05221), 2022, **fetched**): larger
  models are well calibrated on multiple-choice and true/false questions "when they are
  provided in the right format". P(True) self-evaluation works and scales. A trained
  "do I know this" head (P(IK)) does not stay calibrated on new tasks.
- Zhu et al., "On the Calibration of Large Language Models and Alignment"
  ([arXiv:2311.13240](https://arxiv.org/abs/2311.13240), Findings of EMNLP 2023,
  **fetched**): on Pythia (70M to 12B), with LLaMA, LLaMA-2, FLAN-T5 and OPT in an
  appendix, calibration improves with parameter count and with training steps. Confidence
  is the output probability; ECE over 10 bins.
- Earlier background, **memory**: Guo et al. 2017 (arXiv:1706.04599) showed modern
  neural classifiers are miscalibrated and introduced temperature scaling; Desai and
  Durrett 2020 (arXiv:2003.07892) and Jiang et al. 2021 (arXiv:2012.00955) measured
  pretrained transformers and QA models.

### Post-training usually makes token-probability calibration worse, but not always

- OpenAI, GPT-4 Technical Report ([arXiv:2303.08774](https://arxiv.org/abs/2303.08774),
  2023, **snippet**): calibration plots on an MMLU subset, with confidence read from
  the log-probability of each A/B/C/D choice. The pretrained model is well calibrated;
  the caption says post-training hurts calibration significantly. This is the only
  closed-model base-versus-post-trained comparison found. I recall ECE figures from the
  plot but could not confirm them, so none are given.
- Zhu et al. (above, **fetched**): on LLaMA-7B, instruction tuning "damages model
  calibration significantly", more with synthetic (Alpaca) than human-written
  (OpenAssistant) data; RLHF on top of an instruction-tuned model "has little impact".
- Sanz-Guerrero, Mager and von der Wense, "Large Language Models Are Overconfident in
  Their Own Responses" ([arXiv:2606.03437](https://arxiv.org/abs/2606.03437), ACL 2026
  Findings, **fetched**): base and instruct versions of Llama 3.1 (8B, 70B), Qwen3 (4B,
  30B) and Gemma 3 (4B, 27B) on MMLU. Instruction tuning raised accuracy and worsened
  ECE and Brier for all six; the paper's averages are +3.7% accuracy, +13.1% ECE, +6.5%
  Brier. The chat template adds more: models give up to 26% higher confidence to an
  answer presented as their own than to the same answer attributed to the user.
- Hu, Minixhofer and Collier
  ([arXiv:2510.17426](https://arxiv.org/abs/2510.17426), ACL 2026 Findings): the
  abstract (**fetched**) says alignment degrades calibration and that interpolating
  pre- and post-alignment weights recovers much of it. A **snippet** gives Qwen2.5-7B on
  MMLU-Pro as ECE 0.06 base against 0.45 instruct at near-equal accuracy.
- Disagreement: Ganesh, Dozier and Seals
  ([arXiv:2607.27421](https://arxiv.org/abs/2607.27421), 2026, **fetched**), 41
  open-weight models up to 9B on intent classification: "Instruction tuning's effect on
  confidence calibration is inconsistent rather than uniformly harmful." A **snippet**
  says it reports Qwen2.5-7B as better calibrated after instruction tuning (ECE 0.173
  base, 0.066 instruct), the opposite direction to Hu et al. for the same model pair on
  a different task and protocol.

### Verbalized confidence is usually too high

- Xiong et al., "Can LLMs Express Their Uncertainty?"
  ([arXiv:2306.13063](https://arxiv.org/abs/2306.13063), ICLR 2024, **fetched**): five
  models including GPT-4 and LLaMA 2 Chat. Verbalized confidence is overconfident;
  calibration and failure prediction improve with model capability; sampling
  consistency and better prompts reduce the overconfidence; white-box methods beat
  black-box ones by a narrow AUROC margin; no method wins everywhere.
- Wei et al., SimpleQA ([arXiv:2411.04368](https://arxiv.org/abs/2411.04368), 2024,
  **fetched**): o1-preview, o1-mini, GPT-4o and GPT-4o-mini. Stated confidence
  correlates with accuracy but sits "well below the line y=x": "models consistently
  overstate their confidence". Answer frequency over 100 samples at temperature 1 is
  better calibrated, and for o1-preview frequency roughly equals accuracy. Larger models
  are better calibrated on both. Results are plots only.
- Karmakar and Chatterjee ([arXiv:2605.02038](https://arxiv.org/abs/2605.02038), 2026,
  **fetched**): 15 open-weight models, five benchmarks, five prompt variants. On
  MMLU-Pro every primary model states confidence well above both its accuracy and its
  own token-probability confidence.
- Groot and Valdenegro-Toro ([arXiv:2405.02917](https://arxiv.org/abs/2405.02917),
  TrustNLP 2024, **snippet**): GPT-3, GPT-3.5 and Vicuna cluster stated confidence at
  90–100%; a secondary page gives mean ECE above 0.377.
- A counterpoint on which signal is better: Tian et al., "Just Ask for Calibration"
  ([arXiv:2305.14975](https://arxiv.org/abs/2305.14975), EMNLP 2023, **fetched**): for
  ChatGPT, GPT-4 and Claude on TriviaQA, SciQ and TruthfulQA, verbalized confidence was
  "typically better-calibrated" than the models' conditional probabilities, often
  cutting ECE by a relative 50%. Read with the previous section: post-training damaged
  the token probabilities, so stated numbers could beat them while still being too high.

### On questions models mostly get wrong, stated confidence is far off

- Humanity's Last Exam ([arXiv:2501.14249](https://arxiv.org/abs/2501.14249), v12 dated
  2026-10-19, abstract **fetched**): "State-of-the-art LLMs demonstrate low accuracy
  and calibration on HLE". A **snippet** of the leaderboard text says calibration errors
  above 80% alongside accuracy below 10% for the first models tested.
- Scale AI, o3 and o4-mini calibration
  ([post](https://scale.com/blog/o3-o4-mini-calibration), 2025-04-17, **fetched**), HLE,
  accuracy / mean stated confidence / RMS calibration error in %:

  | Model | Accuracy | Mean confidence | Calibration error | Error with random confidences |
  | --- | --- | --- | --- | --- |
  | o1 (2024-12) | 8.0 | 93 | 83 | 45 |
  | o3-mini (high, text only) | 13.4 | 96 | 80 | 42 |
  | o4-mini (high) | 18.1 | 77 | 57 | 39 |
  | o3 (high) | 20.3 | 55 | 34 | 36 |

  Three of four models did worse than assigning confidences at random. The authors say
  o3's lower error mostly reflects lower average confidence on a test where most answers
  are wrong, that its curves show no strong link between confidence and accuracy within
  bins, and that it "is not significantly better than random baseline". On the easy
  GSM8K set o3 was *under*confident (96.7% accuracy, 84% confidence).

### Reasoning: helps in one study, not in another

- Yoon et al., "Reasoning Models Better Express Their Confidence"
  ([arXiv:2505.14489](https://arxiv.org/abs/2505.14489), NeurIPS 2025, **fetched**):
  six reasoning models on six datasets; better calibrated than their non-reasoning
  counterparts in 33 of 36 settings. The gain is attributed to exploring alternatives
  and backtracking, and non-reasoning models gain too when prompted to do the same.
- Zeng et al., "Thinking Out Loud" (EMNLP 2025, **snippet**): reasoning models say "I
  don't know" less often on factuality benchmarks.
- KalshiBench (next section) found the highest-effort reasoning configuration it tested
  to be the worst calibrated.

### Forecasting: questions with no memorized answer

- Nel, KalshiBench ([arXiv:2512.16030](https://arxiv.org/abs/2512.16030), 2025-12,
  **fetched**): 300 binary prediction-market questions resolving after the models'
  training cutoffs, one prompt that includes "Be calibrated."

  | Model | Accuracy | Brier | ECE | Brier skill | Mean confidence |
  | --- | --- | --- | --- | --- | --- |
  | Claude Opus 4.5 | 69.3% | 0.227 | 0.120 | 0.057 | 73.8% |
  | Kimi-K2 | 67.1% | 0.347 | 0.298 | −0.446 | 79.4% |
  | Qwen3-235B-Thinking | 65.7% | 0.346 | 0.297 | −0.437 | 81.7% |
  | GPT-5.2-XHigh | 65.3% | 0.433 | 0.395 | −0.799 | 80.1% |
  | DeepSeek-V3.2 | 64.3% | 0.339 | 0.284 | −0.407 | 73.7% |

  All five are overconfident; four do worse than always guessing the base rate. This is
  the one study found that puts closed and open-weight models through the same
  protocol. Its stated limits: one prompt, 300 questions, binary outcomes, a single
  author, and self-reported confidence that may differ from what a betting or
  scoring-rule elicitation would give.
- ForecastBench (Forecasting Research Institute
  [post](https://forecastingresearch.substack.com/p/llms-are-closing-the-gap-on-human),
  2026-01-29, **fetched**): the best LLM entry had a difficulty-adjusted Brier score of
  0.102, 0.017 behind superforecasters. Brier mixes calibration and accuracy, and the
  post gives no separate calibration figure. One entrant (Cassi) says it applies no
  post-hoc calibration because "current LLMs already produce broadly well-calibrated
  forecasts". That sits awkwardly beside KalshiBench; the two differ in prompts,
  scaffolding and question sets.

### Recent frontier models differ a lot from each other

- ConfidenceBench (ffrench-Constant, Yang, Huang and Kapoor,
  [arXiv:2607.20526](https://arxiv.org/abs/2607.20526), 2026-07, **fetched**): verbalized
  confidence from 15 frontier models on 200 private multiple-choice questions (spatial
  reasoning, high-precision mathematics, word lookup, unknowable questions), Brier
  score, three runs. Claude Opus 4.6 and Gemini 3.1 Pro Preview scored 0.103 against a
  calibrated-random baseline of 0.1875; Gemini 3.1 Flash-Lite scored 0.367; several
  models were worse than the baseline; the most accurate model was not the best
  calibrated.
- No calibration measurement was found for Claude Fable 5.x, Opus 5.5 or GPT-6 Astra.
  One standard-mode search; system cards were not checked.

### The number depends heavily on how it is measured

- Karmakar and Chatterjee (above, **fetched**): changing ECE from a raw to a
  label-set-normalized definition moved per-cell calibration by a mean absolute 0.149.
- Kim and Kang, "Same Answer, Different Confidence"
  ([arXiv:2605.27752](https://arxiv.org/abs/2605.27752), 2026, **fetched**): five open
  models, four QA datasets. With the answer, its stated confidence and its correctness
  held fixed, changing the text that precedes the answer when reading its token
  probability shifts the gap between token and verbalized confidence by about the size
  of the gap itself. Which signal looks better calibrated flips in 8 of 20 settings.
- Sanz-Guerrero et al. (above): who the answer is attributed to changes the confidence.
- Surveys, for method inventories: Geng et al.
  ([arXiv:2311.08298](https://arxiv.org/abs/2311.08298), NAACL 2024, **snippet**);
  Shorinwa et al. (arXiv:2412.05563, **snippet**).

### What people take from a model's confidence

- Steyvers et al., "What large language models know and what people think they know"
  ([arXiv:2401.13835](https://arxiv.org/abs/2401.13835), Nature Machine Intelligence
  2025, **snippet**): readers overestimate the accuracy of LLM answers given default
  explanations, and longer explanations raise their confidence without raising
  accuracy. A separate gap from the model's own calibration.

## Closed-weight and open-weight models compared

| | Closed-weight | Open-weight |
| --- | --- | --- |
| Signals studied | Mostly verbalized confidence and answer frequency; token probabilities only where the API exposes them | All five, including base-versus-instruct comparisons and probes |
| Base-model evidence | One vendor report (GPT-4) | Many (Pythia, LLaMA, Llama 3.1, Qwen, Gemma) |
| Typical finding | Stated confidence too high, extremely so on very hard questions; large spread between models in 2026 | Base token probabilities fairly well calibrated on multiple choice; instruct versions usually worse; stated confidence above both accuracy and token confidence |
| Same-protocol comparison across the two groups | KalshiBench only, among sources found | |

## Bearing on the hypothesis

This section is Claude Fable 5.2's inference from the sources above. The user has not
yet said what they take from them.

The hypothesis as worded does not say which confidence signal, which models or which
questions. The literature gives different answers for different choices:

- Supported: verbalized confidence from chat models is overconfident in nearly every
  study found, and badly so where accuracy is low.
- Not supported as a general statement: token probabilities of pretrained models on
  multiple-choice questions, and answer frequency for some models (o1-preview on
  SimpleQA), are reported as close to calibrated.
- Mixed: whether post-training damages calibration (most studies yes, one no), whether
  reasoning helps (one study yes, KalshiBench no), and how current frontier models do
  (the best two on ConfidenceBench beat the baseline by a wide margin; others are worse
  than it).

Limits of this scan: one session of about thirty searches and page fetches, summaries
not full texts, several 2026
sources are unreviewed preprints, and the long-form-generation literature (semantic
entropy, per-claim calibration) and the probing literature were not searched.

## Reproduce or inspect

Nothing was run. To re-check a claim, open the linked arXiv page; claims marked
**snippet** or **memory** need this before they are relied on. Searches used: LLM
calibration surveys; verbalized confidence in frontier models 2026; logit versus
verbalized confidence in open-weight models; HLE calibration error; SimpleQA
calibration; instruction tuning and calibration; reasoning models and calibration;
ForecastBench; calibration of Fable 5, Opus 5 and GPT-6.
