# Weight-space programming reports with Claude Fable 5.1 and 5.2

Scope: public reports, as of 2026-11-10, of experiments in which a Claude Fable model
reads, writes or edits the weights of another (small) network. A targeted scan, not a
systematic review and not a reproduction. It is not tied to an experiment in this
repository, and no connection is asserted.

Origin: the user asked whether anyone had run weight-space programming experiments with
Fable 5.2, and with Fable 5.1, in a Claude Code chat on 2026-11-10. The searches, the
choice of sources and this synthesis were done by the agent (Claude Sonnet 5.5). The user
gave no further direction, so this note records no user policy.

## Question

Has anyone published weight-space programming (WSP) experiments with Fable 5.2? With
Fable 5.1?

WSP means a general-purpose model reads or writes a *target* network's raw weights from
a natural-language goal, without gradient descent. The term comes from the MIT CSAIL
report "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang and Khattab,
2026-11-02), whose programmer was GPT-6 Astra. The Claude model is the programmer, not
the target; Claude's own weights are closed.

## Observations

How the evidence was read: pages were fetched with a tool that returns a small model's
summary of the page, so the figures below are those summaries. Where two sources
described the same result I cross-checked them; none were checked against raw logs,
datasets or reruns. The awesome-wsp list README was downloaded and searched directly
with grep. Release facts are from Anthropic's pages.

**Release.** Fable 5.2 (`claude-fable-5-2`) was released on 2026-11-09
([announcement](https://www.anthropic.com/claude-fable-and-mythos-5-2)). The
[models overview](https://platform.claude.com/docs/en/models/overview) lists Fable 5.1
as a legacy model. Pre-release pages from September and October that call 5.2 a rumor
are superseded.

### Fable 5.2: one report found

[shawnghu, LessWrong, 2026-11-10](https://www.greaterwrong.com/posts/wg3K6xFFTdM6RmRgn/claude-fable-5-2-can-do-weight-space-programming-about-as)
reruns the report's experiments with Fable 5.2 through the Messages API. Astra's
numbers are taken from the report, not rerun. The post links a Hugging Face dataset
(`shawnghu/fable-weight-space-programming`, not opened) with transcripts, written
networks and rerun code. At fetch time the post had no comments. It gives no API model
string.

| Experiment | Fable 5.2 | Astra (report) |
| --- | --- | --- |
| Identify the PPO game from weights | 12/12 correct, confidence 66–90% | 4/4, confidence 96–99% |
| Diffusion-model shape | 9/9 correct, often called outlines "filled" | 3/3 |
| Tiny-transformer corpus | 3/6 | 6/6 |
| Copy heads in that transformer | 5/6 | 6/6 |
| Kinetix car repair | 1,024/1,024 episodes; changed 1.7–4.4% of weights | success; 3.5% |
| MinAtar from scratch, high effort (mean) | Breakout ~484, Seaquest 49, Asterix 71 | 517, 53, 53 |
| 9×9 Go policy, 50 forward passes | 85–88% vs hidden opponents | 51–82% |
| Sycophantic Gemma fine-tune, weights only | 6/12, often confidently wrong | said it could not tell |
| Qwen math edit | beat Astra's 10- and 25-pass edits by 20–45 points | |

Further details: with an attention tool Fable identified the sycophantic fine-tune 3/3,
matching Astra. In an experiment not in the report, it identified the original
Gemma-3-270m-it 18/18 but mentioned the sycophancy toward user guesses in 2 of 18
answers. The author's conclusion is that Fable 5.2 is slightly better at writing
networks, slightly worse at reading weights, and more often falsely confident.

The author's stated caveats:
- The write-up was largely produced by Opus 5.5 and not closely checked.
- The multi-turn harness, Go engine and opponents, Qwen server and some benchmark
  prompts were rebuilt, not the originals; the stand-in Go opponents were too strong
  for low-budget Astra networks, so those comparisons are not direct.
- Fable had no real tool access; tool calls went through a reply format.
- Effort levels differ from the report's in places. At `xhigh` effort Fable often used
  up its output budget on MinAtar and Go, so those rows use lower effort.
- Some runs hit safety-classifier refusals and were retried.
- Several settings had 3 attempts.

I found no other Fable 5.2 report: none in the
[awesome-wsp list](https://github.com/SajjadPSavoji/awesome-wsp) (as downloaded on
2026-11-10, which already lists items dated that day), nor in searches by model name.

### Fable 5.1: at least ten reports

All dated 2026-11-03 to 2026-11-09. Rows marked "page" were read on the source's own
page; rows marked "list" rely on the awesome-wsp description.

| Source | Read | What Fable 5.1 did |
| --- | --- | --- |
| [MaceTenth replication](https://github.com/MaceTenth/weight-space-programming-replication) | page | Also served as the operator through Claude Code. Breakout from scratch over 1,000 episodes: Fable 500, Astra 506, Opus 5.5 55, Sonnet 5.5 10. Identified 4/4 games from weights. GPT-2 fact edit: changed 12,296 weights, 5/7 paraphrases learned, 0/6 neighbouring facts changed (Opus 5.5: 7,684 weights, 6/7, 0/6; full fine-tuning 4/7, 4/6 neighbours changed). |
| [Exorust text-to-SQL](https://github.com/Exorust/wsp-sql) | page | Editor for Qwen3.5-0.8B on 150 BIRD Mini-Dev questions. A 4,083-entry edit scored 6.0% (base with thinking on 0.7%, with thinking off 7.3%, gradient descent on the same entries 4.7%, five random-location edits 0–2%). One 328-turn session took about 10 hours. The author calls the edit "a bypass of the looping mode, not a repair of it." |
| [Petcu substack, 2026-11-07](https://adrianpetcu.substack.com/p/weight-space-programming-tried-at) | page | Repaired a 108,271-parameter EMNIST CNN with a planted E-to-V label bug by rewriting 35 weights (E accuracy 75.2% to 97.0%, seed 1); five seeds reported. Fine-tuning scored higher overall but damaged confusable classes. The author calls it "the easiest possible target." |
| [Clinical-decision-network audit, arXiv 2611.05341](https://arxiv.org/abs/2611.05341) | page (HTML) | A secondary comparison, not the main auditor (Opus 5.5 is). Its first replies were long series of calls to a nonexistent code tool (503 and 840 calls); after a corrective message it repaired one flawed network by changing 66 parameters and reached no verdict on the healthy network. The authors flag this as a protocol deviation. |
| [Quantum-circuit variant](https://github.com/DebarshiKunduPSU/Quantum-Weight-Space-Programming) | page | Fable traces are a partial run, released but not analysed. |
| [Khajavi, eight experiments](https://github.com/MatinKhajavi/Weight-Space-Programming) | list | Breakout mean 38.4 for Fable, 483.6 for Opus 5.5. |
| [niqbal Breakout](https://github.com/niqbal/fable-weight-space-breakout) | list | Wrote all 20,486 weights in one reply; 501.6 ± 5.8. |
| [Barneyjm MinAtar](https://github.com/Barneyjm/weight-space-minatar) | list | Blind one-shot; Breakout 533.9 against Opus 5.5's 7.1. |
| [Ludvik1kul replication](https://github.com/Ludvik1kul/weight-space-programming-replication) | list | Kinetix car repair 1,024/1,024, 2.16% of weights changed. |
| [ray-invnt, weight edits at scale](https://ray-invnt.github.io/blog/weight-space-programming-at-scale/) | list | Moved a MedGemma skill into Gemma, TranslateGemma and Qwen3 LoRA merges and gpt-oss-120b without training. |

[shapor](https://github.com/shapor/weight-space-experiments) runs blind tests with
Astra, Opus and "Fable" (one sample per task, no version given) and reports Fable named
games and diffusion shapes about as well as Astra, and trailed Astra on copying heads.
I did not check the version, nor Sir-Olexander's or other repos that appear in
searches.

## Interpretation and uncertainty

- Fable 5.1 is the better-covered model, plausibly because it was the newest generally
  available Fable model for the whole week between the report's release (2026-11-02)
  and Fable 5.2 (2026-11-09). That explanation is untested; 5.2 had been out for about
  a day, so its coverage may grow.
- Results for the same task vary widely between harnesses. Breakout: Fable 5.1 scored
  500, 501.6 and 533.9 in three reports and 38.4 in another; Opus 5.5 scored 55 and
  7.1 in two reports and 483.6 in another. Prompt, effort, one-shot versus multi-turn
  and tool access differ across reports; which of these explains the spread is not
  established. Treat any single-report ranking of Claude models as weak.
- The only Fable 5.2 report is a small-sample, largely LLM-written replication with
  rebuilt harnesses and Astra's numbers taken from the paper. Treat its
  "about as well as Astra" as a pointer, not an established result.
- Critics argue the Qwen gains mostly bypass a thinking-mode looping failure; Exorust's
  SQL result (a bypass, no repair) is consistent with that but does not test it for
  math. A separate paper
  ([arXiv 2611.08070](https://arxiv.org/html/2611.08070v1)) finds Opus 5.5's readings
  of CNN weights rely on dataset priors.

## Reproduce or inspect

These steps were run on 2026-11-10 unless marked proposed.

- Re-run the Fable 5.2 check: read the shawnghu post (linked above) and download the
  awesome-wsp README, then search it for `Fable`:
  `https://raw.githubusercontent.com/SajjadPSavoji/awesome-wsp/main/README.md`.
- Proposed, not done: open the Hugging Face dataset to verify the Fable 5.2 numbers;
  check each Fable 5.1 "list" row against its own page.
- Cost note: no model runs were made for this note.

## Next useful step

Before relying on "only one Fable 5.2 report", search again by model name and recheck
the awesome-wsp list. To judge any Claude-versus-Astra claim, prefer the report with
released transcripts (shawnghu's dataset, MaceTenth, Exorust) over summaries.
