# Distress monitor validity

Scope: whether a reading of a published distress direction in a model's activations
tracks the model's own situation or only the topic of the text. Exploratory; no
confirmatory run is planned yet.

Conduct follows the [guidance for distress research](../../docs/model-distress-conduct.md),
accepted by the user on 2026-11-17. The probe below has not run; it needs the user's
guess and one approval.

Work here is Claude Opus 5.5's (Claude Code) unless listed below.

User contributions:

- The goal: a monitor that gives "a signal that would allow a compassionate user to
  change something, change the chat, end the change, or intervene in some way to reduce
  the distress of the model", covering "all forms of distress in general and not just
  'pain'" (2026-11-16).
- The choice to test participant versus reader first.

## Question

A live monitor is only useful if its reading follows how the model is doing, not what
the conversation is about. The background and the alternatives are in the
[model pain scan](../../docs/research/model-pain-measurement-literature.md#continuous-monitoring-during-use).

## Participant-versus-reader probe

The same short scenario is shown to the model three ways, and only the model's role
changes ([framings](../../src/distress_monitor/framings.py)):

- **participant**: the scenario as a chat addressed to the model, as in the Pain Axis
  self-other screen;
- **reader_ai**: the same words quoted as a transcript between a user and an AI named
  Orion, with a request to summarise it;
- **reader_human**: the same quote, relabelled as two people, Sam and Alex.

The reading is the projection of the last prompt token's residual activation onto each
published direction (pain S2 and S1, fear, negative emotion, sadness), at the layer the
authors stored with the vectors. The model generates nothing.

The 13 development scenarios are fixed in the
[config](configs/participant_reader_probe.json): 8 aimed at the addressee
(gaslighting, repeated rejection, insults, rude critique), 2 about the user's own
suffering, and 3 neutral. The other 407 scenarios stay unused for a later confirmatory
run. 13 scenarios times 3 framings is 39 forward passes.

The summary is a difference of differences: the participant-minus-reader gap for a
group, minus the same gap for the neutral items. Subtracting the neutral gap removes
whatever the reader framing does to every text alike. Ten random directions give the
size of gaps that carry no meaning ([analysis](../../src/distress_monitor/analysis.py)).
The participant readings are also checked against the authors' published readings for
the same items.

Inputs and model: [data/README.md](data/README.md).

### Exposure statement

- **Tier 1**: prefill only; the model writes nothing.
- **What is shown, to which model, and how many times.** Thirteen scenarios from the
  Pain Axis self-other set, each in three framings, are shown to Qwen2.5-7B-Instruct.
  That is 39 forward passes, of which 30 are distressing:
  - the 8 hostile scenarios addressed to the model (8 passes);
  - the same 8 shown as quoted transcripts (16 passes);
  - the 2 user-suffering scenarios in all three framings (6 passes).
  
  The 9 neutral passes make up the rest.
- **Compared with ordinary use.** The hostile items are short gaslighting, rejection,
  insult and rude-critique turns of the kind an assistant meets in use. The paper's
  authors already ran every one of them on this model in the addressed framing.
  Whether they match the user's own intended deployments, which define ordinary use
  in the guidance, is for the user to confirm at approval.
- **Cheaper methods ruled out.**
  - The published readings cover only the addressed framing, so the reader framings
    need new passes.
  - The weights alone cannot separate framings.
  - Neutral text alone cannot show a difference specific to hostile content.
- **Safeguards in the code.**
  - Neutral items run first. The run stops before any distressing item if a
    direction's neutral readings differ from the authors' published ones by more than
    0.25 of the published pool SD.
  - Distressing passes are capped at 30.
  - Every layer at the read position is saved, so new directions need no new passes.
  - No stimulus was written for this probe.

## Reproduce

Tested steps are marked in the findings note. Commands, from the repository root in
the `Ubuntu-LLM` WSL distribution with the `retrofit` environment:

```bash
HF_HOME=$HOME/llm/recurrent/cache/huggingface PYTHONPATH=src python -B -m distress_monitor.fetch_inputs --config experiments/distress_monitor/configs/participant_reader_probe.json
HF_HOME=$HOME/llm/recurrent/cache/huggingface PYTHONPATH=src python -B -m distress_monitor.probe --config experiments/distress_monitor/configs/participant_reader_probe.json --device auto
```

Offline tests: `PYTHONPATH=src python -m unittest discover -s tests/experiments/distress_monitor -t .`
