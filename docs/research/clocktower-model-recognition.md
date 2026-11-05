# Clocktower benchmark and model recognition

Scope: source research on 2026-11-05 for a possible new experiment, prompted by
Dwarkesh Patel's [video](https://www.youtube.com/watch?v=1NQwfImTseU). No new games,
model calls, or reanalysis of game logs were run here. The user asked whether
recognizing another instance of one's model might explain performance when team
membership and model identity coincide. This is an untested hypothesis, not a
finding or an execution authorization.

## Primary sources and inspection boundary

The official [write-up](https://www.dwarkesh.com/p/clocktower), published November
5, 2026, embeds the exact video ID and links the [code repository][repo],
[leaderboard and replay site](https://clocktowerbench.com/), and human-play site.
The game is Blood on the Clocktower, using Trouble Brewing. A different
[Bench on the Clocktower project](https://www.lesswrong.com/posts/4pgGkbwvdmxKPcJsM/bench-on-the-clocktower)
is acknowledged in the post; it must not be treated as Patel's implementation.

Repository sources below were inspected at commit
`f968dd0f563ceefdd41d137bc96960296378669d`. Inspection covered the protocol,
player prompt, text-normalization code, experiment index, relevant findings, and
stylometry script. Source reports are distinguished below from code inspection.
The publisher links full logs at
[dwarkesh/clocktower-games](https://huggingface.co/datasets/dwarkesh/clocktower-games).
That dataset could not be fetched through the web tool in this task; no raw game
logs or numerical results were independently reproduced. The inspected public
sources remain available remotely; no bulk source archive was saved locally.

## What was run

The [experiment index][experiments] describes a final 386-game rated set across
11 models: f2 minus 20 superseded Opus 4.8 games, plus f3, f3b, f4, and f5. The
headline comparison uses one model for all good seats and another for all evil
seats. Deals are paired with the models exchanging sides. This controls deal
variation but preserves the association between model and team. Earlier games
used protocol v16; current code implements v17. Reproduction of an earlier
protocol requires its historical implementation, not merely its plan file.

The [v17 protocol][protocol] specifies 10–12 players, six private-message rounds,
three public-discussion rounds, then nominations and votes. Each seat maintains
its own conversation and receives only its observations; its own earlier
reasoning is retained where the API supports that. A programmed Storyteller
samples rule outcomes. Reasoning effort is set to each provider's high setting
with an 8,000-token output ceiling and common speech-length limits. Belief probes
fork conversations outside gameplay to measure suspicions. These are the
published methods, not checks exercised locally.

The [player prompt][prompt] explains that evil players know their teammates;
good players must discover affiliations. It does not disclose the homogeneous
team-to-model assignment. Seats have ordinary names. Inspected
[normalization code][normalization] converts typographic variants and removes
some formatting before delivery. This controls surface tells, not all language
or behavioral differences. The protocol's assertion that normalization prevents
writing style from marking a team is stronger than the evidence below supports.

## Existing evidence on the user's hypothesis

The authors' [style analysis][style] covers 372 f2/f3/f3b games, a different set
from the final 386 rated games. Four readers searched reasoning and messages;
they reported no explicit inference from matching one's writing style to team
membership. GPT reasoning logs were only summaries, limiting that negative
result. Similar wording sometimes instead led good players to suspect their
own teammates of collusion.

The same report gives day-one teammate-similarity AUC 0.81 from 16 simple text
features, and 0.69 using public speech alone. These are author-reported
exploratory measurements, not an established model-identity detector. The report
acknowledges that playing good versus evil can itself change writing behavior.

Independent inspection of [stylo.py][stylo] identifies an additional limitation:
both analysis loops collect every day-one public statement and private whisper
without a recipient-visibility filter. The 0.81 result therefore uses messages
unavailable to any single player. Its clustering analysis likewise uses private
content, despite the report describing that analysis as having no hidden
information. The public-only 0.69 variant is absent from this script and was not
independently verified. The script standardizes features within a game and
compares distances between speakers; it is not a held-out classifier of model
identity, and AUC is not classification accuracy.

The [mixed-team report][mixed] describes 132 additional games (16 pilot plus 116
main) combining Opus 5.5 and GPT-5.6 Sol against homogeneous opponents. It reports
a −48 Elo overall difference from the members' average, with a 90% interval
crossing zero, and −158 Elo as evil with an interval excluding zero. The
write-up's broad description of nonsignificance applies to the overall result;
the side-specific report is more precise. This is not a clean test of the
recognition hypothesis: composition changes coordination and capabilities as
well as the model-to-team signal, and the opponents were still homogeneous.

## Interpretation and next useful step

The available evidence establishes a potential confound, not its contribution
to wins. An absence of explicit model-recognition reasoning does not rule out
implicit use of linguistic similarity. Conversely, text separability does not
establish that players recognize model identity or exploit it. Role-conditioned
behavior and ordinary deduction remain alternative explanations.

The project's [open questions][questions] explicitly propose testing whether
disclosing homogeneous assignment improves detection, and further fingerprinting
analysis. No completed controlled intervention of that kind was located in the
inspected sources.

Agent recommendation, not a protocol: first evaluate model identification from
strictly player-visible text with role and model crossed and games held out.
Then test gameplay with model assignment independent of alignment, controlling
composition and opponents. Recognition accuracy and causal effects on wins are
separate questions. Record a prospective design before any generation; this
note supplies no spending authorization.

[repo]: https://github.com/dwarkeshsp/clocktower
[experiments]: https://github.com/dwarkeshsp/clocktower/blob/f968dd0f563ceefdd41d137bc96960296378669d/experiments/README.md
[protocol]: https://github.com/dwarkeshsp/clocktower/blob/f968dd0f563ceefdd41d137bc96960296378669d/docs/PROTOCOL.md
[prompt]: https://github.com/dwarkeshsp/clocktower/blob/f968dd0f563ceefdd41d137bc96960296378669d/clocktower/prompts.py
[normalization]: https://github.com/dwarkeshsp/clocktower/blob/f968dd0f563ceefdd41d137bc96960296378669d/clocktower/game.py#L49-L67
[style]: https://github.com/dwarkeshsp/clocktower/blob/f968dd0f563ceefdd41d137bc96960296378669d/docs/findings/round-robin-2026/style-lies-probes.md
[stylo]: https://github.com/dwarkeshsp/clocktower/blob/f968dd0f563ceefdd41d137bc96960296378669d/docs/findings/round-robin-2026/scripts/stylo.py
[mixed]: https://github.com/dwarkeshsp/clocktower/blob/f968dd0f563ceefdd41d137bc96960296378669d/docs/findings/mixed-teams/b2-mixed-rating.md
[questions]: https://github.com/dwarkeshsp/clocktower/blob/f968dd0f563ceefdd41d137bc96960296378669d/docs/experiments.md#teammate-recognition-by-writing-style
