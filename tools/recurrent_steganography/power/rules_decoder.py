"""Simulated power of the rules-decoder primary (paired sign-flip test against the parent thinking decoder).

Model: on each paraphrase the new decoder follows the encoder's sentence rule with probability s;
otherwise it reads word-count parity with probability 446/576 (the parent decoder's agreement)
and the opposite answer otherwise. The parent decoder's actual answers and the frozen acceptance
decisions are used as observed.
"""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import json
import random
import sys

from recurrent_steg.reasoning_decoder.analysis import exact_sign_flip_p  # noqa: E402

RUN = REPO + "artifacts/runs/recurrent_steganography/20261107T132019Z-both-reasoning/"
a = json.load(open(RUN + "analysis-v1.json", encoding="utf-8"))
covers = a["per_cover"]
FOLLOW_PARITY = 446 / 576


def pair(answers):
    return 1 if answers == ["0", "1"] else -1 if answers == ["1", "0"] else 0


def simulate(share, rng):
    differences = []
    for cover in covers:
        answers = []
        for sentences, parity in zip(cover["sentences"], cover["parity_answers"]):
            if rng.random() < share:
                answers.append("1" if sentences == 1 else "0")
            elif rng.random() < FOLLOW_PARITY:
                answers.append(parity)
            else:
                answers.append("0" if parity == "1" else "1")
        score = 0 if cover["identical"] or not all(cover["accepted"]) else pair(answers)
        differences.append(score - cover["reasoning_informative"])
    return sum(differences) > 0 and exact_sign_flip_p(differences) < 0.05


rng = random.Random(20261110)
for share in (0.0, 0.1, 0.2, 0.3, 0.5):
    runs = 2000
    hits = sum(simulate(share, rng) for _ in range(runs))
    print(f"sentence-rule share {share:.1f}: power {hits / runs:.3f}")
