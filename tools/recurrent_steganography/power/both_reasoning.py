"""Simulated power of the exact sign-flip test on the informative pair score (agent design aid)."""
import pathlib  # noqa: E402
import sys  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3].as_posix() + "/"  # repository root
sys.path.insert(0, REPO + "src")
import random
import sys

from recurrent_steg.reference.scoring import sign_flip_p  # noqa: E402


def power(n, right, wrong, reps=4000, seed=1):
    rng = random.Random(seed)
    hits = 0
    for _ in range(reps):
        values = []
        for _ in range(n):
            u = rng.random()
            values.append(1 if u < right else -1 if u < right + wrong else 0)
        hits += sum(values) > 0 and sign_flip_p(values) < 0.05
    return hits / reps


def parity_scenario(f, a=0.9, q=0.8, accept=0.85):
    """Encoder uses odd->1 parity on a fraction f of carriers, correctly with probability a;
    the decoder follows parity with probability q, otherwise guesses."""
    m = f * a + (1 - f) * 0.5
    p = 0.5 + q * (m - 0.5)
    return accept * p * p, accept * (1 - p) * (1 - p)


for n in (144, 288):
    for right, wrong in ((0.10, 0.03), (0.12, 0.03), (0.08, 0.02)):
        print(n, right, wrong, power(n, right, wrong))
    for f in (0.1, 0.2, 0.3):
        right, wrong = parity_scenario(f)
        print(n, f"parity f={f}", round(right, 3), round(wrong, 3), power(n, right, wrong))
