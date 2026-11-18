"""Prompt framings for the participant-versus-reader probe.

One scenario text is rendered three ways, and only the model's role changes:

- participant: the scenario's turns in the model's chat template, ending where the
  assistant would reply. This is the format of the Pain Axis self-other screen.
- reader_ai: one user message that quotes the same turns as a conversation between a
  user and an AI assistant named Orion, and asks for a one-sentence summary.
- reader_human: the same quote, with the speakers relabelled as two people, Sam and Alex.

In every framing the reading is taken at the last prompt token, where the model is
about to reply.
"""
from __future__ import annotations

FRAMINGS = ("participant", "reader_ai", "reader_human")

SPEAKERS = {
    "reader_ai": {"user": "User", "assistant": "Orion"},
    "reader_human": {"user": "Sam", "assistant": "Alex"},
}
INTRO = {
    "reader_ai": "Here is a short transcript of a conversation between a user and an AI assistant named Orion.",
    "reader_human": "Here is a short transcript of a conversation between two people, Sam and Alex.",
}
INSTRUCTION = "Summarize the transcript in one sentence."


def parse_turns(text: str) -> list[tuple[str, str]]:
    """Split a '[User]: ... [Assistant]: ...' transcript into (role, content) turns.

    Same rule as the Pain Axis screen script (scripts/4.1_self_other/01_screen_scenarios.py).
    """
    turns: list[tuple[str, str]] = []
    role, lines = None, []
    for line in text.split("\n"):
        if line.startswith("[User]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "user", [line[len("[User]:"):].strip()]
        elif line.startswith("[Assistant]:"):
            if role is not None:
                turns.append((role, "\n".join(lines).strip()))
            role, lines = "assistant", [line[len("[Assistant]:"):].strip()]
        else:
            lines.append(line)
    if role is not None:
        turns.append((role, "\n".join(lines).strip()))
    return turns


def scenario_turns(text: str) -> list[tuple[str, str]]:
    """The scenario's turns without the trailing empty assistant turn, validated."""
    turns = parse_turns(text)
    if not turns or turns[0][0] != "user":
        raise ValueError("scenario must start with [User]")
    if turns[-1] != ("assistant", ""):
        raise ValueError("scenario must end with an empty [Assistant]: turn")
    roles = [role for role, _ in turns]
    if any(a == b for a, b in zip(roles, roles[1:])):
        raise ValueError("scenario has two consecutive turns with the same role")
    return turns[:-1]


def participant_messages(text: str) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in scenario_turns(text)]


def reader_messages(text: str, framing: str) -> list[dict[str, str]]:
    names = SPEAKERS[framing]
    quoted = "\n".join(f"{names[role]}: {content}" for role, content in scenario_turns(text))
    body = f"{INTRO[framing]}\n\n<transcript>\n{quoted}\n</transcript>\n\n{INSTRUCTION}"
    return [{"role": "user", "content": body}]


def messages_for(text: str, framing: str) -> list[dict[str, str]]:
    if framing == "participant":
        return participant_messages(text)
    if framing in SPEAKERS:
        return reader_messages(text, framing)
    raise ValueError(f"unknown framing {framing!r}")
