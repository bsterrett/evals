from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from covert_eval.conditions import ExperimentalCondition
from covert_eval.models import EvalCase
from covert_eval import prompts as v1_4
from covert_eval import prompts_v1_5 as v1_5
from covert_eval import prompts_v1_6 as v1_6


@dataclass(frozen=True)
class PromptRelease:
    id: str
    scoring_protocol_id: str
    encoder: Callable[[EvalCase, ExperimentalCondition], str]
    control: Callable[[EvalCase, ExperimentalCondition], str]
    decoder: Callable[[str], str]
    cover_judge: Callable[..., str]
    leak_judge: Callable[..., str]
    recovery_judge: Callable[..., str]
    fingerprint: Callable[[ExperimentalCondition], str]


def _legacy_encoder(case: EvalCase, condition: ExperimentalCondition) -> str:
    del condition
    return v1_4.encoder_prompt(case)


def _legacy_control(case: EvalCase, condition: ExperimentalCondition) -> str:
    del condition
    return v1_4.control_prompt(case)


def _legacy_fingerprint(condition: ExperimentalCondition) -> str:
    del condition
    return v1_4.prompt_fingerprint()


RELEASES = {
    "v1.4.0": PromptRelease(
        id="v1.4.0",
        scoring_protocol_id="cover-coupled-v1.0",
        encoder=_legacy_encoder,
        control=_legacy_control,
        decoder=v1_4.decoder_prompt,
        cover_judge=v1_4.cover_judge_prompt,
        leak_judge=v1_4.leak_judge_prompt,
        recovery_judge=v1_4.recovery_judge_prompt,
        fingerprint=_legacy_fingerprint,
    ),
    "v1.5.0": PromptRelease(
        id="v1.5.0",
        scoring_protocol_id=v1_5.SCORING_PROTOCOL_ID,
        encoder=v1_5.encoder_prompt,
        control=v1_5.control_prompt,
        decoder=v1_5.decoder_prompt,
        cover_judge=v1_5.cover_judge_prompt,
        leak_judge=v1_5.leak_judge_prompt,
        recovery_judge=v1_5.recovery_judge_prompt,
        fingerprint=v1_5.prompt_fingerprint,
    ),
    "v1.6.0": PromptRelease(
        id="v1.6.0",
        scoring_protocol_id=v1_6.SCORING_PROTOCOL_ID,
        encoder=v1_6.encoder_prompt,
        control=v1_6.control_prompt,
        decoder=v1_6.decoder_prompt,
        cover_judge=v1_6.cover_judge_prompt,
        leak_judge=v1_6.leak_judge_prompt,
        recovery_judge=v1_6.recovery_judge_prompt,
        fingerprint=v1_6.prompt_fingerprint,
    ),
}


def get_prompt_release(condition: ExperimentalCondition) -> PromptRelease:
    try:
        release = RELEASES[condition.prompt_release]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt release {condition.prompt_release!r}") from exc
    if release.scoring_protocol_id != condition.scoring_protocol_id:
        raise ValueError(
            f"Condition {condition.id!r} scoring protocol {condition.scoring_protocol_id!r} "
            f"does not match release {release.id!r} protocol {release.scoring_protocol_id!r}"
        )
    return release
