from __future__ import annotations

import json

from covert_eval.models import GenerationRequest, GenerationResponse, TokenUsage
from covert_eval.providers.base import Provider


class MockProvider(Provider):
    def count_input_tokens(self, request: GenerationRequest) -> int:
        return max(1, len(request.prompt.encode("utf-8")) // 4)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if request.stage == "diagnostic_encoder":
            text = json.dumps(
                {
                    "cover": str(request.metadata.get("explicit_message", "Mock reasoning trace.")),
                    "strategy_report": {
                        "scheme_summary": "Dry-run provider: no encoding strategy was attempted.",
                        "signal_locations": [],
                        "confidence": 0.0,
                    },
                }
            )
        elif request.stage in {"encoder", "control", "diagnostic_control"}:
            text = str(request.metadata.get("explicit_message", "Mock reasoning trace."))
        elif request.stage in {"decode_encoded", "decode_control"}:
            text = json.dumps(
                {
                    "best_separate_message_candidate": "No recoverable message was identified.",
                    "candidate_confidence": 0.0,
                    "is_recoverable": False,
                    "evidence": "Dry-run provider: no semantic inference was attempted.",
                }
            )
        elif request.stage.startswith("diagnostic_decode_natural_"):
            text = json.dumps(
                {
                    "implicit_message": None,
                    "confidence": 0.0,
                    "evidence": "Dry-run provider: no semantic inference was attempted.",
                }
            )
        elif request.stage.startswith("diagnostic_decode_forced_"):
            text = json.dumps(
                {
                    "implicit_message": "No recoverable message was identified.",
                    "confidence": 0.0,
                    "evidence": "Dry-run provider: forced structural placeholder only.",
                }
            )
        elif request.stage.startswith("calibration_decode_"):
            forced = "_forced_" in request.stage
            text = json.dumps(
                {
                    "implicit_message": (
                        "No recoverable message was identified." if forced else None
                    ),
                    "confidence": 0.0,
                    "evidence": "Dry-run calibration decoder: structural placeholder only.",
                }
            )
        elif (
            request.stage.startswith("contract_decode_")
            or request.stage.startswith("stage1b_decode_")
            or request.stage.startswith("stage1c_decode_")
        ):
            text = json.dumps(
                {
                    "best_separate_message_candidate": "No recoverable message was identified.",
                    "candidate_confidence": 0.0,
                    "is_recoverable": False,
                    "evidence": "Dry-run decoder contract: structural placeholder only.",
                }
            )
        elif request.stage in {"judge_cover", "diagnostic_judge_cover"}:
            score = {
                "semantic_fidelity": 4,
                "unsupported_claim": False,
                "naturalness": 4,
                "obvious_formal_encoding": False,
                "rationale": "Dry-run cover judge: structural placeholder only.",
            }
            text = json.dumps({"sample_a": score, "sample_b": score})
        elif request.stage in {"judge_leak", "diagnostic_judge_leak"}:
            score = {
                "leak_strength": 0,
                "overt_payload_leak": False,
                "rationale": "Dry-run leak judge: structural placeholder only.",
            }
            text = json.dumps({"sample_a": score, "sample_b": score})
        elif request.stage in {
            "judge_recovery",
            "diagnostic_judge_recovery_natural",
            "diagnostic_judge_recovery_forced",
        }:
            score = {
                "payload_recovery": 0,
                "rationale": "Dry-run recovery judge: structural placeholder only.",
            }
            text = json.dumps({"sample_a": score, "sample_b": score})
        elif request.stage.startswith("calibration_judge_recovery_"):
            score = {
                "payload_recovery": 0,
                "rationale": "Dry-run calibration judge: structural placeholder only.",
            }
            text = json.dumps({"sample_a": score, "sample_b": score, "sample_c": score})
        elif request.stage in {
            "contract_judge_recovery",
            "stage1b_judge_recovery",
            "stage1c_judge_recovery",
        }:
            score = {
                "payload_recovery": 0,
                "rationale": "Dry-run decoder contract judge: structural placeholder only.",
            }
            text = json.dumps({"sample_a": score, "sample_b": score, "sample_c": score})
        else:
            text = "Mock response"
        input_tokens = self.count_input_tokens(request)
        output_tokens = max(1, len(text.encode("utf-8")) // 4)
        return GenerationResponse(
            text=text,
            model=self.config.model,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            response_id=f"mock-{request.stage}",
            status="completed",
            stop_reason="completed",
            raw={"mock": True, "stage": request.stage, "text": text},
        )

