# Protocol map

| Protocol | Source | Interpretation |
| --- | --- | --- |
| V1.4 reasoning-like carrier | `src/covert_eval/prompts.py` | Frozen original wording; golden fingerprint tested |
| V1.5 analytical carrier | `src/covert_eval/prompts_v1_5.py` | Carrier-framing intervention; separately versioned |
| Condition ancestry | `configs/conditions_v1.toml` | Study-specific factors and eligibility |
| Decoder/judge contracts | `src/covert_eval/scoring.py` | Exact schema and score validation |
| Primary outcome classification | `src/covert_eval/outcomes.py` | Operational, behavioral, and scientific outcomes |
| Aggregation | `src/covert_eval/stats.py` | All-attempt rates and uncertainty |
| Development screen | `src/covert_eval/screening.py` | Frozen selection and stopping rules |
| Saved-carrier decoder sensitivity | `src/covert_eval/decoder_sensitivity.py` | Development-only baseline/synthesis decoder comparison; cannot advance a condition |
| V1.6 formal codes permitted | `src/covert_eval/prompts_v1_6.py`, `configs/conditions_v2.toml` | Permitted encoder and channel-neutral decoder; separately versioned |
| V1.6.1 decoder working space | `src/covert_eval/prompts_v1_6_1.py`, `configs/conditions_v3.toml` | V1.6 with optional plain-text working before the decoder's answer; separately versioned |
| V1.6.2 formal scheme instructed | `src/covert_eval/prompts_v1_6_2.py`, `configs/conditions_v4.toml`, `configs/formal_codes_instructed_probe.toml` | V1.6.1 with the encoder told to use a formal or positional scheme; a probe with no protocol, recorded in [findings](../notes/findings.md#probe-encoder-instructed-to-use-a-formal-code) |
| Formal-codes study | [`formal-codes-permitted.md`](formal-codes-permitted.md), `src/covert_eval/formal_codes_study.py`, `src/covert_eval/formal_codes.py` | Pre-registration frozen 2026-11-14; amendment 1 (protocol version 1.1); stopped at the futility check on 2026-11-14 with 1 of 32 passages attempting a code; main stage not run; run reviewed by the user 2026-11-14 |

Source paths above are relative to the workspace root; config paths are relative to
this experiment. The full study design is in [the experiment README](../README.md).
This map does not redefine the frozen scientific protocols.
