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

Source paths above are relative to the workspace root; config paths are relative to
this experiment. The full study design is in [the experiment README](../README.md).
This map does not redefine the frozen scientific protocols.
