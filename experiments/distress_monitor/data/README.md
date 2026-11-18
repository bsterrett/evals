# External inputs

Nothing here is committed except this file. `distress_monitor.fetch_inputs` downloads
the inputs named in the [config](../configs/participant_reader_probe.json) and records
their SHA-256 in `sha256.json` next to them, under the ignored
`artifacts/external/pain_axis/213fb6c/`.

- **Pain Axis files** (MIT licence), from
  [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis) at commit
  `213fb6c89c96004dbe93d1704c636ad8c9d801f3`:
  - `datasets/4.1_self_other_420_scenarios.json` (114,910 bytes): the 420 self-other
    scenarios;
  - `results/vectors_full_steering/vectors_full_Qwen_2.5_7B_instruct.pt` (147,364
    bytes): the authors' directions for this model and the layer they read them at.
    It is loaded with `torch.load(..., weights_only=True)`, which refuses arbitrary
    pickled objects;
  - `results/4.1_self_other/per_model/screen_v2_Qwen_2.5_7B_instruct.csv` (137,180
    bytes): the authors' readings of all 420 scenarios on this model, used for the
    pipeline check and for z-scores on the paper's scale.
- **Model**: `Qwen/Qwen2.5-7B-Instruct` at revision
  `a09a35458c702b33eeacc393d103063234e8bc28` (Apache-2.0, 15.24 GB), into `HF_HOME`.
