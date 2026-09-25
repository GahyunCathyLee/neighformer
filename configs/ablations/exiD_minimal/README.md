# exiD Minimal Feature Ablation

Generated configs for NeighFormer minimal feature ablation.

Conditions:
- 01_kin: kinematics only (use_s_x=false, use_s_y=false, use_I=false, use_dim=false)
- 02_sx: +sx (use_s_x=true, use_s_y=false, use_I=false, use_dim=false)
- 03_sy: +sy (use_s_x=false, use_s_y=true, use_I=false, use_dim=false)
- 04_sx_sy: +sx + sy (use_s_x=true, use_s_y=true, use_I=false, use_dim=false)
- 05_I: +I (use_s_x=false, use_s_y=false, use_I=true, use_dim=false)
- 06_dim: +dim (use_s_x=false, use_s_y=false, use_I=false, use_dim=true)
- 07_I_dim: +I + dim (use_s_x=false, use_s_y=false, use_I=true, use_dim=true)
- 08_full: +sx + sy + I + dim (use_s_x=true, use_s_y=true, use_I=true, use_dim=true)

See `manifest.csv` for every generated config, seed, and checkpoint tag.

Run all experiments with:

```bash
./scripts/run_exid_minimal_ablation.sh
```
