# exiD Component Ablation

Full feature config is fixed (`pva + sx + sy + dim + I`).
The ablation removes preprocessing-time components from the full setup.

- full neighbor filtering: `--gate_topn 1`
- full slot weighting: `--slotImportance 1`
- `no_nf` is the comparison for not applying neighbor filtering.
- `no_slot` is the comparison for not applying slot weighting.

Conditions:
- 02_no_nf: without neighbor filtering (slot weighting only) (mmap=data/exiD/ablations/slot_a1)
- 03_no_slot: without slot weighting (neighbor filtering top-1 only) (mmap=data/exiD/ablations/nf_top1)
- 04_no_both: without neighbor filtering and without slot weighting (mmap=data/exiD/ablations/base_I)

Prepare the mmap variants first:

```bash
TOPN=1 SLOT_ALPHA=1.0 ./scripts/preprocess_exid_component_ablation_data.sh
```

See `manifest.csv` for every generated config, seed, and checkpoint tag.

Run all experiments with:

```bash
./scripts/run_exid_component_ablation.sh
```
