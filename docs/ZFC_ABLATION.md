# ZFC / SE / FiLM ablation entry

This branch adds configurable alternatives at the **same insertion point** as the
original ZFC module: after `decoder_level3` and before `up3_2`.

## Architecture entry

Set `network_g.dc_mode` in the BasicSR YAML:

| `dc_mode` | Implementation | Conditioning source | Operation |
|---|---|---|---|
| `none` | Identity | — | No calibration module |
| `zfc` | Original `DCFiLM` | `l1_enc` | Bounded scale + shift |
| `se` | Canonical SE, reduction=16 | `l3_dec` itself | Sigmoid channel scaling |
| `se_matched` | Widened canonical SE | `l3_dec` itself | Sigmoid channel scaling |
| `cross_se_matched` | Cross-scale SE | `l1_enc` | Sigmoid channel scaling of `l3_dec` |
| `film` | Identity-centred unbounded FiLM | `l1_enc` | Unbounded scale + shift |
| `film_bottleneck` | Identity-centred unbounded FiLM | `l3_enc` | Unbounded scale + shift |

The legacy `enable_dc` argument remains supported. If `enable_dc: false`, it
forces `dc_mode: none`. For new experiments, set both consistently or let the
provided config-preparation script do so.

## Parameter budget (default dim=48, target channels=192)

The original ZFC has:

- source channels = 48
- hidden channels = 48 * 4 = 192
- output channels = 2 * 192 = 384
- trainable module parameters = **83,520**

The matched alternatives are computed from the ZFC parameter budget rather
than hard-coded for one model width:

| Variant | Hidden width at dim=48 | Module parameters | Difference from ZFC |
|---|---:|---:|---:|
| ZFC | 192 | 83,520 | — |
| Standard SE (r=16) | 12 | 4,812 | -94.24% |
| Parameter-matched SE | 216 | 83,352 | -0.20% |
| Parameter-matched Cross-SE | 346 | 83,578 | +0.07% |
| FiLM, same source as ZFC | 192 | 83,520 | 0.00% |
| FiLM, bottleneck source | 144 | 83,472 | -0.06% |

Interpretation:

1. `se` is the canonical low-parameter SE reference.
2. `se_matched` is the canonical self-conditioned SE with widened hidden
   dimension, satisfying the reviewer's same-parameter-budget request.
3. `cross_se_matched` is the most controlled SE comparison: it uses the same
   shallow condition (`l1_enc`) and deep target (`l3_dec`) as ZFC, changing
   the affine modulation to multiplicative gating.
4. `film` uses exactly the same source, target, MLP widths and parameter count
   as ZFC, but removes the tanh-based bounds. It isolates the effect of
   bounded scale/shift. It is not claimed to be a fundamentally different
   mathematical operator from FiLM: ZFC is a task-specific bounded
   FiLM-style calibration.
5. `film_bottleneck` is an optional source-location control. It tests whether
   the shallow global descriptor is preferable to a bottleneck descriptor.

The `none` model intentionally has fewer parameters. Do not add unused
parameters merely to make its count match. Report its parameter count honestly.

## Create a run config and launch

Use the **actual final training YAML** that generated the paper's GS-Blur result,
not an outdated public template. The helper copies it and changes only the
ablation mode, run name, seed, single-GPU setting and explicitly requested
short-run settings. The original YAML is not modified.

Single variant:

```bash
bash scripts/run_zfc_ablation.sh zfc /path/to/final_GSBlur.yml 42 25000
bash scripts/run_zfc_ablation.sh se_matched /path/to/final_GSBlur.yml 42 25000
bash scripts/run_zfc_ablation.sh cross_se_matched /path/to/final_GSBlur.yml 42 25000
bash scripts/run_zfc_ablation.sh film /path/to/final_GSBlur.yml 42 25000
```

Run the core comparison sequentially:

```bash
bash scripts/run_zfc_ablation.sh all /path/to/final_GSBlur.yml 42 25000
```

Optional single-GPU memory controls:

```bash
NUM_GPU=1 BATCH_SIZE_PER_GPU=1 GT_SIZE=384 \
  bash scripts/run_zfc_ablation.sh all /path/to/final_GSBlur.yml 42 25000
```

The default 25,000 iterations is only a **provisional reduced-budget setting**.
It must be chosen after measuring short-run convergence and single-GPU
iteration time. Do not report reduced-budget results as full 600K convergence.

The script clears `pretrain_network_g` and `resume_state` so each variant
starts independently from scratch. It does not alter dataset roots, loss
weights, optimizer settings, augmentations or model depth.

## Smoke test

```bash
python scripts/check_zfc_ablation.py
```

This checks output shapes, finite outputs, ZFC bounds and expected module
parameter counts on CPU. It does not replace a full end-to-end training run.
