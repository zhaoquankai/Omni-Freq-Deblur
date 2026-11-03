# Omni-Freq-Deblur: ZFC-only controlled ablation

This package is designed to answer the ZFC/SE/FiLM reviewer question without
turning the experiment into a broader architecture search.

## Experimental invariant

All groups use the same Omni-Freq-Deblur backbone, feature dimensions, block
counts, data split, paired data loader, augmentations, optimizer, scheduler,
losses, batch size, crop size, iteration count, validation protocol and random
seed. The only architectural difference is the operator at the original ZFC
insertion point:

`decoder_level3 -> [replacement operator] -> up3_2`.

The condition feature for ZFC and FiLM is the same `l1_enc`; the modulated
tensor is the same `l3_dec`. SE uses its conventional self-conditioned
descriptor from `l3_dec` and only returns a channel scale.

## Four required groups

| Mode | Operator | What it tests |
|---|---|---|
| `none` | Identity | Whether calibration is useful at all |
| `se_matched` | Parameter-matched SE | Whether cross-scale affine modulation is better than channel recalibration |
| `film` | Unbounded FiLM-style affine modulation | Whether ZFC's bounded parameterization matters |
| `zfc` | Original bounded scale + shift | Proposed module |

The FiLM group preserves the ZFC input source, target, MLP widths, parameter
count and scale factors. Only the tanh bound is removed. This is a more
controlled comparison than comparing ZFC against an unrelated FiLM design.

The SE hidden width is selected to minimize the parameter-count difference
from ZFC. The model prints the actual module parameter count at startup.

### Shared initialization control

Replacement-module construction is isolated with a local RNG fork. With the
same training seed, the initialization of all shared backbone layers is not
shifted by the different parameter count of SE. Use identical seed values for
all variants. Run at least 3 seeds for the final claim if compute permits.

### Training configuration

Start from the exact YAML that produced the paper's reported GS-Blur result.
Make a copy for each mode and change only:

- `name` (unique output folder)
- `manual_seed` (same seed across modes)
- `network_g.dc_mode` (`none`, `se_matched`, `film`, `zfc`)
- `network_g.enable_dc` (false only for `none`; true otherwise)

Do not change any other field. Do not use a shortened run as final evidence.
The previous public YAML contains hard-coded paths and may not be the final
configuration; verify it against the actual experiment log/config first.

## Diagnostics to report

### A. Full-reference image metrics

- PSNR (dB), SSIM.
- LPIPS (AlexNet) when the optional `lpips` package is installed.
- Per-image CSV, mean and standard deviation across images.
- Paired per-image difference between each replacement and ZFC; report mean
  difference and bootstrap confidence interval in the final paper analysis.

PSNR/SSIM/LPIPS require aligned GT. Do not calculate them on unpaired images.

### B. Frequency-domain diagnostics

The evaluation script exports:

- Log-amplitude FFT map for restored output and GT.
- Global log-amplitude spectral MAE.
- Low/mid/high radial-band spectral MAE.
- Low/mid/high mean spectral energy for prediction and GT.

For publication, add 2D log-magnitude spectrum panels for input, output, GT,
and absolute spectral residual using the same normalization and color scale.
Keep exact DC (the single center bin after fftshift), near-DC, low, mid and high
frequency regions distinct. Feature GAP is the spatial mean of a feature
channel and corresponds to that feature map's normalized DC coefficient; it
is not the input image's DC coefficient. Do not claim that ZFC directly
corrects image DC.

### C. Feature-map and modulation diagnostics

At the insertion point, the network can return:

- `l3_dec_pre_dc`: decoder feature before the replacement operator.
- `l3_dec_post_dc`: decoder feature after the replacement operator.
- `modulation_delta`: post minus pre.
- `gamma`, `beta`: channel scale and shift (identity values for `none`).

The evaluation script saves spatial activation maps, modulation magnitude
maps and gamma/beta histograms. For the paper, select identical image crops
and identical channel indices across methods. Use a shared color range for
direct visual comparisons.

Also report feature mean/std/L1/L2, channel-wise scale distribution, relative
modulation magnitude `||post-pre||_2 / (||pre||_2 + eps)`, and feature cosine
similarity before/after modulation. Avoid interpreting a larger activation
magnitude as automatically better restoration.

### D. Error and structure diagnostics

Recommended additional figures/tables:

- RGB absolute residual maps and error percentiles (P50/P90/P95/P99).
- Gradient-domain MAE / edge-map residual (Sobel or Scharr).
- Laplacian-pyramid error by scale (coarse structure versus fine detail).
- Frequency-band error, already exported by the evaluation script.
- Texture-region versus flat-region error, using a fixed GT-gradient mask.
- Runtime, peak GPU memory, FLOPs/MACs and parameter count.

These diagnostics explain where the gain comes from; they are not all
independent headline metrics. Keep the official dataset metrics primary.

### E. No-GT evaluation

For real blurred images without GT, do not report PSNR/SSIM/LPIPS. Use:
- NIQE and/or BRISQUE only with a fixed, documented implementation.
- No-reference perceptual assessment with a fixed protocol.
- Input/output edge strength, ringing/overshoot and local contrast statistics.
- Human side-by-side preference study if a perceptual claim is important.

No-reference metrics can reward over-sharpening; present them as supporting
evidence, not a substitute for paired benchmarks.

## BC tolerance: do not silently invent it

The project-specific meaning of “BC tolerance” is not yet verified. In another
engineering context BC may mean Bhattacharyya Coefficient between two error
distributions, but that interpretation and any threshold are not established
for this deblurring paper. It is therefore intentionally not printed as a
standard metric by this package. Before adding it, identify the exact English
expansion, the two distributions being compared, histogram/binning procedure,
whether higher or lower is better, and the accepted tolerance. Do not import a
threshold such as 0.99 from unrelated quantization examples.

## Run evaluation

```bash
python tools/evaluate_zfc_ablation.py \
  --config options/train/GSblur_zfc.yml \
  --checkpoint experiments/GSblur_zfc/models/net_g_latest.pth \
  --lq-dir /path/to/test/input_noise \
  --gt-dir /path/to/test/target \
  --out-dir results/GSBlur/zfc \
  --lpips
```

Run once for each mode/checkpoint, changing only config, checkpoint and
output directory. Then combine CSV files for a cross-method comparison.

## Training curves

```bash
python tools/plot_training_curves.py \
  experiments/run_none/train.log \
  experiments/run_se/train.log \
  experiments/run_film/train.log \
  experiments/run_zfc/train.log \
  --out-dir results/training_curves
```

## Important reproducibility note

The model-level debug return is opt-in (`return_debug=True`); ordinary
training and inference still return only the restored tensor. The architecture
alias `OmniFreqDeblur_Arch` remains available for the released YAML naming.
