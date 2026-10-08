# Manuscript injection profile

The `tdsc-shard-bn-v1` profile implements the injection equations and preprocessing
specified in the manuscript. Choose `--quality-mode stop_gradient` for this profile.
The manifest records the profile, preprocessing, quality mode, code hashes, input
hashes and checkpoint hash. Each modified shard saves its fitted proxy state,
including BN buffers, masks, perturbation, effective patterns and best score.

| Component | Implementation |
|---|---|
| Backbone | Frozen parameters; BN updates during head fitting, evaluation mode during injection |
| Shard state | Each shard starts from the supplied checkpoint and fits its own auxiliary head; classifier and alignment share the resulting backbone |
| Classifier | Resize to 256×256, center crop to 224×224, ImageNet normalization |
| Alignment | Dataset-specific feature preprocessing; target centroid computed after head fitting |
| Mask input | Feature preprocessing followed by the classifier path |
| Mask map | Positive-gradient channel gates, weighted activation sum, min–max, resize, min–max; floor 0.05 |
| Classification | Target cross entropy on the classifier path |
| Alignment loss | 0.6 direction + 0.4 norm + 0.01 raw distance; normalization denominator uses norm + 1e-12 |
| Quality | Equal-weight PSNR/SSIM hinges at 30 dB / 0.9; SSIM uses a 7×7 uniform window, sample covariance and RGB mean |
| Total score | 0.45 classification + 0.45 alignment + 0.10 detached quality + 0.0005 TV |
| Optimizer | Adam, lr 0.05, betas (0.9,0.999); gradient norm clip 50; projection to the epsilon box |
| Scheduler | ReduceLROnPlateau, factor 0.95, patience 1000, minimum lr 1e-5 |
| Output | Best-scoring iterate, including evaluation of the final projected update |

## Reproduction inputs

Supply clean distilled shards with float RGB pixels in [0,1], the compatible proxy
checkpoint, dataset-specific class mapping and attack direction. Downstream VLM
checkpoints, SWIFT configuration and independent evaluation images remain external
assets. See [REPRODUCING.md](REPRODUCING.md) for commands.

## Validation scope

Unit tests check BN phase transitions, frozen backbone weights, preprocessing
routes, objective values and detached-quality gradients, perturbation bounds,
pattern reuse and selection utilities. Smoke tests use generated tensors and
verify execution; they do not measure downstream attack performance. Code alignment
does not change archived experimental artifacts or recalculate manuscript numbers.
