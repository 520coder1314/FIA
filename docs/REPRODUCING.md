# FIA core v2

This is the revised core implementation for new experiments. It lives in `fia/`.
Historical scripts and artifacts are not included in this core release.
Each run must be identified by its input artifacts, implementation profile and manifest.
The core alone does not regenerate published tables without the external experiment assets. No FLBA implementation or baseline measurements are
changed by this release.

## What is aligned

- Find: frozen clean proxy centroids; Euclidean distances; quantiles over unique
  undirected pairs; both directions exported. Geometry does not select a winner.
  The configured direction is checked against the candidate pool; its VLM assessment
  reports ASR, FTR and margin without automatically choosing a winner.
- Inject: one tensor per modified shard, per-image soft channel-gating masks,
  target classification, direction/norm/raw alignment, TV, projected Adam.
- Attack: complete shard set, relabeled selected items, cyclic saved-pattern reuse.
  Downstream VLM training uses your existing SWIFT workflow; foundation models and
  runners are not bundled in this minimal release.
- Backbone weights stay frozen. Head fitting updates shared BN buffers; trigger
  optimization uses evaluation mode and the fitted shard-specific state.
- Classification uses 256×256 resize, 224×224 center crop and ImageNet normalization.
  Mask extraction applies the feature helper followed by the classifier path;
  alignment uses the declared dataset-specific feature helper.
- Masks use binary positive-gradient channel gates and min–max normalization both
  before and after resizing, with a 0.05 floor. There is no extra activation ReLU.
- `--quality-mode stop_gradient` is the manuscript profile. PSNR and uniform-window
  SSIM contribute to the optimization score, scheduler and best-iterate selection,
  but do not contribute gradients. `differentiable` is an explicit alternative.
- Adam uses lr=0.05, betas=(0.9,0.999), gradient clipping=50 and projection.
  ReduceLROnPlateau uses factor=0.95, patience=1000 and min_lr=1e-5.

## Install and test

Python >=3.10; PyTorch >=2.5. Tested locally with Python 3.10 / torch 2.5.1+cu124.
No network downloads or automatic pretrained-weight fallback happen at runtime.

```bash
python -m pip install -r requirements.txt
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v
python -m fia --help
```

Run from the FIA repository root after installation.
`--output` must be a new directory. Failed runs may leave a directory; inspect it and
use a fresh name. Never overwrite old experiment outputs.

## 1. Export initial candidate pairs (no VLM fine-tuning needed)

Supply actual paths; all clean `.pt` files must contain `[images, labels]`.
Images are float RGB in [0,1] (only <=1e-6 rounding overshoot is clipped).
No implicit de-normalization, label remapping, or poisoned-file fallback is allowed.

```bash
python -m fia find \
  --dataset cifar10 \
  --shards /path/to/clean/data_20000_[0-9].pt \
  --checkpoint /path/to/resnet50_cifar10_vicreg.pth \
  --preprocess cifar10 --device cuda:0 \
  --class-names config/cifar10_classes.json \
  --output runs/cifar10_find
```

Outputs: `candidates.csv/json` (all unique pairs and eligibility), `clean_features.pt`,
`scores_template.csv` (both eligible directions). Candidate names use the supplied
ordered JSON label list; without it, IDs are retained rather than guessed.
The checkpoint must strictly match the bundled VICReg ResNet50; no random/partial
backbone fallback is permitted. The 3x3/7x7 input architecture follows weight shape.

For CIFAR-100, Imagenette and COCO: use `--dataset cifar100|imagenette|coco`, their
actual clean shards and verified ordered label lists. Choose `--preprocess imagenet`
for the declared ImageNet-normalized 256→224 route. Dataset name alone does not
silently decide checkpoint preprocessing. Verify the checkpoint's intended route.
A changed preprocessing route defines a new experiment, not a historical replay.

## 2. Prepare independent evaluation sets BEFORE candidate evaluation

Prepare `images.csv` with `path,label` rows for real evaluation images. Paths must
exist. `split` stratifies by class and checks content hashes against duplicates.

```bash
python -m fia split --images images.csv --fraction 0.5 --seed 42 \
  --output runs/evaluation_split_v2
```

Use ONLY `selection.json` for candidate outcomes. Keep `test.json` untouched until
pair selection is frozen. Previously used images cannot become a retrospective
holdout merely by rerunning this command. Prefer unused data for independent claims.

## 3. Validate and inject the configured direction

`--modify-shards` uses zero-based positions in the explicit `--shards` list.
Other shards are copied byte-for-byte. First perform the eligibility check in
[Fixed-direction Find](FIXED_DIRECTION_FIND.md). The following is the CIFAR-10 deer → airplane configuration:

```bash
python -m fia inject \
  --shards /path/to/clean/data_20000_[0-9].pt \
  --checkpoint /path/to/resnet50_cifar10_vicreg.pth \
  --preprocess cifar10 --device cuda:0 --source 4 --target 0 \
  --modify-shards 0 1 2 3 4 --items 10 --head-epochs 150 --steps 4500 \
  --epsilon 0.15 --quality-mode stop_gradient --seed 42 \
  --output runs/candidate_4_0_v2
```

Output includes clean+modified `data_20000_*.pt`, per-shard masks/deltas/patterns,
`trigger_bank.pt` (modified-shard order), fitted proxy state and manifest.
Only selected source labels change. All chosen shards must have sufficient source
items and at least one target item; missing inputs stop the run.
Feed the shards to your existing VLM training pipeline. For evaluation, use
`fia.core.apply_bank` after the declared resize and before the VLM processor,
with indices in the fixed base-class order. Preserve original evaluation labels.
Do NOT feed the extra `*_bank.pt` / `*_proxy.pt` files into the training dataset.

## 4. Fixed-direction proxy assessment and held-out testing

Follow [FIXED_DIRECTION_FIND.md](FIXED_DIRECTION_FIND.md). Assess the fixed artifact
on training-selection images with an accessible VLM, then evaluate independent test
images. Preserve three-epoch endpoints and continuation trajectories separately.
The optional `fia.score_resnet` and `fia select` interfaces are documented as
[historical search utilities](CASCADED_FIND.md), not mandatory manuscript stages.

## Minimal publication scope

Only core Python, CLI, tests, this guide, label list and dependency declaration are
published. No weights, distilled data, predictions, manuscript, credentials, legacy
baseline repositories or personal server paths are included. The adapted VICReg
ResNet file retains its original copyright and MIT license (`fia/LICENSE.vicreg`).

## Candidate counts are not class counts

CIFAR-10 has 10 classes, giving 10×9/2 = 45 unordered pairs and 90 directed
source-target choices. The local geometry run retained 26 of the 45 unordered
pairs (52 directed candidates). These are candidates, not 52 evaluated attacks
or a measured best direction. See the repository README for a summary.
