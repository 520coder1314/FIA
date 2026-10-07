# FIA core v2

This is the revised core implementation for new experiments. It lives in `fia_core/`.
Historical `apply_backdoor_to_distilled_data.py` and its artifacts are retained separately.
**Published manuscript numbers are not outputs of this revision.** New results must be
identified by their own manifest. No FLBA implementation or baseline measurements are
changed by this release.

## What is aligned

- Find: frozen clean proxy centroids; Euclidean distances; quantiles over unique
  undirected pairs; both directions exported. Geometry does not select a winner.
  Candidate injection/fine-tuning must be evaluated; `select` then maximizes mean ASR−FTR.
- Inject: one tensor per modified shard, per-image soft channel-gating masks,
  target classification, direction/norm/raw alignment, TV, projected Adam.
- Attack: complete shard set, relabeled selected items, cyclic saved-pattern reuse.
  Downstream VLM training uses your existing SWIFT workflow; foundation models and
  runners are not bundled in this minimal release.
- Backbone weights and BN buffers remain fixed during head fitting. The classifier,
  saliency calculation and alignment all use one explicit preprocessing route.
- The mask is the paper's channel-gating construction, not standard Grad-CAM++.
- `--quality-mode stop_gradient` matches the current manuscript's *gradient semantics*.
  `--quality-mode differentiable` restores a differentiable PSNR/SSIM objective as
  an explicit alternative. It requires new evidence and a corresponding method update.
  Neither mode is a bit-for-bit historical replay: corrected preprocessing, frozen BN,
  Gaussian-window SSIM and an explicit optimizer schedule define the new implementation.

## Install and test

Python >=3.10; PyTorch >=2.5. Tested locally with Python 3.10 / torch 2.5.1+cu124.
No network downloads or automatic pretrained-weight fallback happen at runtime.

```bash
python -m pip install -r requirements-core.txt
OMP_NUM_THREADS=2 python -m unittest discover -s tests_core -v
python -m fia_core --help
```

Run from NCFM-Lab locally, or from the root of the published core repository.
`--output` must be a new directory. Failed runs may leave a directory; inspect it and
use a fresh name. Never overwrite old experiment outputs.

## 1. Export initial candidate pairs (no VLM fine-tuning needed)

Supply actual paths; all clean `.pt` files must contain `[images, labels]`.
Images are float RGB in [0,1] (only <=1e-6 rounding overshoot is clipped).
No implicit de-normalization, label remapping, or poisoned-file fallback is allowed.

```bash
python -m fia_core find \
  --dataset cifar10 \
  --shards /path/to/clean/data_20000_[0-9].pt \
  --checkpoint /path/to/resnet50_cifar10_vicreg.pth \
  --preprocess cifar10 --device cuda:0 \
  --class-names config/fia_core/cifar10_classes.json \
  --output runs/cifar10_find_v2
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
python -m fia_core split --images images.csv --fraction 0.5 --seed 42 \
  --output runs/evaluation_split_v2
```

Use ONLY `selection.json` for candidate outcomes. Keep `test.json` untouched until
pair selection is frozen. Previously used images cannot become a retrospective
holdout merely by rerunning this command. Prefer unused data for independent claims.

## 3. Inject a candidate and evaluate it on selection data

`--modify-shards` uses zero-based positions in the explicit `--shards` list.
Other shards are copied byte-for-byte. Example direction only; not a claimed winner:

```bash
python -m fia_core inject \
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
`fia_core.core.apply_bank` after the declared resize and before the VLM processor,
with indices in the fixed base-class order. Preserve original evaluation labels.
Do NOT feed the extra `*_bank.pt` / `*_proxy.pt` files into the training dataset.

## 4. Select by measured margin, then evaluate on held-out test data

Copy template to a separate scores CSV; remove unevaluated rows. At least two
eligible directions and the same seed set per direction are required.
Fields: `source,target,seed,split,split_sha256,asr,ftr`.
ASR/FTR are percentages (0–100); split must be `selection` and split_sha256 must be
SHA256 of the exact `selection.json`. Scores must come from actual paired inference.

```bash
sha256sum runs/evaluation_split_v2/selection.json
python -m fia_core select --candidates runs/cifar10_find_v2/candidates.json \
  --scores runs/selection_scores.csv \
  --selection-split runs/evaluation_split_v2/selection.json \
  --output runs/selected_v2
```

This reports best among evaluated candidates, not a global optimum. Only then use
the independent test split for final reporting. Across datasets, export their own
candidate lists; do not copy CIFAR-10 centroids or selection outcomes.

## Minimal publication scope

Only core Python, CLI, tests, this guide, label list and dependency declaration are
published. No weights, distilled data, predictions, manuscript, credentials, legacy
baseline repositories or personal server paths are included. The adapted VICReg
ResNet file retains its original copyright and MIT license (`fia_core/LICENSE.vicreg`).
