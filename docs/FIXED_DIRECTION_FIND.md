# Fixed-direction Find

The current manuscript uses **geometry screening → configured direction → injection
→ VLM proxy assessment → independent downstream evaluation**. The direction is a
configuration input. Geometry checks eligibility; it does not establish a best or
optimal direction. No ResNet ranking or Top-k stage is required.

## 1. Screen clean geometry

Run `fia find` with the actual clean shards, proxy checkpoint, preprocessing and
class mapping, as shown in the main README. Quantile parameters are `--low` and
`--high` (CIFAR-10 configuration: 0.30 and 0.90). The output `pairs` array contains
unordered pairs and an `eligible` flag; each eligible pair permits both directions.
The reported 26/45 pairs is a CIFAR-10 observation, not a fixed pool size.

## 2. Fix and verify the dataset-specific direction

For the CIFAR-10 example, source 4 is deer and target 0 is airplane. From the
repository root, after the README's geometry command, run:

```python
import json
from pathlib import Path

candidate_file = Path("runs/cifar10_find/candidates.json")
source, target = 4, 0
candidates = json.loads(candidate_file.read_text())
assert source != target
eligible = any(
    row["eligible"]
    and {row["source"], row["target"]} == {source, target}
    for row in candidates["pairs"]
)
if not eligible:
    raise SystemExit("Configured direction is outside the candidate pool")
print(f"[ELIGIBLE] {source} -> {target}")
```

Pass the same IDs to `fia inject --source 4 --target 0`. The injection CLI accepts
IDs directly; this explicit check establishes their connection to the geometry
export. Archive the candidate file and injection manifest together. Other datasets
require their own geometry, verified label mapping and configured direction.

## 3. Construct and assess

Follow the injection command in the README. Keep all clean and modified shards in
the downstream training set. Save the trigger bank and exact image order. Use an
external SWIFT/VLM runner to fine-tune an accessible proxy on the supplied artifact.
Measure paired ASR, FTR and margin on training-selection images. This is assessment
of the configured direction, not evidence that it beats untested candidates.

## 4. Independent evaluation and continuation

Use the benchmark's independent test split for final results. Training-selection
images must come from the training partition; `fia split` only partitions its input
CSV and cannot verify benchmark provenance. Do not pool official training and test
partitions before splitting.

The main comparisons use three fine-tuning epochs. The seven-checkpoint study has
an initial stage and a separately configured continuation; the poisoned artifact
remains in use. Preserve each stage's optimizer, learning rate, checkpoint time,
seed, parser and generation limit in its run records. Do not substitute a trajectory
checkpoint for an independently trained main endpoint.

The package supplies artifact construction and saved-pattern application. Model
weights, training runners and evaluation images are external dependencies; see
[assets](ASSETS.md) and [implementation profile](REPRODUCIBILITY.md).
