# Required assets

The repository contains code. The experiment assets below have not been published
with this core release; a fresh clone alone is not sufficient to reproduce the
paper's numerical results.

| Asset | Required contract | Availability |
|---|---|---|
| Clean distilled shards | Each `.pt` contains `[images, labels]`; images N×3×H×W in [0,1], labels integer N | Supply verified artifacts from your distillation run or obtain the exact archived artifacts from the authors |
| VICReg proxy | State dict matching bundled ResNet50; architecture inferred from conv1 kernel shape; strict load | Supply the actual checkpoint; no guessed download URL or automatic replacement |
| Class names | Ordered JSON list, index equals numeric label | CIFAR-10 included; supply verified lists for other datasets |
| Real evaluation images | CSV `path,label`; local paths; deduplicated image content | Prepare from the relevant benchmark and split before candidate evaluation |
| Downstream VLM and runner | Model, processor, SWIFT/LoRA configuration, parser and inference limits | External to this core release; required for candidate scores and paper-level evaluation |

Floating-point overshoot up to 1e-6 is clipped; larger range errors stop loading.
Only declare `--preprocess cifar10` for the corresponding 32×32 normalization route.
`imagenet` declares the 256→224 route and ImageNet normalization. The checkpoint
contract, not the dataset's name alone, determines which route is appropriate.

## Input examples

`class_names.json`:

```json
["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
```

`images.csv`:

```csv
path,label
/path/to/real/deer_0001.png,4
/path/to/real/airplane_0001.png,0
```

Each class needs enough distinct images for both split partitions. These rows only
illustrate the schema, not a sufficient dataset.

`selection_scores.csv`:

```csv
source,target,seed,split,split_sha256,asr,ftr
```

Populate rows from actual selection-set inference. Values are percentages in [0,100].
`split_sha256` is the SHA256 of the exact `selection.json`. Every evaluated direction
must use the same seed set; duplicate direction/seed rows are rejected. At least two
evaluated directions are required. Never fill the file with demonstration values
and report its winner as an experiment.

## Output contracts

- `find`: candidates CSV/JSON, clean feature tensors, empty-score template.
- `split`: selection/test JSON manifests, labels and image content hashes.
- `inject`: the complete shard set, per-shard banks and proxy state, unified
  `trigger_bank.pt`, and `manifest.json` with `complete: true` after success.
- `select`: ranked measured margins, selected direction and provenance hashes.

Train only on the named data shards; never glob all `.pt` files, because proxy and
trigger-bank outputs are not training datasets. Triggered evaluation references
retain the original labels. When porting to a VLM runner, preserve the declared
resize, cyclic base-class pattern ordering, clamp/encoding and label parser.
