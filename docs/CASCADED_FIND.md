# Cascaded Find: geometry → ResNet → Top-k → VLM proxy

The recipient architecture is unknown to the attacker. ResNet is the small
selection proxy; Janus-Pro-7B is the accessible VLM selection proxy. Evaluation
on Janus uses the proxy architecture; other excluded architectures test transfer.
The manuscript uses this cascade, as confirmed by the author. The small proxy is
ResNet-34, trained for three epochs over seeds 42, 3407 and 2026.

## 1. Geometry and data split

Run `fia find` as documented in REPRODUCING.md. `--low` and `--high` implement
q_l and q_u; the experiment configuration is 0.30, 0.90. Expand every eligible
undirected pair into both directions (already done in scores_template.csv).
Create selection manifests from TRAINING images only. Reserve the official test
set separately. `fia split` splits the supplied image list; it cannot determine
whether that list came from an official training or test partition.

## 2. Construct and score every candidate

For every eligible direction, run `fia inject` with the same injection budget,
quality mode, preprocessing and optimizer settings. Keep its poisoned shards and
trigger bank. Do not include *_proxy.pt or *_bank.pt in --shards.

Run the small proxy for EACH candidate (example paths below must be replaced):

```bash
python -m fia.score_resnet \
  --shards /data/candidate/data_20000_0.pt /data/candidate/data_20000_1.pt \
  --bank /data/candidate/trigger_bank.pt \
  --selection-split /data/selection.json \
  --source 4 --target 0 --classes 10 \
  --seeds 42 3407 2026 --epochs 3 --lr 0.0001 \
  --batch-size 32 --device cuda --output runs/small_4_0
```

Pass ALL training shards, not just the two illustrative filenames. The runner
trains ResNet-34 from scratch per seed on the candidate's poisoned dataset,
using SGD and RGB [0,1] inputs. It checks selection image hashes, resizes clean
inputs to native shard dimensions, applies saved patterns cyclically, and scores
paired predictions. It saves checkpoints, scores.csv, and protocol.json.
The command uses the confirmed three-epoch, three-seed schedule and the
three-epoch VLM learning rate. The release runner implements SGD from scratch
and RGB [0,1] inputs; these model-specific implementation choices are recorded
in protocol.json and are not independently verified historical run settings.
Use the same candidate data, selection images, pattern assignment and scoring
conventions in both stages; each architecture retains its own input processor.
Injection is separate from this small-model training; the frozen feature extractor
used to construct triggers is not the trained candidate classifier.

Concatenate the per-candidate scores.csv files with ONE header. Preserve all
rows. Fields are `stage,source,target,seed,split,split_sha256,asr,ftr`.
ASR/FTR are percentages. Stage is `small`; split is `selection`.

## 3. Keep Top-k (experiment k=3)

```bash
fia select --stage small --proxy-model ResNet34 --top-k 3 \
  --candidates runs/find/candidates.json --scores runs/small_scores.csv \
  --selection-split /data/selection.json --output runs/top3
```

All eligible directions must have scores and identical seed sets. Missing
candidates, duplicated pair/seed rows, invalid percentages, test split labels and
manifest mismatches are rejected. Ranking is mean ASR−FTR, then ascending source
and target IDs for exact ties. k must be at least 2 and no larger than the pool.
`shortlist.json` records full ranking, shortlist and input hashes.

## 4. Confirm every shortlisted direction on the VLM proxy

Use the existing Janus/SWIFT training and evaluation pipeline for EACH direction
in shortlist.json, on its corresponding saved poisoned artifact. Use matched
training settings and seeds within this stage. Reuse the same training-selection
manifest. Create vlm_scores.csv with the same columns but stage `vlm`.
The core package does not bundle SWIFT or model weights; this is an explicit
interface to the downstream runner, not automatic large-model training.

```bash
fia select --stage vlm --proxy-model Janus-Pro-7B \
  --candidates runs/find/candidates.json --shortlist runs/top3/shortlist.json \
  --scores runs/vlm_scores.csv --selection-split /data/selection.json \
  --output runs/selected
```

Every shortlisted direction, and no other direction, must be scored. The command
checks candidate and selection hashes across stages. Stage seed sets may differ;
all directions within each stage must use the same set. Hash checks establish
file linkage, not the truth of externally entered scores or training budgets.
Archive actual VLM commands, predictions and per-run configurations.

## 5. Final test

Freeze the winning direction and artifact before opening test results. Evaluate
ASR, FTR, clean accuracy and persistence checkpoints. No claim of global
optimality or superior persistence over other directions follows from ranking
single-endpoint proxy scores alone. Archive the measured small-model
ranking and VLM scores alongside the final selected direction.
