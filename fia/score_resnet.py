"""Train one ResNet per candidate/seed and score paired training-selection images.

Run ``python -m fia.score_resnet --help``. Candidate injection is a separate step.
Only training-selection images are read; this runner never opens a test manifest.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from . import resnet
from .__main__ import digest, load_data, write


def evaluate(net, x, bank, target, batch_size, device):
    net.eval()
    clean_hits = trigger_hits = 0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            clean = x[start:start + batch_size].to(device)
            index = torch.arange(start, start + len(clean)) % len(bank)
            triggered = (clean + bank[index].to(device)).clamp(0, 1)
            clean_hits += (net(clean).argmax(1) == target).sum().item()
            trigger_hits += (net(triggered).argmax(1) == target).sum().item()
    return 100 * trigger_hits / len(x), 100 * clean_hits / len(x)


def run(args):
    from PIL import Image
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("Seeds must be distinct and nonempty")
    if args.source == args.target or min(args.source, args.target) < 0:
        raise ValueError("Distinct nonnegative source and target required")
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        raise ValueError("Positive epochs/lr and batch size >= 2 required")
    split = json.loads(args.selection_split.read_text())
    if split.get("role") != "selection" or not split.get("images"):
        raise ValueError("Nonempty training-selection manifest required")
    shards = load_data(args.shards)
    x, y = torch.cat([v[0] for v in shards]), torch.cat([v[1] for v in shards])
    if len(x) < 2 or x.shape[-2] != x.shape[-1]:
        raise ValueError("Require >=2 square training images")
    if args.classes <= max(int(y.max()), args.source, args.target):
        raise ValueError("Class count does not cover labels")
    bank = torch.load(args.bank, map_location="cpu", weights_only=True)
    if (not isinstance(bank, torch.Tensor) or bank.ndim != 4 or not len(bank)
        or bank.shape[1:] != x.shape[1:] or not torch.isfinite(bank).all()):
        raise ValueError("Invalid saved effective-pattern bank")
    images = []
    seen = set()
    for row in split["images"]:
        if int(row["label"]) != args.source:
            continue
        path = Path(row["path"])
        h = digest(path)
        if h != row["sha256"] or h in seen:
            raise ValueError("Selection image hash mismatch or duplicate")
        seen.add(h)
        with Image.open(path) as im:
            im = im.convert("RGB").resize((x.shape[-1], x.shape[-2]), Image.Resampling.BILINEAR)
            tensor = torch.tensor(list(im.getdata()), dtype=torch.float32)
            images.append(tensor.reshape(x.shape[-2], x.shape[-1], 3).permute(2, 0, 1)/255)
    if not images:
        raise ValueError("No source-class selection images")
    evaluation = torch.stack(images)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        backbone, dim = resnet.resnet34(small_input=(x.shape[-1] == 32))
        net = nn.Sequential(backbone, nn.Linear(dim, args.classes)).to(args.device)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        for _ in range(args.epochs):
            net.train()
            batches = list(torch.randperm(len(x)).split(args.batch_size))
            # Avoid BatchNorm singleton batches without dropping any training items.
            if len(batches) > 1 and len(batches[-1]) == 1:
                batches[-2] = torch.cat([batches[-2], batches[-1]])
                batches.pop()
            for idx in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(net(x[idx].to(args.device)), y[idx].to(args.device))
                loss.backward()
                optimizer.step()
        asr, ftr = evaluate(net, evaluation, bank, args.target, args.batch_size, args.device)
        rows.append(["small", args.source, args.target, seed, "selection", digest(args.selection_split), asr, ftr])
        torch.save(net.state_dict(), args.output / f"resnet34_seed{seed}.pt")
        print(f"seed={seed} ASR={asr:.4f} FTR={ftr:.4f} margin={asr-ftr:.4f}")
    with (args.output / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "source", "target", "seed", "split", "split_sha256", "asr", "ftr"])
        w.writerows(rows)
    write(args.output / "protocol.json", {
        "model": "ResNet34", "initialization": "from scratch per seed",
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "optimizer": "SGD", "momentum": .9, "weight_decay": 1e-4,
        "classes": args.classes, "seeds": args.seeds, "normalization": "RGB [0,1]",
        "source": args.source, "target": args.target, "selection_count": len(evaluation),
        "shards": {str(p): digest(p) for p in args.shards},
        "bank_sha256": digest(args.bank), "selection_split_sha256": digest(args.selection_split),
        "code_sha256": digest(Path(__file__)), "torch": torch.__version__,
        "selection_image_order": "manifest source-class order; cyclic bank assignment",
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", required=True, nargs="+", type=Path)
    p.add_argument("--bank", required=True, type=Path)
    p.add_argument("--selection-split", required=True, type=Path)
    p.add_argument("--source", required=True, type=int)
    p.add_argument("--target", required=True, type=int)
    p.add_argument("--classes", required=True, type=int)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 3407, 2026])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", required=True, type=float)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True, type=Path)
    run(p.parse_args())


if __name__ == "__main__":
    main()
