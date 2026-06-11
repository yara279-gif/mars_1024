#!/usr/bin/env python3
import argparse
import sys
from collections import Counter
from pathlib import Path

EXPECTED = {
    "SumMe": {"num_videos": 25, "train": 20, "val": 5},
    "TVSum": {"num_videos": 50, "train": 40, "val": 10},
}

def parse_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    try:
        dataset, split, rest = line.split("/", 2)
        train_str, val_str = rest.split("/")
    except ValueError:
        raise ValueError(f"Invalid line format: {line}")
    train = [x.strip() for x in train_str.split(",") if x.strip()]
    val = [x.strip() for x in val_str.split(",") if x.strip()]
    return dataset, split, train, val


def load_splits(file_path: Path):
    splits_by_dataset = {}
    with file_path.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            parsed = parse_line(raw)
            if parsed is None:
                continue
            dataset, split, train, val = parsed
            splits_by_dataset.setdefault(dataset, []).append({
                "split": split,
                "train": train,
                "val": val,
            })
    return splits_by_dataset


def validate_dataset(dataset: str, splits: list):
    ok = True
    msgs = []
    exp = EXPECTED[dataset]

    # Basic per-split checks
    all_val = []
    for s in splits:
        split_name = s["split"]
        train = s["train"]
        val = s["val"]

        # counts
        if len(train) != exp["train"]:
            ok = False
            msgs.append(f"[{dataset} {split_name}] train count {len(train)} != {exp['train']}")
        if len(val) != exp["val"]:
            ok = False
            msgs.append(f"[{dataset} {split_name}] val count {len(val)} != {exp['val']}")

        # duplicates within split
        tdup = [k for k, c in Counter(train).items() if c > 1]
        vdup = [k for k, c in Counter(val).items() if c > 1]
        if tdup:
            ok = False
            msgs.append(f"[{dataset} {split_name}] duplicate in train: {tdup}")
        if vdup:
            ok = False
            msgs.append(f"[{dataset} {split_name}] duplicate in val: {vdup}")

        # disjoint
        inter = set(train) & set(val)
        if inter:
            ok = False
            msgs.append(f"[{dataset} {split_name}] train ∩ val not empty: {sorted(inter)}")

        all_val.extend(val)

    # Across splits: each video appears exactly once in validation
    val_counts = Counter(all_val)
    if any(c != 1 for c in val_counts.values()):
        offenders = {k: c for k, c in val_counts.items() if c != 1}
        ok = False
        msgs.append(f"[{dataset}] validation appearance counts != 1: {offenders}")

    # Coverage: validation should cover the whole set exactly once
    # We assume canonical IDs are video_1..video_N
    expected_all = {f"video_{i}" for i in range(1, exp["num_videos"] + 1)}
    observed_val = set(val_counts.keys())
    missing = sorted(expected_all - observed_val)
    extra = sorted(observed_val - expected_all)
    if missing:
        ok = False
        msgs.append(f"[{dataset}] validation missing videos: {missing}")
    if extra:
        ok = False
        msgs.append(f"[{dataset}] validation unexpected videos: {extra}")

    return ok, msgs


def main():
    ap = argparse.ArgumentParser(description="Validate split files for SumMe and TVSum")
    ap.add_argument("summe", type=Path, help="Path to SumMe_splits.txt")
    ap.add_argument("tvsum", type=Path, help="Path to TVSum_splits.txt")
    args = ap.parse_args()

    overall_ok = True

    # Load and validate SumMe
    summe_splits = load_splits(args.summe)
    if "SumMe" not in summe_splits:
        print("ERROR: SumMe dataset lines not found in", args.summe)
        return 2
    ok, msgs = validate_dataset("SumMe", summe_splits["SumMe"]) 
    print("SumMe: ", "PASS" if ok else "FAIL")
    for m in msgs:
        print("-", m)
    overall_ok &= ok

    # Load and validate TVSum
    tvsum_splits = load_splits(args.tvsum)
    if "TVSum" not in tvsum_splits:
        print("ERROR: TVSum dataset lines not found in", args.tvsum)
        return 2
    ok, msgs = validate_dataset("TVSum", tvsum_splits["TVSum"]) 
    print("TVSum:", "PASS" if ok else "FAIL")
    for m in msgs:
        print("-", m)
    overall_ok &= ok

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
