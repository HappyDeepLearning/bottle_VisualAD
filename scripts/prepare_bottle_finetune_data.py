from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare anomaly-only bottle-cap finetuning metadata from own_datasets."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("own_datasets"),
        help="Root directory that contains source folders such as 70G/90g/YJ.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("prepared_data/bottle_positive_only/split.json"),
        help="Path to the generated metadata json file.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for each source/type group.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    return parser.parse_args()


def collect_anomaly_items(source_root: Path) -> List[Dict]:
    items = []
    for image_path in sorted(source_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        relative_path = image_path.relative_to(source_root)
        if len(relative_path.parts) < 3:
            continue

        source_name = relative_path.parts[0]
        defect_type = relative_path.parts[1]

        items.append(
            {
                "path": str(image_path.as_posix()),
                "relative_path": str(relative_path.as_posix()),
                "label": 1,
                "source_name": source_name,
                "defect_type": defect_type,
                "group_key": f"{source_name}/{defect_type}",
            }
        )
    return items


def split_grouped_items(items: List[Dict], val_ratio: float, seed: int):
    rng = random.Random(seed)
    grouped_items = defaultdict(list)
    for item in items:
        grouped_items[item["group_key"]].append(item)

    train_items = []
    val_items = []

    for group_key in sorted(grouped_items):
        group = grouped_items[group_key]
        rng.shuffle(group)

        if len(group) == 1:
            train_items.extend(group)
            continue

        val_count = int(round(len(group) * val_ratio))
        val_count = max(1, min(len(group) - 1, val_count))

        val_items.extend(group[:val_count])
        train_items.extend(group[val_count:])

    if not val_items and len(train_items) > 1:
        val_items.append(train_items.pop())

    return train_items, val_items


def build_stats(train_items: List[Dict], val_items: List[Dict]):
    def summarize(items: List[Dict]):
        by_group = defaultdict(int)
        by_source = defaultdict(int)
        by_type = defaultdict(int)

        for item in items:
            by_group[item["group_key"]] += 1
            by_source[item["source_name"]] += 1
            by_type[item["defect_type"]] += 1

        return {
            "count": len(items),
            "by_group": dict(sorted(by_group.items())),
            "by_source": dict(sorted(by_source.items())),
            "by_defect_type": dict(sorted(by_type.items())),
        }

    return {
        "train": summarize(train_items),
        "val": summarize(val_items),
        "total_count": len(train_items) + len(val_items),
        "labels": {"normal": 0, "anomaly": 1},
        "note": (
            "Current metadata only contains anomaly images from own_datasets. "
            "This supports positive-only finetuning on top of the pretrained VisualAD checkpoint."
        ),
    }


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    source_root = args.source_root
    if not source_root.is_absolute():
        source_root = project_root / source_root
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    output_json = args.output_json
    if not output_json.is_absolute():
        output_json = project_root / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)

    items = collect_anomaly_items(source_root)
    if not items:
        raise RuntimeError(f"No image files found under {source_root}")

    train_items, val_items = split_grouped_items(items, val_ratio=args.val_ratio, seed=args.seed)
    stats = build_stats(train_items, val_items)

    payload = {
        "project_root": str(project_root.as_posix()),
        "source_root": str(source_root.as_posix()),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "train": train_items,
        "val": val_items,
        "stats": stats,
    }

    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    print(f"Saved metadata to: {output_json}")
    print(f"Total anomaly images: {stats['total_count']}")
    print(f"Train images: {stats['train']['count']}")
    print(f"Val images: {stats['val']['count']}")
    print("Train by group:")
    for key, value in stats["train"]["by_group"].items():
        print(f"  {key}: {value}")
    print("Val by group:")
    for key, value in stats["val"]["by_group"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
