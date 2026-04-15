import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path


CLASS_NAME_MAP = {
    "勺子": "spoon",
    "标签": "label",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def natural_sort_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def parse_folder(folder_name: str):
    lower_name = folder_name.lower()
    cls_name = None
    for zh_name, en_name in CLASS_NAME_MAP.items():
        if zh_name in folder_name:
            cls_name = en_name
            break
    if cls_name is None:
        raise ValueError(f"Unsupported class folder name: {folder_name}")

    if "ok" in lower_name:
        specie_name = "good"
        anomaly = 0
    elif "ng" in lower_name:
        specie_name = "ng"
        anomaly = 1
    else:
        raise ValueError(f"Unsupported status folder name: {folder_name}")

    return cls_name, specie_name, anomaly


def collect_images(folder: Path):
    return sorted(
        [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_sort_key,
    )


def split_paths(paths, train_ratio: float, seed: int):
    if not paths:
        return [], []

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []

    train_count = int(len(shuffled) * train_ratio)
    train_count = max(1, min(train_count, len(shuffled) - 1))
    return sorted(shuffled[:train_count], key=natural_sort_key), sorted(shuffled[train_count:], key=natural_sort_key)


def materialize_image(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    try:
        if mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_record(dataset_root: Path, image_path: Path, cls_name: str, specie_name: str, anomaly: int):
    record = {
        "img_path": image_path.relative_to(dataset_root).as_posix(),
        "mask_path": "",
        "cls_name": cls_name,
        "specie_name": specie_name,
        "anomaly": anomaly,
    }
    return record


def ensure_ground_truth_dir(dataset_root: Path, cls_name: str, specie_name: str):
    if specie_name != "good":
        (dataset_root / cls_name / "ground_truth" / specie_name).mkdir(parents=True, exist_ok=True)


def convert_dataset(src_root: Path, dst_root: Path, train_ratio: float, seed: int, mode: str):
    grouped_paths = {}
    for folder in sorted(src_root.iterdir(), key=lambda p: p.name):
        if not folder.is_dir():
            continue
        cls_name, specie_name, anomaly = parse_folder(folder.name)
        grouped_paths.setdefault((cls_name, specie_name, anomaly), []).extend(collect_images(folder))

    if not grouped_paths:
        raise RuntimeError(f"No supported image folders found under {src_root}")

    meta = {"train": {}, "test": {}}
    summary = {}

    for (cls_name, specie_name, anomaly), paths in sorted(grouped_paths.items()):
        train_paths, test_paths = split_paths(paths, train_ratio=train_ratio, seed=seed)
        summary.setdefault(cls_name, {})[specie_name] = {
            "total": len(paths),
            "train": len(train_paths),
            "test": len(test_paths),
        }
        ensure_ground_truth_dir(dst_root, cls_name, specie_name)

        for phase, phase_paths in (("train", train_paths), ("test", test_paths)):
            phase_dir = dst_root / cls_name / phase / specie_name
            phase_dir.mkdir(parents=True, exist_ok=True)
            meta[phase].setdefault(cls_name, [])
            for src_path in phase_paths:
                dst_path = phase_dir / src_path.name
                materialize_image(src_path, dst_path, mode=mode)
                meta[phase][cls_name].append(
                    build_record(dst_root, dst_path, cls_name=cls_name, specie_name=specie_name, anomaly=anomaly)
                )

    for phase in ("train", "test"):
        for cls_name in sorted(meta[phase].keys()):
            meta[phase][cls_name].sort(key=lambda item: (item["specie_name"], item["img_path"]))

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    (dst_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    print(f"Converted dataset written to: {dst_root}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser("Prepare datasets_v2 for bottle_VisualAD")
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path("/Users/majingzhe/Desktop/瓶盖缺陷检测论文整理/数据集/datasets_v2"),
        help="source dataset root with Chinese folder names",
    )
    parser.add_argument(
        "--dst_root",
        type=Path,
        default=Path("/Users/majingzhe/Desktop/瓶盖缺陷检测论文整理/数据集/datasets_v2_visualad"),
        help="output dataset root for bottle_VisualAD",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="split ratio for train set")
    parser.add_argument("--seed", type=int, default=42, help="random seed for split")
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="how to materialize images into output dataset",
    )
    args = parser.parse_args()

    if not args.src_root.exists():
        raise FileNotFoundError(f"Source dataset root does not exist: {args.src_root}")
    if args.src_root.resolve() == args.dst_root.resolve():
        raise ValueError("src_root and dst_root must be different directories")

    convert_dataset(args.src_root, args.dst_root, args.train_ratio, args.seed, args.mode)


if __name__ == "__main__":
    main()
