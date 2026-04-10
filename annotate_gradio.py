from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# This local annotation app only serves files from disk and localhost.
# Clearing proxy env vars avoids Gradio/httpx import failures in shells that
# export SOCKS proxies without the optional socks extras installed.
for proxy_key in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
]:
    os.environ.pop(proxy_key, None)

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OUTPUT_DIRNAME = "_annotations"
CSV_COLUMNS = [
    "image_rel_path",
    "label",
    "anomaly",
    "mask_rel_path",
    "image_width",
    "image_height",
    "updated_at",
]


@dataclass
class AnnotationState:
    image_root: str = ""
    output_root: str = ""
    csv_path: str = ""
    images: List[str] = field(default_factory=list)
    annotations: Dict[str, Dict[str, str]] = field(default_factory=dict)
    current_index: int = 0


def resolve_root(image_root: str) -> Path:
    path = Path(image_root).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"图片目录不存在: {path}")
    if not path.is_dir():
        raise ValueError(f"输入必须是目录: {path}")
    return path


def scan_images(image_root: Path) -> List[Path]:
    images = []
    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        if OUTPUT_DIRNAME in path.parts:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    return sorted(images)


def ensure_output_paths(image_root: Path) -> tuple[Path, Path]:
    """
    labels.csv 仍保存在图片目录下的 _annotations 中
    """
    output_root = image_root / OUTPUT_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root, output_root / "labels.csv"


def ensure_mask_root(image_root: Path) -> Path:
    """
    mask 改为保存到：图片目录的上一层 / _annotations / masks
    """
    mask_root = image_root.parent / OUTPUT_DIRNAME / "masks"
    mask_root.mkdir(parents=True, exist_ok=True)
    return mask_root


def load_annotations(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        annotations = {}
        for row in reader:
            image_rel_path = row.get("image_rel_path", "").strip()
            if image_rel_path:
                annotations[image_rel_path] = row
        return annotations


def write_annotations(csv_path: Path, annotations: Dict[str, Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for key in sorted(annotations.keys()):
            row = annotations[key]
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def pil_rgba(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGBA")


def pil_rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def make_mask_path(image_root: Path, image_path: Path) -> tuple[Path, str]:
    """
    mask 保存到 image_root 的上一层目录：
    image_root.parent / _annotations / masks / <原图相对路径>.png

    CSV 中依然保存成相对于 image_root 的相对路径，例如：
    ../_annotations/masks/xxx.png
    """
    image_rel = image_path.relative_to(image_root)
    masks_root = ensure_mask_root(image_root)
    mask_path = masks_root / image_rel.with_suffix(".png")
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    mask_rel_path = os.path.relpath(mask_path, image_root).replace("\\", "/")
    return mask_path, mask_rel_path


def mask_to_overlay(mask: Image.Image) -> Image.Image:
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    overlay = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 3] = (mask_np > 0).astype(np.uint8) * 180
    return Image.fromarray(overlay, mode="RGBA")


def build_editor_value(background: Image.Image, mask: Optional[Image.Image]) -> Dict[str, object]:
    layers: List[Image.Image] = []
    composite = background.copy()

    if mask is not None:
        mask = mask.convert("L")
        if np.max(np.array(mask, dtype=np.uint8)) > 0:
            overlay = mask_to_overlay(mask)
            layers.append(overlay)
            composite = Image.alpha_composite(composite, overlay)

    return {
        "background": background,
        "layers": layers,
        "composite": composite,
    }


def load_image_editor_value(image_path: Path, mask_path: Optional[Path]) -> Dict[str, object]:
    background = pil_rgba(Image.open(image_path))
    mask_image = None

    if mask_path is not None and mask_path.exists():
        mask_image = Image.open(mask_path).convert("L")
        if mask_image.size != background.size:
            mask_image = mask_image.resize(background.size, Image.Resampling.NEAREST)

    return build_editor_value(background, mask_image)


def load_layer_rgba(layer: object) -> Optional[np.ndarray]:
    if layer is None:
        return None
    if isinstance(layer, Image.Image):
        return np.array(layer.convert("RGBA"))
    if isinstance(layer, np.ndarray):
        if layer.ndim == 2:
            rgba = np.zeros((layer.shape[0], layer.shape[1], 4), dtype=np.uint8)
            rgba[..., 3] = (layer > 0).astype(np.uint8) * 255
            return rgba
        if layer.ndim == 3 and layer.shape[2] == 4:
            return layer.astype(np.uint8)
        if layer.ndim == 3 and layer.shape[2] == 3:
            rgba = np.zeros((layer.shape[0], layer.shape[1], 4), dtype=np.uint8)
            rgba[..., :3] = layer.astype(np.uint8)
            rgba[..., 3] = 255
            return rgba
    return None


def extract_mask_from_editor(editor_value: object, image_size: tuple[int, int]) -> Image.Image:
    width, height = image_size
    merged_mask = np.zeros((height, width), dtype=np.uint8)

    if isinstance(editor_value, dict):
        for layer in editor_value.get("layers", []) or []:
            layer_rgba = load_layer_rgba(layer)
            if layer_rgba is None:
                continue

            # 关键修复：不再用 alpha>0，而是只识别红色画笔区域
            painted_mask = extract_red_brush_mask(layer_rgba)

            if painted_mask.shape != (height, width):
                painted_mask = np.array(
                    Image.fromarray(painted_mask, mode="L").resize(
                        (width, height), Image.Resampling.NEAREST
                    )
                )

            merged_mask = np.maximum(merged_mask, painted_mask)

    return Image.fromarray(merged_mask, mode="L")


def fill_enclosed_regions(mask: Image.Image) -> Image.Image:
    """
    将当前 mask 中“闭合轮廓”内部的区域全部填满。
    做法：
    - 当前已画的像素视为障碍/边界
    - 在其反相图上，从外部做 flood fill
    - 无法从外部到达的区域，视为闭合区域内部
    """
    mask_np = np.array(mask.convert("L"), dtype=np.uint8)
    boundary = mask_np > 0

    if not np.any(boundary):
        return mask

    free_space = np.where(boundary, 0, 255).astype(np.uint8)

    # 外围补一圈白边，保证从 (0,0) 一定代表“图外区域”
    padded = np.pad(free_space, ((1, 1), (1, 1)), mode="constant", constant_values=255)

    try:
        import cv2
        # 优先使用 OpenCV，速度更快且绝对稳定
        h, w = padded.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(padded, flood_mask, (0, 0), 128)
    except ImportError:
        # 关键修复：加入 .copy()，避免 PIL 的 floodfill 因直接操作 NumPy 内存视图而静默失效
        padded_img = Image.fromarray(padded, mode="L").copy()
        ImageDraw.floodfill(padded_img, (0, 0), 128)
        padded = np.array(padded_img, dtype=np.uint8)

    filled = padded[1:-1, 1:-1]
    enclosed = filled == 255

    result = np.where(boundary | enclosed, 255, 0).astype(np.uint8)
    return Image.fromarray(result, mode="L")


def is_mask_empty(mask: Image.Image) -> bool:
    return np.max(np.array(mask, dtype=np.uint8)) == 0


def current_image_path(state: AnnotationState) -> Path:
    return Path(state.images[state.current_index])


def current_image_rel_path(state: AnnotationState) -> str:
    return current_image_path(state).relative_to(Path(state.image_root)).as_posix()


def find_first_unlabeled_index(state: AnnotationState) -> int:
    for idx, image_path in enumerate(state.images):
        rel_path = Path(image_path).relative_to(Path(state.image_root)).as_posix()
        if rel_path not in state.annotations:
            return idx
    return 0


def count_summary(state: AnnotationState) -> str:
    total = len(state.images)
    labeled = len(state.annotations)
    normal = sum(1 for row in state.annotations.values() if row.get("anomaly") == "0")
    anomaly = sum(1 for row in state.annotations.values() if row.get("anomaly") == "1")
    unlabeled = total - labeled
    return (
        f"总图像: **{total}** | 已标注: **{labeled}** | 未标注: **{unlabeled}** | "
        f"正常: **{normal}** | 异常: **{anomaly}**"
    )


def current_position_text(state: AnnotationState) -> str:
    if not state.images:
        return "当前没有可标注图片"
    return f"第 **{state.current_index + 1} / {len(state.images)}** 张"


def saved_status_text(state: AnnotationState) -> str:
    if not state.images:
        return "未加载目录"
    rel_path = current_image_rel_path(state)
    row = state.annotations.get(rel_path)
    if not row:
        return "当前图片: **未标注**"
    label = row.get("label", "")
    updated_at = row.get("updated_at", "")
    return f"当前图片: **已标注为 {label}**  | 最近保存: `{updated_at}`"


def render_current_view(state: AnnotationState, message: str = ""):
    if not state.images:
        empty_editor = None
        return (
            state,
            "未加载目录",
            "无图片",
            "",
            gr.update(value=None),
            gr.update(value=empty_editor),
            gr.update(minimum=1, maximum=1, step=1, value=1, interactive=False),
            message,
        )

    image_path = current_image_path(state)
    image_rel_path = current_image_rel_path(state)
    row = state.annotations.get(image_rel_path, {})
    label_value = row.get("label") or None
    mask_rel_path = row.get("mask_rel_path", "").strip()
    mask_path = Path(state.image_root) / mask_rel_path if mask_rel_path else None
    editor_value = load_image_editor_value(image_path, mask_path if mask_path and mask_path.exists() else None)

    return (
        state,
        current_position_text(state),
        count_summary(state),
        f"`{image_rel_path}`",
        gr.update(value=label_value),
        gr.update(value=editor_value),
        gr.update(minimum=1, maximum=len(state.images), step=1, value=state.current_index + 1, interactive=True),
        message or saved_status_text(state),
    )


def normalize_label(label: Optional[str]) -> Optional[str]:
    if label in {"正常", "异常"}:
        return label
    return None


def save_current_annotation(
    state: AnnotationState,
    label: Optional[str],
    editor_value: object,
    move_next: bool = False,
):
    if not state.images:
        return render_current_view(state, "请先加载图片目录")

    label = normalize_label(label)
    if label is None:
        return render_current_view(state, "请先选择当前图片的标签")

    image_root = Path(state.image_root)
    csv_path = Path(state.csv_path)
    image_path = current_image_path(state)
    image_rel_path = image_path.relative_to(image_root).as_posix()
    image_rgb = pil_rgb(Image.open(image_path))
    image_width, image_height = image_rgb.size

    existing_row = state.annotations.get(image_rel_path, {})
    old_mask_rel_path = existing_row.get("mask_rel_path", "").strip()
    old_mask_path = image_root / old_mask_rel_path if old_mask_rel_path else None

    mask_rel_path = ""
    if label == "异常":
        mask_image = extract_mask_from_editor(editor_value, image_rgb.size)
        if is_mask_empty(mask_image):
            return render_current_view(state, "异常图片必须绘制 mask 后才能保存")
        mask_path, mask_rel_path = make_mask_path(image_root, image_path)
        mask_image.save(mask_path)
    elif old_mask_path is not None and old_mask_path.exists():
        old_mask_path.unlink()

    state.annotations[image_rel_path] = {
        "image_rel_path": image_rel_path,
        "label": label,
        "anomaly": "1" if label == "异常" else "0",
        "mask_rel_path": mask_rel_path,
        "image_width": str(image_width),
        "image_height": str(image_height),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    write_annotations(csv_path, state.annotations)

    if move_next and state.current_index < len(state.images) - 1:
        state.current_index += 1
        return render_current_view(state, "已保存，已跳转到下一张")

    return render_current_view(state, "已保存当前标注")


def load_folder(image_root: str):
    try:
        root = resolve_root(image_root)
        images = scan_images(root)
        if not images:
            raise ValueError(f"目录下没有找到支持的图片文件: {root}")

        output_root, csv_path = ensure_output_paths(root)
        mask_root = ensure_mask_root(root)

        annotations = load_annotations(csv_path)
        state = AnnotationState(
            image_root=str(root),
            output_root=str(output_root),
            csv_path=str(csv_path),
            images=[str(path) for path in images],
            annotations=annotations,
            current_index=0,
        )
        state.current_index = find_first_unlabeled_index(state)
        return render_current_view(
            state,
            f"目录已加载: `{root}` | CSV: `{csv_path}` | 掩码目录: `{mask_root}`",
        )
    except Exception as exc:
        empty_state = AnnotationState()
        return render_current_view(empty_state, f"加载失败: {exc}")


def go_previous(state: AnnotationState):
    if not state.images:
        return render_current_view(state, "请先加载图片目录")
    state.current_index = max(0, state.current_index - 1)
    return render_current_view(state)


def go_next(state: AnnotationState):
    if not state.images:
        return render_current_view(state, "请先加载图片目录")
    state.current_index = min(len(state.images) - 1, state.current_index + 1)
    return render_current_view(state)


def go_to_index(state: AnnotationState, index_value: float):
    if not state.images:
        return render_current_view(state, "请先加载图片目录")
    index = int(index_value) - 1
    state.current_index = min(max(index, 0), len(state.images) - 1)
    return render_current_view(state)


def go_next_unlabeled(state: AnnotationState):
    if not state.images:
        return render_current_view(state, "请先加载图片目录")

    total = len(state.images)
    for offset in range(1, total + 1):
        idx = (state.current_index + offset) % total
        rel_path = Path(state.images[idx]).relative_to(Path(state.image_root)).as_posix()
        if rel_path not in state.annotations:
            state.current_index = idx
            return render_current_view(state, "已跳转到下一张未标注图片")

    return render_current_view(state, "所有图片都已完成标注")


def reset_mask_canvas(state: AnnotationState):
    if not state.images:
        return gr.update(value=None), "请先加载图片目录"
    image_path = current_image_path(state)
    editor_value = load_image_editor_value(image_path, None)
    return gr.update(value=editor_value), "已清空当前 mask 图层"


def fill_current_region_to_mask(state: AnnotationState, editor_value: object):
    """
    点击按钮后，将当前已圈出的闭合区域内部全部填成 mask
    """
    if not state.images:
        return gr.update(value=None), "请先加载图片目录"

    image_path = current_image_path(state)
    image_rgba = pil_rgba(Image.open(image_path))

    # 这里只提取红色画笔内容
    current_mask = extract_mask_from_editor(editor_value, image_rgba.size)

    if is_mask_empty(current_mask):
        return gr.update(value=editor_value), "请先用红色画笔圈出一个闭合区域"

    filled_mask = fill_enclosed_regions(current_mask)

    current_np = np.array(current_mask, dtype=np.uint8)
    filled_np = np.array(filled_mask, dtype=np.uint8)

    if np.array_equal(current_np, filled_np):
        return gr.update(value=editor_value), "没有检测到闭合区域，请把轮廓画闭合一些"

    new_editor_value = build_editor_value(image_rgba, filled_mask)
    return gr.update(value=new_editor_value), "已将闭合区域内部全部设为 mask"


def create_demo(default_root: str = "") -> gr.Blocks:
    with gr.Blocks(title="瓶盖缺陷标注系统") as demo:
        gr.Markdown(
            """
            # 瓶盖缺陷标注系统
            - 输入一个图片目录，系统会递归扫描其中所有图片
            - 每张图片标注为 `正常` 或 `异常`
            - `异常` 时需要在右侧画出缺陷 mask
            - 可以先圈出区域，再点击 **区域内全部设为 mask**
            - 标注结果保存到 `图片目录/_annotations/labels.csv`
            - mask 保存到 `图片目录的上一层/_annotations/masks/<原图相对路径>.png`
            """
        )

        state = gr.State(AnnotationState())

        with gr.Row():
            image_root_input = gr.Textbox(
                label="图片目录",
                value=default_root,
                placeholder="输入要标注的图片文件夹路径",
                scale=8,
            )
            load_button = gr.Button("加载目录", variant="primary", scale=1)

        with gr.Row():
            position_md = gr.Markdown("未加载目录")
            summary_md = gr.Markdown("无图片")

        current_path_md = gr.Markdown("")
        message_md = gr.Markdown("等待加载目录")

        with gr.Row():
            with gr.Column(scale=2):
                label_radio = gr.Radio(
                    choices=["正常", "异常"],
                    label="图像级标签",
                    value=None,
                )

                with gr.Row():
                    prev_button = gr.Button("上一张")
                    next_button = gr.Button("下一张")
                    next_unlabeled_button = gr.Button("下一张未标注", variant="secondary")

                with gr.Row():
                    save_button = gr.Button("保存", variant="primary")
                    save_next_button = gr.Button("保存并下一张", variant="primary")
                    clear_mask_button = gr.Button("清空 mask")

                with gr.Row():
                    fill_region_button = gr.Button("区域内全部设为 mask", variant="secondary")

                index_slider = gr.Slider(
                    minimum=1,
                    maximum=1,
                    step=1,
                    value=1,
                    label="跳转到第几张",
                    interactive=False,
                )

                gr.Markdown(
                    """
                    **标注优化说明**
                    - 默认自动恢复已保存的标签和 mask
                    - `保存并下一张` 适合连续标注
                    - `下一张未标注` 适合断点续标
                    - 标为 `正常` 时，mask 会被忽略
                    - 标为 `异常` 时，必须绘制 mask 才能保存
                    - 先用画笔把区域边界圈出来，再点击 `区域内全部设为 mask`
                    """
                )

            with gr.Column(scale=5):
                image_editor = gr.ImageEditor(
                    label="异常区域标注（在原图上直接涂抹）",
                    type="pil",
                    image_mode="RGBA",
                    height=720,
                    brush=gr.Brush(colors=["#ff0000"], color_mode="fixed", default_size=24),
                    eraser=gr.Eraser(default_size=28),
                    interactive=True,
                )

        output_targets = [
            state,
            position_md,
            summary_md,
            current_path_md,
            label_radio,
            image_editor,
            index_slider,
            message_md,
        ]

        load_button.click(load_folder, inputs=[image_root_input], outputs=output_targets)
        prev_button.click(go_previous, inputs=[state], outputs=output_targets)
        next_button.click(go_next, inputs=[state], outputs=output_targets)
        next_unlabeled_button.click(go_next_unlabeled, inputs=[state], outputs=output_targets)
        index_slider.release(go_to_index, inputs=[state, index_slider], outputs=output_targets)
        save_button.click(save_current_annotation, inputs=[state, label_radio, image_editor], outputs=output_targets)
        save_next_button.click(
            lambda s, l, e: save_current_annotation(s, l, e, move_next=True),
            inputs=[state, label_radio, image_editor],
            outputs=output_targets,
        )
        clear_mask_button.click(reset_mask_canvas, inputs=[state], outputs=[image_editor, message_md])
        fill_region_button.click(
            fill_current_region_to_mask,
            inputs=[state, image_editor],
            outputs=[image_editor, message_md],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio image annotation tool for bottle defects")
    parser.add_argument(
        "--image-root",
        type=str,
        default="/Users/majingzhe/Desktop/瓶盖缺陷检测论文整理/检测样本",
        help="Default image root shown in the UI",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Gradio server")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser.parse_args()



def extract_red_brush_mask(layer_rgba: np.ndarray) -> np.ndarray:
    """
    只提取用户用红色画笔画出来的区域，避免把整层 alpha>0 都当成 mask。
    这里按“明显偏红”的像素判定。
    """
    rgba = layer_rgba.astype(np.uint8)
    r = rgba[..., 0]
    g = rgba[..., 1]
    b = rgba[..., 2]
    a = rgba[..., 3]

    painted = (
        (a > 0)
        & (r >= 160)
        & (r >= g + 60)
        & (r >= b + 60)
        & (g <= 140)
        & (b <= 140)
    )
    return painted.astype(np.uint8) * 255
def main() -> None:
    args = parse_args()
    demo = create_demo(default_root=args.image_root)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()