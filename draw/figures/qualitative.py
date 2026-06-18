"""Qualitative paired edit grids for SD3 v3.1 experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .io import find_image_samples, to_float, to_int
from .style import (
    TOKENS,
    add_center_header,
    add_manifest_records,
    format_axis,
    label_mode,
    mode_sort_key,
    save_figure,
    slugify,
    use_paper_style,
)


MAIN_METHODS = ["textual_only", "E", "O", "A", "B", "C", "D", "FMP_single", "FMP_multi"]
GRID_COLUMNS = ["Original", "Protected", "Clean edit", "Protected edit", "Abs. difference"]


def _load_image(path: Path | None, size: tuple[int, int] | None = None) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def _diff_heatmap(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return None
    diff = np.mean(np.abs(a - b), axis=2)
    hi = np.percentile(diff, 99)
    if hi <= 1e-8:
        norm = np.zeros_like(diff)
    else:
        norm = np.clip(diff / hi, 0, 1)
    cmap = plt.get_cmap("magma")
    return cmap(norm)[..., :3]


def _sample_priority(sample: dict, target_sigma: float | None) -> tuple:
    sigma = to_float(sample.get("sigma"))
    seed = to_int(sample.get("seed"))
    return (
        -(abs((sigma or 0.0) - target_sigma) if target_sigma is not None and sigma is not None else 0.0),
        to_int(sample.get("epsilon")) or -1,
        to_int(sample.get("steps")) or -1,
        1 if seed == 0 else 0,
        -(abs(seed or 0)),
    )


def _select_samples(samples: list[dict], image: str | None, sigma: str | None, methods: list[str]) -> dict[str, list[dict]]:
    target_sigma = to_float(sigma)
    filtered = []
    for sample in samples:
        if image and sample.get("image_id") != image:
            continue
        if methods and sample.get("mode") not in methods:
            continue
        if sample.get("clean_edit_path") is None or sample.get("adv_edit_path") is None:
            continue
        filtered.append(sample)
    by_image_mode: dict[tuple[str, str], dict] = {}
    priorities: dict[tuple[str, str], tuple] = {}
    for sample in filtered:
        key = (str(sample.get("image_id")), str(sample.get("mode")))
        priority = _sample_priority(sample, target_sigma)
        if key not in by_image_mode or priority > priorities[key]:
            by_image_mode[key] = sample
            priorities[key] = priority
    by_image: dict[str, list[dict]] = {}
    for (image_id, _mode), sample in by_image_mode.items():
        by_image.setdefault(image_id, []).append(sample)
    for image_id in by_image:
        by_image[image_id].sort(key=lambda s: mode_sort_key(str(s.get("mode"))))
    return by_image


def plot_qualitative_grid(
    image_id: str,
    samples: list[dict],
    out_dir: Path,
    *,
    formats: list[str],
    dpi: int,
    manifest: list[dict] | None = None,
    source: str = "experiment outputs",
) -> list[Path]:
    if not samples:
        return []
    rows = len(samples)
    fig, axes = plt.subplots(rows, len(GRID_COLUMNS), figsize=(10.6, max(2.2, 1.55 * rows)), squeeze=False)
    for row_idx, sample in enumerate(samples):
        protected = _load_image(Path(sample["attacked_path"]))
        size = None
        if protected is not None:
            size = (protected.shape[1], protected.shape[0])
        original = _load_image(Path(sample["original_path"]) if sample.get("original_path") else None, size=size)
        clean = _load_image(Path(sample["clean_edit_path"]))
        adv = _load_image(Path(sample["adv_edit_path"]), size=(clean.shape[1], clean.shape[0]) if clean is not None else None)
        diff = _diff_heatmap(clean, adv)
        images = [original, protected, clean, adv, diff]
        mode = str(sample.get("mode", ""))
        row_label = f"{label_mode(mode)}\neps={sample.get('epsilon', '')}, seed={sample.get('seed', '')}, sigma={sample.get('sigma', '')}"
        for col_idx, (ax, arr) in enumerate(zip(axes[row_idx], images)):
            if arr is not None:
                ax.imshow(np.clip(arr, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(GRID_COLUMNS[col_idx], fontsize=8.2, fontweight="semibold", color=TOKENS["ink"], pad=5)
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=7.0, color=TOKENS["ink"], rotation=0, labelpad=42, va="center")
        format_axis(axes[row_idx, 0], x_grid=False, y_grid=False)
    add_center_header(
        fig,
        f"{image_id} paired SDEdit qualitative grid",
        "Rows are methods; columns show original, protected image, paired clean edit, protected edit, and absolute edit difference heatmap.",
        title_size=10.2,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.86, bottom=0.035, wspace=0.045, hspace=0.12)
    sigma = samples[0].get("sigma", "mixed")
    figure_id = f"paired_examples_{slugify(image_id)}_sigma{slugify(sigma)}"
    paths = save_figure(fig, out_dir / figure_id, formats=formats, dpi=dpi)
    plt.close(fig)
    if manifest is not None:
        add_manifest_records(
            manifest,
            figure_id=figure_id,
            paths=paths,
            source=source,
            family="Tables & Scorecards / Image Grid",
            description="Qualitative paired SDEdit image grid with absolute difference heatmap.",
            expected_effect="Protected images should remain visually close to originals while FMP protected edits show stronger deviations from clean edits.",
        )
    return paths


def generate_qualitative_figures(
    *,
    root: str | Path,
    out_dir: str | Path,
    image: str | None = None,
    sigma: str | None = "0.3",
    methods: list[str] | None = None,
    formats: list[str] | None = None,
    dpi: int = 300,
    manifest: list[dict] | None = None,
) -> list[Path]:
    use_paper_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["png", "svg"]
    samples = find_image_samples(root)
    by_image = _select_samples(samples, image, sigma, methods or MAIN_METHODS)
    if not by_image:
        print(f"No paired qualitative samples found under {root}")
        return []
    saved: list[Path] = []
    for image_id, selected in sorted(by_image.items()):
        saved.extend(plot_qualitative_grid(image_id, selected, out_dir, formats=formats, dpi=dpi, manifest=manifest, source=str(root)))
    for path in saved:
        print(f"Saved: {path}")
    return saved

