#!/usr/bin/env python3
"""One-command SD3 v3.1 paper figure generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from .ablations import generate_ablation_figures
from .diagnostics import generate_diagnostic_figures
from .io import read_csv_rows
from .loss_curves import generate_loss_figures
from .metric_comparison import generate_metric_figures
from .qualitative import generate_qualitative_figures
from .style import write_manifest


def _csv_has_rows(path: Path) -> bool:
    return bool(read_csv_rows(path))


def _parse_formats(value: str) -> list[str]:
    return [part.strip().lower().lstrip(".") for part in value.split(",") if part.strip()]


def generate_all_figures(
    *,
    root: Path,
    metrics: Path,
    analysis_dir: Path,
    out_dir: Path,
    loss_root: Path | None,
    legacy_loss_root: Path | None,
    loss_image: str | None,
    all_loss_images: bool,
    qualitative_image: str | None,
    qualitative_sigma: str | None,
    formats: list[str],
    dpi: int,
    skip_metrics: bool = False,
    skip_loss: bool = False,
    skip_qualitative: bool = False,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    paired = analysis_dir / "paired_differences.csv"

    if not skip_metrics:
        if metrics.exists() and _csv_has_rows(metrics):
            generate_metric_figures(
                metrics_path=metrics,
                paired_differences_path=paired if paired.exists() else None,
                out_dir=out_dir / "main",
                formats=formats,
                dpi=dpi,
                manifest=manifest,
            )
            generate_ablation_figures(
                metrics_path=metrics,
                out_dir=out_dir / "ablation",
                formats=formats,
                dpi=dpi,
                manifest=manifest,
            )
            generate_diagnostic_figures(
                metrics_path=metrics,
                out_dir=out_dir / "diagnostics",
                formats=formats,
                dpi=dpi,
                manifest=manifest,
            )
        else:
            print(f"Skip metric figures: metrics file missing or empty: {metrics}")

    if not skip_loss:
        for candidate_root, label in [(loss_root or root, "v3"), (legacy_loss_root, "legacy")]:
            if not candidate_root or not candidate_root.exists():
                continue
            try:
                generate_loss_figures(
                    root=candidate_root,
                    output=out_dir / "loss" / label,
                    image=loss_image,
                    all_images=all_loss_images,
                    components=True,
                    formats=formats,
                    dpi=dpi,
                    manifest=manifest,
                )
            except ValueError as exc:
                print(f"Skip loss figures for {candidate_root}: {exc}")

    if not skip_qualitative:
        if root.exists():
            generate_qualitative_figures(
                root=root,
                out_dir=out_dir / "qualitative",
                image=qualitative_image,
                sigma=qualitative_sigma,
                formats=formats,
                dpi=dpi,
                manifest=manifest,
            )
        else:
            print(f"Skip qualitative grid: root missing: {root}")

    write_manifest(manifest, out_dir / "figure_manifest.csv")
    print(f"figure manifest written to {out_dir / 'figure_manifest.csv'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all SD3 v3.1 paper figures")
    parser.add_argument("--root", default="out_sd3_v3", help="v3 experiment output root")
    parser.add_argument("--metrics", default=None, help="full_metrics.csv path")
    parser.add_argument("--analysis-dir", default=None, help="analysis table directory")
    parser.add_argument("--out-dir", default=None, help="figure output directory")
    parser.add_argument("--loss-root", default=None, help="primary loss-curve root; defaults to --root")
    parser.add_argument("--legacy-loss-root", default="", help="optional legacy loss root for v2/origin comparison figures")
    parser.add_argument("--loss-image", default="suzume", help="image id for representative loss curves")
    parser.add_argument("--all-loss-images", action="store_true", help="plot every available image loss curve")
    parser.add_argument("--qualitative-image", default=None, help="optional image id for qualitative grids")
    parser.add_argument("--qualitative-sigma", default="0.3", help="preferred paired SDEdit sigma for qualitative grids")
    parser.add_argument("--formats", default="png,svg", help="comma-separated output formats")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--skip-qualitative", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    metrics = Path(args.metrics) if args.metrics else root / "full_metrics.csv"
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else root / "analysis"
    out_dir = Path(args.out_dir) if args.out_dir else root / "figures"
    loss_root = Path(args.loss_root) if args.loss_root else None
    legacy_loss_root = Path(args.legacy_loss_root) if args.legacy_loss_root else None
    formats = _parse_formats(args.formats) or ["png"]
    generate_all_figures(
        root=root,
        metrics=metrics,
        analysis_dir=analysis_dir,
        out_dir=out_dir,
        loss_root=loss_root,
        legacy_loss_root=legacy_loss_root,
        loss_image=args.loss_image,
        all_loss_images=args.all_loss_images,
        qualitative_image=args.qualitative_image,
        qualitative_sigma=args.qualitative_sigma,
        formats=formats,
        dpi=args.dpi,
        skip_metrics=args.skip_metrics,
        skip_loss=args.skip_loss,
        skip_qualitative=args.skip_qualitative,
    )


if __name__ == "__main__":
    main()
