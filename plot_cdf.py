#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical CDF coordinates for a 1D array."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values, values
    xs = np.sort(values)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def load_ranks(summary_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load gene and variant ranks from all summary JSON files in a directory.

    Expects fields:
      - gene_rank_after_prelimin8
      - variant_rank_after_round_robin
    """
    gene_ranks: list[float] = []
    variant_ranks: list[float] = []

    for path in sorted(summary_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        if "gene_rank_after_prelimin8" in summary:
            gene_ranks.append(summary["gene_rank_after_prelimin8"])
        if "variant_rank_after_round_robin" in summary:
            variant_ranks.append(summary["variant_rank_after_round_robin"])

    return np.array(gene_ranks, dtype=float), np.array(variant_ranks, dtype=float)


def plot_and_save_cdf(values: np.ndarray, xlabel: str, title: str, out_path: Path) -> None:
    if values.size == 0:
        print(f"Skipping plot {out_path.name}: no data.")
        return

    xs, ys = empirical_cdf(values)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.step(xs, ys, where="post")
    plt.scatter(xs, ys, s=20)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.xlim(left=0)
    plt.ylim(0, 1)
    
    # Set integer ticks only
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))  # Ticks at 0.1 intervals
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {title} to {out_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    summary_dir = base_dir / "data" / "summary"
    plots_dir = base_dir / "plots"

    if not summary_dir.is_dir():
        print(f"No summary directory found at {summary_dir}")
        return

    gene_ranks, variant_ranks = load_ranks(summary_dir)

    print(f"Found {gene_ranks.size} gene ranks and {variant_ranks.size} variant ranks.")

    plot_and_save_cdf(
        gene_ranks,
        xlabel="Gene rank after Prelimin8",
        title="CDF of gene ranks after Prelimin8",
        out_path=plots_dir / "gene_rank_prelimin8_cdf.png",
    )

    plot_and_save_cdf(
        variant_ranks,
        xlabel="Variant rank after Round Robin",
        title="CDF of variant ranks after Round Robin",
        out_path=plots_dir / "variant_rank_round_robin_cdf.png",
    )


if __name__ == "__main__":
    main()


