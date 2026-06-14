# -*- coding: utf-8 -*-
"""
Plot Monte Carlo results saved by validation.py and prediction.py.
"""

import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "uqchallenge2-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "uqchallenge2-cache")
)

import matplotlib.pyplot as plt

import uqlib as uq


DEFAULT_DATA_ROOT = "results"
DEFAULT_PLOT_ROOT = "plots"


def plot_case(results, case_idx, output_dir, include_truth):
    x0 = results["x_points"][case_idx]
    output_labels = results.get("output_labels", uq.OUTPUT_LABELS)
    aoa = int(x0[1])
    flap = int(x0[2])

    for output_idx, label in enumerate(output_labels):
        plt.figure(figsize=(7, 5))

        plt.hist(
            results["y_base"][case_idx, :, output_idx],
            bins=40,
            alpha=0.5,
            label="Base",
        )
        plt.hist(
            results["y_combined"][case_idx, :, output_idx],
            bins=40,
            alpha=0.5,
            label="Corrected",
        )

        if include_truth and "y_true" in results:
            plt.axvline(
                results["y_true"][case_idx, output_idx],
                color="black",
                linewidth=2,
                label="True",
            )

        plt.axvline(
            results["y_center_base"][case_idx, output_idx],
            color="blue",
            linestyle="--",
            label="Base @ point",
        )
        plt.axvline(
            results["y_center_combined"][case_idx, output_idx],
            color="orange",
            linestyle="--",
            label="Corrected @ point",
        )

        plt.xlabel(str(label))
        plt.ylabel("Frequency")
        plt.title(f"MC ({aoa} deg, flap={flap} deg) - {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mc_aoa{aoa}_flap{flap}_{label}.png"))
        plt.close()


def plot_dataset(name, data_root=DEFAULT_DATA_ROOT, plot_root=DEFAULT_PLOT_ROOT):
    input_dir = os.path.join(data_root, name)
    output_dir = os.path.join(plot_root, name)
    os.makedirs(output_dir, exist_ok=True)

    results = uq.load_npy_results(input_dir)
    include_truth = name == "validation"

    for case_idx in range(len(results["x_points"])):
        plot_case(results, case_idx, output_dir, include_truth)

    print(f"Saved {name} plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot saved Monte Carlo .npy results."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=["validation", "prediction", "all"],
        help="Which saved result set to plot.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--plot-root", default=DEFAULT_PLOT_ROOT)
    args = parser.parse_args()

    datasets = ["validation", "prediction"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        plot_dataset(dataset, args.data_root, args.plot_root)


if __name__ == "__main__":
    main()
