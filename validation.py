# -*- coding: utf-8 -*-
"""
Generate Monte Carlo validation data for the seven provided truth points.

This script intentionally does not plot. It writes numerical results to .npy
files that can be loaded by plot_results.py.
"""

import os

import uqlib as uq


N_SAMPLES = 1000
OUTPUT_DIR = os.path.join("results", "validation")


def main():
    xfoil_model, corrector, true_x, true_y = uq.build_default_models()
    results = uq.run_monte_carlo(true_x, xfoil_model, corrector, N_SAMPLES)
    results["y_true"] = true_y.detach().cpu().numpy()
    uq.save_npy_results(results, OUTPUT_DIR)
    print(f"Saved validation .npy files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
