# -*- coding: utf-8 -*-
"""
Generate Monte Carlo prediction data for the requested target points.

This script intentionally does not plot. It writes numerical results to .npy
files that can be loaded by plot_results.py.
"""

import os

import uqlib as uq


N_SAMPLES = 1000
OUTPUT_DIR = os.path.join("results", "prediction")


def main():
    xfoil_model, corrector, _, _ = uq.build_default_models()
    desired_path = uq.existing_path(uq.DEFAULT_DESIRED_PATH, uq.DEFAULT_DESIRED_FALLBACK_PATH)
    pred_x = uq.loadTrainingData(desired_path, [0, 1, 2], [])[0]
    results = uq.run_monte_carlo(pred_x, xfoil_model, corrector, N_SAMPLES)
    uq.save_npy_results(results, OUTPUT_DIR)
    print(f"Saved prediction .npy files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
