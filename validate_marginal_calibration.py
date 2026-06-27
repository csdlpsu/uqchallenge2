# -*- coding: utf-8 -*-
"""
Validate marginal distribution calibration at the overlapping truth/prediction point.
"""

import os
import sys

import numpy as np


RESULTS_DIR = os.path.join("results", "distributional_calibrated_prediction")
MEAN_TOL = 5e-12
STD_TOL = 5e-12
QUANTILE_TOL_FACTOR = 0.25


def load_result(name):
    return np.load(os.path.join(RESULTS_DIR, f"{name}.npy"), allow_pickle=True)


def main():
    x_points = load_result("x_points")
    samples = load_result("y_marginal_calibrated_samples")
    left_edges = load_result("interval_left")
    right_edges = load_result("interval_right")
    output_labels = load_result("output_labels")

    overlap_indices = np.where(
        np.isclose(x_points[:, 1], 0.0) & np.isclose(x_points[:, 2], 0.0)
    )[0]
    if len(overlap_indices) == 0:
        raise AssertionError("No alpha=0, flap=0 prediction point found.")

    case_idx = int(overlap_indices[0])
    case_samples = samples[case_idx]
    target_center = 0.5 * (left_edges[case_idx] + right_edges[case_idx])
    target_std = (right_edges[case_idx] - left_edges[case_idx]) / (2 * 1.96)
    sample_mean = np.mean(case_samples, axis=0)
    sample_std = np.std(case_samples, axis=0)
    sample_quantiles = np.quantile(case_samples, [0.025, 0.975], axis=0)
    quantile_tol = QUANTILE_TOL_FACTOR * (right_edges[case_idx] - left_edges[case_idx])

    failures = []
    for output_idx, label in enumerate(output_labels):
        mean_error = abs(sample_mean[output_idx] - target_center[output_idx])
        std_error = abs(sample_std[output_idx] - target_std[output_idx])
        left_error = abs(sample_quantiles[0, output_idx] - left_edges[case_idx, output_idx])
        right_error = abs(sample_quantiles[1, output_idx] - right_edges[case_idx, output_idx])
        print(
            f"{label}: mean_error={mean_error:.3e}, "
            f"std_error={std_error:.3e}, "
            f"left_q_error={left_error:.3e}, "
            f"right_q_error={right_error:.3e}"
        )
        if mean_error > MEAN_TOL:
            failures.append(f"{label} mean mismatch")
        if std_error > STD_TOL:
            failures.append(f"{label} std mismatch")
        if left_error > quantile_tol[output_idx]:
            failures.append(f"{label} left quantile mismatch")
        if right_error > quantile_tol[output_idx]:
            failures.append(f"{label} right quantile mismatch")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print("Marginal calibration check passed for alpha=0, flap=0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
