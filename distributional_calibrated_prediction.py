# -*- coding: utf-8 -*-
"""
Generate distributionally calibrated corrected GP prediction data.

This script uses the epistemic bounds in
uq_challenge_ground_truth_epistemic_bounds.xlsx as target posterior moments for
the corrected GP, then saves a separate set of Monte Carlo results for the four
prediction points.
"""

import os

import numpy as np
import openpyxl
import torch

import uqlib as uq


N_SAMPLES = 1000
N_MARGINAL_CALIBRATION_SAMPLES = 1000
BOUNDS_XLSX = "uq_challenge_ground_truth_epistemic_bounds.xlsx"
OUTPUT_DIR = os.path.join("results", "distributional_calibrated_prediction")


def load_prediction_bounds(workbook_path=BOUNDS_XLSX):
    workbook = openpyxl.load_workbook(workbook_path, data_only=True)
    sheet = workbook["Wide Format"]

    headers = None
    rows = []
    for row in sheet.iter_rows(values_only=True):
        if "alpha (deg)" in row and "CL left edge" in row:
            headers = {value: idx for idx, value in enumerate(row) if value is not None}
            continue

        if headers is None or row[headers["alpha (deg)"]] is None:
            continue

        rows.append(row)

    x_points = []
    left_edges = []
    right_edges = []
    for row in rows:
        alpha = float(row[headers["alpha (deg)"]])
        flap = float(row[headers["delta_flap (deg)"]])
        x_points.append([7e5, alpha, flap])
        left_edges.append(
            [
                float(row[headers["CL left edge"]]),
                float(row[headers["CD left edge"]]),
                float(row[headers["CM left edge"]]),
            ]
        )
        right_edges.append(
            [
                float(row[headers["CL right edge"]]),
                float(row[headers["CD right edge"]]),
                float(row[headers["CM right edge"]]),
            ]
        )

    return (
        torch.tensor(x_points, dtype=torch.float64),
        np.array(left_edges, dtype=float),
        np.array(right_edges, dtype=float),
    )


def run_distributionally_calibrated_monte_carlo(
    pred_x,
    xfoil_model,
    corrector,
    calibrated_model,
    marginal_model,
    n_samples,
):
    cases = []
    for case_idx, x0 in enumerate(pred_x):
        x_samples = uq.uniform_input_samples(x0, n_samples)
        y_base = xfoil_model.query(x_samples)
        y_corr = corrector.query(x_samples)
        y_uncalibrated = y_base + y_corr
        y_calibrated_mean, y_calibrated_var = calibrated_model.mean_variance(x_samples)
        y_calibrated_samples = (
            y_calibrated_mean
            + torch.randn_like(y_calibrated_mean) * torch.sqrt(y_calibrated_var)
        )
        y_marginal_calibrated_mean, y_marginal_calibrated_var = (
            marginal_model.mean_variance_for_case(case_idx, x_samples)
        )
        y_marginal_calibrated_samples = marginal_model.sample_for_case(
            case_idx,
            x_samples,
            match_sample_moments=True,
        )

        center_mean, center_var = calibrated_model.mean_variance(x0.unsqueeze(0))
        cases.append(
            {
                "x_samples": x_samples.detach().cpu().numpy(),
                "y_base": y_base.detach().cpu().numpy(),
                "y_corr": y_corr.detach().cpu().numpy(),
                "y_uncalibrated_combined": y_uncalibrated.detach().cpu().numpy(),
                "y_conditional_calibrated_mean": y_calibrated_mean.detach().cpu().numpy(),
                "y_conditional_calibrated_var": y_calibrated_var.detach().cpu().numpy(),
                "y_conditional_calibrated_samples": y_calibrated_samples.detach().cpu().numpy(),
                "y_marginal_calibrated_mean": y_marginal_calibrated_mean.detach().cpu().numpy(),
                "y_marginal_calibrated_var": y_marginal_calibrated_var.detach().cpu().numpy(),
                "y_marginal_calibrated_samples": y_marginal_calibrated_samples.detach().cpu().numpy(),
                "y_center_calibrated_mean": center_mean.detach().cpu().numpy()[0],
                "y_center_calibrated_var": center_var.detach().cpu().numpy()[0],
            }
        )

    return {
        "x_points": pred_x.detach().cpu().numpy(),
        "x_samples": np.stack([case["x_samples"] for case in cases]),
        "y_base": np.stack([case["y_base"] for case in cases]),
        "y_corr": np.stack([case["y_corr"] for case in cases]),
        "y_uncalibrated_combined": np.stack(
            [case["y_uncalibrated_combined"] for case in cases]
        ),
        "y_conditional_calibrated_mean": np.stack(
            [case["y_conditional_calibrated_mean"] for case in cases]
        ),
        "y_conditional_calibrated_var": np.stack(
            [case["y_conditional_calibrated_var"] for case in cases]
        ),
        "y_conditional_calibrated_samples": np.stack(
            [case["y_conditional_calibrated_samples"] for case in cases]
        ),
        "y_marginal_calibrated_mean": np.stack(
            [case["y_marginal_calibrated_mean"] for case in cases]
        ),
        "y_marginal_calibrated_var": np.stack(
            [case["y_marginal_calibrated_var"] for case in cases]
        ),
        "y_marginal_calibrated_samples": np.stack(
            [case["y_marginal_calibrated_samples"] for case in cases]
        ),
        "y_center_calibrated_mean": np.stack(
            [case["y_center_calibrated_mean"] for case in cases]
        ),
        "y_center_calibrated_var": np.stack(
            [case["y_center_calibrated_var"] for case in cases]
        ),
        "input_labels": np.array(uq.INPUT_LABELS),
        "output_labels": np.array(uq.OUTPUT_LABELS),
    }


def print_overlap_diagnostic(results, left_edges, right_edges):
    output_labels = results["output_labels"]
    x_points = results["x_points"]
    overlap_indices = np.where(
        np.isclose(x_points[:, 1], 0.0) & np.isclose(x_points[:, 2], 0.0)
    )[0]
    if len(overlap_indices) == 0:
        print("No alpha=0, flap=0 overlap point found for marginal diagnostic.")
        return

    case_idx = int(overlap_indices[0])
    samples = results["y_marginal_calibrated_samples"][case_idx]
    sample_mean = np.mean(samples, axis=0)
    sample_std = np.std(samples, axis=0)
    sample_quantiles = np.quantile(samples, [0.025, 0.975], axis=0)
    target_center = 0.5 * (left_edges[case_idx] + right_edges[case_idx])
    target_std = (right_edges[case_idx] - left_edges[case_idx]) / (2 * 1.96)

    print("Marginal calibration diagnostic for alpha=0, flap=0:")
    for output_idx, label in enumerate(output_labels):
        print(
            "  "
            f"{label}: mean {sample_mean[output_idx]:.8g} "
            f"(target {target_center[output_idx]:.8g}), "
            f"std {sample_std[output_idx]:.8g} "
            f"(target {target_std[output_idx]:.8g}), "
            f"q2.5 {sample_quantiles[0, output_idx]:.8g} "
            f"(left {left_edges[case_idx, output_idx]:.8g}), "
            f"q97.5 {sample_quantiles[1, output_idx]:.8g} "
            f"(right {right_edges[case_idx, output_idx]:.8g})"
        )


def main():
    xfoil_model, corrector, _, _ = uq.build_default_models()
    pred_x, left_edges, right_edges = load_prediction_bounds()
    calibrated_model = uq.DistributionallyCalibratedCorrectedBatch(
        xfoil_model,
        corrector,
        pred_x,
        left_edges,
        right_edges,
    )
    marginal_model = uq.MarginallyCalibratedCorrectedBatch(
        calibrated_model,
        pred_x,
        left_edges,
        right_edges,
        n_calibration_samples=N_MARGINAL_CALIBRATION_SAMPLES,
    )
    results = run_distributionally_calibrated_monte_carlo(
        pred_x,
        xfoil_model,
        corrector,
        calibrated_model,
        marginal_model,
        N_SAMPLES,
    )
    results["interval_left"] = left_edges
    results["interval_right"] = right_edges
    results["interval_center"] = 0.5 * (left_edges + right_edges)
    results["interval_variance"] = ((right_edges - left_edges) / (2 * 1.96)) ** 2
    results["source_marginal_mean"] = (
        marginal_model.source_marginal_mean.detach().cpu().numpy()
    )
    results["source_marginal_variance"] = (
        marginal_model.source_marginal_variance.detach().cpu().numpy()
    )
    results["marginal_std_scale"] = marginal_model.std_scale.detach().cpu().numpy()
    uq.save_npy_results(results, OUTPUT_DIR)
    print_overlap_diagnostic(results, left_edges, right_edges)
    print(f"Saved distributionally calibrated .npy files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
