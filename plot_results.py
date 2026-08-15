"""
Plot Monte Carlo results saved by validation.py and prediction.py.
"""

import argparse
import os
import tempfile

import numpy as np
import openpyxl
from scipy.stats import gaussian_kde

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "uqchallenge2-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "uqchallenge2-cache")
)

import matplotlib.pyplot as plt

try:
    import uqlib as uq
except ImportError:
    uq = None

MODEL_OUTPUT_LABELS = getattr(uq, "OUTPUT_LABELS", ["CL", "CD", "CM"])

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
    }
)

DEFAULT_DATA_ROOT = "results"
DEFAULT_PLOT_ROOT = "plots"
DEFAULT_BOUNDS_XLSX = "uq_challenge_ground_truth_epistemic_bounds.xlsx"
DEFAULT_GIVEN_TRUTHS_CSV = "given_truths.csv"
DEFAULT_DESIRED_TRUTHS_CSV = "desired_truths.csv"
DEFAULT_TRAINING_DATA = os.path.join("data", "training", "training_100.npy")

LATEX_OUTPUT_LABELS = {
    "CL": r"$C_l$",
    "CD": r"$C_d$",
    "CM": r"$C_m$",
}


def latent_gp_truth(x):
    """Smooth one-dimensional function used only for the latent-input schematic."""
    return (
        0.46 * np.sin(2.25 * np.pi * x + 0.15)
        + 0.34 * np.sin(4.45 * np.pi * x - 0.45)
        + 0.36 * np.exp(-0.5 * ((x - 0.34) / 0.12) ** 2)
        - 0.10
    )


def latex_output_label(label):
    return LATEX_OUTPUT_LABELS.get(str(label).upper(), str(label))


def save_standalone_legend(handles, labels, output_dir, filename="legend.png"):
    if not handles:
        return

    legend_fig = plt.figure(figsize=(7, 1.25))
    legend_fig.legend(
        handles,
        labels,
        loc="center",
        ncol=min(len(labels), 3),
        frameon=False,
    )
    legend_fig.tight_layout()
    legend_fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(legend_fig)


def empirical_cdf(samples):
    samples = np.sort(np.asarray(samples, dtype=float))
    probabilities = np.arange(1, len(samples) + 1, dtype=float) / len(samples)
    return samples, probabilities


def load_truth_locations(csv_path):
    values = np.genfromtxt(csv_path, delimiter=",", names=True)
    return np.column_stack((values["alpha"], values["flap"]))


def load_npy_results(input_dir):
    results = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            name = os.path.splitext(filename)[0]
            results[name] = np.load(
                os.path.join(input_dir, filename), allow_pickle=True
            )
    return results


def plot_desiderata_figure(
    output_path=os.path.join("paper", "desired_pts.png"),
    given_truths_csv=DEFAULT_GIVEN_TRUTHS_CSV,
    desired_truths_csv=DEFAULT_DESIRED_TRUTHS_CSV,
):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    given_points = load_truth_locations(given_truths_csv)
    desired_points = load_truth_locations(desired_truths_csv)

    fig, ax = plt.subplots(figsize=(5.0, 3.65))
    ax.scatter(
        given_points[:, 0],
        given_points[:, 1],
        marker="x",
        s=65,
        linewidths=1.5,
        color="tab:blue",
        label=r"\textrm{Given truth points}",
        zorder=3,
    )
    ax.scatter(
        desired_points[:, 0],
        desired_points[:, 1],
        marker="o",
        s=65,
        facecolors="white",
        edgecolors="tab:orange",
        linewidths=1.5,
        label=r"\textrm{Test points}",
        zorder=4,
    )

    ax.set_xlabel(r"\textrm{Angle of attack}")
    ax.set_ylabel(r"\textrm{Flap setting}")
    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.07, 0.5)
    ax.set_xlim(-10, 15)
    ax.set_ylim(-10, 20)
    ax.set_xticks([-10, -5, 0, 5, 10, 15])
    ax.set_yticks([-10, -5, 0, 5, 10, 15, 20])
    ax.grid(True, color="0.84", linewidth=0.8)

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")
    ax.spines["right"].set_color("0.82")
    ax.spines["top"].set_color("0.82")
    ax.spines["left"].set_color("0.72")
    ax.spines["bottom"].set_color("0.72")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="both", colors="0.35", direction="inout", pad=6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved desiderata figure to {output_path}")


def rbf_kernel(x_left, x_right, lengthscale=0.18, outputscale=0.85):
    x_left = np.atleast_2d(x_left).T
    x_right = np.atleast_2d(x_right).T
    sqdist = (x_left - x_right.T) ** 2
    return outputscale**2 * np.exp(-0.5 * sqdist / lengthscale**2)


def gp_posterior(x_train, y_train, x_grid, noise=0.06):
    k_xx = rbf_kernel(x_train, x_train) + noise**2 * np.eye(len(x_train))
    k_xs = rbf_kernel(x_train, x_grid)
    k_ss = rbf_kernel(x_grid, x_grid)
    chol = np.linalg.cholesky(k_xx + 1e-10 * np.eye(len(x_train)))
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
    mean = k_xs.T @ alpha
    v = np.linalg.solve(chol, k_xs)
    variance = np.maximum(np.diag(k_ss) - np.sum(v * v, axis=0), 0.0)
    return mean, np.sqrt(variance)


def plot_latent_gp_figure(output_path=os.path.join("paper", "figures", "1dlatent_.png")):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    x_grid = np.linspace(0.0, 1.0, 600)
    y_true = latent_gp_truth(x_grid)

    x_nominal = np.array(
        [0.14, 0.17, 0.19, 0.22, 0.27, 0.43, 0.58, 0.66, 0.69, 0.82, 0.90, 0.97]
    )
    x_latent = np.array(
        [
            0.12,
            0.16,
            0.20,
            0.235,
            0.265,
            0.505,
            0.535,
            0.635,
            0.685,
            0.725,
            0.895,
            0.905,
        ]
    )
    half_width = np.array(
        [0.055, 0.055, 0.060, 0.070, 0.065, 0.080, 0.055, 0.080, 0.070, 0.080, 0.050, 0.060]
    )
    y_obs = latent_gp_truth(x_nominal) + np.array(
        [0.08, -0.03, 0.03, 0.12, -0.02, 0.06, -0.10, -0.02, -0.15, 0.10, -0.07, -0.03]
    )

    mean, std = gp_posterior(x_latent, y_obs, x_grid)

    fig, ax = plt.subplots(figsize=(10.5, 6.75))
    ax.fill_between(
        x_grid,
        mean - 1.96 * std,
        mean + 1.96 * std,
        color="tab:blue",
        alpha=0.24,
        linewidth=0,
        label=r"$95\%$ CI",
    )
    ax.plot(x_grid, mean, color="tab:blue", linewidth=2.0, label=r"\textrm{GP mean}")
    ax.scatter(
        x_nominal,
        y_obs,
        marker="x",
        s=80,
        linewidths=2.0,
        color="black",
        label=r"\textrm{noisy obs.}",
        zorder=4,
    )
    ax.scatter(
        x_latent,
        y_obs,
        facecolors="white",
        edgecolors="tab:orange",
        s=90,
        linewidths=2.0,
        label=r"\textrm{inferred input location}",
        zorder=5,
    )

    interval_colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(x_nominal)))
    for idx, (x_center, x_inferred, dx, y_value) in enumerate(
        zip(x_nominal, x_latent, half_width, y_obs)
    ):
        line_start = max(0.0, x_center - dx)
        line_stop = min(1.0, x_center + dx)
        ax.hlines(
            y_value,
            line_start,
            line_stop,
            color=interval_colors[idx],
            alpha=0.55,
            linewidth=2.0,
            zorder=3,
        )
        if not (line_start <= x_inferred <= line_stop):
            ax.hlines(
                y_value,
                min(line_start, x_inferred),
                max(line_stop, x_inferred),
                color=interval_colors[idx],
                alpha=0.35,
                linewidth=2.0,
                zorder=2,
            )

    ax.plot(
        x_grid,
        y_true,
        color="tab:red",
        linestyle="--",
        linewidth=2.0,
        alpha=0.8,
        label=r"$f(x)$",
    )

    ax.set_title(r"\textrm{1D Latent-input GP}", pad=12)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.05, 1.40)
    ax.legend(loc="upper right", frameon=True, fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved latent GP figure to {output_path}")


def plot_true_vs_predicted_panel(
    true_values,
    predicted_values,
    output_path,
    label,
    markers=None,
    colors=None,
):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(predicted_values, dtype=float)

    if predicted_values.ndim == 1:
        predicted_values = predicted_values[:, None]

    markers = markers or ["o"] * predicted_values.shape[1]
    colors = colors or ["tab:blue"] * predicted_values.shape[1]
    for series_idx in range(predicted_values.shape[1]):
        marker = markers[min(series_idx, len(markers) - 1)]
        color = colors[min(series_idx, len(colors) - 1)]
        if marker in ("s", "^", "D", "v"):
            ax.scatter(
                true_values,
                predicted_values[:, series_idx],
                marker=marker,
                s=35,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                zorder=3,
            )
        else:
            ax.scatter(
                true_values,
                predicted_values[:, series_idx],
                marker=marker,
                s=28,
                color=color,
                edgecolors="black",
                linewidths=0.25,
                zorder=3,
            )

    limits = np.concatenate((true_values, predicted_values.ravel()))
    lower = float(np.min(limits))
    upper = float(np.max(limits))
    padding = max(0.04 * (upper - lower), 1.0e-6)
    lower -= padding
    upper += padding

    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.2, zorder=2)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(rf"\textrm{{True }}{latex_output_label(label)}")
    ax.set_ylabel(rf"\textrm{{Predicted }}{latex_output_label(label)}")
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    fig.tight_layout(pad=0.25)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_validation_true_vs_predict(
    data_root=DEFAULT_DATA_ROOT,
    plot_root=os.path.join("paper", "figures"),
    training_data_path=DEFAULT_TRAINING_DATA,
):
    output_dir = os.path.join(plot_root, "validations", "true_vs_predict")
    os.makedirs(output_dir, exist_ok=True)

    training = np.load(training_data_path)
    xfoil_by_label = {
        "CL": training[:, 3],
        "CD": training[:, 4],
        "CM": training[:, 5],
    }
    validation_dir = os.path.join(data_root, "validation")
    true_y = np.load(os.path.join(validation_dir, "y_true.npy"))
    base_center = np.load(os.path.join(validation_dir, "y_center_base.npy"))
    combined_center = np.load(os.path.join(validation_dir, "y_center_combined.npy"))
    validation_by_label = {label: idx for idx, label in enumerate(MODEL_OUTPUT_LABELS)}

    for label in ("CD", "CL", "CM"):
        # The repository stores the XFOIL validation coefficients, but not a
        # separate persisted XFOIL-GP posterior mean for this held-out set.
        plot_true_vs_predicted_panel(
            xfoil_by_label[label],
            xfoil_by_label[label],
            os.path.join(output_dir, f"{label.capitalize()}_xfoilgp_validation.png"),
            label,
        )

        idx = validation_by_label[label]
        plot_true_vs_predicted_panel(
            true_y[:, idx],
            np.column_stack((base_center[:, idx], combined_center[:, idx])),
            os.path.join(output_dir, f"{label.capitalize()}_truegp_validation.png"),
            label,
            markers=["s", "^"],
            colors=["mediumpurple", "tab:green"],
        )

    print(f"Saved Figure 4 true-vs-predicted panels to {output_dir}")


def plot_case(results, case_idx, output_dir, include_truth):
    x0 = results["x_points"][case_idx]
    output_labels = results.get("output_labels", MODEL_OUTPUT_LABELS)
    aoa = int(x0[1])
    flap = int(x0[2])
    legend_handles = None
    legend_labels = None

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

        plt.xlabel(latex_output_label(label))
        plt.ylabel(r"\textrm{Frequency}")
        if legend_handles is None:
            legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"mc_aoa{aoa}_flap{flap}_{label}.png"),
            dpi=300,
        )
        plt.close()

    return legend_handles, legend_labels


def plot_dataset(name, data_root=DEFAULT_DATA_ROOT, plot_root=DEFAULT_PLOT_ROOT):
    input_dir = os.path.join(data_root, name)
    output_dir = os.path.join(plot_root, name)
    os.makedirs(output_dir, exist_ok=True)

    results = load_npy_results(input_dir)
    include_truth = name == "validation"
    legend_saved = False

    for case_idx in range(len(results["x_points"])):
        handles, labels = plot_case(results, case_idx, output_dir, include_truth)
        if not legend_saved:
            save_standalone_legend(handles, labels, output_dir)
            legend_saved = True

    print(f"Saved {name} plots to {output_dir}")


def load_epistemic_bounds(workbook_path=DEFAULT_BOUNDS_XLSX):
    workbook = openpyxl.load_workbook(workbook_path, data_only=True)
    sheet = workbook["Bounds"]

    headers = None
    bounds = {}
    for row in sheet.iter_rows(values_only=True):
        if "Left edge" in row and "Right edge" in row:
            headers = {value: idx for idx, value in enumerate(row) if value is not None}
            continue

        if headers is None or row[headers["Coefficient"]] is None:
            continue

        alpha = int(row[headers["alpha (deg)"]])
        flap = int(row[headers["delta_flap (deg)"]])
        coefficient = str(row[headers["Coefficient"]]).upper()
        bounds[(alpha, flap, coefficient)] = (
            float(row[headers["Left edge"]]),
            float(row[headers["Right edge"]]),
        )

    return bounds


def plot_prediction_bounds(
    data_root=DEFAULT_DATA_ROOT,
    plot_root=DEFAULT_PLOT_ROOT,
    workbook_path=DEFAULT_BOUNDS_XLSX,
):
    input_dir = os.path.join(data_root, "prediction")
    output_dir = os.path.join(plot_root, "prediction_bounds")
    os.makedirs(output_dir, exist_ok=True)

    results = load_npy_results(input_dir)
    bounds = load_epistemic_bounds(workbook_path)
    output_labels = results.get("output_labels", MODEL_OUTPUT_LABELS)
    legend_handles = None
    legend_labels = None

    for case_idx, x0 in enumerate(results["x_points"]):
        aoa = int(x0[1])
        flap = int(x0[2])

        for output_idx, label in enumerate(output_labels):
            label = str(label).upper()
            key = (aoa, flap, label)
            if key not in bounds:
                raise KeyError(
                    f"No epistemic bounds found for alpha={aoa}, flap={flap}, {label}"
                )

            left_edge, right_edge = bounds[key]
            corrected = results["y_combined"][case_idx, :, output_idx]
            corrected_mean = float(np.mean(corrected))

            plt.figure(figsize=(7, 5))
            _, bin_edges, _ = plt.hist(
                corrected,
                bins=40,
                alpha=0.7,
                color="tab:orange",
                label=r"\textrm{First moment calibrated}",
            )
            if np.std(corrected) > 0:
                x_grid = np.linspace(bin_edges[0], bin_edges[-1], 300)
                bin_width = bin_edges[1] - bin_edges[0]
                density = gaussian_kde(corrected)(x_grid)
                plt.plot(
                    x_grid,
                    density * len(corrected) * bin_width,
                    color="tab:blue",
                    linewidth=2,
                    label="Density estimate",
                )
            plt.axvline(
                corrected_mean,
                color="black",
                linewidth=2,
                linestyle="--",
                label=r"\textrm{Mean estimate}",
            )
            plt.axvline(
                left_edge,
                color="tab:red",
                linewidth=2,
            )
            plt.axvline(
                right_edge,
                color="tab:red",
                linewidth=2,
                linestyle=":",
            )
            plt.axvspan(
                left_edge,
                right_edge,
                color="tab:red",
                alpha=0.08,
                label=r"\textrm{True epistemic interval}",
            )

            plt.xlabel(latex_output_label(label))
            plt.ylabel(r"\textrm{Frequency}")
            if legend_handles is None:
                legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir, f"corrected_bounds_aoa{aoa}_flap{flap}_{label}.png"
                ),
                dpi=300,
            )
            plt.close()

    save_standalone_legend(legend_handles, legend_labels, output_dir)
    print(f"Saved prediction bounds plots to {output_dir}")


def plot_distributional_calibrated_bounds(
    data_root=DEFAULT_DATA_ROOT,
    plot_root=DEFAULT_PLOT_ROOT,
):
    input_dir = os.path.join(data_root, "distributional_calibrated_prediction")
    output_dir = os.path.join(plot_root, "distributional_calibrated_prediction_bounds")
    os.makedirs(output_dir, exist_ok=True)

    results = load_npy_results(input_dir)
    output_labels = results.get("output_labels", MODEL_OUTPUT_LABELS)
    legend_handles = None
    legend_labels = None

    for case_idx, x0 in enumerate(results["x_points"]):
        aoa = int(x0[1])
        flap = int(x0[2])

        for output_idx, label in enumerate(output_labels):
            label = str(label).upper()
            calibrated_samples = results["y_marginal_calibrated_samples"][
                case_idx, :, output_idx
            ]
            calibrated_mean = float(np.mean(calibrated_samples))
            left_edge = float(results["interval_left"][case_idx, output_idx])
            right_edge = float(results["interval_right"][case_idx, output_idx])

            plt.figure(figsize=(7, 5))
            _, bin_edges, _ = plt.hist(
                calibrated_samples,
                bins=40,
                alpha=0.7,
                color="tab:green",
                label=r"\textrm{Distribution calibrated}",
            )
            if np.std(calibrated_samples) > 0:
                x_grid = np.linspace(bin_edges[0], bin_edges[-1], 300)
                bin_width = bin_edges[1] - bin_edges[0]
                density = gaussian_kde(calibrated_samples)(x_grid)
                plt.plot(
                    x_grid,
                    density * len(calibrated_samples) * bin_width,
                    color="tab:blue",
                    linewidth=2,
                    label="Density estimate",
                )

            plt.axvline(
                calibrated_mean,
                color="black",
                linewidth=2,
                linestyle="--",
                label=r"\textrm{Mean estimate}",
            )
            plt.axvline(
                left_edge,
                color="tab:red",
                linewidth=2,
            )
            plt.axvline(
                right_edge,
                color="tab:red",
                linewidth=2,
                linestyle=":",
            )
            plt.axvspan(
                left_edge,
                right_edge,
                color="tab:red",
                alpha=0.08,
                label=r"\textrm{True epistemic interval}",
            )

            plt.xlabel(latex_output_label(label))
            plt.ylabel(r"\textrm{Frequency}")
            if legend_handles is None:
                legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir,
                    f"distributional_calibrated_aoa{aoa}_flap{flap}_{label}.png",
                ),
                dpi=300,
            )
            plt.close()

    save_standalone_legend(legend_handles, legend_labels, output_dir)
    print(f"Saved distributionally calibrated plots to {output_dir}")


def plot_calibration_cdfs(
    data_root=DEFAULT_DATA_ROOT,
    plot_root=DEFAULT_PLOT_ROOT,
    calibration="first-moment",
    workbook_path=DEFAULT_BOUNDS_XLSX,
):
    if calibration == "first-moment":
        input_dir = os.path.join(data_root, "prediction")
        output_dir = os.path.join(plot_root, "first_moment_calibrated_cdfs")
        sample_key = "y_combined"
        file_prefix = "first_moment_calibrated_cdf"
        line_color = "tab:orange"
        line_label = r"\textrm{First moment calibrated}"
        bounds = load_epistemic_bounds(workbook_path)
    elif calibration == "distributional":
        input_dir = os.path.join(data_root, "distributional_calibrated_prediction")
        output_dir = os.path.join(plot_root, "distributional_calibrated_cdfs")
        sample_key = "y_marginal_calibrated_samples"
        file_prefix = "distributional_calibrated_cdf"
        line_color = "tab:green"
        line_label = r"\textrm{Distribution calibrated}"
        bounds = None
    else:
        raise ValueError(f"Unknown calibration mode: {calibration}")

    os.makedirs(output_dir, exist_ok=True)
    results = load_npy_results(input_dir)
    if (
        calibration == "distributional"
        and sample_key not in results
        and "y_conditional_calibrated_samples" in results
    ):
        sample_key = "y_conditional_calibrated_samples"
    elif (
        calibration == "distributional"
        and sample_key not in results
        and "y_calibrated_samples" in results
    ):
        sample_key = "y_calibrated_samples"
    output_labels = results.get("output_labels", MODEL_OUTPUT_LABELS)
    legend_handles = None
    legend_labels = None

    for case_idx, x0 in enumerate(results["x_points"]):
        aoa = int(x0[1])
        flap = int(x0[2])

        for output_idx, label in enumerate(output_labels):
            label = str(label).upper()
            samples = results[sample_key][case_idx, :, output_idx]
            x_cdf, y_cdf = empirical_cdf(samples)

            if calibration == "first-moment":
                key = (aoa, flap, label)
                if key not in bounds:
                    raise KeyError(
                        f"No epistemic bounds found for alpha={aoa}, flap={flap}, {label}"
                    )
                left_edge, right_edge = bounds[key]
            else:
                left_edge = float(results["interval_left"][case_idx, output_idx])
                right_edge = float(results["interval_right"][case_idx, output_idx])

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.step(
                x_cdf,
                y_cdf,
                where="post",
                color=line_color,
                linewidth=2,
                label=line_label,
            )
            ax.axvline(
                left_edge,
                color="tab:red",
                linewidth=2,
            )
            ax.axvline(
                right_edge,
                color="tab:red",
                linewidth=2,
                linestyle=":",
            )
            ax.axvspan(
                left_edge,
                right_edge,
                color="tab:red",
                alpha=0.08,
                label=r"\textrm{True epistemic interval}",
            )

            x_min = min(float(x_cdf[0]), left_edge, right_edge)
            x_max = max(float(x_cdf[-1]), left_edge, right_edge)
            x_pad = 0.04 * (x_max - x_min) if x_max > x_min else 1.0
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(latex_output_label(label))
            ax.set_ylabel(r"\textrm{Empirical CDF}")
            ax.grid(True, color="0.86", linewidth=0.8)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_dir, f"{file_prefix}_aoa{aoa}_flap{flap}_{label}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    save_standalone_legend(legend_handles, legend_labels, output_dir)
    print(f"Saved {calibration} CDF plots to {output_dir}")


def plot_combined_calibration_cdfs(
    data_root=DEFAULT_DATA_ROOT,
    plot_root=DEFAULT_PLOT_ROOT,
    workbook_path=DEFAULT_BOUNDS_XLSX,
):
    first_results = load_npy_results(os.path.join(data_root, "prediction"))
    distributional_results = load_npy_results(
        os.path.join(data_root, "distributional_calibrated_prediction")
    )
    bounds = load_epistemic_bounds(workbook_path)
    output_dir = os.path.join(plot_root, "cdf_overview")
    os.makedirs(output_dir, exist_ok=True)

    distributional_sample_key = (
        "y_marginal_calibrated_samples"
        if "y_marginal_calibrated_samples" in distributional_results
        else "y_conditional_calibrated_samples"
    )
    output_labels = first_results.get("output_labels", MODEL_OUTPUT_LABELS)
    legend_handles = None
    legend_labels = None
    distributional_case_indices = {
        (int(x_point[1]), int(x_point[2])): idx
        for idx, x_point in enumerate(distributional_results["x_points"])
    }

    for case_idx, x0 in enumerate(first_results["x_points"]):
        aoa = int(x0[1])
        flap = int(x0[2])
        case_key = (aoa, flap)
        if case_key not in distributional_case_indices:
            raise KeyError(
                f"No distributional prediction found for alpha={aoa}, flap={flap}"
            )
        distributional_case_idx = distributional_case_indices[case_key]

        for output_idx, label in enumerate(output_labels):
            label = str(label).upper()
            key = (aoa, flap, label)
            if key not in bounds:
                raise KeyError(
                    f"No epistemic bounds found for alpha={aoa}, flap={flap}, {label}"
                )

            first_x, first_y = empirical_cdf(
                first_results["y_combined"][case_idx, :, output_idx]
            )
            distributional_x, distributional_y = empirical_cdf(
                distributional_results[distributional_sample_key][
                    distributional_case_idx, :, output_idx
                ]
            )
            left_edge, right_edge = bounds[key]

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.step(
                first_x,
                first_y,
                where="post",
                color="tab:orange",
                linewidth=2,
                label=r"\textrm{First moment calibrated}",
            )
            ax.step(
                distributional_x,
                distributional_y,
                where="post",
                color="tab:green",
                linewidth=2,
                label=r"\textrm{Distribution calibrated}",
            )
            ax.axvline(
                left_edge,
                color="tab:red",
                linewidth=2,
            )
            ax.axvline(
                right_edge,
                color="tab:red",
                linewidth=2,
                linestyle=":",
            )
            ax.axvspan(
                left_edge,
                right_edge,
                color="tab:red",
                alpha=0.08,
                label=r"\textrm{True epistemic interval}",
            )

            x_min = min(
                float(first_x[0]),
                float(distributional_x[0]),
                left_edge,
                right_edge,
            )
            x_max = max(
                float(first_x[-1]),
                float(distributional_x[-1]),
                left_edge,
                right_edge,
            )
            x_pad = 0.04 * (x_max - x_min) if x_max > x_min else 1.0
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(latex_output_label(label))
            ax.set_ylabel(r"\textrm{Empirical CDF}")
            ax.grid(True, color="0.86", linewidth=0.8)

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"calibrated_cdf_aoa{aoa}_flap{flap}_{label}.png",
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    save_standalone_legend(legend_handles, legend_labels, output_dir)
    print(f"Saved combined calibration CDF plots to {output_dir}")


def plot_calibration_cdf_overview(
    data_root=DEFAULT_DATA_ROOT,
    plot_root=DEFAULT_PLOT_ROOT,
    workbook_path=DEFAULT_BOUNDS_XLSX,
):
    first_results = load_npy_results(os.path.join(data_root, "prediction"))
    distributional_results = load_npy_results(
        os.path.join(data_root, "distributional_calibrated_prediction")
    )
    bounds = load_epistemic_bounds(workbook_path)
    output_dir = os.path.join(plot_root, "cdf_overview")
    os.makedirs(output_dir, exist_ok=True)

    output_labels = first_results.get("output_labels", MODEL_OUTPUT_LABELS)
    distributional_case_indices = {
        (int(x_point[1]), int(x_point[2])): idx
        for idx, x_point in enumerate(distributional_results["x_points"])
    }
    fig, axes = plt.subplots(
        len(first_results["x_points"]),
        len(output_labels),
        figsize=(10.5, 11.0),
        sharey=True,
    )

    for case_idx, x0 in enumerate(first_results["x_points"]):
        aoa = int(x0[1])
        flap = int(x0[2])
        case_key = (aoa, flap)
        if case_key not in distributional_case_indices:
            raise KeyError(
                f"No distributional prediction found for alpha={aoa}, flap={flap}"
            )
        distributional_case_idx = distributional_case_indices[case_key]

        for output_idx, label in enumerate(output_labels):
            label = str(label).upper()
            ax = axes[case_idx, output_idx]

            first_x, first_y = empirical_cdf(
                first_results["y_combined"][case_idx, :, output_idx]
            )
            distributional_x, distributional_y = empirical_cdf(
                distributional_results["y_marginal_calibrated_samples"][
                    distributional_case_idx, :, output_idx
                ]
            )
            left_edge, right_edge = bounds[(aoa, flap, label)]

            ax.step(
                first_x,
                first_y,
                where="post",
                color="tab:orange",
                linewidth=1.5,
                label=r"\textrm{First moment}",
            )
            ax.step(
                distributional_x,
                distributional_y,
                where="post",
                color="tab:green",
                linewidth=1.5,
                label=r"\textrm{Distribution}",
            )
            ax.axvline(left_edge, color="tab:red", linewidth=1.2)
            ax.axvline(right_edge, color="tab:red", linewidth=1.2, linestyle=":")
            ax.axvspan(left_edge, right_edge, color="tab:red", alpha=0.08)
            ax.grid(True, color="0.88", linewidth=0.7)
            ax.set_ylim(0.0, 1.0)

            if case_idx == 0:
                ax.set_title(latex_output_label(label))
            if output_idx == 0:
                ax.set_ylabel(
                    rf"$\alpha={aoa}^\circ$, $\beta_{{\mathrm{{flap}}}}={flap}^\circ$"
                    "\n"
                    r"\textrm{CDF}"
                )
            if case_idx == len(first_results["x_points"]) - 1:
                ax.set_xlabel(latex_output_label(label))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    output_path = os.path.join(output_dir, "calibration_cdf_overview.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration CDF overview to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot saved Monte Carlo .npy results."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=[
            "validation",
            "prediction",
            "prediction-bounds",
            "distributional-calibrated-bounds",
            "first-moment-cdfs",
            "distributional-calibrated-cdfs",
            "calibration-cdfs",
            "calibrated-cdfs",
            "calibration-cdf-overview",
            "desiderata-figure",
            "latent-gp-figure",
            "validation-true-vs-predict",
            "all",
        ],
        help="Which saved result set to plot.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--plot-root", default=DEFAULT_PLOT_ROOT)
    parser.add_argument("--bounds-xlsx", default=DEFAULT_BOUNDS_XLSX)
    parser.add_argument("--given-truths-csv", default=DEFAULT_GIVEN_TRUTHS_CSV)
    parser.add_argument("--desired-truths-csv", default=DEFAULT_DESIRED_TRUTHS_CSV)
    parser.add_argument("--training-data", default=DEFAULT_TRAINING_DATA)
    args = parser.parse_args()

    if args.dataset == "prediction-bounds":
        plot_prediction_bounds(args.data_root, args.plot_root, args.bounds_xlsx)
        return
    if args.dataset == "distributional-calibrated-bounds":
        plot_distributional_calibrated_bounds(args.data_root, args.plot_root)
        return
    if args.dataset == "first-moment-cdfs":
        plot_calibration_cdfs(
            args.data_root,
            args.plot_root,
            calibration="first-moment",
            workbook_path=args.bounds_xlsx,
        )
        return
    if args.dataset == "distributional-calibrated-cdfs":
        plot_calibration_cdfs(
            args.data_root,
            args.plot_root,
            calibration="distributional",
            workbook_path=args.bounds_xlsx,
        )
        return
    if args.dataset in ("calibration-cdfs", "calibrated-cdfs"):
        plot_combined_calibration_cdfs(
            args.data_root, args.plot_root, args.bounds_xlsx
        )
        return
    if args.dataset == "calibration-cdf-overview":
        plot_calibration_cdf_overview(args.data_root, args.plot_root, args.bounds_xlsx)
        return
    if args.dataset == "desiderata-figure":
        plot_desiderata_figure(
            os.path.join(args.plot_root, "desired_pts.png"),
            args.given_truths_csv,
            args.desired_truths_csv,
        )
        return
    if args.dataset == "latent-gp-figure":
        plot_latent_gp_figure(os.path.join(args.plot_root, "1dlatent_.png"))
        return
    if args.dataset == "validation-true-vs-predict":
        plot_validation_true_vs_predict(
            args.data_root, args.plot_root, args.training_data
        )
        return

    datasets = ["validation", "prediction"] if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        plot_dataset(dataset, args.data_root, args.plot_root)


if __name__ == "__main__":
    main()
