"""
This is the main library for the UQ challenge problem. It contains code for 
running XFOIL executables using subprocess and a botorch wrapper for 
creating Gaussian process models. This is the 2nd version of the code, featuring
speed and ease-of-use improvements for XFOIL subprocess and simpler wrapper 
design. 
"""

import pandas as pd
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

import numpy as np
import time
import os
import subprocess

INPUT_LABELS = ["Re", "alpha", "flap"]
OUTPUT_LABELS = ["CL", "CD", "CM"]

DEFAULT_BOUNDS = torch.tensor(
    [[7e5 * 0.995, -5.0, -5.0], [7e5 * 1.005, 10.0, 15.0]],
    dtype=torch.float64,
)
DEFAULT_INPUT_DELTA = torch.tensor([7e5 * 0.005, 0.02, 0.1], dtype=torch.float64)
DEFAULT_BASE_YVAR = [1e-5, 1e-8, 1e-7]
DEFAULT_CORRECTION_YVAR = [1e-6, 1e-10, 1e-9]
DEFAULT_TRAINING_PATH = "data/training/training_100.npy"
DEFAULT_TRAINING_FALLBACK_PATH = "training/training_100.csv"
DEFAULT_TRUTH_PATH = "data/truth/given_truths.npy"
DEFAULT_TRUTH_FALLBACK_PATH = "given_truths.csv"
DEFAULT_DESIRED_PATH = "data/truth/desired_truths.npy"
DEFAULT_DESIRED_FALLBACK_PATH = "desired_truths.csv"

def read_xfoil_polar(re,alpha,flap, filename):
    """
    Read the final converged coefficient row from an XFOIL polar file.

    Parameters
    ----------
    re: float
        reynolds number
    alpha : float
        Angle of attack of airfoil.
    flap : float
        Angle of flap.
    filename : string
        path to polarfile. 

    Returns
    -------
    tuple
        The coefficient row and a boolean indicating whether parsing failed.
    """
    
    max_wait = 2
    elapsed = 0
    stamp = time.time()
    while (not os.path.exists(filename) or (os.path.exists(filename) and os.path.getsize(filename) < 750)) and elapsed < max_wait:
        time.sleep(.05)
        elapsed = time.time() - stamp
        
    with open(filename, 'r') as file:
        lines = file.readlines()

    data_lines = []
    
    for line in lines:
        try:
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure enough columns]
                cl = float(parts[1])
                cd = float(parts[2])
                cm = float(parts[4])
                if abs(alpha-float(parts[0])) < 1e-3:
                    data_lines=[np.array([re,alpha,flap,cl, cd, cm])]    
        except ValueError:
            continue  # Skip non-numeric lines
    error = False
    if not data_lines:
        error = True
    else:
        data_lines = data_lines[0]
    return data_lines,error



class XFOIL:
    def __init__(self, xfoil_path, re, flap, polarpath):
        if os.path.exists(polarpath):
            os.remove(polarpath)
            time.sleep(.1)
        self.proc = subprocess.Popen(
           [xfoil_path],
           stdin=subprocess.PIPE,
           stdout=subprocess.PIPE,
           stderr=subprocess.PIPE,
           text=True
       )
        self.re = re
        self.flap = flap
        self.polarpath = polarpath
    
        self.proc.stdin.write("PLOP\n")
        self.proc.stdin.write("G F\n")
        self.proc.stdin.write("\n")
    
        self.proc.stdin.write("NACA 2412\n")
        self.proc.stdin.write("gdes\n")
        self.proc.stdin.write("flap\n")
        self.proc.stdin.write("0.70\n")
        self.proc.stdin.write("999\n")
        self.proc.stdin.write("0.5\n")
        self.proc.stdin.write(f"{self.flap}\n")
        self.proc.stdin.write("exec\n")
        self.proc.stdin.write("\n")
        
        self.proc.stdin.write("ppar\n")
        self.proc.stdin.write("n\n")
        self.proc.stdin.write("100\n")
        self.proc.stdin.write("\n")
        self.proc.stdin.write("\n")
       
    
        self.proc.stdin.write("oper\n")
        self.proc.stdin.write("Mach\n")
        self.proc.stdin.write("0\n")
        self.proc.stdin.write("visc\n")
        self.proc.stdin.write(f"{self.re}\n")
        self.proc.stdin.write("vpar\n")
        self.proc.stdin.write("N\n")
        self.proc.stdin.write("0.01\n")
        self.proc.stdin.write("\n")
       
        self.proc.stdin.write("pacc\n")
        self.proc.stdin.write(f"{self.polarpath}\n")
        self.proc.stdin.write("\n")
        self.proc.stdin.flush()
       
    def run_alpha(self,alpha):
        self.proc.stdin.write(f"alfa {alpha}\n")
        self.proc.stdin.flush()
        time.sleep(.2)
        self.wait_time = 0
    def sweep(self,alpha):
        alphas = np.arange(0,alpha)
        alphas = np.append(alphas,alpha)
        for alpha_sweep in alphas:
            self.run_alpha(alpha_sweep)
        self.wait_time = abs(alpha)/10
    def done(self):
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc.stderr.close()
        time.sleep(self.wait_time) #give time for polar to flush
                



class Model():
    def __init__(self, xdata, ydata, bounds, train_yvar):
        self.xdata = xdata.double()
        self.bounds = bounds.double()

        ydata = ydata.squeeze(-1)
        self.ydata = ydata.unsqueeze(-1).double()

        self.xdim = xdata.shape[1]
        self.train_yvar = torch.full_like(self.ydata, train_yvar, dtype=torch.double)

        self.gp = SingleTaskGP(
            self.xdata,
            self.ydata,
            self.train_yvar,
            input_transform=Normalize(d=self.xdim, bounds=self.bounds),
            outcome_transform=Standardize(m=1)
        )

        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)
        self.gp.eval()

    def query(self, test_x):
        pred = self.gp.posterior(test_x)
        return pred.mean.detach().squeeze(-1)

    def posterior_mean_cov(self, test_x):
        pred = self.gp.posterior(test_x.double())
        return (
            pred.mean.detach().squeeze(-1),
            pred.mvn.covariance_matrix.detach(),
        )

class Batch():
    def __init__(self, xdata, ydata, bounds, train_yvar):
        """
        ydata: (N, m)
        train_yvar: list of length m
        """
        self.models = []
        for var in range(ydata.shape[1]):
            self.models.append(Model(xdata, ydata[:, var], bounds, train_yvar[var]))

    def query(self, test_x):
        return torch.stack([model.query(test_x) for model in self.models], dim=1)

    def posterior_mean_cov(self, test_x):
        means = []
        covariances = []
        for model in self.models:
            mean, covariance = model.posterior_mean_cov(test_x)
            means.append(mean)
            covariances.append(covariance)
        return torch.stack(means, dim=1), torch.stack(covariances, dim=0)


class DistributionallyCalibratedCorrectedBatch:
    """
    Moment-calibrated corrected GP.

    The calibration follows the moment-replacement construction in
    distributional_calibration.pdf. At calibration locations, the corrected GP
    mean matches each epistemic interval centroid and the pointwise 95% band
    matches the interval edges.
    """

    def __init__(
        self,
        xfoil_model,
        correction_model,
        calibration_x,
        interval_left,
        interval_right,
        z_value=1.96,
        jitter=1e-12,
    ):
        self.xfoil_model = xfoil_model
        self.correction_model = correction_model
        self.calibration_x = calibration_x.double()
        self.interval_left = torch.tensor(interval_left, dtype=torch.float64)
        self.interval_right = torch.tensor(interval_right, dtype=torch.float64)
        self.target_center = 0.5 * (self.interval_left + self.interval_right)
        half_width = 0.5 * (self.interval_right - self.interval_left)
        self.target_variance = (half_width / z_value) ** 2
        self.jitter = jitter

    def _uncalibrated_mean_cov(self, test_x):
        base_mean, base_cov = self.xfoil_model.posterior_mean_cov(test_x)
        corr_mean, corr_cov = self.correction_model.posterior_mean_cov(test_x)
        return base_mean + corr_mean, base_cov + corr_cov

    def mean_variance(self, test_x):
        test_x = test_x.double()
        n_test = test_x.shape[0]
        n_calibration = self.calibration_x.shape[0]
        joint_x = torch.cat([test_x, self.calibration_x], dim=0)
        joint_mean, joint_cov = self._uncalibrated_mean_cov(joint_x)

        calibrated_means = []
        calibrated_variances = []

        for output_idx in range(joint_mean.shape[1]):
            mean_x = joint_mean[:n_test, output_idx]
            mean_t = joint_mean[n_test:, output_idx]
            cov = joint_cov[output_idx]

            k_xx = cov[:n_test, :n_test]
            k_xt = cov[:n_test, n_test:]
            k_tt = cov[n_test:, n_test:]
            k_tt = k_tt + self.jitter * torch.eye(
                n_calibration, dtype=torch.float64
            )

            chol = torch.linalg.cholesky(k_tt)
            residual = self.target_center[:, output_idx] - mean_t
            alpha = torch.cholesky_solve(residual.unsqueeze(-1), chol).squeeze(-1)
            calibrated_mean = mean_x + k_xt @ alpha

            k_tx = k_xt.transpose(0, 1)
            ktt_inv_ktx = torch.cholesky_solve(k_tx, chol)
            kriging_reduction = torch.sum(k_xt * ktt_inv_ktx.transpose(0, 1), dim=1)
            target_variance = self.target_variance[:, output_idx]
            replacement_variance = torch.sum(
                (ktt_inv_ktx**2) * target_variance.unsqueeze(-1),
                dim=0,
            )
            calibrated_variance = (
                torch.diag(k_xx) - kriging_reduction + replacement_variance
            ).clamp_min(0.0)

            calibrated_means.append(calibrated_mean)
            calibrated_variances.append(calibrated_variance)

        return (
            torch.stack(calibrated_means, dim=1),
            torch.stack(calibrated_variances, dim=1),
        )

    def query(self, test_x):
        mean, _ = self.mean_variance(test_x)
        return mean.detach()

    def sample(self, test_x):
        mean, variance = self.mean_variance(test_x)
        return mean + torch.randn_like(mean) * torch.sqrt(variance)


class MarginallyCalibratedCorrectedBatch:
    """
    Case-wise moment calibration for input-marginalized prediction samples.

    The conditional calibrated GP matches the supplied bounds at the center
    input. This wrapper rescales that predictive distribution so the response
    marginalized over each input uncertainty box matches the same target mean
    and Gaussian-equivalent variance.
    """

    def __init__(
        self,
        conditional_model,
        calibration_x,
        interval_left,
        interval_right,
        n_calibration_samples=1000,
        input_delta=DEFAULT_INPUT_DELTA,
        z_value=1.96,
        seed=12345,
        min_variance=1e-18,
    ):
        self.conditional_model = conditional_model
        self.calibration_x = calibration_x.double()
        self.interval_left = torch.tensor(interval_left, dtype=torch.float64)
        self.interval_right = torch.tensor(interval_right, dtype=torch.float64)
        self.target_center = 0.5 * (self.interval_left + self.interval_right)
        half_width = 0.5 * (self.interval_right - self.interval_left)
        self.target_variance = (half_width / z_value) ** 2
        self.input_delta = input_delta.double()
        self.min_variance = min_variance

        rng = np.random.default_rng(seed)
        marginal_means = []
        marginal_variances = []

        for x0 in self.calibration_x:
            design = self._uniform_input_samples(x0, n_calibration_samples, rng)
            mean, variance = self.conditional_model.mean_variance(design)
            marginal_mean = mean.mean(dim=0)
            marginal_variance = variance.mean(dim=0) + mean.var(dim=0, unbiased=False)
            marginal_means.append(marginal_mean)
            marginal_variances.append(marginal_variance.clamp_min(min_variance))

        self.source_marginal_mean = torch.stack(marginal_means, dim=0)
        self.source_marginal_variance = torch.stack(marginal_variances, dim=0)
        self.mean_shift = self.target_center - self.source_marginal_mean
        self.std_scale = torch.sqrt(
            self.target_variance
            / self.source_marginal_variance.clamp_min(min_variance)
        )

    def _uniform_input_samples(self, x0, n_samples, rng):
        x_min = x0 - self.input_delta
        x_max = x0 + self.input_delta
        samples = rng.uniform(
            low=x_min.detach().cpu().numpy(),
            high=x_max.detach().cpu().numpy(),
            size=(n_samples, x0.numel()),
        )
        return torch.from_numpy(samples).double()

    def mean_variance_for_case(self, case_idx, test_x):
        mean, variance = self.conditional_model.mean_variance(test_x)
        source_mean = self.source_marginal_mean[case_idx]
        target_mean = self.target_center[case_idx]
        scale = self.std_scale[case_idx]
        calibrated_mean = target_mean + scale * (mean - source_mean)
        calibrated_variance = variance * scale.pow(2)
        return calibrated_mean, calibrated_variance.clamp_min(0.0)

    def sample_for_case(self, case_idx, test_x, match_sample_moments=False):
        mean, variance = self.mean_variance_for_case(case_idx, test_x)
        samples = mean + torch.randn_like(mean) * torch.sqrt(variance)

        if match_sample_moments and samples.shape[0] > 1:
            sample_mean = samples.mean(dim=0)
            sample_std = samples.std(dim=0, unbiased=False).clamp_min(
                np.sqrt(self.min_variance)
            )
            target_std = torch.sqrt(self.target_variance[case_idx])
            samples = self.target_center[case_idx] + (
                samples - sample_mean
            ) * (target_std / sample_std)

        return samples


def loadTrainingData(filename, x_cols, y_col):
    if filename.endswith(".npy"):
        values = np.load(filename)
    else:
        values = pd.read_csv(filename).values

    x_data = torch.tensor(values[:, x_cols], dtype=torch.float64)
    y_data = torch.tensor(values[:, y_col], dtype=torch.float64)
    return x_data, y_data


def existing_path(preferred_path, fallback_path):
    return preferred_path if os.path.exists(preferred_path) else fallback_path




def makeCorrectionModel(xdata, ydata, model, bounds, train_yvar):
    """
    Train a correction model on residuals.
    """
    yevals = model.query(xdata)  # (N,)
    ydata = ydata.squeeze(-1)    # ensure (N,)
    deltas = ydata - yevals      # safe subtraction
    if isinstance(model, Model):
        return Model(xdata, deltas, bounds, train_yvar)

    elif isinstance(model, Batch):
        return Batch(xdata, deltas, bounds, train_yvar)

    else:
        raise TypeError("model must be Model or Batch")


def build_default_models(
    training_path=None,
    truth_path=None,
    bounds=DEFAULT_BOUNDS,
):
    """
    Build the base XFOIL surrogate and the truth-data residual correction model.
    """
    if training_path is None:
        training_path = existing_path(DEFAULT_TRAINING_PATH, DEFAULT_TRAINING_FALLBACK_PATH)
    if truth_path is None:
        truth_path = existing_path(DEFAULT_TRUTH_PATH, DEFAULT_TRUTH_FALLBACK_PATH)

    xfoil_x, xfoil_y = loadTrainingData(training_path, [0, 1, 2], [3, 4, 5])
    xfoil_model = Batch(xfoil_x, xfoil_y, bounds, DEFAULT_BASE_YVAR)

    true_x, true_y = loadTrainingData(truth_path, [0, 1, 2], [3, 4, 5])
    corrector = makeCorrectionModel(
        true_x,
        true_y,
        xfoil_model,
        bounds,
        DEFAULT_CORRECTION_YVAR,
    )
    return xfoil_model, corrector, true_x, true_y


def uniform_input_samples(x0, n_samples, delta=DEFAULT_INPUT_DELTA):
    """
    Sample uniformly around a center point using the UQ Challenge input intervals.
    """
    x_min = x0 - delta
    x_max = x0 + delta
    samples = np.random.uniform(
        low=x_min.detach().cpu().numpy(),
        high=x_max.detach().cpu().numpy(),
        size=(n_samples, x0.numel()),
    )
    return torch.from_numpy(samples).double()


def monte_carlo_case(x0, xfoil_model, corrector, n_samples):
    """
    Run one Monte Carlo case and return tensors needed for plotting and analysis.
    """
    x_samples = uniform_input_samples(x0, n_samples)
    y_base = xfoil_model.query(x_samples)
    y_corr = corrector.query(x_samples)
    y_combined = y_base + y_corr

    x0_batch = x0.unsqueeze(0)
    y_center_base = xfoil_model.query(x0_batch)
    y_center_corr = corrector.query(x0_batch)

    return {
        "x_samples": x_samples.detach().cpu().numpy(),
        "y_base": y_base.detach().cpu().numpy(),
        "y_corr": y_corr.detach().cpu().numpy(),
        "y_combined": y_combined.detach().cpu().numpy(),
        "y_center_base": y_center_base.detach().cpu().numpy()[0],
        "y_center_corr": y_center_corr.detach().cpu().numpy()[0],
        "y_center_combined": (
            y_center_base + y_center_corr
        ).detach().cpu().numpy()[0],
    }


def run_monte_carlo(points, xfoil_model, corrector, n_samples):
    cases = [monte_carlo_case(point, xfoil_model, corrector, n_samples) for point in points]
    return {
        "x_points": points.detach().cpu().numpy(),
        "x_samples": np.stack([case["x_samples"] for case in cases]),
        "y_base": np.stack([case["y_base"] for case in cases]),
        "y_corr": np.stack([case["y_corr"] for case in cases]),
        "y_combined": np.stack([case["y_combined"] for case in cases]),
        "y_center_base": np.stack([case["y_center_base"] for case in cases]),
        "y_center_corr": np.stack([case["y_center_corr"] for case in cases]),
        "y_center_combined": np.stack(
            [case["y_center_combined"] for case in cases]
        ),
        "input_labels": np.array(INPUT_LABELS),
        "output_labels": np.array(OUTPUT_LABELS),
    }


def save_npy_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, values in results.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), values)


def load_npy_results(input_dir):
    results = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            name = os.path.splitext(filename)[0]
            results[name] = np.load(os.path.join(input_dir, filename), allow_pickle=True)
    return results

def save_model(obj, path):
    """
    Save Model or Batch object to file.
    """

    if isinstance(obj, Model):
        torch.save({
            "type": "Model",
            "xdata": obj.xdata,
            "ydata": obj.ydata,
            "bounds": obj.bounds,
            "train_yvar": obj.train_yvar,
            "state_dict": obj.gp.state_dict()
        }, path)

    elif isinstance(obj, Batch):
        torch.save({
            "type": "Batch",
            "models": [
                {
                    "xdata": m.xdata,
                    "ydata": m.ydata,
                    "bounds": m.bounds,
                    "train_yvar": m.train_yvar,
                    "state_dict": m.gp.state_dict()
                }
                for m in obj.models
            ]
        }, path)

    else:
        raise TypeError("Object must be Model or Batch")



if __name__ == "__main__":
    re=697915.5270597762
    alpha=3.960168038192652
    flap = -1.3707409713137082
    xfoil = XFOIL("xfoil.exe",re,flap,"test.dat")
    xfoil.sweep(alpha) 
    xfoil.done()
    out = read_xfoil_polar(re,alpha,flap,"test.dat")
