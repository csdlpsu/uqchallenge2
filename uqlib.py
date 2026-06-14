# -*- coding: utf-8 -*-
"""
@author: Geoffrey C Davis, B.S. Aerospace Engineering, The Pennsylvania State University, 2026. 
Researcher at Computational complex engineered Systems Design Laboratory

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
    output : TYPE
        DESCRIPTION.

    """
    
    #this is logic to wait for XFOIL to flush its info to the polarpath
    max_wait = 2 #prevent infinite loops
    elapsed = 0
    stamp = time.time()
    #filesize of 750 is a tad over the base size for an unfilled polar. So if the base polar has been created,
    #but not values put into it, it will be detected
    while (not os.path.exists(filename) or (os.path.exists(filename) and os.path.getsize(filename) < 750)) and elapsed < max_wait:
        time.sleep(.05)
        elapsed = time.time() - stamp
        
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Skip header lines
    data_lines = []
    
    #this goes through every line in polar, finds coeffs and stores
    #get overwritten until last line, so this only returns last run in an accumulated polar file
    for line in lines:
        try:
            #print(line)
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure enough columns]
                cl = float(parts[1])
                cd = float(parts[2])
                cm = float(parts[4])
                if abs(alpha-float(parts[0])) < 1e-3: #checking to make sure that alpha is same as expected 
                    data_lines=[np.array([re,alpha,flap,cl, cd, cm])]    
        except ValueError:
            continue  # Skip non-numeric lines
    error = False
    if not data_lines:
        error = True
    else:
        data_lines = data_lines[0] #if data_lines not empty, return the part inside
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
