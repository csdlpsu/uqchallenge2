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
    df = pd.read_csv(filename)
    x_data = torch.tensor(df.iloc[:, x_cols].values, dtype=torch.float64)
    y_data = torch.tensor(df.iloc[:, y_col].values, dtype=torch.float64)
    return x_data, y_data




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
