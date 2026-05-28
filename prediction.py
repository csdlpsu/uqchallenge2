# -*- coding: utf-8 -*-
"""
@author: Geoffrey C Davis, B.S. Aerospace Engineering, The Pennsylvania State University, 2026. 
Researcher at Computational complex engineered Systems Design Laboratory

This code is for validating model training using monte-carlo distributions. 
"""

import uqlib as uq
import torch

from scipy.stats import uniform
import numpy as np
import matplotlib.pyplot as plt
import os
save_dir = "prediction_plots"
os.makedirs(save_dir, exist_ok=True)


########### Loading in training data and fitting the models ##################
xfoil_x,xfoil_y = uq.loadTrainingData("training/training_100.csv",[0,1,2],[3,4,5])
bounds = torch.tensor([[7e5*.995,-5.0,-5.0],[7e5*1.005,10.0,15.0]])

xfoil_model = uq.Batch(xfoil_x,xfoil_y,bounds,[1e-5,1e-8,1e-7])

true_x,true_y = uq.loadTrainingData("given_truths.csv",[0,1,2],[3,4,5])
corrector = uq.makeCorrectionModel(true_x,true_y,xfoil_model, bounds, [1e-6,1e-10,1e-9])
#####################################################################################

n_samples = 1000 #monte-carlo samples

'''
This loop is going to do a monte-carlo simulation at each of the 4 desired 
points requested by Boeing. Essentially, an interval is created based on the 
epistemic uncertainty intervals provided, and is placed at each point: For instance,
Re = 7e5 +- .5%. This interval is treated as epistemic, and is sampled assuming 
a uniform distribution. The sampled points are then propagated through the GPs,
and the results are plotted, showing both the XFOIL Gp by itself, and the corrected
GP (XFOIL + correction GP). 
'''
pred_x = uq.loadTrainingData("desired_truths.csv",[0,1,2],[])[0]

for idx in range(len(pred_x)):

    x0 = pred_x[idx]

    delta = torch.tensor([7e5 * 0.005, 0.02, 0.1]) 

    x_min = x0 - delta #sampling intervals
    x_max = x0 + delta


    samples = []
    #creating the random samples for the 3 dimensions (Cl, Cd, Cm)
    for i in range(len(x0)):
        
        samples.append(
            uniform.rvs(
                loc=x_min[i].item(),
                scale=(x_max[i] - x_min[i]).item(),
                size=n_samples
            )
        )

    x_samples_np = np.stack(samples, axis=1)
    x_samples = torch.from_numpy(x_samples_np).double()

    # -------------------------
    # batch predictions (3 outputs)
    # -------------------------
    y_base = xfoil_model.query(x_samples)       # (N, 3)
    y_corr = corrector.query(x_samples)         # (N, 3)
    y_combined = y_base + y_corr                # (N, 3)

    # -------------------------
    # center point eval
    # -------------------------
    x0_batch = x0.unsqueeze(0)

    y_center_base = xfoil_model.query(x0_batch).detach().cpu().numpy()[0]
    y_center_combined = (
        y_center_base + corrector.query(x0_batch).detach().cpu().numpy()[0]
    )

    # -------------------------
    # plot all 3 outputs
    # -------------------------
    labels = ["CL", "CD", "CM"]

    for j in range(3):

        plt.figure(figsize=(7, 5))

        plt.hist(y_base[:, j].detach().numpy(), bins=40, alpha=0.5, label="Base")
        plt.hist(y_combined[:, j].detach().numpy(), bins=40, alpha=0.5, label="Corrected")

        plt.axvline(y_center_base[j], color='blue', linestyle='--', label="Base @ point")
        plt.axvline(y_center_combined[j], color='orange', linestyle='--', label="Corrected @ point")

        plt.xlabel(labels[j])
        plt.ylabel("Frequency")
        aoa = int(x0[1].item())
        flap = int(x0[2].item())
        
        plt.title(f"MC ({aoa}°, flap={flap}°) - {labels[j]}")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"mc_aoa{aoa}_flap{flap}_{labels[j]}.png"))
        plt.close()