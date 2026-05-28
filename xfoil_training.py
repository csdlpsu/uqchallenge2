# -*- coding: utf-8 -*-
"""
@author: Geoffrey C Davis, B.S. Aerospace Engineering, The Pennsylvania State University, 2026. 
Researcher at Computational complex engineered Systems Design Laboratory

This code is for generating training data for XFOIL surrogate. To change data 
ranges and number of training points, modify the highlighted blocks at the top. 
"""

import uqlib as uq
from pyDOE import lhs    
import numpy as np
import pandas as pd
##################################
#Modify stuff here
re_range = [7e5*.995, 7e5 * 1.005]
alpha_range = [-5,10]
flap_range = [-5,15]
num = 100
outpath = f"training/training_{num}.csv"
#################################

#desired sampling points
bounds = np.array([re_range,alpha_range,flap_range])
samples = lhs(3, samples=num) 
scaled_samples = samples * (bounds[:,1] - bounds[:,0]) + bounds[:,0]

#logging
training_list = []
fail_count = 0;

for i in range(num):
    print(f"{i+1}/{num}")
    params = scaled_samples[i,:]
    re = params[0]
    alpha = params[1]
    flap = params[2]
    xfoil = uq.XFOIL("xfoil.exe", re, flap, "out.dat")
    xfoil.run_alpha(alpha)
    xfoil.done()
    data,error = uq.read_xfoil_polar(re,alpha,flap,"out.dat")
    #if it errors first time, we try a sweep. If sweep doesn't work, move on. 
    if error:
        print("Attempting sweep")
        xfoil = uq.XFOIL("xfoil.exe", re, flap, "out.dat") #need to reinit
        xfoil.sweep(alpha)
        xfoil.done()
        data,error = uq.read_xfoil_polar(re,alpha,flap,"out.dat")
    if error:
        print(f"Failed to converge at ({re},{alpha},{flap})")
        fail_count +=1
        continue
    else:
        training_list.append(data)
print(f"{fail_count}/{num} failed to converge")
if training_list:
    print("Exporting to csv...")
    training_array = np.array(training_list)
    df = pd.DataFrame(training_array, columns=["re","alpha","flap","cl", "cd", "cm"])
    df.to_csv(outpath,index=False)
else:
    print("Empty data. Nothing written to file")
        
        
