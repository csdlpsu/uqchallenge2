# UQ Challenge 2 by CSDL
Architected by Geoffrey Davis and Ashwin Renganathan. 

## Background
The UQ Challenge problems are a series of two problems posed by Boeing investigating uncertainty quantification for aerospace
CFD applications. In the second problem, the goal was to predict the important 2D aerodynamic coefficients (Cl, Cd, Cm)
for the NACA 2412 airfoil under a variety of conditions. A nominal Reynolds number of 700,000 was set, and flap angle and angle 
of attack were varied between -5 and 15 degrees, and -5 and 10 degrees, respectively. The precise conditions are shown In
desired_truths.csv. "Truth" information regarding these coefficients were provided at seven combinations of angle of attack and flap angle, as shown in given_truths.csv.
However, the truth data had uncertainties in the conditions (angle of attack, flap angle, and Reynolds number) and desired quantities (Cl, Cd, Cm). Our work
attempts to use Gaussian process models and Monte-Carlo simulations to model the system and predict the coefficients and uncertainties in those coefficients.

## Dependencies
Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

XFOIL is also required for generating new aerodynamic training data:
https://web.mit.edu/drela/Public/web/xfoil/

## Workflow
The scripts separate numerical data generation from plotting. Monte Carlo
outputs are saved as `.npy` files, and the plotting script reads those saved
arrays instead of rerunning the models.

1. Convert the checked-in CSV files to the `.npy` data layout:

   ```bash
   python prepare_data.py
   ```

2. Generate Monte Carlo validation arrays:

   ```bash
   python validation.py
   ```

3. Generate Monte Carlo prediction arrays:

   ```bash
   python prediction.py
   ```

4. Plot saved `.npy` results:

   ```bash
   python plot_results.py all
   ```

Source `.npy` files are saved under `data/`. Monte Carlo result arrays are
saved under `results/`. Figures are saved under `plots/`.

## Repository layout
- `uqlib.py`: shared XFOIL, Gaussian process, correction model, Monte Carlo,
  and `.npy` helper functions.
- `prepare_data.py`: converts checked-in CSV data to `.npy`.
- `xfoil_training.py`: generates new XFOIL training data and saves it to
  `data/training/training_100.npy`.
- `validation.py`: saves validation Monte Carlo arrays to `results/validation/`.
- `prediction.py`: saves requested prediction Monte Carlo arrays to
  `results/prediction/`.
- `plot_results.py`: creates plots from saved `.npy` arrays.

## Papers
https://pure.psu.edu/en/publications/uncertainty-quantification-via-latent-gaussian-process-surrogates/
