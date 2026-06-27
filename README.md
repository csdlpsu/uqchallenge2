# UQ Challenge 2 by CSDL
Architected by Geoffrey Davis and Ashwin Renganathan. 

## Background
The UQ Challenge problems are a series of two problems posed by Boeing investigating uncertainty quantification for aerospace
CFD applications. In the second problem, the goal was to predict the important 2D aerodynamic coefficients (`C_l`, `C_d`, `C_m`)
for the NACA 2412 airfoil under a variety of conditions. A nominal Reynolds number of 700,000 was set, and flap angle and angle
of attack were varied between -5 and 15 degrees, and -5 and 10 degrees, respectively. The precise conditions are shown in
[`desired_truths.csv`](desired_truths.csv). "Truth" information for these coefficients was provided at seven combinations of angle of attack and flap angle, as shown in [`given_truths.csv`](given_truths.csv).
However, the truth data had uncertainties in the conditions (angle of attack, flap angle, and Reynolds number) and desired quantities (`C_l`, `C_d`, `C_m`). Our work
uses Gaussian process models and Monte Carlo simulations to model the system and predict the coefficients and their uncertainties.

## Dependencies
Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

XFOIL is also required for generating new aerodynamic training data:
[https://web.mit.edu/drela/Public/web/xfoil/](https://web.mit.edu/drela/Public/web/xfoil/)

The paper-style plots use Matplotlib's LaTeX text rendering, so a working LaTeX
installation is needed when regenerating those figures.

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

4. Generate the distributionally calibrated prediction arrays used by the paper
   comparison figures:

   ```bash
   python distributional_calibrated_prediction.py
   python validate_marginal_calibration.py
   ```

   These commands require the epistemic bounds workbook. By default, the scripts
   look for `uq_challenge_ground_truth_epistemic_bounds.xlsx` in the repository
   root.

5. Plot saved `.npy` results:

   ```bash
   python plot_results.py all
   ```

Source `.npy` files are saved under `data/`. Monte Carlo result arrays are
saved under `results/`. Figures are saved under `plots/`.

Generated figures and local paper build products are intentionally ignored by
git. To regenerate the paper figures into the default local output directory,
run:

```bash
python plot_results.py desiderata-figure --plot-root plots/paper
python plot_results.py latent-gp-figure --plot-root plots/paper
python plot_results.py validation-true-vs-predict --plot-root plots/paper
python plot_results.py prediction-bounds --plot-root plots/paper
python plot_results.py first-moment-cdfs --plot-root plots/paper
python plot_results.py distributional-calibrated-bounds --plot-root plots/paper
python plot_results.py distributional-calibrated-cdfs --plot-root plots/paper
python plot_results.py calibration-cdfs --plot-root plots/paper
python plot_results.py calibration-cdf-overview --plot-root plots/paper
```

If you want the generated files next to a manuscript checkout instead, pass that
figure directory as `--plot-root`, for example `--plot-root paper/arxiv/figures`.

## Repository layout
- `uqlib.py`: shared XFOIL, Gaussian process, correction model, Monte Carlo,
  and `.npy` helper functions.
- `prepare_data.py`: converts checked-in CSV data to `.npy`.
- `xfoil_training.py`: generates new XFOIL training data and saves it to
  `data/training/training_100.npy`.
- `validation.py`: saves validation Monte Carlo arrays to `results/validation/`.
- `prediction.py`: saves requested prediction Monte Carlo arrays to
  `results/prediction/`.
- `distributional_calibrated_prediction.py`: saves distributionally calibrated
  prediction arrays to `results/distributional_calibrated_prediction/`.
- `validate_marginal_calibration.py`: checks the marginal calibration moments at
  the overlapping prediction/truth point.
- `plot_results.py`: creates plots from saved `.npy` arrays.

## Papers
[Uncertainty quantification via latent Gaussian process surrogates](https://pure.psu.edu/en/publications/uncertainty-quantification-via-latent-gaussian-process-surrogates/)
