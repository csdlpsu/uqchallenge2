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
Matplotlib: pip install matplotlib
Pandas: pip install pandas
Numpy: pip install numpy
Scipy: pip install scipy
Botorch: pip install botorch
XFOIL: https://web.mit.edu/drela/Public/web/xfoil/

## Papers
https://pure.psu.edu/en/publications/uncertainty-quantification-via-latent-gaussian-process-surrogates/