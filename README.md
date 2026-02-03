# ADHP-maintains-pyloricness
Organized code to accompany the paper: When can local activity-dependent homeostatic plasticity maintain circuit-level dynamic properties? (Stolting & Beer, 2026)
(pre-print can be found here: )
(stable code base can be found here: )

The scripts and notebooks in this repository can be used to reproduce experiments and recreate figures in the corresponding paper. Due to file size restrictions, it does not include all necessary data, but does include our two datasets of 100 evolved pyloric CTRNNs and our dataset of evolved ADHP mechanisms. For parameter space data or further clarification, contact the corresponding author (lstoltin@iu.edu). 

## Prerequisite files
The following files contain necessary functions, class definitions, and utilities.
- CTRNN.cpp
- CTRNN.h
- metaparres.dat
- plasticpars.dat (always ensure that the regulated parameter vector in this file matches the settings used in any ADHP files)
- pyloric.h
- random.cpp
- random.h
- TSearch.cpp
- TSearch.h
- VectorMatrix.h

## Simulation Scripts and Analysis Notebooks
All scripts are compiled using Makefile___ with the same suffix as the main___.cpp script. Jupyter notebooks are used for visualization and light data processing of output files. 

- mainpyloricevol.cpp: Evolve pyloric CTRNNs, collect information about the best evolved individuals' burst ordering and phase
  - Orderingarchetypes.ipynb
- mainADHPevol.cpp: Evolve ADHP mechanisms of specified dimension to regulate one or several pyloric networks and examine their parameters
  - ADHPstatistics.ipynb
- mainavgpyloric.cpp: Gather data about the average value taken by circuits in a given section of parameter space and their pyloricness, then use to predict the performance of ADHP mechanisms
  - ADHPperformanceprediction.ipynb (2D)
  - 3Dpredictionvsperformance.ipynb (3D)
- mainsimulateADHP.cpp: Test a given ADHP mechanism in detail and record its effect on network parameters, then visualize trajectories through parameter space as they relate to predicted nullclines
  - VisualizeADHPtrajectories.ipynb
- mainADHPmetaparspace.cpp: Test all possible ADHP mechanisms which regulate given network parameters, on one section of parameter network parameter space, and visualize for comparison to predicted performance
  - VisualizeADHPmetaparameterspace.ipynb

## Data File Layouts
Pyloric base circuits are CTRNNs specified by parameters in the "pyloriccircuit.ns" files of the following format:
#neurons (N)
neural time constants (taus)
neural biases (thetas)
neural gains (ones)
synaptic weight matrix (NxN)

Neural output space trajectories over one cycle are recorded in corresponding "pylorictrajectory.ns" files, with each row listing neural outputs
at each timestep 
"pyloricbursttimes.dat" files summarize timestamps for LPstart, LPend, PYstart, PYend, PDstart, and PDend relative to PDstart, followed by total period in seconds. 

ADHP mechanisms, evolved or otherwise, are specified by parameters in "bestind.dat" files with the following format:
Specify parameters being regulated using 1's and fixed parameters with 0's. Vector lists all biases (theta_1, theta_2,...,theta_n) then all weights (w_11,..., w_1n,......,w_n1,...,w_nn)
Evolutionary Genotype (irrelevant unless evolved)
Parameter time constants in order (tau_theta's), neuron lower bounds in order (LB_i), neuron ranges in order (Delta_i), parameter sliding windows in order (seconds)
Fitness value (irrelevant unless evolved)

"recoverytest.dat" files record the effects of the relevant ADHP mechanism acting on the relevant pyloric circuit over time, after being
perturbed to various initial points throughout the theta_LP, theta_PD plane. Each line lists perturbed theta_LP, perturbed theta_PD, initial
pyloric fitness, and then re-tests the pyloric fitness at intervals of 5000 seconds.
