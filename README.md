# ADHP-maintains-pyloricness
Organized code to accompany the paper: Local activity-dependent homeostatic plasticity maintains circuit-level dynamic properties (Stolting & Beer, 2026)
(pre-print can be found here: )

The scripts and notebooks in this repository can be used to reproduce all experiments and recreate all figures in the corresponding paper. Due to space considerations, it does not include data. To examine pre-produced data, contact the author (lstoltin@iu.edu). All scripts are compiled using Makefile___ with the same suffix as the main___.cpp scrpt. Jupyter notebooks are used for visualization and light data processing of output files. 

- Evolve pyloric CTRNNs, collect information about the best evolved individuals' burst ordering and phase
  - mainpyloricevol.cpp
  - Orderingarchetypes.ipynb
- Evolve ADHP mechanisms of specified dimension to regulate one or several pyloric networks
  - mainADHPevol.cpp
  - ADHPstatistics.ipynb
- Gather data about the average value taken by circuits in a given section of parameter space and their pyloricness, then predict the performance of ADHP mechanisms
  - mainavgslice.cpp
  - mainpyloricslice.cpp
  - ADHPperformanceprediction.ipynb
- Test a given ADHP mechanism in detail and record its effect on network parameters, then visualize trajectories through parameter space as they relate to predicted nullclines
  - mainsimulateADHP.cpp
  - VisualizeADHPtrajectories.ipynb
- Test all possible ADHP mechanisms which regulate given network parameters, on one section of parameter network parameter space, and visualize for comparison to predicted performance
  - mainADHPmetaparameterspace.cpp
  - VisualizeADHPmetaparameterspace.ipynb


