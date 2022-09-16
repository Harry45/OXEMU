"""
The main setting file for the emulator

Author: Arrykrishna Mootoovaloo
Collaborators: David, Pedro, Jaime
Date: March 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: Emulator for computing the linear matter power spectrum
"""

# number of wavenumber to consider [equal to the number of GPs]
NWAVE = 40

# minimum and maximum redshift and wavenumber [1/Mpc]
Z_MIN = 0.0
Z_MAX = 5.0
K_MIN = 1e-4
K_MAX = 7.0

# CLASS output
# KiDS sets P_k_max_h/Mpc to 50 (if I remember correctly)
CLASS_ARGS = {"output": "mPk", "P_k_max_1/Mpc": K_MAX, "z_max_pk": Z_MAX}

# neutrino settings
NEUTRINO = {"N_ncdm": 1.0, "deg_ncdm": 3.0, "T_ncdm": 0.71611, "N_ur": 0.00641}

# cosmological parameters (DES)
COSMO = ["omega_cdm", "Omega_b", "sigma8", "n_s", "h"]

# cosmological parameters (KiDS)
# COSMO = ["omega_cdm", "omega_b", "ln10^{10}A_s", "n_s", "h"]

# priors (DES)
PRIORS = {
    "omega_cdm": {"distribution": "uniform", "specs": [0.05, 0.45]},
    "Omega_b": {"distribution": "uniform", "specs": [0.03, 0.015]},
    "sigma8": {"distribution": "uniform", "specs": [0.6, 0.30]},
    "n_s": {"distribution": "uniform", "specs": [0.87, 0.30]},
    "h": {"distribution": "uniform", "specs": [0.60, 0.31]},
}

# priors (KiDS)
# PRIORS = {
#     "omega_cdm": {"distribution": "uniform", "specs": [0.06, 0.34]},
#     "omega_b": {"distribution": "uniform", "specs": [0.019, 0.007]},
#     "ln10^{10}A_s": {"distribution": "uniform", "specs": [1.70, 3.30]},
#     "n_s": {"distribution": "uniform", "specs": [0.70, 0.60]},
#     "h": {"distribution": "uniform", "specs": [0.64, 0.18]},
# }

# the Gaussian Process settings
LEARN_RATE = 1e-2
NRESTART = 2
NITER = 1000
