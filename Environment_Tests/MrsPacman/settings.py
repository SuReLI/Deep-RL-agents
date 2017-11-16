# -*- coding: utf-8 -*-
ENV = "ms_pacman.bin"
ACTION_SIZE = 5

DISPLAY = False
LOAD = False

NB_THREADS = 8 # parallel thread size

GAMMA = 0.99
UPDATE_FREQ = 20

# RMSProp parameters
RMSP_ALPHA = 0.99
RMSP_EPSILON = 0.1

# Learning rate distribution parameters
INITIAL_ALPHA_LOW = 1e-4
INITIAL_ALPHA_HIGH = 1e-2
INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)

ENTROPY_REG = 0.01

MAX_TIME_STEP = 10 * 10**7
MAX_GRADIENT_NORM = 40.0

LOG_FILE = 'summary/'


# Display settings
DISP_REWARD_FREQ = 5
PLOT_FREQ = 25
RENDER_FREQ = 50
PERF_FREQ = 100
