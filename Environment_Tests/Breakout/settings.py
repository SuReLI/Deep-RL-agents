# -*- coding: utf-8 -*-
ENV = "breakout.bin"
ACTION_SIZE = 4

DISPLAY = False
LOAD = True

NB_THREADS = 12 # parallel thread size

GAMMA = 0.99
UPDATE_FREQ = 25

# RMSProp parameters
RMSP_ALPHA = 0.99
RMSP_EPSILON = 0.1

# Learning rate distribution parameters
INITIAL_ALPHA_LOW = 1e-7
INITIAL_ALPHA_HIGH = 1e-5
INITIAL_ALPHA_LOG_RATE = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4)

ENTROPY_REG = 0.01

MAX_TIME_STEP = 10 * 10**12
MAX_GRADIENT_NORM = 40.0

LOG_FILE = 'summary/'


# Display settings
DISP_REWARD_FREQ = 5
PLOT_FREQ = 10000
RENDER_FREQ = 50
PERF_FREQ = 100
