
ENV = "FrozenLake-v0"

LOAD = False
DISPLAY = True

FRAME_SKIP = 1

DISCOUNT = 0.99
LEARNING_RATE = 1e-4

EPSILON_START = 0.8
EPSILON_STOP = 0.05
EPSILON_STEPS = 100000

BUFFER_SIZE = 200000
PRIOR_ALPHA = 0.5
PRIOR_BETA_START = 0.4
PRIOR_BETA_STOP = 1
PRIOR_BETA_STEPS = 100000

BATCH_SIZE = 32

# Number of episodes of game environment to train with
TRAINING_STEPS = 100000
PRE_TRAIN_STEPS = 100

# Maximal number of steps during one episode
MAX_EPISODE_STEPS = 250
TRAINING_FREQ = 4

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.001
