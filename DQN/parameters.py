
ENV = "SpaceInvaders-v0"

LOAD = True
DISPLAY = True


DISCOUNT = 0.99

EPSILON_START = 0.8
EPSILON_STOP = 0.01
EPSILON_STEPS = 50000

LEARNING_RATE = 1e-4

BUFFER_SIZE = 100000
PRIOR_ALPHA = 0.5
PRIOR_BETA_START = 0.4
PRIOR_BETA_STOP = 1
PRIOR_BETA_STEPS = 100000

BATCH_SIZE = 32

# Number of episodes of game environment to train with
TRAINING_STEPS = 1000000
PRE_TRAIN_STEPS = 5000

# Maximal number of steps during one episode
MAX_EPISODE_STEPS = 600
TRAINING_FREQ = 4

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.001
