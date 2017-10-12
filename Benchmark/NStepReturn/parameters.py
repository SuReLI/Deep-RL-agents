
ENV = "Acrobot-v1"

LOAD = False
DISPLAY = True

BUFFER_SAVE = False

CONV = False


DISCOUNT = 0.99

FRAME_SKIP = 0


EPSILON_START = 0.8
EPSILON_STOP = 0.01
EPSILON_STEPS = 100000

LEARNING_RATE = 7.5e-4

BUFFER_SIZE = 100000
PRIOR_ALPHA = 0.5
PRIOR_BETA_START = 0.4
PRIOR_BETA_STOP = 1
PRIOR_BETA_STEPS = 25000

BATCH_SIZE = 32

# Number of episodes of game environment to train with
TRAINING_STEPS = 25000
PRE_TRAIN_STEPS = 1000

# Maximal number of steps during one episode
MAX_EPISODE_STEPS = 600
TRAINING_FREQ = 4

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.001
