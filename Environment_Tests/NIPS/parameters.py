
ENV = "NIPS"

LOAD = True
DISPLAY = True


DISCOUNT = 0.99

FRAME_SKIP = 3
EPSILON_START = 0.8
EPSILON_STOP = 0.08
EPSILON_STEPS = 50000


ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3

# Memory size
BUFFER_SIZE = 100000
BATCH_SIZE = 512

# Number of episodes of game environment to train with
TRAINING_STEPS = 1500000

# Maximal number of steps during one episode
MAX_EPISODE_STEPS = 125
TRAINING_FREQ = 1

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.01

# Display Frequencies
DISP_EP_REWARD_FREQ = 50
PLOT_FREQ = 100

SAVE_FREQ = 100
EP_ELONGATION = 50
