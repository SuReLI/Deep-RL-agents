
ENV = "BipedalWalker-v2"

LOAD = True
DISPLAY = True


DISCOUNT = 0.99

FRAME_SKIP = 4
EPSILON_START = 0.1
EPSILON_STOP = 0.05
EPSILON_STEPS = 25000


ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3

# Memory size
BUFFER_SIZE = 100000
BATCH_SIZE = 1024

# Number of episodes of game environment to train with
TRAINING_STEPS = 1500000

# Maximal number of steps during one episode
MAX_EPISODE_STEPS = 125
TRAINING_FREQ = 4

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.01
