
ENV = "NIPS"

LOAD = False
DISPLAY = True


DISCOUNT = 0.99

FRAME_SKIP = 0
EPSILON_START = 0.8
EPSILON_STOP = 0.1
EPSILON_STEPS = 10000


ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3

# Memory size
BUFFER_SIZE = 100000
BATCH_SIZE = 1024

# Number of episodes of game environment to train with
TRAINING_STEPS = 150000

# Maximal number of steps during one episode
MAX_EPISODE_STEPS = 200
TRAINING_FREQ = 1

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.01
