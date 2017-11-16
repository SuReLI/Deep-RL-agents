
ENV = "BipedalWalker-v2"

LOAD = False
DISPLAY = False


DISCOUNT = 0.99

FRAME_SKIP = 0

EPSILON_START = 0.6
EPSILON_STOP = 0.01
EPSILON_STEPS = 15000
EPSILON_DECAY = (EPSILON_START - EPSILON_STOP) / EPSILON_STEPS

ACTOR_LEARNING_RATE = 5e-4
CRITIC_LEARNING_RATE = 5e-4
CRITIC_REG = 0.01

# Memory size
BUFFER_SIZE = 100000
BATCH_SIZE = 64

# Number of episodes of game environment to train with
TRAINING_STEPS = 15000

# Maximal number of steps during one episode
MAX_EPISODE_STEPS = 1000
TRAINING_FREQ = 4

# Rate to update target network toward primary network
UPDATE_TARGET_RATE = 0.1


# scale of the exploration noise process (1.0 is the range of each action
# dimension)
NOISE_SCALE_INIT = 0.1

# decay rate (per episode) of the scale of the exploration noise process
NOISE_DECAY = 0.99

# parameters for the exploration noise process:
# dXt = theta*(mu-Xt)*dt + sigma*dWt
EXPLO_MU = 0.0
EXPLO_THETA = 0.15
EXPLO_SIGMA = 0.2


# Display settings
DISP_REWARD_FREQ = 5
PLOT_FREQ = 10000
RENDER_FREQ = 100
SAVE_FREQ = 100
