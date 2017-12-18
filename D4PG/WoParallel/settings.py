
ENV = "Pendulum-v0"

LOAD = False
DISPLAY = True

NB_ATOMS = 12
MIN_VALUE = -2000
MAX_VALUE = 0

DISCOUNT = 0.99

# Memory settings
MEMORY_SIZE = 1000000
BATCH_SIZE = 64

# Learning rates
ACTOR_LEARNING_RATE = 5e-4
CRITIC_LEARNING_RATE = 5e-4



# Maximal number of steps during one episode
TRAINING_EPS = 1000
MAX_EPISODE_STEPS = 100000

# Rate to update target network toward primary network
TRAINING_FREQ = 1
UPDATE_TARGET_RATE = 0.05

NOISE_SCALE = 0.3
NOISE_DECAY = 0.99


# Display Frequencies
DISP_EP_REWARD_FREQ = 1
PLOT_FREQ = 5000
RENDER_FREQ = 50
GIF_FREQ = 1000
