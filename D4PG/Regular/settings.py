
from multiprocessing import cpu_count
import gym

ENV = "Pendulum-v0"

setting_env = gym.make(ENV)
print()

ACTION_SIZE = setting_env.action_space.shape[0]
STATE_SIZE = list(setting_env.observation_space.shape)
BOUNDS = (setting_env.action_space.low, setting_env.action_space.high)

del setting_env

DISPLAY = False
LOAD = False
INTERFACE = False

NB_ACTORS = 3  # cpu_count() - 2
NB_ATOMS = 51
MIN_VALUE = -2000
MAX_VALUE = 0

N_STEP_RETURN = 3
DISCOUNT = 0.99
DISCOUNT_N = DISCOUNT ** N_STEP_RETURN

MEMORY_SIZE = 1000000
BATCH_SIZE = 64


CRITIC_LEARNING_RATE = 5e-4
ACTOR_LEARNING_RATE = 5e-4

UPDATE_TARGET_RATE = 0.05

UPDATE_TARGET_FREQ = 1
UPDATE_ACTORS_FREQ = 1

NOISE_SCALE = 0.3
NOISE_DECAY = 0.99


MAX_STEPS = 2000

# Display settings
RENDER_FREQ = 1000
EP_REWARD_FREQ = 50
PLOT_FREQ = 100
PERF_FREQ = 100
SAVE_FREQ = 0
