
import gym

class Settings:

    ENV = "Pendulum-v0"

    setting_env = gym.make(ENV)
    print()

    ACTION_SIZE = setting_env.action_space.shape[0]
    STATE_SIZE = list(setting_env.observation_space.shape)
    LOW_BOUND = setting_env.action_space.low
    HIGH_BOUND = setting_env.action_space.high

    del setting_env

    LOAD = False
    DISPLAY = True
    GUI = True

    DISCOUNT = 0.99
    FRAME_SKIP = 0
    BUFFER_SIZE = 100000
    BATCH_SIZE = 1024

    ACTOR_LEARNING_RATE = 5e-4
    CRITIC_LEARNING_RATE = 5e-4

    # Number of episodes of game environment to train with
    TRAINING_STEPS = 1000

    # Maximal number of steps during one episode
    MAX_EPISODE_STEPS = 1000
    TRAINING_FREQ = 1

    # Rate to update target network toward primary network
    UPDATE_TARGET_RATE = 0.05

    NOISE_SCALE_INIT = 0.1
    NOISE_DECAY = 0.99

    # settings for the exploration noise process:
    # dXt = theta*(mu-Xt)*dt + sigma*dWt
    EXPLO_MU = 0.0
    EXPLO_THETA = 0.15
    EXPLO_SIGMA = 0.2

    # Display Frequencies
    EP_REWARD_FREQ = 50
    PLOT_FREQ = 100
    RENDER_FREQ = 500
    GIF_FREQ = 2000
    SAVE_FREQ = 1000

    MAX_NB_GIF = 5
    GIF_PATH = 'results/gif/'
