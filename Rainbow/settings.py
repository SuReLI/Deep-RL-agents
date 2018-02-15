
import gym

class Settings:

    ENV = "LunarLander-v2"

    setting_env = gym.make(ENV)
    print()

    ACTION_SIZE = setting_env.action_space.n
    STATE_SIZE = list(setting_env.observation_space.shape)

    del setting_env

    LOAD = True
    DISPLAY = True
    GUI = True

    CONV = False

    DISCOUNT = 0.99
    N_STEP_RETURN = 3

    LEARNING_RATE = 7.5e-4

    FRAME_SKIP = 0
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64

    # Number of episodes of game environment to train with
    TRAINING_STEPS = 100000
    PRE_TRAIN_STEPS = 1000

    # Maximal number of steps during one episode
    MAX_EPISODE_STEPS = 200
    TRAINING_FREQ = 4

    # Rate to update target network toward primary network
    UPDATE_TARGET_RATE = 0.1

    EPSILON_START = 0.8
    EPSILON_STOP = 0.01
    EPSILON_STEPS = 5000
    EPSILON_DECAY = (EPSILON_START - EPSILON_STOP) / EPSILON_STEPS

    ALPHA = 0.5
    BETA_START = 0.4
    BETA_STOP = 1
    BETA_STEPS = 25000
    BETA_INCR = (BETA_STOP - BETA_START) / BETA_STEPS

    # Display Frequencies
    EP_REWARD_FREQ = 50
    PLOT_FREQ = 100
    RENDER_FREQ = 500
    GIF_FREQ = 2000
    SAVE_FREQ = 1000

    MAX_NB_GIF = 5
    GIF_PATH = 'results/gif/'
    EP_ELONGATION = 10
