
from multiprocessing import cpu_count

class Settings:

    ###########################################################################
    # Environment settings

    ENV = "Pendulum-v0"

    LOAD    = False
    DISPLAY = False
    GUI     = False

    TRAINING_EPS  = 250

    MAX_EPISODE_STEPS = 2000
    FRAME_SKIP        = 0
    EP_ELONGATION     = 10


    ###########################################################################
    # Network settings


    # CONV_LAYERS = [
    #                 {'filters': 32, 'kernel_size': [8, 8], 'strides': [4, 4]},
    #                 {'filters': 64, 'kernel_size': [4, 4], 'strides': [2, 2]},
    #                 {'filters': 64, 'kernel_size': [3, 3], 'strides': [1, 1]}
    #               ]

    HIDDEN_ACTOR_LAYERS  = [8, 8, 8]
    HIDDEN_CRITIC_LAYERS = [8, 8, 8]

    NB_ATOMS             = 51
    ACTOR_LEARNING_RATE  = 5e-4
    CRITIC_LEARNING_RATE = 5e-4


    ###########################################################################
    # Algorithm hyper-parameters

    NB_ACTORS = cpu_count() - 2

    DISCOUNT      = 0.99
    N_STEP_RETURN = 5
    DISCOUNT_N    = DISCOUNT ** N_STEP_RETURN

    MIN_Q = -2000
    MAX_Q = 0

    BUFFER_SIZE = 100000
    BATCH_SIZE  = 64

    UPDATE_TARGET_FREQ = 4
    UPDATE_TARGET_RATE = 0.001

    UPDATE_ACTORS_FREQ = 5


    ###########################################################################
    # Exploration settings

    NOISE_SCALE = 0.3
    NOISE_DECAY = 0.99
    

    ###########################################################################
    # Features frequencies
    
    EP_REWARD_FREQ = 50
    PLOT_FREQ      = -1
    RENDER_FREQ    = -1
    GIF_FREQ       = -1
    SAVE_FREQ      = -1
    PERF_FREQ      = 100


    ###########################################################################
    # Save settings

    RESULTS_PATH = 'results/'
    MODEL_PATH   = 'model/'
    GIF_PATH     = 'results/gif/'
    MAX_NB_GIF   = 5


    ###########################################################################

    import gym
    setting_env = gym.make(ENV)

    if 'CONV_LAYERS' in locals():
        STATE_SIZE = [84, 84, 4]
    else:
        STATE_SIZE  = list(setting_env.observation_space.shape)
    ACTION_SIZE = setting_env.action_space.shape[0]
    LOW_BOUND   = setting_env.action_space.low
    HIGH_BOUND  = setting_env.action_space.high

    del setting_env
