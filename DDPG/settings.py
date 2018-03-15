
class Settings:

    ###########################################################################
    # Environment settings

    ENV = "Pendulum-v0"

    LOAD    = True
    DISPLAY = True
    GUI     = True

    TRAINING_EPS  = 1000
    PRE_TRAIN_EPS = 1000

    MAX_EPISODE_STEPS = 200
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

    ACTOR_LEARNING_RATE  = 5e-4
    CRITIC_LEARNING_RATE = 5e-4


    ###########################################################################
    # Algorithm hyper-parameters

    DISCOUNT = 0.99

    BUFFER_SIZE = 100000
    BATCH_SIZE  = 1024

    TRAINING_FREQ      = 1
    UPDATE_TARGET_RATE = 0.05


    ###########################################################################
    # Exploration settings

    NOISE_SCALE_INIT = 0.1
    NOISE_DECAY      = 0.99

    EXPLO_MU    = 0.0
    EXPLO_THETA = 0.15
    EXPLO_SIGMA = 0.2
    

    ###########################################################################
    # Features frequencies
    
    EP_REWARD_FREQ = 50
    PLOT_FREQ      = 100
    RENDER_FREQ    = 500
    GIF_FREQ       = 2000
    SAVE_FREQ      = 1000


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
