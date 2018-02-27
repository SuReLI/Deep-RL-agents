
class Settings:

    ###########################################################################
    # Environment settings

    ENV = "Pong-v0"

    LOAD    = True
    DISPLAY = True
    GUI     = True

    TRAINING_EPS  = 100000
    PRE_TRAIN_EPS = 100

    MAX_EPISODE_STEPS = 250
    FRAME_SKIP        = 4
    EP_ELONGATION     = 5


    ###########################################################################
    # Network settings

    CONV_LAYERS = [
                    {'filters': 32, 'kernel_size': [8, 8], 'strides': [4, 4]},
                    {'filters': 64, 'kernel_size': [4, 4], 'strides': [2, 2]},
                    {'filters': 64, 'kernel_size': [3, 3], 'strides': [1, 1]}
                  ]

    HIDDEN_LAYERS = [512]

    NB_ATOMS      = 51
    LEARNING_RATE = 7.5e-4


    ###########################################################################
    # Algorithm hyper-parameters

    DISCOUNT = 0.99
    MIN_Q    = -21
    MAX_Q    = 21

    BUFFER_SIZE = 100000
    BATCH_SIZE  = 64

    TRAINING_FREQ      = 1
    UPDATE_TARGET_RATE = 0.05


    ###########################################################################
    # Exploration settings

    EPSILON_START = 0.6
    EPSILON_STOP  = 0.01
    EPSILON_STEPS = 2500
    EPSILON_DECAY = (EPSILON_START - EPSILON_STOP) / EPSILON_STEPS
    

    ###########################################################################
    # Features frequencies
    
    EP_REWARD_FREQ = 10
    PLOT_FREQ      = 10
    RENDER_FREQ    = 500
    GIF_FREQ       = 25
    SAVE_FREQ      = 100


    ###########################################################################
    # Save settings

    RESULTS_PATH = 'results/'
    MODEL_PATH   = 'model/'
    GIF_PATH     = 'results/gif/'
    MAX_NB_GIF   = 50


    ###########################################################################

    import gym
    setting_env = gym.make(ENV)
    print()

    if 'CONV_LAYERS' in locals():
        STATE_SIZE = [84, 84, 4]
    else:
        STATE_SIZE  = list(setting_env.observation_space.shape)
    ACTION_SIZE = setting_env.action_space.n

    del setting_env
