
class Settings:

    ###########################################################################
    # Environment settings

    ENV = "LunarLander-v2"

    LOAD    = True
    DISPLAY = True
    GUI     = True

    TRAINING_EPS  = 100000
    PRE_TRAIN_EPS = 1000

    MAX_EPISODE_STEPS = 200
    FRAME_SKIP        = 0
    EP_ELONGATION     = 10


    ###########################################################################
    # Network settings

    CONV = False
    if CONV:
        CONV_LAYERS = [32, 32, 32]

    HIDDEN_LAYERS = [32, 32]

    NB_ATOMS = 51
    LEARNING_RATE = 7.5


    ###########################################################################
    # Algorithm hyper-parameters

    DISCOUNT = 0.99
    MIN_Q    = -100
    MAX_Q    = 100

    BUFFER_SIZE = 100000
    BATCH_SIZE  = 64

    TRAINING_FREQ      = 4
    UPDATE_TARGET_RATE = 0.1


    ###########################################################################
    # Exploration settings

    EPSILON_START = 0.8
    EPSILON_STOP  = 0.01
    EPSILON_STEPS = 5000
    EPSILON_DECAY = (EPSILON_START - EPSILON_STOP) / EPSILON_STEPS
    

    ###########################################################################
    # Features frequencies
    
    EP_REWARD_FREQ = 50
    PLOT_FREQ      = 100
    RENDER_FREQ    = 500
    GIF_FREQ       = 2000
    SAVE_FREQ      = 1000


    ###########################################################################
    # Gif settings

    MAX_NB_GIF = 5
    GIF_PATH = 'results/gif/'


    ###########################################################################

    import gym
    setting_env = gym.make(ENV)
    print()

    ACTION_SIZE = setting_env.action_space.n
    STATE_SIZE = list(setting_env.observation_space.shape)

    del setting_env
