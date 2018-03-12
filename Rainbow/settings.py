
class Settings:

    ###########################################################################
    # Environment settings

    ENV = "CartPole-v0"

    LOAD    = False
    DISPLAY = False
    GUI     = False

    TRAINING_EPS  = 100000
    PRE_TRAIN_EPS = 1000

    MAX_EPISODE_STEPS = 200
    FRAME_SKIP        = 0
    EP_ELONGATION     = 10


    ###########################################################################
    # Switches

    DOUBLE_DQN     = True
    DUELING_DQN    = True
    PRIORITIZED_ER = True
    DISTRIBUTIONAL = True
    N_STEP         = True


    ###########################################################################
    # Network settings

    # CONV_LAYERS = [
    #                 {'filters': 32, 'kernel_size': [8, 8], 'strides': [4, 4]},
    #                 {'filters': 64, 'kernel_size': [4, 4], 'strides': [2, 2]},
    #                 {'filters': 64, 'kernel_size': [3, 3], 'strides': [1, 1]}
    #               ]

    HIDDEN_LAYERS = [32, 32]

    NB_ATOMS = 51
    LEARNING_RATE = 7.5e-4


    ###########################################################################
    # Algorithm hyper-parameters

    DISCOUNT      = 0.99
    N_STEP_RETURN = 5
    DISCOUNT_N    = DISCOUNT ** N_STEP_RETURN

    MIN_Q = 0
    MAX_Q = 200

    BUFFER_SIZE = 100000
    BATCH_SIZE  = 64

    TRAINING_FREQ      = 4
    UPDATE_TARGET_RATE = 0.005


    ###########################################################################
    # Exploration settings

    EPSILON_START = 0.6
    EPSILON_STOP  = 0.01
    EPSILON_STEPS = 5000
    EPSILON_DECAY = (EPSILON_START - EPSILON_STOP) / EPSILON_STEPS
    

    ###########################################################################
    # Prioritized Experience Buffer settings
    
    ALPHA = 0.5
    BETA_START = 0.4
    BETA_STOP = 1
    BETA_STEPS = 25000
    BETA_INCR = (BETA_STOP - BETA_START) / BETA_STEPS


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
    MODEL_PATH = 'model/'
    GIF_PATH = 'results/gif/'
    MAX_NB_GIF = 5


    ###########################################################################

    import gym
    setting_env = gym.make(ENV)
    print()

    ACTION_SIZE = setting_env.action_space.n
    STATE_SIZE = list(setting_env.observation_space.shape)

    del setting_env

    if not N_STEP:
        N_STEP_RETURN = 1
