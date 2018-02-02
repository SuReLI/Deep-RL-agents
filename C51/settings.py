
import gym

class Settings:
    
    def __init__(self):

        self.ENV = "LunarLander-v2"

        setting_env = gym.make(self.ENV)
        print()

        self.ACTION_SIZE = setting_env.action_space.n
        self.STATE_SIZE = list(setting_env.observation_space.shape)

        del setting_env

        self.LOAD = True
        self.DISPLAY = True
        self.GUI = True

        self.CONV = False

        self.DISCOUNT = 0.99
        self.NB_ATOMS = 51
        self.MIN_Q = -100
        self.MAX_Q = 100

        self.LEARNING_RATE = 7.5


        self.FRAME_SKIP = 0
        self.BUFFER_SIZE = 100000
        self.BATCH_SIZE = 64

        # Number of episodes of game environment to train with
        self.TRAINING_STEPS = 100000
        self.PRE_TRAIN_STEPS = 1000

        # Maximal number of steps during one episode
        self.MAX_EPISODE_STEPS = 200
        self.TRAINING_FREQ = 4

        # Rate to update target network toward primary network
        self.UPDATE_TARGET_RATE = 0.1


        self.EPSILON_START = 0.8
        self.EPSILON_STOP = 0.01
        self.EPSILON_STEPS = 5000
        self.EPSILON_DECAY = (self.EPSILON_START - self.EPSILON_STOP) / self.EPSILON_STEPS

        # Display Frequencies
        self.EP_REWARD_FREQ = 50
        self.PLOT_FREQ = 100
        self.RENDER_FREQ = 500
        self.GIF_FREQ = 2000
        self.SAVE_FREQ = 1000

        self.MAX_NB_GIF = 5
        self.GIF_PATH = 'results/gif/'
        self.EP_ELONGATION = 10
