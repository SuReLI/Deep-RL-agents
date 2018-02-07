
import gym

class Settings:
    
    def __init__(self):

        self.ENV = "Pendulum-v0"

        setting_env = gym.make(self.ENV)
        print()

        self.ACTION_SIZE = setting_env.action_space.shape[0]
        self.STATE_SIZE = list(setting_env.observation_space.shape)
        self.LOW_BOUND = setting_env.action_space.low
        self.HIGH_BOUND = setting_env.action_space.high

        del setting_env

        self.LOAD = False
        self.DISPLAY = True
        self.GUI = True

        self.DISCOUNT = 0.99
        self.FRAME_SKIP = 0
        self.BUFFER_SIZE = 100000
        self.BATCH_SIZE = 1024

        self.ACTOR_LEARNING_RATE = 5e-4
        self.CRITIC_LEARNING_RATE = 5e-4

        # Number of episodes of game environment to train with
        self.TRAINING_STEPS = 1000

        # Maximal number of steps during one episode
        self.MAX_EPISODE_STEPS = 1000
        self.TRAINING_FREQ = 1

        # Rate to update target network toward primary network
        self.UPDATE_TARGET_RATE = 0.05

        self.NOISE_SCALE_INIT = 0.1
        self.NOISE_DECAY = 0.99

        # settings for the exploration noise process:
        # dXt = theta*(mu-Xt)*dt + sigma*dWt
        self.EXPLO_MU = 0.0
        self.EXPLO_THETA = 0.15
        self.EXPLO_SIGMA = 0.2

        # Display Frequencies
        self.EP_REWARD_FREQ = 50
        self.PLOT_FREQ = 100
        self.RENDER_FREQ = 500
        self.GIF_FREQ = 2000
        self.SAVE_FREQ = 1000

        self.MAX_NB_GIF = 5
        self.GIF_PATH = 'results/gif/'
