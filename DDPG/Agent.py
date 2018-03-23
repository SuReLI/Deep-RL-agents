
import tensorflow as tf
import numpy as np

from QNetwork import Network
from ExperienceBuffer import ExperienceBuffer
from Environment import Environment

from settings import Settings


class Agent:
    """
    This class builds an agent with its own Network, memory buffer and
    environment to learn a policy.
    """

    def __init__(self, sess, gui, displayer, saver):
        """
        Build a new instance of Environment, QNetwork and ExperienceBuffer.

        Args:
            sess     : the tensorflow session in which to build the network
            gui      : a GUI instance to manage the control of the agent
            displayer: a Displayer instance to keep track of the episode rewards
            saver    : a Saver instance to save periodically the network
        """
        print("Initializing the agent...")

        self.sess = sess
        self.gui = gui
        self.displayer = displayer
        self.saver = saver

        self.env = Environment()
        self.network = Network(sess)
        self.buffer = ExperienceBuffer()

        self.create_summaries()

        self.best_run = -1e10
        self.n_gif = 0

        print("Agent initialized !")

    def create_summaries(self):

        self.ep_reward_ph = tf.placeholder(tf.float32)
        ep_reward_summary = tf.summary.scalar("Episode/Episode reward", self.ep_reward_ph)

        self.steps_ph = tf.placeholder(tf.float32)
        steps_summary = tf.summary.scalar("Episode/Nb steps", self.steps_ph)

        self.noise_ph = tf.placeholder(tf.float32)
        noise_summary = tf.summary.scalar("Settings/Noise", self.noise_ph)

        self.ep_summary = tf.summary.merge([ep_reward_summary,
                                            noise_summary,
                                            steps_summary])

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

    def pre_train(self):
        """
        Method to run a random agent in the environment to fill the memory
        buffer.
        """
        print("Beginning of the pre-training...")

        for i in range(Settings.PRE_TRAIN_EPS):

            s = self.env.reset()
            done = False
            episode_reward = 0
            episode_step = 0

            while episode_step < Settings.MAX_EPISODE_STEPS and not done:

                a = self.env.act_random()
                s_, r, done, info = self.env.act(a)
                self.buffer.add((s, a, r, s_, 1 if not done else 0))

                s = s_
                episode_reward += r
                episode_step += 1

            if Settings.PRE_TRAIN_EPS > 5 and i % (Settings.PRE_TRAIN_EPS // 5) == 0:
                print("Pre-train step n", i)

            # Set the best score to at least the max score the random agent got
            self.best_run = max(self.best_run, episode_reward)

        print("End of the pre training !")

    def save_best(self, episode_reward):
        self.best_run = episode_reward
        print("Save best", episode_reward)
        self.saver.save('best')
        # self.play(1, 'best')

    def run(self):
        """
        Method to run the agent in the environment to collect experiences and
        learn on these experiences by gradient descent.
        """
        print("Beginning of the run...")

        self.pre_train()
        self.network.init_target()

        self.total_steps = 0
        self.nb_ep = 1

        while self.nb_ep < Settings.TRAINING_EPS and not self.gui.STOP:

            s = self.env.reset()
            episode_reward = 0
            done = False

            episode_step = 1
            # The more episodes the agent performs, the longer they are
            max_step = Settings.MAX_EPISODE_STEPS
            if Settings.EP_ELONGATION > 0:
                max_step += self.nb_ep // Settings.EP_ELONGATION

            # Initialize exploration noise process
            noise_process = np.zeros(Settings.ACTION_SIZE)
            noise_scale = (Settings.NOISE_SCALE_INIT *
                           Settings.NOISE_DECAY**self.nb_ep) * \
                (Settings.HIGH_BOUND - Settings.LOW_BOUND)

            # Render settings
            self.env.set_render(self.gui.render.get(self.nb_ep))
            self.env.set_gif(self.gui.gif.get(self.nb_ep))

            while episode_step <= max_step and not done:

                # Choose action based on deterministic policy
                a = self.network.act(s)

                # Add temporally-correlated exploration noise to action
                noise_process = Settings.EXPLO_THETA * \
                    (Settings.EXPLO_MU - noise_process) + \
                    Settings.EXPLO_SIGMA * np.random.randn(Settings.ACTION_SIZE)

                a += noise_scale * noise_process
                s_, r, done, info = self.env.act(a)
                episode_reward += r

                self.buffer.add((s, a, r, s_, 1 if not done else 0))

                if self.total_steps % Settings.TRAINING_FREQ == 0:
                    batch = self.buffer.sample()
                    self.network.train(np.asarray(batch))
                    self.network.update_target()

                s = s_
                episode_step += 1
                self.total_steps += 1

            self.displayer.add_reward(episode_reward, plot=self.gui.plot.get(self.nb_ep))
            # if episode_reward > self.best_run:
            #     self.save_best(episode_reward)

            # Episode display
            if self.gui.ep_reward.get(self.nb_ep):
                print('Episode %2i, Reward: %7.3f, Steps: %i, Final noise scale: %7.3f' %
                      (self.nb_ep, episode_reward, episode_step, noise_scale))

            # Write the summary
            feed_dict = {self.ep_reward_ph: episode_reward,
                         self.noise_ph: noise_scale[0],
                         self.steps_ph: episode_step}
            summary = self.sess.run(self.ep_summary, feed_dict=feed_dict)
            self.writer.add_summary(summary, self.nb_ep)

            # Save the model
            if self.gui.save.get(self.nb_ep):
                self.saver.save(self.nb_ep)

            self.nb_ep += 1

        self.env.close()

    def play(self, number_run, name=None):
        """
        Method to evaluate the policy without exploration.

        Args:
            number_run: the number of episodes to perform
            name      : the name of the gif that will be saved
        """
        print("Playing for", number_run, "runs")

        self.env.set_render(Settings.DISPLAY)
        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False
                self.env.set_gif(True, name)

                while not done:
                    a = self.network.act(s)
                    s, r, done, info = self.env.act(a)

                    episode_reward += r

                print("Episode reward :", episode_reward)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            print("End of the demo")

    def stop(self):
        self.env.close()
