
import tensorflow as tf
import numpy as np

from QNetwork import Network
from ExperienceBuffer import ExperienceBuffer
from Environment import Environment


class Agent:

    def __init__(self, settings, sess, gui, displayer, saver):
        print("Initializing the agent...")

        self.settings = settings
        self.sess = sess
        self.gui = gui
        self.displayer = displayer
        self.saver = saver

        self.env = Environment(settings)
        self.network = Network(settings, sess)
        self.buffer = ExperienceBuffer(settings)

        self.best_run = -1e10
        self.n_gif = 0

    def run(self):

        self.network.init_target_update()
        self.total_steps = 0
        self.nb_ep = 1

        while self.nb_ep < self.settings.TRAINING_STEPS and not self.gui.STOP:

            s = self.env.reset()
            episode_reward = 0
            episode_step = 0
            done = False

            # Initialize exploration noise process
            noise_process = np.zeros(self.settings.ACTION_SIZE)
            noise_scale = (self.settings.NOISE_SCALE_INIT *
                           self.settings.NOISE_DECAY**self.nb_ep) * \
                (self.settings.HIGH_BOUND - self.settings.LOW_BOUND)

            # Render settings
            self.env.set_render(self.gui.render.get(self.nb_ep))
            self.env.set_gif(self.gui.gif.get(self.nb_ep))

            while episode_step < self.settings.MAX_EPISODE_STEPS and not done:

                # choose action based on deterministic policy
                a, = self.sess.run(self.network.actions,
                                   feed_dict={self.network.state_ph: s[None]})

                # add temporally-correlated exploration noise to action
                noise_process = self.settings.EXPLO_THETA * \
                    (self.settings.EXPLO_MU - noise_process) + \
                    self.settings.EXPLO_SIGMA * np.random.randn(self.settings.ACTION_SIZE)

                a += noise_scale * noise_process

                s_, r, done, info = self.env.act(a)
                episode_reward += r

                self.buffer.add((s, a, r, s_, 1 if not done else 0))

                # update network weights to fit a minibatch of experience
                if self.total_steps % self.settings.TRAINING_FREQ == 0:
                    batch = self.buffer.sample()
                    self.network.train(np.asarray(batch))
                    self.network.target_update()

                s = s_
                episode_step += 1
                self.total_steps += 1


            self.displayer.add_reward(episode_reward, self.gui.plot.get(self.nb_ep))
#            if self.nb_ep > 50 and episode_reward > self.best_run:
#                print("Saving best")
#                self.play(1, 'results/gif/gif_best')
#                self.best_run = episode_reward

            # Episode display
            if self.gui.ep_reward.get(self.nb_ep):
                print('Episode %2i, Reward: %7.3f, Steps: %i, Final noise scale: %7.3f' %
                      (self.nb_ep, episode_reward, episode_step, noise_scale))

            # Periodically save the model
            if self.gui.save.get(self.nb_ep):
                self.saver.save(self.nb_ep)

            self.nb_ep += 1

    def play(self, number_run, name=None):
        print("Playing for", number_run, "runs")

        self.env.set_render(self.settings.DISPLAY)
        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False
                self.env.set_gif(True, name)

                while not done:
                    a, = self.sess.run(self.network.actions,
                                       feed_dict={self.network.state_ph: [s]})
                    s, r, done, info = self.env.act(a)

                    episode_reward += r

                print("Episode reward :", episode_reward)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            print("End of the demo")

    def close(self):
        self.env.close()
