
import tensorflow as tf
import numpy as np

from Model import build_actor, get_vars
from ExperienceBuffer import BUFFER
from Environment import Environment

from Displayer import DISPLAYER
import settings


STOP_REQUESTED = False


def request_stop(signal, frame):
    global STOP_REQUESTED
    print('End of training')
    STOP_REQUESTED = True


class Actor:

    def __init__(self, sess, n_actor):
        print("Initializing actor %i..." % n_actor)

        self.n_actor = n_actor
        self.sess = sess
        self.env = Environment()
        self.state_size = self.env.get_state_size()[0]
        self.action_size = self.env.get_action_size()
        self.bounds = self.env.get_bounds()

        self.build_actor()

    def get_env_features(self):
        return self.state_size, self.action_size, self.bounds

    def build_actor(self):

        scope = 'worker_actor_' + str(self.n_actor)
        self.state_ph = tf.placeholder(dtype=tf.float32,
                                       shape=[None, self.state_size],
                                       name='state_ph')

        # Get the policy prediction network
        self.policy = build_actor(self.state_ph, self.bounds, self.action_size,
                                  trainable=False, scope=scope)

        self.vars = get_vars(scope, trainable=False)

    def predict_action(self, s):
        return self.sess.run(self.policy, feed_dict={self.state_ph: s[None]})[0]

    def run(self):

        total_eps = 1
        while not STOP_REQUESTED:

            episode_reward = 0
            episode_step = 0
            done = False

            noise_scale = settings.NOISE_SCALE * settings.NOISE_DECAY**total_eps

            s = self.env.reset()

            render = (self.n_actor == 1 and settings.DISPLAY and
                      total_eps % settings.RENDER_FREQ == 0)
            self.env.set_render(render)

            while not done and not STOP_REQUESTED:

                noise = np.random.normal(size=self.action_size)
                scaled_noise = noise_scale * noise

                a = np.clip(self.predict_action(s) + scaled_noise, *self.bounds)

                s_, r, done, _ = self.env.act(a)

                episode_reward += r

                BUFFER.add(s, a, r, s_, 0 if done else 1)

                s = s_
                episode_step += 1

            if self.n_actor == 1 and total_eps % 10 == 0:
            	BUFFER.stats()
            
            if self.n_actor == 1 and total_eps % settings.EP_REW_FREQ == 0:
                print("Episode %i : reward %i, steps %i, noise scale %f" % (total_eps, episode_reward, episode_step, noise_scale))

            if not STOP_REQUESTED:
                DISPLAYER.add_reward(episode_reward, self.n_actor)
            
            total_eps += 1

        self.env.close()
