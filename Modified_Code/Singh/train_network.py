
import sys
sys.path.append("game/")

import os
os.makedirs('saved_models', exist_ok=True)

import numpy as np

import skimage
from skimage import transform, color, exposure

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Activation, Input
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
import keras.backend as K
from keras.callbacks import LearningRateScheduler, History
import tensorflow as tf

import wrapped_flappy_bird as game

import threading




LOAD = False


GAMMA = 0.99  # discount value
BETA = 0.01  # regularisation coefficient
IMAGE_ROWS = 85
IMAGE_COLS = 84
IMAGE_CHANNELS = 4
LEARNING_RATE = 7e-4
THREADS = 16
MAX_STEP = 5



episode_reward = []
episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
episode_output = []
episode_critic = []


# loss function for policy output
def policy_loss(y_true, y_pred):  # policy loss
    return -K.sum(K.log(y_true * y_pred + (1 - y_true) * (1 - y_pred) + 1e-5), axis=-1)
    # BETA * K.sum(y_pred * K.log(y_pred + const) + (1-y_pred) *
    # K.log(1-y_pred + const))   #regularisation term


# loss function for critic output
def value_loss(y_true, y_pred):  # critic loss
    return K.sum(K.square(y_pred - y_true), axis=-1)


# function buildmodel() to define the structure of the neural network in use
def buildmodel():
    print("Model building begins")

    model = Sequential()
    keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)

    S = Input(shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS, ), name='Input')
    h0 = Convolution2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu',
                       kernel_initializer='random_uniform', bias_initializer='random_uniform')(S)
    h1 = Convolution2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu',
                       kernel_initializer='random_uniform', bias_initializer='random_uniform')(h0)
    h2 = Flatten()(h1)
    h3 = Dense(256, activation='relu', kernel_initializer='random_uniform',
               bias_initializer='random_uniform')(h2)
    P = Dense(1, name='o_P', activation='sigmoid',
              kernel_initializer='random_uniform', bias_initializer='random_uniform')(h3)
    V = Dense(1, name='o_V', kernel_initializer='random_uniform',
              bias_initializer='random_uniform')(h3)

    model = Model(inputs=S, outputs=[P, V])
    rms = RMSprop(lr=LEARNING_RATE, rho=0.99, epsilon=0.1)
    model.compile(loss={'o_P': policy_loss, 'o_V': value_loss},
                  loss_weights={'o_P': 1., 'o_V': 0.5}, optimizer=rms)
    return model


# function to preprocess an image before giving as input to the neural network
def preprocess(image):
    image = skimage.color.rgb2gray(image)
    image = skimage.transform.resize(
        image, (IMAGE_ROWS, IMAGE_COLS), mode='constant')
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
    image = image.reshape(1, image.shape[0], image.shape[1], 1)
    return image

# initialize a new model using buildmodel() or use load_model to resume
# training an already trained model
if LOAD:
    model = load_model("saved_models/model_checkpoint",
                       custom_objects={'policy_loss': policy_loss,
                                       'value_loss': value_loss})
else:
    model = buildmodel()

model._make_predict_function()
graph = tf.get_default_graph()

game_state = []
for i in range(0, THREADS):
    game_state.append(game.GameState(30000))


# function to decrease the learning rate after every epoch. In this
# manner, the learning rate reaches 0, by 20,000 epochs
def step_decay(epoch):
    decay = 3.2e-8
    lrate = LEARNING_RATE - epoch * decay
    return max(lrate, 0)


class actorthread(threading.Thread):

    def __init__(self, thread_id, s_t):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.next_state = s_t

    def run(self):
        global episode_output
        global episode_reward
        global episode_critic
        global episode_state

        threadLock.acquire()
        state_buffer, action_buffer, reward_buffer, value_buffer = self.runprocess()

        episode_reward = np.append(episode_reward, reward_buffer)
        episode_output = np.append(episode_output, action_buffer)
        episode_state = np.append(episode_state, state_buffer, axis=0)
        episode_critic = np.append(episode_critic, value_buffer)

        threadLock.release()

    def runprocess(self):
        global model

        episode_step = 0
        done = False
        reward = 0
        reward_buffer = []
        state_buffer = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
        action_buffer = []
        value_buffer = []
        self.next_state = self.next_state.reshape(1, *self.next_state.shape)

        while episode_step < MAX_STEP and not done:
            episode_step += 1

            with graph.as_default():
                out = model.predict(self.next_state)[0]

            if np.random.rand() < out:
                action = [0, 1]
            else:
                action = [1, 0]

            state, reward, done = game_state[self.thread_id].frame_step(action)
            state = preprocess(state)

            with graph.as_default():
                critic_reward = model.predict(self.next_state)[1]

            y = action[1]

            reward_buffer = np.append(reward_buffer, reward)
            state_buffer = np.append(state_buffer, self.next_state, axis=0)
            action_buffer = np.append(action_buffer, y)
            value_buffer = np.append(value_buffer, critic_reward)

            self.next_state = np.append(state,
                                        self.next_state[:, :, :, :3], axis=3)
            # print("Frame = " + str(T) + ", Updates = " +
            #       str(EPISODE) + ", Thread = " + str(thread_id))

        if not done:
            reward_buffer[-1] = value_buffer[-1]
        else:
            reward_buffer[-1] = -1
            self.next_state = np.concatenate(
                (state, state, state, state), axis=3)

        self.next_state = self.next_state.reshape(self.next_state.shape[1],
                                                  self.next_state.shape[2],
                                                  self.next_state.shape[3])

        for i in range(len(reward_buffer) - 2, -1, -1):
            reward_buffer[i] = reward_buffer[i] + GAMMA * reward_buffer[i + 1]

        return state_buffer, action_buffer, reward_buffer, value_buffer


states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, 4))

# initializing state of each thread
for i in range(0, len(game_state)):
    image = game_state[i].getCurrentFrame()
    image = preprocess(image)
    state = np.concatenate((image, image, image, image), axis=3)
    states = np.append(states, state, axis=0)


EPISODE = 1

try:
    while True:
        threadLock = threading.Lock()
        threads = []
        for i in range(0, THREADS):
            threads.append(actorthread(i, states[i]))

        states = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, 4))

        for i in range(0, THREADS):
            threads[i].start()

        # thread.join() ensures that all threads fininsh execution before
        # proceeding further
        for i in range(0, THREADS):
            threads[i].join()

        for i in range(0, THREADS):
            state = threads[i].next_state
            state = state.reshape(
                1, state.shape[0], state.shape[1], state.shape[2])
            states = np.append(states, state, axis=0)

        # advantage calculation for each action taken
        advantage = episode_reward - episode_critic
        print("backpropagating")

        weights = {'o_P': advantage, 'o_V': np.ones(len(advantage))}
        # backpropagation
        history = model.fit(episode_state, [episode_output, episode_reward],
                            epochs=EPISODE + 1,
                            batch_size=len(episode_output),
                            callbacks=[LearningRateScheduler(step_decay)],
                            sample_weight=weights,
                            initial_epoch=EPISODE)


        mean_reward = np.mean(episode_reward)

        episode_reward = []
        episode_output = []
        episode_state = np.zeros((0, IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
        episode_critic = []

        f = open("rewards.txt", "a")
        f.write("Update: " + str(EPISODE) + ", Reward_mean: " +
                str(mean_reward) + ", Loss: " +
                str(history.history['loss']) + "\n")
        f.close()

        if EPISODE % 50 == 0:
            model.save("saved_models/model_updates" + str(EPISODE))
            print("Episode", EPISODE, " model saved")
        EPISODE += 1

except KeyboardInterrupt:
    model.save("saved_models/model_checkpoint")
    print("Model saved")
