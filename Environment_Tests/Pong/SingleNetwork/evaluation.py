
import sys
sys.path.append("game/")

import skimage
from skimage import transform, color, exposure

import keras.backend as K
from keras.models import load_model
import numpy as np
import gym

try:
    number = int(sys.argv[1])
except:
    number = 1


def policy_loss(y_true, y_pred):
    return -K.sum( K.log(y_true*y_pred + (1-y_true)*(1-y_pred) + 1e-5), axis=-1) 

def value_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

def preprocess(image):
    image = skimage.color.rgb2gray(image)
    image = skimage.transform.resize(image, (85,84), mode = 'constant')
    image = skimage.exposure.rescale_intensity(image, out_range=(0,255))
    image = image.reshape(1, image.shape[0], image.shape[1], 1)
    return image

model = load_model("saved_models/model_checkpoint", custom_objects={'policy_loss': policy_loss, 'value_loss': value_loss})
game_state = gym.make("PongDeterministic-v4")

topScore = 0

try:
    for i in range(number):
        state = preprocess(game_state.reset())
        states = np.concatenate((state, state, state, state), axis=3)

        done = False
        reward = 0
        currentScore = 0

        while not done:

            out = model.predict(states)[0]
            action = np.argmax(out)
            
            state, reward, done, _ = game_state.step(action)
            state = preprocess(state)
            states = np.append(state, states[:, :, :, :3], axis=3)

            game_state.render()

except KeyboardInterrupt:
    pass
