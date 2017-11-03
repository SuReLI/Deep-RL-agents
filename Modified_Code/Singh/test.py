
import sys
sys.path.append("game/")

import wrapped_flappy_bird as game

import skimage
from skimage import transform, color, exposure

import keras.backend as K
from keras.models import load_model
import numpy as np

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

model = load_model("saved_models/model_checkpoint", custom_objects={'logloss': policy_loss, 'sumofsquares': value_loss})
game_state = game.GameState(30)

topScore = 0

try:
    for i in range(number):
        state = preprocess(game_state.getCurrentFrame())
        states = np.concatenate((state, state, state, state), axis=3)

        done = False
        reward = 0
        currentScore = 0

        while not done:

            y = model.predict(states)[0]

            if np.random.rand() < y:
                action = [0, 1]
            else:
                action = [1, 0]
            
            state, reward, done = game_state.frame_step(action)
            state = preprocess(state)
            states = np.append(state, states[:, :, :, :3], axis=3)
            
            if reward == 1:
                currentScore += 1
                topScore = max(topScore, currentScore)
                print("Current Score: " + str(currentScore) + " Top Score: " + str(topScore))

except KeyboardInterrupt:
    pass
