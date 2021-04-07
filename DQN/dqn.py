
from ple.games.pixelcopter import Pixelcopter
from ple import PLE
import numpy as np
import random
import pygame
import json
from skimage import transform, exposure
from collections import deque
import tensorflow as tf

# tf.compat.v1.disable_v2_behavior()

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

OBSERVATION = 3200  # timesteps to observe before training
INITIAL_EPSILON = 0.1  # starting value of epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
num_actions = 2
FRAME_PER_ACTION = 1
EXPLORE = 3000000.  # frames over which to reduce epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 32  # size of minibatch
GAMMA = 0.99  # decay rate of past observations
img_rows, img_cols = 80, 80
img_channels = 4  # We stack 4 frames


class Bot():
    """
            This is our Test agent. It's gonna pick some actions after training!
    """

    def __init__(self, lr):

        self.lr = lr
        self.game = Pixelcopter(width=480, height=480)
        self.p = PLE(self.game, fps=60, display_screen=True)
        self.actions = self.p.getActionSet()

    # def pickAction(self, reward, obs):
    #   return random.choice(self.actions)

    def frame_step(self, act_inp):
        terminal = False
        reward = self.p.act(act_inp)
        if self.p.game_over():
            self.p.reset_game()
            terminal = True
            reward = -1
        else:
            reward = 1

        self.score = self.p.score()
        img = self.p.getScreenGrayscale()
        img = transform.resize(img, (80, 80))
        img = exposure.rescale_intensity(img, out_range=(0, 255))
        img = img / 255.0

        return img, reward, terminal

    def build_model(self):
        print("Building the model..")
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same',
                                input_shape=(img_rows, img_cols, img_channels)))  # 80*80*4
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(2))

        adam = Adam(lr=self.lr)
        model.compile(loss='mse', optimizer=adam)
        self.model = model
        print("Finished building the model..")

    def trainNetwork(self, mode):
        D = deque()

        x_t, r_0, terminal = self.frame_step(self.actions[1])

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        # print (s_t.shape)

        # need to reshape for keras
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*80*80*4

        if mode == 'Run':
            OBSERVE = 999999999  # We keep observe, never train
            epsilon = FINAL_EPSILON
            print("Now we load weight")
            self.model.load_weights("model.h5")
            adam = Adam(lr=self.lr)
            self.model.compile(loss='mse', optimizer=adam)
            print("Weight load successfully")
        else:  # We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        t = 0
        while (True):
            loss = 0
            Q_sa = 0
            action_index = 0
            r_t = 0
            # choose an action epsilon greedy
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
                    action_index = random.randrange(num_actions)
                    chosen_act = self.actions[action_index]
                else:
                    q = self.model.predict(s_t)  # input a stack of 4 images, get the prediction
                    max_Q = np.argmax(q)
                    action_index = max_Q
                    chosen_act = self.actions[action_index]

            # We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observed next state and reward
            x_t1, r_t, terminal = self.frame_step(chosen_act)

            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            # store the transition in D
            D.append((s_t, action_index, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # Now we do the experience replay
                state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                targets = self.model.predict(state_t)
                Q_sa = self.model.predict(state_t1)
                targets[range(BATCH), action_t] = reward_t + GAMMA * np.max(Q_sa, axis=1) * np.invert(terminal)

                loss += self.model.train_on_batch(state_t, targets)

            s_t = s_t1
            t = t + 1

            # save progress every 10000 iterations
            if t % 1000 == 0:
                print("Now we save model")
                self.model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(self.model.to_json(), outfile)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, \
                  "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
                  "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

        print("Episode finished!")
        print("************************")

    def playGame(self, mode):
        self.build_model()
        self.trainNetwork(mode)

    def main(self):
        modes = ["Train", "Run"]
        enter = int(input("Do you wanna Train(0) or Run(1): "))
        print(enter)
        print(type(enter))
        mode = modes[enter]
        self.playGame(mode)


lr = 0.01

if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    # from keras import backend as K
    from tensorflow.python.keras import backend as K

    K.set_session(sess)

    b1 = Bot(lr)
    b1.main()
    pygame.quit()
