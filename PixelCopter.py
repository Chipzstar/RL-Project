#!/usr/bin/env python
# coding: utf-8

# ## **SET UP PYGAME ENVIRONMENT**

# In[ ]:


# ONLY NEED TO RUN ONCE
# import os
# get_ipython().system('git clone https://github.com/ntasfi/PyGame-Learning-Environment/')
# os.chdir("PyGame-Learning-Environment")
# print(f"Current directory {os.getcwd()}")
# get_ipython().system('pip install -e .')
# get_ipython().system('pip install pygame')
# get_ipython().system('pip install -q tensorflow')
# get_ipython().system('pip install -q keras')


# # Imports

import os
import pickle
import platform
import sys
import time
from collections import deque
from pprint import pprint

import keras
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import MaxPooling2D, Dense, Dropout, Activation
from keras.optimizers import Adam
from ple import PLE
from ple.games.pixelcopter import Pixelcopter

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version} {platform.system()}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")

# Sets the initial window position of the game
x = 100
y = 0
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)


# # Create Game

class NaiveAgent:
    def __init__(self, allowed_actions=[]):
        self.moves = ["up", "fall"]
        self.actions = allowed_actions
        self.epsilon = 0.5
        print(self.actions)

    def pick_action(self):
        action_index = np.random.randint(len(self.actions))
        return self.actions[action_index], self.moves[action_index]


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrides, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrides
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrides, so won't close writer
    def on_train_end(self, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 32
    INPUT_SIZE = 7
    LAYER_SIZE = 500
    OUTPUT_SIZE = 2
    EPSILON = 0.15
    DECAY_RATE = 0.005
    MIN_EPSILON = 0.1
    GAMMA = 0.99
    MEMORIES = deque()
    MEMORY_SIZE = 300
    MODEL_NAME = f"DQN model LR={LEARNING_RATE} BATCH={BATCH_SIZE}"

    # based on documentation, state has 7 features
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, screen=False, forcefps=True):
        self.game = Pixelcopter(width=250, height=250)
        self.env = PLE(self.game, fps=30, display_screen=screen, force_fps=forcefps)
        self.env.init()
        # main model  # gets trained every step
        self.model = self.create_model()
        self.action_map = {
            0: None,
            1: 119
        }
        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.rewards = []

    def create_model(self):
        model = Sequential()

        model.add(Dense(32, input_shape=(self.INPUT_SIZE, ), activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(self.OUTPUT_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE), metrics=['accuracy'])
        return model

    def update_replay_memory(self, state, action, reward, new_state, done):
        self.MEMORIES.append((state, action, reward, new_state, done))
        if len(self.MEMORIES) > self.MEMORY_SIZE:
            self.MEMORIES.popleft()

    def select_action(self, state):
        # chose random action with probability epsilon
        if np.random.uniform() < self.EPSILON:
            action = np.random.randint(self.OUTPUT_SIZE)
        # otherwise chose epsilon-greedy action from neural net
        else:
            action = self.get_predicted_action([state])
        actual_action = self.action_map[action]
        return actual_action

    def get_qs(self, state, step):
        return self.model.predict(np.array(state))

    def _construct_memories(self, replay):
        states = np.array([a[0] for a in replay])
        new_states = np.array([a[3] for a in replay])
        Q = self.model.predict(states)
        Q_new = self.model.predict(new_states)
        replay_size = len(replay)
        X = np.empty((replay_size, self.INPUT_SIZE))
        Y = np.empty((replay_size, self.OUTPUT_SIZE))
        for i in range(replay_size):
            state_r, action_r, reward_r, new_state_r, done_r = replay[i]
            target = Q[i]
            target[action_r] = reward_r
            if not done_r:
                target[action_r] += self.GAMMA * np.amax(Q_new[i])
            X[i] = state_r
            Y[i] = target
        return X, Y

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))[0]
        print("Prediction", prediction)
        return np.argmax(prediction)

    def get_state(self):
        state = self.env.getGameState()
        return np.array(list(state.values()))


# init
def main():
    agent = DQNAgent()
    agent.env.reset_game()
    print("State attributes", agent.env.getGameState().keys())
    print("All actions", agent.env.getActionSet())
    done = False
    total_reward = 0.0
    while not done:
        if agent.env.game_over():
            print("GAME OVER!")
            done = True
        state = agent.get_state()
        print("State:", state)
        action = agent.select_action(state)
        print("Action:", action)
        reward = agent.env.act(action)
        print("Reward:", reward)
        total_reward += reward
        # prevent infinite loop
        done = True


if __name__ == "__main__":
    main()
