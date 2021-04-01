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

import sys
import os
import time
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
import keras
import platform
from pprint import pprint
from ple.games.pixelcopter import Pixelcopter
from ple import PLE

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

    def go_up(self):
        action = self.actions[0]
        return action


# init
def main():
    game = Pixelcopter(width=300, height=300)
    env = PLE(game, fps=30, display_screen=True)
    env.init()
    agent = NaiveAgent(allowed_actions=env.getActionSet())
    print("All actions", env.getActionSet())
    print("Start Lives:", env.lives())
    reward = 0.0
    nb_frames = 100
    start_time = time.time()

    for i in range(nb_frames):
        print("Frame:", env.getFrameNumber())
        if env.game_over():
            print("GAME OVER!")
            break
        action, move = agent.pick_action()
        print("New move:", move)
        reward = env.act(action)
        env.saveScreen(f"screenshots/Frame {i + 1}.png")
        state = env.getGameState()
        print("State:")
        pprint(state)
        print("Reward:", reward)

    screenState = env.getScreenRGB()
    print(screenState.shape)


def testing():
    game = Pixelcopter()
    env = PLE(game, fps=30, display_screen=True)
    env.init()
    agent = NaiveAgent(allowed_actions=env.getActionSet())
    print("All actions", env.getActionSet())
    print("Start Lives:", env.lives())
    total_reward = 0
    start_time = time.time()

    while not env.game_over():
        duration = int(time.time() - start_time)
        if duration >= 10:
            break
        # chosen action by agent
        action = agent.go_up()
        print(action)
        # agent makes move in the game
        reward = env.act(action)
        total_reward += reward
        print(env.getFrameNumber(), action, reward)

    print("GAME OVER!")
    print("Total Reward", total_reward)


if __name__ == "__main__":
    main()
