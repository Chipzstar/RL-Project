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


import datetime
import os
import platform
import random
import sys
import time
from collections import deque
from tqdm import tqdm
import pickle

import matplotlib
import keras
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf
from keras import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras import initializers
from ple import PLE
from ple.games.pixelcopter import Pixelcopter
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version} {platform.system()}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")


# Sets the initial window position of the game
# WIDTH = GetSystemMetrics(0)
# HEIGHT = GetSystemMetrics(1)
# x = WIDTH - 500
# y = 200
# os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x, y)


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.log_write_dir = self.log_dir
        self._train_dir = os.path.join(self.log_dir + 'train')
        self._val_dir = os.path.join(self.log_dir, 'validation')
        self._should_write_train_graph = False

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model
        self._train_step = self.model._train_counter
        self._val_step = self.model._test_counter

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
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
                self.writer.flush()


class PrioritizedReplayBuffer:
    def __init__(self, maxlen=5000):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance_weights(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=0.1):
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=batch_size, weights=sample_probs)
        samples = np.array(self.buffer, dtype=object)[sample_indices]
        importance_weights = self.get_importance_weights(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance_weights, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


class DDQNAgent:
    # hyper-parameters
    INPUT_SIZE = 7
    OUTPUT_SIZE = 2
    EPSILON = 1
    MIN_EPSILON = 0.05
    GAMMA = 0.99
    MIN_MEMORY_SIZE = 1000
    UPDATE_TARGET_LIMIT = 3

    # based on documentation, state has 7 features
    # output is 2 dimensions, 0 = do nothing, 1 = jump

    def __init__(self, mode="train", nodes=32, memory_size=10000, final_act="linear", minibatch=32, lr=1e-3,
                 num_episodes=1000):
        # depending on what mode the agent is in, will determine how the agent chooses actions
        # if agent is training, EPSILON = 1 and will decay over time with epsilon probability of exploring
        # if agent is playing (using trained model), EPSILON = 0 and only choose actions based on Q network
        self.DECAY_RATE = 5 / num_episodes
        self.HIDDEN_NODES = nodes
        self.MEMORY_SIZE = memory_size
        self.FINAL_ACTIVATION = final_act
        self.MINIBATCH_SIZE = minibatch
        self.LEARNING_RATE = lr
        self.MODEL_NAME = f"model - ({lr} {minibatch} {memory_size} {nodes} {final_act} {num_episodes})"
        self.LOAD_MODEL = "per/best/model - (0.01 128 10000 49 linear 5000)____91.97max___30.01 avg___-1.67min.h5"
        # Set to LOAD_MODEL to NONE to train from scratch

        self.model = self.create_model(self.LOAD_MODEL)
        print(self.model.summary())
        self.action_map = {
            0: None,
            1: 119
        }
        # Target model this is what we .predict against every step
        self.target_model = self.create_model(self.LOAD_MODEL)
        print("Finished building target model..")
        self.target_model.set_weights(self.model.get_weights())

        self.replay_buffer = PrioritizedReplayBuffer(maxlen=memory_size)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.errors = []

    def create_model(self, model_file):
        if model_file:
            print("Loading model...")
            model = keras.models.load_model(model_file)
            self.EPSILON = 0
        else:
            model = Sequential()

            model.add(Dense(
                14,
                input_shape=(self.INPUT_SIZE,),
                activation="relu"
            ))

            model.add(Dense(self.HIDDEN_NODES, activation="relu"))
            model.add(Dropout(0.1))

            model.add(Dense(self.OUTPUT_SIZE, activation=self.FINAL_ACTIVATION))  # OUTPUT_SIZE = how many actions (2)
            model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE), metrics=['mae'])
            print("Finished building baseline model..")
        return model

    def update_replay_memory(self, state, action, reward, new_state, done):
        self.replay_buffer.add((state, action, reward, new_state, done))

    def select_action(self, state):
        # chose random action with probability epsilon
        if np.random.uniform() < self.EPSILON:
            # to speed up training give higher probability to action 0 (no jump)
            action_index = np.random.choice([0, 1], size=1, p=[0.75, 0.25])[0]
            # action_index = np.random.randint(self.OUTPUT_SIZE)
        # otherwise chose epsilon-greedy action from neural net
        else:
            action_index = self.get_predicted_action([state])
        actual_action = self.action_map[action_index]
        return action_index, actual_action

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))[0]
        # print("Prediction", prediction)
        return np.argmax(prediction)

    def get_qs(self, state):
        return self.model.predict(np.array(state))

    def construct_memories_double_dqn(self, states, actions, rewards, next_states, dones):
        # selection of action is from model
        # update is from target
        Q_values = self.model.predict(np.array(states))
        Q_next = self.model.predict(np.array(next_states))
        Q_target = self.target_model.predict(np.array(next_states))

        X = []
        Y = []

        for i in range(self.MINIBATCH_SIZE):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not dones[i]:
                max_a = np.argmax(Q_next[i])
                new_Q = rewards[i] + self.GAMMA * Q_target[i][max_a]
            else:
                new_Q = rewards[i]

            # Update the Q value for given state
            target = Q_values[i]
            target[actions[i]] = new_Q

            # Append new values to training data
            X.append(states[i])
            Y.append(target)
        return np.array(X), np.array(Y)

    def calculate_errors(self, transitions):
        # print(transitions[0])
        errors = []
        for s, a, r, s_, done in transitions:
            # Q value of sample
            state = np.array([s])
            Q = self.model.predict(state)[0][a]
            # T(S) = target value of sample
            next_state = np.array([s_])
            next_Q = self.target_model.predict(next_state)[0][a]
            T = r + self.GAMMA * next_Q
            errors.append(abs(Q - T))
        return np.array(errors)

    def train(self, is_terminal, step):
        if not os.path.isdir('per'):
            os.makedirs('per')
        # Start training only if certain number of samples is already saved

        if len(self.replay_buffer) < self.MIN_MEMORY_SIZE:
            return

        # Step 1: obtain random minibatch from replay memory
        (states, actions, rewards, next_states, dones), importance_weights, indices = self.replay_buffer.sample(
            self.MINIBATCH_SIZE)
        # constructs training data for training of the neural network
        X, y, = self.construct_memories_double_dqn(states, actions, rewards, next_states, dones)

        if is_terminal:
            self.model.fit(
                X,
                y,
                sample_weight=importance_weights,
                batch_size=self.MINIBATCH_SIZE,
                verbose=1,
                shuffle=False,
                callbacks=[self.tensorboard]
            )
            self.target_update_counter += 1
            errors = self.calculate_errors(list(zip(states, actions, rewards, next_states, dones)))
            self.replay_buffer.set_priorities(indices, errors)

        # If counter reaches a set value, update the target network with weights of main network
        if self.target_update_counter >= self.UPDATE_TARGET_LIMIT:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def main(num_episodes=2000, nodes=32, memory_size=10_000, final_act="linear", minibatch=32, lr=1e-3):
    print(f"LEARNING PARAMS: nodes={nodes} num_episodes={num_episodes} memory_size={memory_size} " \
          f"final_act={final_act} batch={minibatch} lr={lr}")
    game = Pixelcopter(width=250, height=250)
    env = PLE(game, fps=30, display_screen=True, force_fps=True)
    env.init()
    episode_rewards = []
    agent = DDQNAgent("train", nodes, memory_size, final_act, minibatch, lr, num_episodes)
    interval = 50
    print("State attributes", env.getGameState().keys())
    print("All actions", env.getActionSet())
    # Save model, but only when min reward is greater or equal a set value
    if not os.path.isdir(os.path.join(os.getcwd(), f"ddqn/{agent.MODEL_NAME}")):
        os.mkdir(os.path.join(os.getcwd(), f"ddqn/{agent.MODEL_NAME}"))
    for episode in range(1, num_episodes + 1):
        print("Episode : ", episode)
        agent.tensorboard.step = episode
        done = False
        step = 1
        total_reward = 0.0
        # initial state
        state = np.array(list(env.getGameState().values()))
        # print("State:", state)
        while not done:
            if env.game_over():
                print("GAME OVER!")
                print("Total reward:", total_reward)
                done = True

            action_index, action = agent.select_action(state)
            action_string = 'jump!' if action_index == 1 else 'chill'
            # print("Action:", action, action_string)
            reward = env.act(action)
            # print("Reward:", reward)
            new_state = np.array(list(env.getGameState().values()))
            # update total reward
            total_reward += reward
            # update replay memory
            agent.update_replay_memory(state, action_index, reward, new_state, done)
            # update q_network
            agent.train(done, step)
            # update current state with new state
            state = new_state
            # increment time step
            step += 1
        # perform update only if a new max. reward was obtained after an episode
        can_update = total_reward > np.max(episode_rewards) if len(episode_rewards) > 0 else False
        # Append episode rewards to list of all episode rewards
        episode_rewards.append(total_reward)
        if can_update or episode == 1:
            average_reward = np.mean(episode_rewards[-interval:])
            min_reward = np.min(episode_rewards[-interval:])
            max_reward = np.max(episode_rewards[-interval:])
            agent.tensorboard.update_stats(
                reward_avg=average_reward,
                reward_min=min_reward,
                reward_max=max_reward,
                epsilon=agent.EPSILON
            )
            # Save model, but only when min reward is greater or equal a set value
            agent.model.save(
                f'per/{agent.MODEL_NAME}/{agent.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f} avg_{min_reward:_>7.2f}min.h5')
        # Decay epsilon
        if agent.EPSILON > agent.MIN_EPSILON:
            agent.EPSILON -= agent.DECAY_RATE
            print("EPSILON", agent.EPSILON)
            # ensure epsilon does not subside below minimum value
            agent.EPSILON = max(agent.MIN_EPSILON, agent.EPSILON)
        env.reset_game()
    plot_graph(episode_rewards, num_episodes, nodes, memory_size, final_act, minibatch, lr)


def plot_graph(episode_rewards, num_episodes, nodes, memory_size, final_act, minibatch, lr):
    if not os.path.isdir(os.path.join(os.getcwd(), f"per/graphs")):
        os.mkdir(os.path.join(os.getcwd(), f"per/graphs"))
    fig, ax = plt.subplots(nrows=1, figsize=(12, 15))
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards)
    ax.set_title(f"PERDDQN agent Learning Curve - ({num_episodes} {nodes} {memory_size} {minibatch} {final_act} {lr})")
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Total Reward")
    plt.savefig(
        f"per/graphs/PERDDQN Agent learning curve - num_episodes={num_episodes} hidden_nodes={nodes} mem_size={memory_size} final_act={final_act} minibatch={minibatch} lr={lr}.png")
    return True


# run the game using best DQN model
def play():
    game = Pixelcopter(width=250, height=250)
    env = PLE(game, fps=30, display_screen=True)
    env.init()
    agent = DDQNAgent("play")
    step = 0
    total_reward = 0
    state = np.array(list(env.getGameState().values()))
    while True:
        if env.game_over():
            print("===========================")
            print("TOTAL REWARD: ", total_reward)
            step = 0
            total_reward = 0
            env.reset_game()

        action_index, action = agent.select_action(state)
        # action_string = 'jump!' if action_index == 1 else 'chill'
        reward = env.act(action)
        new_state = np.array(list(env.getGameState().values()))
        state = new_state
        step += 1
        total_reward += reward


if __name__ == "__main__":
    # init()
    # pass in specific params to function, otherwise uses default ones
    play()
