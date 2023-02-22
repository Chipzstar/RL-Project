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


import os
import platform
import random
import pickle
import sys
import time
from collections import deque
from tqdm import tqdm

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,5"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['MPLCONFIGDIR'] = "./graphs"

import matplotlib
matplotlib.use("Agg")
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

GPUS = ["/GPU:0", "/GPU:5"]
strategy = tf.distribute.MirroredStrategy(devices=GPUS)
WIDTH = 250
HEIGHT = 250

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print()
print(f"Python {sys.version} {platform.system()}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("GPU is", "available" if gpu else "NOT AVAILABLE")


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

    def __init__(self, mode="train", nodes=32, num_episodes=1000, memory_size=500, final_act="linear", minibatch=32,
                 lr=1e-3):
        # depending on what mode the agent is in, will determine how the agent chooses actions
        # if agent is training, EPSILON = 1 and will decay over time with epsilon probability of exploring
        # if agent is playing (using trained model), EPSILON = 0 and only choose actions based on Q network
        self.DECAY_RATE = 5 / num_episodes
        self.EPSILON = 1 if mode == "train" else 0
        self.HIDDEN_NODES = nodes
        self.MEMORY_SIZE = memory_size
        self.FINAL_ACTIVATION = final_act
        self.MINIBATCH_SIZE = minibatch
        self.GLOBAL_BATCH_SIZE = minibatch * strategy.num_replicas_in_sync  # TODO
        self.LEARNING_RATE = lr
        self.MODEL_NAME = f"model - ({lr} {minibatch} {memory_size} {nodes} {final_act} {num_episodes})"

        with strategy.scope():
            self.model = self.create_model()
            print("Finished building baseline model..")
            # Target model this is what we .predict against every step
            self.target_model = self.create_model()
            print("Finished building target model..")
            self.target_model.set_weights(self.model.get_weights())
        print(self.model.summary())
        self.action_map = {
            0: None,
            1: 119
        }
        self.replay_memory = deque(maxlen=self.MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0
        self.rewards = []

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                14,
                input_shape=(self.INPUT_SIZE,),
                activation="relu",
            ),
            tf.keras.layers.Dense(self.HIDDEN_NODES, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(self.OUTPUT_SIZE, activation=self.FINAL_ACTIVATION)
            # OUTPUT_SIZE = how many actions (2)
        ])

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=Adam(lr=self.LEARNING_RATE),
            metrics=['mae']
        )
        return model

    def update_replay_memory(self, state, action, reward, new_state, done):
        self.replay_memory.append((state, action, reward, new_state, done))
        if len(self.replay_memory) > self.MEMORY_SIZE:
            self.replay_memory.popleft()

    def select_action(self, state):
        # chose random action with probability epsilon
        if np.random.uniform() < self.EPSILON:
            # to speed up training give higher probability to action 0 (no jump)
            action_index = np.random.choice([0, 1], size=1, p=[0.8, 0.2])[0]
            # action_index = np.random.randint(self.OUTPUT_SIZE)
        else:
            # otherwise chose epsilon-greedy action from neural net
            action_index = self.get_predicted_action([state])
        actual_action = self.action_map[action_index]
        return action_index, actual_action

    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.array(sequence))[0]
        return np.argmax(prediction)

    def construct_memories_DoubleDQN(self):
        # Step 1: obtain random minibatch from replay memory
        replay = random.sample(self.replay_memory, self.GLOBAL_BATCH_SIZE)
        # Get current states from minibatch, then query NN model for Q values
        states = np.array([step[0] for step in replay])
        # Get future states from minibatch, then query NN model for Q values
        next_states = np.array([step[3] for step in replay])
        next_action = np.argmax(next_states)
        # selection of action is from model
        # update is from target model
        Q_values = self.model.predict(states)
        Q_next = self.model.predict(next_states)
        Q_target = self.target_model.predict(next_states)

        X = []
        Y = []

        for index, (state, action, reward, state_, done) in enumerate(replay):
            if not done:
                max_a = np.argmax(Q_next[index])
                new_Q = reward + self.GAMMA * Q_target[index][max_a]
            else:
                new_Q = reward

            # Update the Q value for given state
            target = Q_values[index]
            target[action] = new_Q

            # Append new values to training data
            X.append(state)
            Y.append(target)
        return np.array(X), np.array(Y)

    def train(self, is_terminal, step):
        if not os.path.isdir('models'):
            os.makedirs('models')

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_MEMORY_SIZE:
            return

        # constructs training data for training of the neural network
        X, y = self.construct_memories_DoubleDQN()

        train_data = tf.data.Dataset.from_tensor_slices((X, y))
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        train_data = train_data.batch(self.GLOBAL_BATCH_SIZE)

        if is_terminal:
            self.model.fit(
                train_data,
                verbose=1,
                shuffle=False
            )
            self.target_update_counter += 1

        # If counter reaches a set value, update the target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_LIMIT:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


def learn(nodes=49, num_episodes=10_000, memory_size=10_000, final_act="linear", minibatch=32, lr=1e-3):
    game = Pixelcopter(width=WIDTH, height=HEIGHT)
    env = PLE(game, fps=30, display_screen=False, force_fps=True)
    env.init()
    episode_rewards = []
    agent = DDQNAgent("train", nodes, num_episodes, memory_size, final_act, minibatch, lr)
    interval = 50
    for episode in tqdm(range(1, num_episodes + 1)):
        print("Episode :", episode)
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
                print("Total Reward: ", total_reward)
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
        if len(episode_rewards) > 0 and total_reward > np.max(episode_rewards):
            can_update = True
        else:
            can_update = False
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
            if not os.path.isdir(os.path.join(os.getcwd(), f"models/{agent.MODEL_NAME}")):
                os.mkdir(os.path.join(os.getcwd(), f"models/{agent.MODEL_NAME}"))
            agent.model.save(
                f'models/{agent.MODEL_NAME}/{agent.MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_model.h5')
        # Decay epsilon
        if agent.EPSILON > agent.MIN_EPSILON:
            agent.EPSILON -= agent.DECAY_RATE
            print("EPSILON", agent.EPSILON)
            # ensure epsilon does not subside below minimum value
            agent.EPSILON = max(agent.MIN_EPSILON, agent.EPSILON)
        env.reset_game()
    plot_graph(episode_rewards, num_episodes, nodes, memory_size, final_act, minibatch, lr)


def plot_graph(episode_rewards, num_episodes, nodes, memory_size, final_act, minibatch, lr):
    fig, ax = plt.subplots(nrows=1, figsize=(12, 15))
    episodes = np.arange(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards)
    ax.set_title(f"DDQN Agent Learning Curve - ({num_episodes} {nodes} {memory_size} {minibatch} {final_act} {lr})")
    ax.set_xlabel("Number of episodes")
    ax.set_ylabel("Total Reward")
    plt.savefig(
        f"graphs/Agent learning curve - num_episodes={num_episodes} hidden_nodes={nodes} mem_size={memory_size} final_act={final_act} minibatch={minibatch} lr={lr}.png")
    return True


if __name__ == "__main__":
    learn(num_episodes=10_000, memory_size=10_000, nodes=49, minibatch=64, lr=1e-2)
