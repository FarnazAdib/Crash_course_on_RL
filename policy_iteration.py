import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
from collections import deque


class Q:
    def __init__(self, hparams, epsilon_min=0.01, epsilon_decay=0.995):
        self.hparams = hparams
        np.random.seed(hparams['Rand_Seed'])
        tf.random.set_seed(hparams['Rand_Seed'])
        random.seed(hparams['Rand_Seed'])
        self.memory = deque(maxlen=100000)
        self.epsilon = hparams['epsilon']
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # The Q network
        self.network = keras.Sequential([
            keras.layers.Dense(self.hparams['hidden_size'], input_dim=self.hparams['num_state'], activation='relu',
                               kernel_initializer=keras.initializers.he_normal(), dtype='float64'),
            keras.layers.Dense(self.hparams['hidden_size'], activation='relu',
                               kernel_initializer=keras.initializers.he_normal(), dtype='float64'),
            keras.layers.Dense(self.hparams['hidden_size'], activation='relu',
                               kernel_initializer=keras.initializers.he_normal(), dtype='float64'),
            keras.layers.Dense(self.hparams['num_actions'], dtype='float64')
        ])

        # The cost function for the Q network
        self.network.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(
            epsilon=self.hparams['adam_eps'], learning_rate=self.hparams['learning_rate_adam']))

    def get_action(self, state, env):
        state = self._process_state(state)

        # Exploration
        if np.random.random() <= self.epsilon:
            selected_action = env.action_space.sample()

        # Exploitation
        else:
            selected_action = np.argmax(self.network(state))
        return selected_action

    def remember(self, state, action, reward, next_state, done):
        # Adding data to the history batch
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # Sampling the history batch
        batch = random.sample(self.memory, min(len(self.memory), batch_size))
        states, actions, rewards, new_states, dones = list(map(lambda i: [j[i] for j in batch], range(5)))
        loss = self.update_network(states, actions, rewards, new_states, dones)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def update_network(self, states, actions, rewards, next_states, dones):
        eps_length = len(states)
        states = np.vstack(states)
        q_target = self.network(states).numpy()
        for i in range(eps_length):
            if dones[i]:
                q_target[i, actions[i]] = rewards[i]
            else:
                next_state = self._process_state(next_states[i])
                q_target[i, actions[i]] = rewards[i] + \
                                        self.hparams['GAMMA'] * tf.math.reduce_max(self.network(next_state)).numpy()
        loss = self.network.train_on_batch(states, q_target)
        return loss

    def _process_state(self, state):
        return state.reshape([1, self.hparams['num_state']])

