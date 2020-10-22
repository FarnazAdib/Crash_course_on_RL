import numpy as np
import tensorflow as tf
from tensorflow import keras


class PG:
    def __init__(self, hparams):
        self.hparams = hparams
        np.random.seed(hparams['Rand_Seed'])
        tf.random.set_seed(hparams['Rand_Seed'])

        # The policy network
        self.network = keras.Sequential([
            keras.layers.Dense(self.hparams['hidden_size'], input_dim=self.hparams['num_state'], activation='relu',
                               kernel_initializer=keras.initializers.he_normal(), dtype='float64'),
            keras.layers.Dense(self.hparams['hidden_size'], activation='relu',
                               kernel_initializer=keras.initializers.he_normal(), dtype='float64'),
            keras.layers.Dense(self.hparams['num_actions'], activation='softmax', dtype='float64')
        ])
        self.network.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(
            epsilon=self.hparams['adam_eps'], learning_rate=self.hparams['learning_rate_adam']))

    def get_action(self, state, env):
        softmax_out = self.network(state.reshape((1, -1)))
        selected_action = np.random.choice(self.hparams['num_actions'], p=softmax_out.numpy()[0])
        return selected_action

    def update_network(self, states, actions, rewards):
        reward_sum = 0
        discounted_rewards = []
        for reward in rewards[::-1]:  # reverse buffer r
            reward_sum = reward + self.hparams['GAMMA'] * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)
        # standardise the rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        states = np.vstack(states)
        target_actions = np.array([1 - np.array(actions), np.array(actions)]).T
        loss = self.network.train_on_batch(states, target_actions, sample_weight=discounted_rewards)
        return loss
