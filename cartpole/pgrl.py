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
        rewards_to_go = []
        for reward in rewards[::-1]:  # reverse buffer r
            reward_sum = reward + self.hparams['GAMMA'] * reward_sum
            rewards_to_go.append(reward_sum)
        rewards_to_go.reverse()
        rewards_to_go = np.array(rewards_to_go)
        # standardise the rewards
        rewards_to_go -= np.mean(rewards_to_go)
        rewards_to_go /= np.std(rewards_to_go)
        states = np.vstack(states)
        target_actions = tf.keras.utils.to_categorical(np.array(actions), self.hparams['num_actions'])
        loss = self.network.train_on_batch(states, target_actions, sample_weight=rewards_to_go)
        return loss
