import numpy as np
import tensorflow as tf
import datetime as dt
from collections import deque
import gym
from gym import wrappers, logger
import argparse
import json, sys, os
from os import path
from pgrl import PG

Rand_Seed = 1
STORE_PATH = '/tmp/cartpole_exp1'
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/PG_data_{dt.datetime.now().strftime('%d%m%Y%H%M')}")

logger.set_level(logger.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--display', action='store_true')
parser.add_argument('target', nargs="?", default="CartPole-v0")
args = parser.parse_args()
env = gym.make(args.target)
env.seed(0)
env = wrappers.Monitor(env, STORE_PATH + f"/vid_{dt.datetime.now().strftime('%d%m%Y%H%M')}",
                       video_callable=lambda episode_id: episode_id % 100 == 0, force=True)

# hyper parameters
hparams = {
        'input_size': env.observation_space.shape[0],
        'hidden_size': 30,
        'num_actions': env.action_space.n,
        'GAMMA': 1,
        'num_episodes': 1000,
        'Rand_Seed': Rand_Seed
}

# ifo of the system for saving
info = {
    'params': hparams,
    'argv': sys.argv,
    'env_id': env.spec.id,
    'mean_100rew_thre': 195.0
}

policy = PG(hparams)
tot_rews = []
mean_100ep = 0
for episode in range(hparams['num_episodes']):
    state = env.reset()
    rewards = []
    states = []
    actions = []
    done = False
    while not done:
        action = policy.get_action(state)
        new_state, reward, done, _ = env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        state = new_state
    loss = policy.update_network(rewards, states, actions)
    tot_reward = sum(rewards)
    tot_rews.append(tot_reward)
    if episode > 100:
        mean_100ep = np.mean(tot_rews[-101:-1])
        print(f"Episode: {episode}, Reward: {tot_reward}, Mean of 100 cons episodes: {mean_100ep}, "
              f"avg loss: {loss:.5f}")
    # ------- save data --------------
    # with train_writer.as_default():
    #     tf.summary.scalar('reward', tot_reward, step=episode)
    #     tf.summary.scalar('avg loss', loss, step=episode)
    if mean_100ep > info['mean_100rew_thre']:
        print(f"Problem solved after {episode} Episode with the mean reward {mean_100ep} over the last 100 episodes ")
        policy.network.save(STORE_PATH + f"/PG_agent_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
        break
    #---------------------------------

with open(path.join(STORE_PATH, 'info.json'), 'w') as fh:
    fh.write(json.dumps(info))
env.close()