import numpy as np
import tensorflow as tf
import datetime as dt
# from collections import deque
# import gym
# from gym import wrappers, logger
# import argparse
import json, sys, os
from os import path
from policy_iteration import PI
from dynamics import CartPole

# ----------locations for saving data ----------------------
STORE_PATH = '/tmp/cartpole_exp1/Q'
data_path = STORE_PATH + f"/data_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
agent_path = STORE_PATH + f"/agent_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
info_path = STORE_PATH + f"/info_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
os.makedirs(info_path)
train_writer = tf.summary.create_file_writer(data_path)

# ---------- environment -------------------------------------
Rand_Seed = 1
CP = CartPole(Rand_Seed, STORE_PATH)
info = {
    'argv': sys.argv,
    'env_id': CP.env.spec.id,
    'mean_100rew_thre': 195.0
}

# ---------- agent --------------------------------------------
# hyper parameters
hparams = {
        'input_size': CP.env.observation_space.shape[0],
        'hidden_size': 30,
        'num_actions': CP.env.action_space.n,
        'GAMMA': 1,
        'num_episodes': 5000,
        'Rand_Seed': Rand_Seed,
        'epsilon': 0.1,
        'adam_eps': 0.1
}
policy = PI(hparams)


# Running the algorithm for a maximum number of iteration until it is solved
tot_rews = []
mean_100ep = 0
for episode in range(hparams['num_episodes']):
    states, actions, rewards, new_states, dones = CP.one_rollout(policy) # one rollout
    loss = policy.update_network(states, actions, rewards, new_states, dones) # update the network

    # check if the problem is solved
    mean_100ep = np.mean(tot_rews[-101:-1])
    tot_reward = sum(rewards)
    tot_rews.append(tot_reward)
    print(f"Episode: {episode}, Reward: {tot_reward}, Mean of 100 cons episodes: {mean_100ep}, avg loss: {loss:.5f}")
    if mean_100ep > info['mean_100rew_thre']:
        print(f"Problem solved after {episode} Episode with the mean reward {mean_100ep} over the last 100 episodes ")
        policy.network.save(agent_path)
        break

    # save data
    with train_writer.as_default():
        tf.summary.scalar('reward', tot_reward, step=episode)
# save info
with open(path.join(info_path, 'info.json'), 'w') as fh:
    fh.write(json.dumps(info))
CP.env.close()