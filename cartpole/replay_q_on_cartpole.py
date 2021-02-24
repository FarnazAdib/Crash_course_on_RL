import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from cartpole.policy_iteration import Q
from cartpole.dynamics import CartPole
from cartpole.pltlib import PLTLIB

# ----------locations for saving data ----------------------
STORE_PATH = '/tmp/cartpole_exp1/Q_replay'
data_path = STORE_PATH + f"/data_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
agent_path = STORE_PATH + f"/agent_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
train_writer = tf.summary.create_file_writer(data_path)

# ---------- environment -------------------------------------
Rand_Seed = 1
env_par = {
    'Rand_Seed': Rand_Seed,
    'STORE_PATH': STORE_PATH,
    'monitor': False,
    'threshold': 195.0
}
Rand_Seed = 1
CP = CartPole(env_par)

# ---------- agent --------------------------------------------
agent_par = {
    'num_state': CP.env.observation_space.shape[0],
    'num_actions': CP.env.action_space.n,
    'Rand_Seed': Rand_Seed,
    'hidden_size': 30,
    'GAMMA': 1.0,
    'num_episodes': 5000,
    'batch_size': 200,
    'epsilon': 0.1,  # exploration rate
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'learning_rate_adam': 0.01,
    'adam_eps': 0.1,
}
policy = Q(agent_par)


# Running the algorithm for a maximum number of iteration until it is solved
tot_rews = []
mean_100ep = 0.0
for episode in range(agent_par['num_episodes']):

    # Do one rollout. When remember=True, data is saved in policy.memory
    _, _, rewards, _, _ = CP.one_rollout(policy, remember=True)

    # Update the network using experience replay
    loss = policy.replay(agent_par['batch_size'])

    # Check if the problem is solved
    if episode > 100:
        mean_100ep = np.mean(tot_rews[-101:-1])

    tot_reward = sum(rewards)
    tot_rews.append(tot_reward)
    print(f"Episode: {episode}, Reward: {tot_reward}, Mean of 100 cons episodes: {mean_100ep}")
    if mean_100ep > env_par['threshold']:
        print(f"Problem is solved.")
        policy.network.save(agent_path)
        break

    # Save data
    with train_writer.as_default():
        tf.summary.scalar('reward', tot_reward, step=episode)

# Close the environment
CP.env.close()

# Print the results if the problem is solved
if mean_100ep > env_par['threshold']:
    # Print the summary of the solution
    print(f"\n\nProblem is solved after {episode} Episode with the mean reward {mean_100ep} over the last 100 episodes")

    # Plot the result
    MyPlot = PLTLIB()
    MyPlot.reward_it(tot_rews)
