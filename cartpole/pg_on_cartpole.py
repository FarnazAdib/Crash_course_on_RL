import numpy as np
import tensorflow as tf
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
from cartpole.pgrl import PG
from cartpole.dynamics import CartPole


# ----------locations for saving data ----------------------
STORE_PATH = '/tmp/cartpole_exp1/PG'
data_path = STORE_PATH + f"/data_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
agent_path = STORE_PATH + f"/agent_{dt.datetime.now().strftime('%d%m%Y%H%M')}"
train_writer = tf.summary.create_file_writer(data_path)

# ---------- environment -------------------------------------
Rand_Seed = 1
env_par = {
    'Rand_Seed': Rand_Seed,
    'STORE_PATH': STORE_PATH,
    'monitor': True,
    'threshold': 195.0
}
Rand_Seed = 1
CP = CartPole(env_par)

# ---------- agent --------------------------------------------
# hyper parameters
agent_par = {
    'num_state': CP.env.observation_space.shape[0],
    'num_actions': CP.env.action_space.n,
    'Rand_Seed': Rand_Seed,
    'hidden_size': 30,
    'GAMMA': 1.0,
    'num_episodes': 1000,
    'learning_rate_adam': 0.001,
    'adam_eps': 1e-7,
}
policy = PG(agent_par)


# Running the algorithm for a maximum number of iteration until it is solved
tot_rews = []
mean_100ep = 0
for episode in range(agent_par['num_episodes']):

    # Do one rollout
    states, actions, rewards, _, _ = CP.one_rollout(policy)

    # Update the network
    loss = policy.update_network(states, actions, rewards)

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

# Print the summary of the solution
if mean_100ep > env_par['threshold']:
    print(f"\n\nProblem is solved after {episode} Episode with the mean reward {mean_100ep} over the last 100 episodes")
