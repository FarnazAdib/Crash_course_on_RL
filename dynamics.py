import tensorflow as tf
import datetime as dt
import gym
from gym import wrappers, logger
import argparse
class CartPole():
    def __init__(self, rand_seed, store_path):
        self.Rand_Seed = rand_seed
        self.STORE_PATH = store_path

        logger.set_level(logger.INFO)
        parser = argparse.ArgumentParser()
        parser.add_argument('--display', action='store_true')
        parser.add_argument('target', nargs="?", default="CartPole-v0")
        args = parser.parse_args()
        self.env = gym.make(args.target)
        self.env.seed(0)
        self.env.action_space.seed(self.Rand_Seed)
        self.env = wrappers.Monitor(self.env, self.STORE_PATH + f"/vid_{dt.datetime.now().strftime('%d%m%Y%H%M')}",
                                    video_callable=lambda episode_id: episode_id % 100 == 0, force=True)

    def one_rollout(self, agent):
        """Run one episode."""
        states, actions, rewards, new_states, dones = [], [], [], [], []
        state = self.env.reset()
        done = False
        while not done:
            action = agent.get_action(state, self.env)
            new_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(new_state)
            dones.append(done)
            state = new_state
        return states, actions, rewards, new_states, dones
