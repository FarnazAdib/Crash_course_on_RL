import datetime as dt
import gym
from gym import wrappers, logger
import argparse


class CartPole:
    def __init__(self, par):
        self.Rand_Seed = par['Rand_Seed']
        self.STORE_PATH = par['STORE_PATH']

        # logger.set_level(logger.INFO)
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--display', action='store_true')
        # parser.add_argument('target', nargs="?", default="CartPole-v0")
        # args = parser.parse_args()
        # self.env = gym.make(args.target)
        self.env = gym.make('CartPole-v0')
        self.env.seed(0)
        self.env.action_space.seed(self.Rand_Seed)
        if par['monitor']:
            self.env = wrappers.Monitor(self.env, self.STORE_PATH + f"/vid_{dt.datetime.now().strftime('%d%m%Y%H%M')}",
                                        video_callable=lambda episode_id: episode_id % 100 == 0, force=True)

    def one_rollout(self, agent, remember=False):
        '''
        Run one rollout using agent
        :param agent: The policy  
        :param remember: if it is true, the data will be stored in the agent memory
        :return: states, actions, rewards, new_states, dones of the rollout
        '''''
        states, actions, rewards, new_states, dones = [], [], [], [], []
        state = self.env.reset()
        done = False
        while not done:
            action = agent.get_action(state, self.env)
            new_state, reward, done, _ = self.env.step(action)
            if remember:
                agent.remember(state, action, reward, new_state, done)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(new_state)
            dones.append(done)
            state = new_state
        return states, actions, rewards, new_states, dones
