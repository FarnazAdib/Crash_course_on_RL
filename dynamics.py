def policy_rollout(env, agent):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []

    while not done:

        env.render()
        obs.append(observation)

        action = agent.act(observation)
        observation, reward, done, _ = env.step(action)

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews

def process_rewards(rews):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return [len(rews)] * len(rews)