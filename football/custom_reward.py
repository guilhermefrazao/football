import gfootball.env as football_env
import os
import gym


# Class to better reward my RL model
class FootballShapedReward(gym.Wrapper):
    def __init__(self, env, step_penalty=0.001, goal_bonus=1):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.goal_bonus = goal_bonus

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaped_reward = reward - self.step_penalty

        if reward > 0:
            shaped_reward += self.goal_bonus

        return obs, shaped_reward, done, info