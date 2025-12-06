import gfootball.env as football_env
import os
import numpy as np
import gym


# Class to better reward my RL model
class FootballShapedReward(gym.Wrapper):
    def __init__(self, env, step_penalty=0.001, goal_bonus=1, progress_reward=0.05, reward_type="light"):
        super().__init__(env)
        self.step_penalty = step_penalty
        self.goal_bonus = goal_bonus
        self.reward_type = reward_type
        self.progress_reward = progress_reward

        self.ball_last_place = None

    def check_ball_status(self, obs):
        ball_x = obs[88]
        ball_y = obs[89]
        ball_z = obs[90]
        
        out_lateral = abs(ball_y) > 0.42
        
        out_fundo = abs(ball_x) > 1.01 and abs(ball_y) > 0.05 

        is_physically_out = out_lateral or out_fundo

        ball_owned_team = obs[94:97]

        owner_index = np.argmax(ball_owned_team)
        
        is_opponent_ball = (owner_index == 2)

        return is_physically_out, is_opponent_ball 

    
    def checkpoint_reward(self, obs):
        ball_x = obs[88]

        if self.ball_last_place is None:
            self.ball_last_place = ball_x

        progress = ball_x - self.ball_last_place

        self.ball_last_place = ball_x

        return progress


    def goal_rewards(self, obs):
        ball_x = obs[88]
        ball_y = obs[89]

        in_goal_area = abs(ball_y) < 0.044 

        passed_right_line = ball_x > 1.0

        passed_left_line = ball_x < -1.0
        
        we_scored = passed_right_line and in_goal_area
        
        they_scored = passed_left_line and in_goal_area

        return we_scored, they_scored



    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        shaped_reward = reward - self.step_penalty

        if reward > 0:
            shaped_reward += self.goal_bonus

        """we_scored, they_scored = self.goal_rewards(obs)

        if we_scored:
            shaped_reward += self.goal_bonus

        if they_scored:
            shaped_reward -= self.goal_bonus"""

        if self.reward_type == "light":
            ball_out, is_opponent_ball = self.check_ball_status(obs)

            if ball_out and not is_opponent_ball:
                print("Bola saiu, punição aplicada.")
                shaped_reward -= 0.2

        elif self.reward_type == "advanced":
            progress = self.checkpoint_reward(obs)
            
            if progress > 0:
                shaped_reward += self.progress_reward * progress
            else:
                shaped_reward -= self.progress_reward * abs(progress) * 0.5
            

        return obs, shaped_reward, done, info