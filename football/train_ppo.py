import gfootball.env as football_env
import os
from gfootball.env import players
import gym
import wandb
import argparse

from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from datetime import datetime

import gfootball.env.players.agent as agent_module
from custom_agent import Player as CustomPlayer
from custom_reward import FootballShapedReward


parser = argparse.ArgumentParser(description='Visualiza um agente treinado.')

parser.add_argument('--stage', type=int, required=True, help='Nome do modelo a ser carregado')

args = parser.parse_args()



STAGES = {
    1: {
        "scenario": "1_vs_1_easy",
        "timesteps": 1_000_000,
        "reward": "light",
        "adversary": "default"
    },
    2: {
        "scenario": "academy_pass_and_shoot_with_keeper",
        "timesteps": 1_000_000,
        "reward": "light",
        "adversary": "default"
    },
    3: {
        "scenario": "academy_3_vs_1_with_keeper",
        "timesteps": 1_000_000,
        "reward": "medium",
        "adversary": "default"
    },
    4: {
        "scenario": "academy_counterattack_hard",
        "timesteps": 1_000_000,
        "reward": "advanced",
        "adversary": "custom"
    },
    5: {
        "scenario": "5_vs_5",
        "timesteps": 1_000_000,
        "reward": "advanced",
        "adversary": "self_play"
    }

}


class FootballMetricsCallback(BaseCallback):
    def __init__(self, window_size=50):
        super().__init__()
        self.window_size = window_size
        self.recent_success = []

    def _on_step(self):
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        if done:
            ep = info["episode"]
            truncated = info.get("TimeLimit.truncated", False)
            score_reward = info.get("score_reward", 0)

            self.logger.record("custom/episode_reward", ep["r"])
            self.logger.record("custom/episode_length", ep["l"])

            success = 1 if (score_reward > 0 and not truncated) else 0 # Change score_reward according to stage

            self.recent_success.append(success)

            if len(self.recent_success) > self.window_size:
                self.recent_success.pop(0)

            success_rate = sum(self.recent_success) / len(self.recent_success)
            efficiency = success / ep["l"]

            self.logger.record("custom/success_rate", success_rate)
            self.logger.record("custom/success_last_ep", success)
            self.logger.record("custom/efficiency", efficiency)

        return True



def setup_wandb(scenario_name, STAGE, total_timesteps):
    run = wandb.init(
    project="RL_Fut_PPO",
    entity="guilhermefrazao-ufg", 
    name=f"train_{STAGE}_{scenario_name}_-{datetime.now().strftime('%d - %H:%M')}_3050",
    sync_tensorboard=True,   
    save_code=True,  
    config={
        "exp": 1,
        "algo": "PPO",
        "scenario": scenario_name,
        "total_steps": total_timesteps,
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "gamma": 0.97,
        "step_penalty": 0.001,
        "goal_bonus": 0.5,
        }   
    )


    wandb_callback = WandbCallback(
        model_save_path=f"models/{run.id}",
        model_save_freq=10000,
        verbose=2
    )

    log_path = f"./logs_gfootball/{run.id}"
    
    new_logger = configure(log_path, ["stdout", "tensorboard"])

    return new_logger, wandb_callback



def setup_env(config, scenario_name):
    if config["adversary"] == "custom":
        players=[
            'agent:left_players=5',
            'agent:right_players=5'
                ]

        env = football_env.create_environment(
            env_name=scenario_name, 
            stacked=True, 
            representation='simple115v2',
            render=False, 
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            write_video=False,
            players=players
        )

        agent_module.Player = CustomPlayer

    else:
        env = football_env.create_environment(
            env_name=scenario_name, 
            stacked=True, 
            representation='simple115v2',
            render=False, 
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            write_video=False
        )


    if config["reward"] == "none":
        pass
    elif config["reward"] == "light":
        env = FootballShapedReward(env, step_penalty=1e-4, reward_type=config["reward"])
    elif config["reward"] == "medium":
        env = FootballShapedReward(env, step_penalty=1e-4, goal_bonus=0.5, progress_reward=0.05, reward_type=config["reward"])
    elif config["reward"] == "advanced":
        env = FootballShapedReward(env, step_penalty=5e-5, goal_bonus=1.0, progress_reward=0.05, reward_type=config["reward"])

    return env


def linear_lr_decay(initial_lr):
    def schedule(progress_remaining):
        return progress_remaining * initial_lr
    return schedule



def setup_model(STAGE, load_path, env):
    # Change to use the model pretrained from another stage
    # Change hyperparameters to favor exploration.
    if STAGE > 1 and os.path.exists(load_path):
        print(f"🔁 Carregando pesos do estágio anterior: {load_path}")
        model = PPO.load(load_path, env=env, device="cpu")
    else:    
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log="./logs_gfootball/",
            learning_rate=linear_lr_decay(3e-4),
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            device="cpu",
        )    

    return model



def run_agent():
    STAGE = args.stage

    config = STAGES[STAGE]

    scenario_name = config["scenario"]

    total_timesteps = config["timesteps"]

    print(f"--> Configuração Python: Cenário={scenario_name}")

    new_logger, wandb_callback = setup_wandb(scenario_name, STAGE, total_timesteps)

    env = setup_env(config, scenario_name)

    prev_stage = STAGE - 1

    load_path = f"./models/exp{prev_stage}_{STAGES[prev_stage]['scenario']}_model.zip" if STAGE > 1 else None

    model = setup_model(STAGE, load_path, env)

    model.set_logger(new_logger)

    callback = CallbackList([wandb_callback, FootballMetricsCallback()])

    print(f"Iniciando treinamento no cenário: {scenario_name}...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    model_path = f"models/exp{STAGE}_{scenario_name}_model"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.save(model_path)


    wandb.finish()
    env.close()



if __name__ == "__main__":
    run_agent()