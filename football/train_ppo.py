import gfootball.env as football_env
import os
from gfootball.env import players
from gfootball.env.football_action_set import action_bottom
import gym
from numpy import diff
import wandb
import torch 
import zipfile
import io
import argparse

from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from datetime import datetime

import gfootball.env.players.agent as agent_module
from custom_agent import Player as CustomPlayer
from custom_reward import FootballShapedReward
from action_wrapper import ActionCurriculumCallback, ActionCurriculumWrapper


parser = argparse.ArgumentParser(description='Visualiza um agente treinado.')

parser.add_argument('--stage', type=int, required=True, help='Nome do modelo a ser carregado')

args = parser.parse_args()



STAGES = {
    1: {
        "scenario": "academy_empty_goal",
        "timesteps": 1_000_000,
        "reward": "checkpoint",
        "adversary": "angle",
        "initial_diff" : 0.01
    },
    2: {
        "scenario": "academy_pass_and_shoot_with_keeper",
        "timesteps": 2_000_000,
        "reward": "angle",
        "adversary": "default",
        "initial_diff" : 0.05
    },
    3: {
        "scenario": "academy_counterattack_easy",
        "timesteps": 2_000_000,
        "reward": "counterattack",
        "adversary": "default",
        "initial_diff" : 0.01
    },
    4: {
        "scenario": "5_vs_5",
        "timesteps": 1_000_000,
        "reward": "advanced",
        "adversary": "custom",
        "initial_diff" : 0.1
    },
    5: {
        "scenario": "5_vs_5",
        "timesteps": 1_000_000,
        "reward": "advanced",
        "adversary": "self_play",
        "initial_diff" : 0.3
    }


}


class FootballMetricsCallback(BaseCallback):
    def __init__(self, window_size=50, initial_diff=0.1):
        super().__init__()
        self.window_size = window_size
        self.recent_success = []
        self.current_diff = initial_diff

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

            if success_rate >= 0.80 and self.current_diff < 1.0:
                print(f"\n--- Alterando Dificuldade ---")
                print(f"Total timesteps: {self.num_timesteps}")
                print(f"Nova Dificuldade (θ): {self.current_diff:.3f}")
                if self.current_diff >= 1.0:
                    self.current_diff = 1.0

                else:
                    self.current_diff = self.current_diff + 0.1
                
                self.model.env.set_attr('other_config_options', [{'difficulty': self.current_diff}])

        return True


def setup_wandb(scenario_name, STAGE, total_timesteps):
    run = wandb.init(
    project="RL_Fut_PPO",
    entity="guilhermefrazao-ufg", 
    name=f"train_{STAGE}_{scenario_name}_-{datetime.now().strftime('%d - %H:%M')}_3050_bs-1",
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
    print(f"difficulty = {config['initial_diff']}")
    if config["adversary"] == "self_play":
        env = football_env.create_environment(
            env_name=scenario_name, 
            stacked=True, 
            representation='simple115v2',
            render=False, 
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            write_video=False,
            number_of_left_players_agent_controls=5,
            number_of_right_players_agent_controls=5
        )

        agent_module.Player = CustomPlayer

    elif config["adversary"] == "custom":
        env = football_env.create_environment(
            env_name=scenario_name, 
            stacked=True, 
            representation='simple115v2',
            render=False, 
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            write_video=False,
            number_of_left_players_agent_controls=5,
            other_config_options={'difficulty': config["initial_diff"]}
        )

    else:
        env = football_env.create_environment(
            env_name=scenario_name, 
            stacked=True, 
            representation='simple115v2',
            render=False, 
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            write_video=False,
            other_config_options={'difficulty': config["initial_diff"]}
        )

        env = ActionCurriculumWrapper(env)


    if config["reward"] == "none":
        pass
    elif config["reward"] == "angle":
        env = FootballShapedReward(env, step_penalty=1e-4, reward_type=config["reward"], adversary_type=config["adversary"])
    elif config["reward"] == "checkpoint":
        env = FootballShapedReward(env, step_penalty=1e-4, goal_bonus=1.0, progress_reward=0.1, reward_type=config["reward"], adversary_type=config["adversary"])
    elif config["reward"] == "counterattack":
        env = FootballShapedReward(env, step_penalty=1e-4, goal_bonus=0.5, progress_reward=0.05, reward_type=config["reward"], adversary_type=config["adversary"])
    elif config["reward"] == "advanced":
        env = FootballShapedReward(env, step_penalty=5e-5, goal_bonus=0.5, progress_reward=0.05, reward_type=config["reward"], adversary_type=config["adversary"])

    return env


def linear_lr_decay(initial_lr):
    def schedule(progress_remaining):
        return progress_remaining * initial_lr
    return schedule



def setup_model(STAGE, load_path, env):
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs_gfootball/",
        learning_rate=linear_lr_decay(3e-4),
        n_steps=2048,
        gamma=0.993,
        gae_lambda=0.95,
        ent_coef=0.00155,
        batch_size=128,
        clip_range=0.115,
        max_grad_norm=0.76,
        vf_coef=0.5,
        device="cpu",
    )    

    if STAGE > 1 and os.path.exists(load_path):
        print(f"🔁 Carregando pesos do estágio anterior: {load_path}")

        old_model = PPO.load(load_path, device="cpu")

        old_sd = old_model.policy.state_dict()
        new_sd = model.policy.state_dict()

        for k in new_sd.keys():
            if k in old_sd and old_sd[k].shape == new_sd[k].shape:
                new_sd[k] = old_sd[k]
            else:
                print(f"Ignorando camada incompatível: {k}")

        model.policy.load_state_dict(new_sd)

        print("Pesos transferidos com sucesso! Iniciando treinamento...")

    return model



def run_agent():
    STAGE = args.stage

    config = STAGES[STAGE]

    scenario_name = config["scenario"]

    total_timesteps = config["timesteps"]

    print(f"--> Configuração Python: Cenário = {scenario_name}")

    new_logger, wandb_callback = setup_wandb(scenario_name, STAGE, total_timesteps)

    env = setup_env(config, scenario_name)

    prev_stage = STAGE - 1

    load_path = f"./models/exp_{prev_stage}_{STAGES[prev_stage]['scenario']}_model.zip" if STAGE > 1 else None

    model = setup_model(STAGE, load_path, env)

    model.set_logger(new_logger)

    callback = CallbackList([wandb_callback, FootballMetricsCallback(initial_diff=config["initial_diff"]), ActionCurriculumCallback(activation_timestep=10000)])

    print(f"Iniciando treinamento no cenário: {scenario_name}...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    model_path = f"models/exp_{STAGE}_{scenario_name}_model"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model.save(model_path)


    wandb.finish()
    env.close()



if __name__ == "__main__":
    run_agent()