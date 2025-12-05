from ast import List
import gfootball.env as football_env
from stable_baselines3 import PPO
import os
import wandb
import time
import sys
import numpy as np
import argparse


from datetime import datetime



parser = argparse.ArgumentParser(description='Visualiza um agente treinado.')

parser.add_argument('--model_name', type=str, required=True, help='Nome do modelo a ser carregado')
parser.add_argument('--scenario', type=str, default="5_vs_5")

args = parser.parse_args()


def eval_metrics(info, ep_reward, step, episode, max_episodes, all_success, all_rewards, all_lengths, all_goals_for, all_goals_against):
    score_reward = info.get("score_reward", (0, 0))
    goals_for = score_reward[0] if type(score_reward) == List else score_reward
    goals_against = score_reward[1] if type(score_reward) == List else 0

    print(f"goals_for {goals_for}")
    print(f"goals_against {goals_against}")

    success = 1 if goals_for > 0 else 0


    all_success.append(success)
    all_rewards.append(ep_reward)
    all_lengths.append(step)
    all_goals_for.append(goals_for)
    all_goals_against.append(goals_against)

    success_rate = np.mean(all_success)

    print(
            f"[Episódio {episode}/{max_episodes}] "
            f"R_total={ep_reward:.2f} | len={step} | "
            f"goals_for={goals_for} | goals_against={goals_against} | "
            f"success_rate={success_rate:.2f}"
        )


    wandb.log({
        "episode": episode,
        "ep_reward": ep_reward,
        "episode_length": step,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "success": success,
        "success_rate": success_rate,
        "goal_diff": goals_for - goals_against,
    })



model_path = args.model_name

scenario_name = args.scenario

if not os.path.exists(model_path):
    print(f"❌ Modelo não encontrado: {model_path}")
    sys.exit(1)


print(f"--> Carregando agente: {model_path} no cenário {scenario_name}")


wandb.init(
    project="RL_Fut_PPO", 
    entity="guilhermefrazao-ufg", 
    name=f"evaluation_{scenario_name}_{datetime.now().strftime('%d - %H:%M')}",
    config={
        "scenario": scenario_name,
        "model": model_path,
        "mode": "evaluation"
    }
)


env = football_env.create_environment(
    env_name=scenario_name, 
    stacked=True, 
    representation='simple115v2',
    render=True, 
    write_video=False
)


model = PPO.load(model_path)

all_rewards = []
all_lengths = []
all_goals_for = []
all_goals_against = []
all_success = []

episode = 0
step = 0
ep_reward = 0
max_episodes = 20
obs = env.reset()

print("Rodando o agente treinado...")
while episode < max_episodes:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    time.sleep(0.09)

    ep_reward += reward
    step += 1

    wandb.log({
        "step_reward": reward,
        "step": step,
        "episode_idx": episode,
    })
    
    if done:
        episode += 1

        eval_metrics(info, ep_reward, step, episode, max_episodes, all_success, all_rewards, all_lengths, all_goals_for, all_goals_against)

        ep_reward = 0.0
        step = 0
        obs = env.reset()

# ---------------- FIM DA AVALIAÇÃO ----------------
env.close()


mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
mean_len = float(np.mean(all_lengths)) if all_lengths else 0.0
mean_goals_for = float(np.mean(all_goals_for)) if all_goals_for else 0.0
total_goals_for = float(np.sum(all_goals_for)) if all_goals_for else 0.0
mean_goals_against = float(np.mean(all_goals_against)) if all_goals_against else 0.0
total_goals_agaist = float(np.sum(all_goals_against)) if all_goals_against else 0.0
final_success_rate = float(np.mean(all_success)) if all_success else 0.0

summary = {
    "eval/mean_reward": mean_reward,
    "eval/mean_length": mean_len,
    "eval/mean_goals_for": mean_goals_for,
    "eval/total_goals_for": total_goals_for,
    "eval/mean_goals_against": mean_goals_against,
    "eval/total_goals_agaist": total_goals_agaist,
    "eval/final_success_rate": final_success_rate,
}

print("\n📊 RESUMO FINAL DA AVALIAÇÃO:")
for k, v in summary.items():
    print(f"  {k}: {v:.3f}")

wandb.log(summary)

wandb.finish()

print("✅ Avaliação concluída.")

            