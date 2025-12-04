import gfootball.env as football_env
from stable_baselines3 import PPO
import os
import wandb
import time
import sys
import argparse


from datetime import datetime



parser = argparse.ArgumentParser(description='Visualiza um agente treinado.')

parser.add_argument('--model_name', type=str, required=True, help='Nome do modelo a ser carregado')
parser.add_argument('--scenario', type=str, default="5_vs_5")

args = parser.parse_args()


model_path = args.model_name

scenario_name = args.scenario

if not os.path.exists(model_path):
    print(f"❌ Modelo não encontrado: {model_path}")
    sys.exit(1)


print(f"--> Carregando agente: {model_path} no cenário {scenario_name}")


wandb.init(
    project="RL_Fut_PPO", 
    entity="guilhermefrazao-ufg", 
    name=f"evaluation_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "scenario": scenario_name,
        "model": model_path,
        "mode": "evaluation"
    }
)


env = football_env.create_environment(
    env_name=scenario_name, # Agora está sincronizado com o treino!
    stacked=True, 
    representation='pixels',
    render=False, 
    write_video=False
)

# Carrega usando a variável
model = PPO.load(model_path)

obs = env.reset()
done = False

episode = 0
step = 0
ep_reward = 0
max_episodes = 20

print("Rodando o agente treinado...")
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    time.sleep(0.03)

    ep_reward += reward
    step += 1

    wandb.log({
        "step_reward": reward,
        "step": step
    })
    
    if done:
        episode += 1

        print(f"Episódio {episode} finalizado | Reward Total: {ep_reward}")

        wandb.log({
            "episode": episode,
            "ep_reward": ep_reward,
            "episode_length": step
        })

        ep_reward = 0
        step = 0


        obs = env.reset()
        print("Episódio terminou. Reiniciando...")

        wandb.save("*.mp4")
        print("Video salvo no WandB")

        if episode >= max_episodes:
            env.close()
            break
            