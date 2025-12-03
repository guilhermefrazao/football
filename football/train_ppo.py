import gfootball.env as football_env
import os
import wandb

from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from datetime import datetime

# --- LENDO VARIÁVEIS DO SHELL SCRIPT ---
# O segundo parâmetro é um valor padrão caso não venha do shell
scenario_name = os.getenv("SCENARIO_NAME", "5_vs_5")
model_path = os.getenv("MODEL_NAME", "meu_agente_gfootball")
total_timesteps = int(os.getenv("TOTAL_STEPS", "100000"))

print(f"--> Configuração Python: Cenário={scenario_name} | Modelo={model_path}")

run = wandb.init(
    project="RL_Fut_PPO",
    entity="guilhermefrazao-ufg", 
    name=f"train_{scenario_name}-{datetime.now().strftime('%d - %H:%M')}",
    sync_tensorboard=True,  
    monitor_gym=True,    
    save_code=True,  
    config={
        "algo": "PPO",
        "scenario": scenario_name,
        "total_steps": total_timesteps,
        "learning_rate": 0.0003,
        "n_steps": 2048
    }
)

wandb_callback = WandbCallback(
    model_save_path=f"./models/{run.id}",
    model_save_freq=10000,
    verbose=2
)

log_path = f"./logs_gfootball/{run.id}"
new_logger = configure(log_path, ["stdout", "tensorboard"])


env = football_env.create_environment(
    env_name=scenario_name, 
    stacked=True, 
    representation='simple115v2',
    render=False, 
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    write_video=False
)

model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log="./logs_gfootball/",
    learning_rate=0.0003,
    n_steps=2048,
    device='cpu',  # MlpPolicy funciona melhor na CPU
)

model.set_logger(new_logger)

print(f"Iniciando treinamento no cenário: {scenario_name}...")

model.learn(
    total_timesteps=total_timesteps,
    callback=wandb_callback
)


model.save(model_path)
print(f"Modelo salvo em {model_path}.zip")

final_model_path = f"./models/{run.id}/final_model"
os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

model.save(final_model_path)
print(f"Modelo final salvo em {final_model_path}.zip")


if os.path.exists(final_model_path + ".zip"):
    artifact = wandb.Artifact("final-model", type="model")
    artifact.add_file(final_model_path + ".zip")
    wandb.log_artifact(artifact)
    print("Modelo adicionado ao wandb artifact")
else:
    print(f"⚠️  Aviso: Arquivo {final_model_path}.zip não encontrado. Pulando upload para wandb.")


env.close()