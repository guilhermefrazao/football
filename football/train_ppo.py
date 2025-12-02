import gfootball.env as football_env
from stable_baselines3 import PPO
import os

# --- LENDO VARIÁVEIS DO SHELL SCRIPT ---
# O segundo parâmetro é um valor padrão caso não venha do shell
scenario_name = os.getenv("SCENARIO_NAME", "academy_empty_goal_close")
model_path = os.getenv("MODEL_NAME", "meu_agente_gfootball")
total_timesteps = int(os.getenv("TOTAL_STEPS", "100000"))

print(f"--> Configuração Python: Cenário={scenario_name} | Modelo={model_path}")

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
)

print(f"Iniciando treinamento no cenário: {scenario_name}...")

model.learn(total_timesteps=total_timesteps)

model.save(model_path)
print(f"Modelo salvo em {model_path}.zip")

env.close()