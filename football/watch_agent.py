import gfootball.env as football_env
from stable_baselines3 import PPO
import os

# --- LENDO VARIÁVEIS DO SHELL SCRIPT ---
scenario_name = os.getenv("SCENARIO_NAME", "academy_empty_goal_close")
model_path = os.getenv("MODEL_NAME", "meu_agente_gfootball")

print(f"--> Carregando agente: {model_path} no cenário {scenario_name}")

env = football_env.create_environment(
    env_name=scenario_name, # Agora está sincronizado com o treino!
    stacked=True, 
    representation='simple115v2',
    render=True, 
    write_video=True
)

# Carrega usando a variável
model = PPO.load(model_path)

obs = env.reset()
done = False

print("Rodando o agente treinado...")
while True:
    action, _states = model.predict(obs)
    
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
        print("Episódio terminou. Reiniciando...")

env.close()