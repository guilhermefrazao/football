import numpy as np
from gym import spaces
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gym.core import Wrapper 

RESTRICTED_ACTIONS = {
    9: 'LONG_PASS', 10: 'HIGH_PASS', 11: 'SHORT_PASS', 
    12: 'SHOT', 16: 'DRIBBLE', 17: 'STOP_DRIBBLE', 18: 'SLIDING'
}


SUBSTITUTE_ACTION = 0 # IDLE

class ActionCurriculumWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.allow_restricted = False # Começa com ações restritas
        
    def step(self, action):      
        if not self.allow_restricted and action in RESTRICTED_ACTIONS:
            action = SUBSTITUTE_ACTION 

        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def set_allow_restricted(self, value: bool):
        """Método para ser chamado pelo Callback para alterar a restrição."""
        self.allow_restricted = value
        if self.allow_restricted:
            print("Curriculum: Ações complexas ATIVADAS.")
        else:
            print("Curriculum: Ações complexas DESATIVADAS.")


class ActionCurriculumCallback(BaseCallback):
    def __init__(self, activation_timestep: int, verbose: int = 0):
        super().__init__(verbose)
        self.activation_timestep = activation_timestep
        self.activated = False

    def _on_step(self) -> bool:
        if not self.activated and self.num_timesteps >= self.activation_timestep:

            self.model.env.set_attr('allow_restricted', True)

            self.activated = True

            if self.verbose > 0:
                print(f"--- 🔓 Ações Complexas Liberadas em {self.num_timesteps} timesteps! ---")
                
        return True 