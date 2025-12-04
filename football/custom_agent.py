import gfootball

from gfootball.env.players import agent

class Player(agent.Player):
    """Classe Player customizada que herda da Player do repositório."""
    
    def __init__(self, player_config, env_config):
        super().__init__(player_config, env_config)
        self._custom_data = {}
        
    def set_action(self, action):
        super().set_action(action)
        
    def take_action(self, observations):
        action = super().take_action(observations)
        return action
        
    def reset(self):
        if hasattr(agent.Player, 'reset'):
            super().reset()
