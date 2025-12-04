from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from custom_reward import FootballShapedReward
import gfootball.env as football_env

class CurriculumCallback(BaseCallback):
    def __init__(self, thresholds, scenarios, env):
        super().__init__()
        self.thresholds = thresholds
        self.scenarios = scenarios
        self.env = env
        self.index = 0
        self.recent_scores = []

    def _on_step(self):
        if self.locals["dones"][0]:
            goals = self.locals["infos"][0].get("score", (0,0))[0]
            self.recent_scores.append(goals)

            print(f"recent_goals {self.recent_scores}")

            if len(self.recent_scores) > 20:
                self.recent_scores.pop(0)

            avg = sum(self.recent_scores)/len(self.recent_scores)

            if self.index + 1 >= len(self.scenarios):
                print("[CURRICULUM] Último nível atingido.")
                return True

            if avg >= self.thresholds[self.index]:
                self.index += 1
                print(f"\n Subindo dificuldade → {self.scenarios[self.index]}")

                old_env = self.model.get_env()
                old_env.close()
                new_env = football_env.create_environment(env_name=self.scenarios[self.index])
                wrapped = FootballShapedReward(new_env)
                self.model.set_env(wrapped)
                self.env = wrapped
        
        return True