from stable_baselines3.common.callbacks import CallbackList, BaseCallback

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=20, min_delta=0.01, window_size=50, verbose=1):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float("inf")
        self.window_size = window_size

        self.success_window = []
        self.best_rate = 0.0
        self.counter = 0

    def _on_step(self):
        done = self.locals["dones"][0]
        info = self.locals["infos"][0]

        if done:
            
            truncated = info.get("TimeLimit.truncated", False)
            score_reward = info.get("score_reward", 0)

            success = 1 if (score_reward > 0 and not truncated) else 0

            self.success_window.append(success)

            if len(self.success_window) > self.window_size:
                self.success_window.pop(0)

            if len(self.success_window) == self.window_size:
                success_rate = sum(self.success_window) / self.window_size

                if success_rate > self.best_rate + self.min_delta:
                    self.best_rate = success_rate
                    self.counter = 0
                else:
                    self.counter += 1

                if self.verbose:
                    print(f"[EarlyStop] SuccessRate={success_rate:.2f} | Best={self.best_rate:.2f} | Count={self.counter}/{self.patience}")

                if self.counter >= self.patience:
                    print("⛔ Early stopping acionado por estagnação da taxa de sucesso!")
                    return False

        return True
