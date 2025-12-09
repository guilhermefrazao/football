import wandb

run = wandb.init()
artifact = run.use_artifact('guilhermefrazao-ufg/RL_Fut_PPO/run-1ekqd8ah-history:v0', type='wandb-history')
artifact_dir = artifact.download()