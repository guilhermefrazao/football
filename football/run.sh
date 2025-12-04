#!/bin/bash

#SBATCH --job-name=ppo_agent_football

#SBATCH --output=saida_%j.log

#SBATCH --error=erro_%j.log

#SBATCH --time=03:00:00

#SBATCH --partition=h100n3

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source ~/venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

set -e

echo "🔥 INICIANDO PIPELINE COMPLETO"

STAGES=(1 2 3 4 5)


for STAGE in "${STAGES[@]}"
do
    echo "===================================="
    echo "🚀 EXECUTANDO STAGE $STAGE"
    echo "===================================="
    # 1) Treino
    python3 train_ppo.py --stage $STAGE

    # 2) Avaliação
    echo "🎥 VISUALIZANDO ESTÁGIO $STAGE"

    MODEL_PATH="models/exp${STAGE}_model.zip"

    SCENARIOS=("academy_empty_goal_close" \
           "academy_empty_goal" \
           "academy_run_to_score" \
           "academy_3_vs_1" \
           "5_vs_5")

    SCENARIO=${SCENARIOS[$((STAGE-1))]}

    python3 watch_agent.py --model_name $MODEL_PATH --scenario $SCENARIO

done

