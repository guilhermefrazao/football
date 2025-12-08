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

    SCENARIOS=("academy_run_to_score_with_keeper" \
           "academy_pass_and_shoot_with_keeper" \
           "academy_counterattack_easy" \
           "5_vs_5" \
           "5_vs_5")

    SCENARIO=${SCENARIOS[$((STAGE-1))]}

    python3 watch_agent.py --model_name $MODEL_PATH --scenario $SCENARIO

done

