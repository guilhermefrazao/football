#!/bin/bash

# --- 1. Configurações Compartilhadas ---
# Defina aqui o nome do cenário e do modelo uma única vez
export SCENARIO_NAME="academy_empty_goal_close"
export MODEL_NAME="meu_agente_v1"
export TOTAL_STEPS=100000

# --- 2. Fix de Dependência (MPI/HPCX) ---
# Garante que não dê erro de "undefined symbol"
export LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

echo "=================================================="
echo "Iniciando processo para o cenário: $SCENARIO_NAME"
echo "Modelo será salvo como: $MODEL_NAME"
echo "=================================================="

# --- 3. Executar Treinamento ---
echo "[1/2] Iniciando Treinamento..."
python3 train_ppo.py

# Verifica se o treino rodou com sucesso (código de saída 0)
if [ $? -eq 0 ]; then
    echo "Treinamento concluído com sucesso!"
    
    # --- 4. Executar Visualização ---
    echo "[2/2] Iniciando Visualização do Agente..."
    python3 watch_agent.py
else
    echo "ERRO: O treinamento falhou. A visualização não será iniciada."
    exit 1
fi