Fork do repositório https://github.com/BrunoBSM/football, para realização do trabalho final da diciplina de Aprendizado por Reforço.

Para rodar localmente antes de utilizar o docker é necessário baixar o requirements, mais tem algumas questões que podem gerar problemas.

**bugs**

Não recomendo rodar localmente utilizando Windows
Não é recomendando utilizar pacotes de ambiente virtuais como "uv" para rodar localmente

Passo a passo para rodar o repositório (localmente)

```bash
pip install -r requirements.txt
```

```bash
cd football
``` 

```bash
./run_agents.sh
``` 

ou 

```bash
python3 train_ppo.py
``` 


Passo a passo para rodar o repositório (utilizando docker: Recomendado)


```bash
cd rl-course-gfootball-helpers
``` 

É necessário criar um arquivo .env dentro dessa pasta, que vai conter 

WANDB_API_KEY=chave-wandb
UID=1000
GID=1000


buildar e subir o docker 

```bash
docker compose up --build
```


Pode também Entrar dentro do container para executar o código

```bash
docker compose exec gfootball bash
```

Rodar o código python desejado dentro do container

```bash
ls
python3 train_ppo.py --stage 1
```

Para verificar a capacidade do modelo treinado deve rodar watch_agent.py

```bash
python3 watch_agent.py --model_name /models/exp/1_model.zip --scenario academy_empty_goal_close
```



O código implementado utilizou curriculum learning para aumentar a dificuldade dos bots gradualmente dividido em stages, o melhor modelo deve ser treinado em sequência pelo run.sh, além disso foi implementado "Early Stopping e Learning Rate Decay", assim como métricas explicitas para melhorar o entendimento do treinamento.




