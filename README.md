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
docker compose up -d --build
```


Entrar dentro do container para executar o código

```bash
docker compose exec gfootball bash
```

Rodar o código python desejado dentro do container

```bash
ls
python3 train_ppo.py
```

