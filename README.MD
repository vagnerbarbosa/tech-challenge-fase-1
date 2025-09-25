# Diagnóstico de Diabetes com Machine Learning

## Como rodar

### Pré-requisitos

- Python: Certifique-se de ter o [Python](https://www.python.org/downloads/release/python-3119/) instalado na versão 3.11 ou inferior.
- Jupyter: O [Jupyter](https://docs.jupyter.org/pt-br/latest/install/notebook-classic.html) Notebook deve estar instalado e corretamente configurado nas variáveis de ambiente do Python.
- Docker (opcional): Caso opte por executar via [Docker](https://docs.docker.com/engine/install/), garanta que o Docker esteja devidamente instalado e configurado em seu ambiente.

### Passo a passo

1. Clone o repositório
2. Instale as dependências: `pip install --only-binary=:all: -r requirements.txt`
3. Execute o notebook: `jupyter notebook Diabetes_Analysis.ipynb`
4. Ou rode via Docker:

```bash
docker build -t diabetes-ml . && docker run -p 8888:8888 diabetes-ml
```

## Dataset - [Link para download](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data) 

## Resultados - Accuracy: XX% - Recall: XX% - F1-score: XX%

## Importante
Por questões de compatibilidade entre as versões das bibliotecas utilizadas, certifique-se de que o Python 3.10 ou 3.11 esteja instalado em seu ambiente.