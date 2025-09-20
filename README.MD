# Diagnóstico de Diabetes com Machine Learning

## Como rodar

1. Clone o repositório
2. Instale as dependências: `pip install --only-binary=:all: -r requirements.txt`
3. Execute o notebook: `jupyter notebook Diabetes_Analysis.ipynb`
4. Ou rode via Docker:

```bash
docker build -t diabetes-ml .
docker run -p 8888:8888 diabetes-ml
```

## Dataset - [Link para download](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data) 

## Resultados - Accuracy: XX% - Recall: XX% - F1-score: XX%

## Importante
Por questões de compatibilidade entre as versões das bibliotecas utilizadas, certifique-se de que o Python 3.10 ou 3.11 esteja instalado em seu ambiente.