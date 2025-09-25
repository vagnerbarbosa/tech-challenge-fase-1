# 🧪 Diagnóstico de Diabetes com Machine Learning

Este projeto utiliza técnicas de machine learning para prever o diagnóstico de diabetes com base em dados clínicos. A maneira **recomendada** de execução é via **Docker**, garantindo um ambiente isolado e livre de conflitos de dependências.

---

## 🚀 Como Executar com Docker (Recomendado)

### ✅ Pré-requisitos

- Ter o [Docker](https://docs.docker.com/engine/install/) instalado e funcionando corretamente.

### ▶️ Passos

1. Clone este repositório:
   ```bash
   git clone https://github.com/vagnerbarbosa/tech-challenge-fase-1.git
   cd tech-challenge-fase-1
   ```

2. Construa a imagem Docker:
   ```bash
   docker build -t diabetes-ml .
   ```

3. Execute o container:
   ```bash
   docker run -p 8888:8888 diabetes-ml
   ```

4. Acesse o Jupyter Notebook via navegador:
   ```
   http://localhost:8888
   ```

---

## 💻 Execução Local (Alternativa)

### ✅ Pré-requisitos

- [Python 3.10 ou 3.11](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install)

### ▶️ Passos

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Instale as dependências:
   ```bash
   pip install --only-binary=:all: -r requirements.txt
   ```

3. Execute o Jupyter Notebook:
   ```bash
   jupyter notebook Diabetes_Analysis.ipynb
   ```

---

## 📊 Dataset

- O dataset utilizado pode ser baixado [neste link do Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)

---

## 📈 Resultados Esperados

- **Accuracy:** XX%
- **Recall:** XX%
- **F1-Score:** XX%

---

## ⚠️ Observações

- O uso do Docker evita problemas de compatibilidade de bibliotecas e versões do Python.
- Para execução local, garanta que a versão do Python seja **3.10 ou 3.11**, pois versões superiores podem causar erros de compatibilidade com algumas bibliotecas.
