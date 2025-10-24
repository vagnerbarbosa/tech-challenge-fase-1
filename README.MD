# 🧪 Diagnóstico de Diabetes com Machine Learning: Comparação de Pipelines de Análise

Este projeto utiliza técnicas de machine learning para prever o diagnóstico de diabetes com base em dados clínicos, **comparando duas diferentes abordagens (Pipelines A e B) de pré-processamento, modelagem e avaliação**. A maneira **recomendada** de execução é via **Docker**, garantindo um ambiente isolado e livre de conflitos de dependências.

O objetivo principal do **Notebook** (`Diabetes_Analysis.ipynb`) é:
1.  Executar o **Pipeline A (Baseline/MLOps)**: Usa KNN Imputer e um conjunto de modelos clássicos.
2.  Executar o **Pipeline B (Otimizado/Deep Dive)**: Adiciona **Feature Engineering**, remove features e inclui XGBoost, usando um `threshold` de 0.3 para priorizar o Recall.
3.  **Comparar** detalhadamente as métricas de desempenho e a importância das *features* de ambas as abordagens.

---

## 🚀 Como Executar com Docker (Recomendado)

### ✅ Pré-requisitos

- Ter o [Docker](https://docs.docker.com/engine/install/) instalado e funcionando corretamente.

### ▶️ Passos

1. Clone este repositório:
   ```bash
   git clone [https://github.com/vagnerbarbosa/tech-challenge-fase-1.git](https://github.com/vagnerbarbosa/tech-challenge-fase-1.git)
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

4. Acesse o Jupyter Notebook via navegador e abra o arquivo `Diabetes_Analysis.ipynb`:
   ```
   http://localhost:8888
   ```
> **Instrução no Notebook:** Execute todas as células sequencialmente para rodar as duas esteiras (A e B) e gerar a comparação final.

---

## 💻 Execução Local (Alternativa)

### ✅ Pré-requisitos

- [Python 3.10 ou 3.11](https://www.python.org/downloads/)
- [Jupyter Notebook](https://jupyter.org/install)

### ▶️ Passos

1. Clone este repositório:
   ```bash
   git clone [https://github.com/vagnerbarbosa/tech-challenge-fase-1.git](https://github.com/vagnerbarbosa/tech-challenge-fase-1.git)
   cd tech-challenge-fase-1
   ```

2. Instale as dependências. O arquivo `requirements.txt` deve conter as bibliotecas dos dois pipelines (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `imblearn`, `xgboost`, `kagglehub`, `shap`):
   ```bash
   pip install --only-binary=:all: -r requirements.txt
   ```

3. Execute o Jupyter Notebook e abra o arquivo:
   ```bash
   jupyter notebook Diabetes_Analysis.ipynb
   ```

---

## 📊 Dataset

- O dataset utilizado é o **Pima Indians Diabetes Dataset**, carregado diretamente do Kaggle via `kagglehub` para ambos os pipelines.
- Link da fonte: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)

---

## 📈 Resultados Chave e Comparação de Pipelines

O notebook se encerra com a Seção 4, que compara o desempenho dos modelos dos Pipelines A e B. Os resultados podem variar ligeiramente dependendo do ambiente, mas a tendência observada (o ponto central da análise) é a seguinte:

| Métrica de Desempenho | Pipeline A (Baseline) | Pipeline B (Otimizado) |
| :--- | :--- | :--- |
| **Melhor Modelo (por Recall)** | DecisionTree | Random Forest |
| **Melhor Recall Máximo** | $\approx 0.82 - 0.87$ | $\approx 0.89 - 0.91$ |
| **Melhor F1-score** | $\approx 0.65 - 0.70$ | $\approx 0.70 - 0.72$ |

### 🔍 Destaques da Comparação

- **Recall (Sensibilidade):** O Pipeline B é projetado para maximizar o Recall (detectar mais casos positivos) através do uso de **Feature Engineering** e um **`threshold` de classificação mais baixo (0.3)**. Espera-se que o Pipeline B atinja um Recall superior, indicando maior sensibilidade para o diagnóstico.
- **F1-score:** O Pipeline A pode apresentar um bom F1-score devido à imputação por KNN e à padronização robusta, enquanto o Pipeline B pode ter um **aumento no Recall à custa de uma ligeira queda na Precisão** (e, consequentemente, um F1-score potencialmente similar ou superior).
- **Interpretabilidade:** O Pipeline B tem um conjunto de *features* diferentes, incluindo variáveis de interação (`idade_bmi`, `glucose_bmi`), cuja importância será visível nos gráficos de **Feature Importance e SHAP**, provando o valor das etapas de otimização.

---

### 🧮 Interpretação das Métricas

- **Recall (Sensibilidade):** Percentual de pacientes com diabetes corretamente identificados. **Essencial** para evitar diagnósticos falsos negativos (FN).
- **Precision (Precisão):** Percentual de previsões positivas que estão corretas. Reduz diagnósticos falsos positivos (FP).
- **F1-score:** Equilíbrio entre precisão e recall, importante em casos de desbalanceamento de classes.
- **Matriz de Confusão:** Permite visualizar detalhadamente os acertos e erros do modelo.

---

## Limitações dos Modelos e do Dataset
- O modelo foi treinado em dados públicos, que podem não representar toda a população.
- Possível desbalanceamento entre classes (o problema é mitigado pelo uso de SMOTE em ambos os pipelines).
- Não foi realizada validação clínica direta.
- O dataset não contém variáveis socioeconômicas ou contextuais relevantes.

---

## ⚠️ Observações

- **Compatibilidade Python:** Para execução local, garanta que a versão do Python seja **3.10 ou 3.11**, pois versões superiores podem causar erros de compatibilidade com algumas bibliotecas de ML/Visualização.
- O uso do Docker é a forma mais robusta de garantir um ambiente sem conflitos.

---

## Conformidade e Ética dos Dados
Este projeto utiliza apenas dados públicos