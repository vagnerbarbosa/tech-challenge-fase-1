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

1. Clone este repositório:
   ```bash
   git clone https://github.com/vagnerbarbosa/tech-challenge-fase-1.git
   cd tech-challenge-fase-1
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

### 🧮 Interpretação das Métricas

- **Acurácia:** Percentual de diagnósticos corretos.
- **Recall (Sensibilidade):** Percentual de pacientes com diabetes corretamente identificados. Essencial para evitar diagnósticos falsos negativos.
- **F1-score:** Equilíbrio entre precisão e recall, importante em casos de desbalanceamento de classes.
- **Matriz de Confusão:** Permite visualizar detalhadamente os acertos e erros do modelo.

---

## Limitações do Modelo e do Dataset
- O modelo foi treinado em dados públicos, que podem não representar toda a população.
- Possível desbalanceamento entre classes (ex: mais negativos do que positivos).
- Não foi realizada validação clínica direta.
- O dataset não contém variáveis socioeconômicas ou contextuais relevantes.

---

## ⚠️ Observações

- O uso do Docker evita problemas de compatibilidade de bibliotecas e versões do Python.
- Para execução local, garanta que a versão do Python seja **3.10 ou 3.11**, pois versões superiores podem causar erros de compatibilidade com algumas bibliotecas.

---

## Conformidade com a LGPD
Este projeto utiliza apenas dados públicos e anonimizados, em conformidade com a Lei Geral de Proteção de Dados (LGPD).  
Nenhuma informação pessoal identificável é processada.  
Para uso em ambientes reais, recomenda-se revisão jurídica e anonimização rigorosa dos dados.

---

## Segurança e Ética dos Dados
Todos os dados utilizados são públicos e anonimizados (Fonte: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)).  
O projeto segue boas práticas para evitar vazamento ou reidentificação, mesmo em ambiente local.  
**Recomendação:** Em projetos reais, é imprescindível anonimizar informações sensíveis e aderir à [LGPD](https://www.gov.br/cidadania/pt-br/acesso-a-informacao/lgpd).
