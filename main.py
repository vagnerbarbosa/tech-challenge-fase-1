# Importando módulos do pipeline (organização do projeto)
from pipeline import preprocessing
from pipeline import visualization
from pipeline import modeling
from pipeline import evaluation

def main():
    # 1. Carregar e limpar dados
    print("🔹 Carregando os dados...")
    df = preprocessing.load_data('data/diabetes.csv')
    print("Primeiras linhas do dataset:")
    print(df.head())
    print("\nInformações gerais do dataset:")
    print(df.info())

    print("\n🔹 Limpando dados inválidos (valores zero em colunas clínicas)...")
    df = preprocessing.clean_data_remove_invalid(df)
    print("Dados após limpeza:")
    print(df.describe())

    # 2. Visualização dos dados
    print("\n🔹 Visualizando a distribuição do diagnóstico (target)...")
    visualization.plot_target_distribution(df)
    print("\n🔹 Visualizando histogramas das variáveis...")
    visualization.plot_histograms(df)
    print("\n🔹 Visualizando a matriz de correlação...")
    visualization.plot_correlation_matrix(df)

    # 3. Pré-processamento: separação e padronização dos dados
    print("\n🔹 Separando dados em treino e teste, e aplicando padronização...")
    X_train, X_test, y_train, y_test = preprocessing.split_and_scale(df)

    # 4. Modelagem: definição e treinamento dos modelos
    print("\n🔹 Definindo e treinando modelos de classificação...")
    models = modeling.get_models()
    models = modeling.train_models(models, X_train, y_train)

    # 5. Avaliação dos modelos
    print("\n🔹 Avaliando o desempenho dos modelos...")
    evaluation.evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()