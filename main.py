import preprocessing
import visualization
import modeling
import evaluation

def main():
    # 1. Carregar e limpar dados
    df = preprocessing.load_data('data/diabetes.csv')
    print(df.head())
    print(df.info())
    df = preprocessing.clean_data(df)

    # 2. Visualização
    visualization.plot_target_distribution(df)
    visualization.plot_histograms(df)
    visualization.plot_correlation_matrix(df)

    # 3. Pré-processamento
    X_train, X_test, y_train, y_test = preprocessing.split_and_scale(df)

    # 4. Modelagem
    models = modeling.get_models()
    models = modeling.train_models(models, X_train, y_train)

    # 5. Avaliação
    evaluation.evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()