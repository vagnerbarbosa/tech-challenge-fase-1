from pipeline import preprocessing, visualization, modeling, evaluation
from pipeline.data_split import split_and_oversample  # NOVO

def main():
    # 1. Carregar dados
    df = preprocessing.load_data('data/diabetes.csv')

    # 2. Pré-processar dados (limpeza + scaling)
    df_ready, X_scaled, y, scaler = preprocessing.preprocess_diabetes_data(df)

    # 3. Visualização
    visualization.plot_target_distribution(df_ready)
    visualization.plot_histograms(df_ready)
    visualization.plot_correlation_matrix(df_ready)

    # 4. Split e Oversampling
    X_train_res, X_test, y_train_res, y_test = split_and_oversample(X_scaled, y)

    # 5. Modelagem e avaliação
    models = modeling.get_models()
    models = modeling.train_models(models, X_train_res, y_train_res)
    evaluation.evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()