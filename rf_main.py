from random_forest_pipeline import preprocessing, visualization, modeling, evaluation
from random_forest_pipeline.data_split import split_and_oversample
import pandas as pd

def main():
    # 1. Carregar dados
    df = preprocessing.load_data('random_forest_pipeline/data/diabetes.csv')

    # 2. Pré-processar dados (limpeza + scaling)
    df_ready, X_scaled, y, scaler, feat_names = preprocessing.preprocess_diabetes_data(df)

    # 3. Visualização
    visualization.plot_target_distribution(df_ready)
    visualization.plot_histograms(df_ready)
    visualization.plot_correlation_matrix(df_ready)

    # 4. Split e Oversampling
    X_train_res, X_test, y_train_res, y_test = split_and_oversample(X_scaled, y)

    # 5. Modelagem e treinamento
    models = modeling.get_models()
    models = modeling.train_models(models, X_train_res, y_train_res)
    
    # 6. Avaliação e determinação do melhor modelo
    best_model_name, best_model = evaluation.evaluate_models(models, X_test, y_test, feat_names)

    # 7. Interpretação do Melhor Modelo
    if best_model:
        X_test_df = pd.DataFrame(X_test, columns=feat_names) #
        evaluation.interpret_best_model(best_model, X_test_df, feat_names)
    else:
        print("Nenhum modelo foi avaliado ou encontrado para interpretação.")

if __name__ == "__main__":
    main()