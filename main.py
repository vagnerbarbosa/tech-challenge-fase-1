from pipeline import preprocessing
from pipeline import visualization
from pipeline import modeling
from pipeline import evaluation

def main():
    # 1. Carregar dados
    df = preprocessing.load_data('data/diabetes.csv')

    # 2. Pré-processar dados (limpeza + scaling)
    df_ready, X_scaled, y, scaler = preprocessing.preprocess_diabetes_data(df)

    # 3. Visualização
    visualization.plot_target_distribution(df_ready)
    visualization.plot_histograms(df_ready)
    visualization.plot_correlation_matrix(df_ready)

    # 4. Split dos dados prontos para modelagem
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Modelagem e avaliação
    models = modeling.get_models()
    models = modeling.train_models(models, X_train, y_train)
    evaluation.evaluate_models(models, X_test, y_test)
    # Se quiser exportar o melhor modelo para uso futuro, implemente depois!

if __name__ == "__main__":
    main()