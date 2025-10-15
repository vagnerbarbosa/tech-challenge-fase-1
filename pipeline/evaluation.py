from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(models, X_test, y_test, threshold=0.3):
    """Avalia múltiplos modelos de classificação e exibe métricas e matriz de confusão."""
    for name, model in models.items():
        # Verifica se existe o método predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        else:
            # fallback para modelos sem predict_proba
            y_pred = model.predict(X_test)
        print(f"\n--- {name} ---")
        print(f"Accuracy (Acurácia): {accuracy_score(y_test, y_pred):.2f}")
        print(f"Recall (Sensibilidade): {recall_score(y_test, y_pred):.2f}")
        print(f"F1-score: {f1_score(y_test, y_pred):.2f}")
        print("\nRelatório de classificação detalhado:")
        print(classification_report(y_test, y_pred))
        print("Matriz de Confusão (linhas = real, colunas = predito):")
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {name} (Threshold={threshold})')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.show()