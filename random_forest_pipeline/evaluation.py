from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd

def evaluate_models(models, X_test, y_test, feat_names, threshold=0.3):
    """
    Avalia múltiplos modelos de classificação, exibe métricas e matriz de confusão,
    e retorna o modelo com o melhor Recall.
    """
    best_recall = -1
    best_f1 = -1
    best_model_name = None
    best_model = None

    for name, model in models.items():

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            
        current_recall = recall_score(y_test, y_pred)
        current_f1 = f1_score(y_test, y_pred)
        
        # Lógica de rastreamento do melhor modelo (Baseado no Recall)
        if current_recall > best_recall:
            best_recall = current_recall
            best_f1 = current_f1 
            best_model_name = name
            best_model = model

        print(f"\n--- {name} ---")
        print(f"Recall (Sensibilidade): {current_recall:.2f}")
        print(f"F1-score: {current_f1:.2f}")
        print("\nRelatório de classificação detalhado:")
        print(classification_report(y_test, y_pred))
        print("Matriz de Confusão (linhas = real, colunas = predito):")
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {name} (Threshold={threshold})')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.show()

    print("==================================================")
    print(f"🏆 Melhor Modelo Encontrado ({best_model_name})")
    print(f"   Critério de Seleção: Maior Recall")
    print(f"   Recall (Sensibilidade): {best_recall:.2f}")
    print(f"   F1-score: {best_f1:.2f}")
    print("==================================================")
    
    return best_model_name, best_model

def interpret_best_model(best_model, X_test_df, feat_names):
    """
    Realiza a interpretação do melhor modelo utilizando Feature Importance e SHAP.
    """
    
    print("\n\n=== 🔎 INTERPRETAÇÃO DO MELHOR MODELO ===")

    # 1. FEATURE IMPORTANCE CLÁSSICA (Para modelos baseados em árvore)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis")
        plt.title(f'Importância das Features - {type(best_model).__name__}')
        plt.show()
        
        print("\n--- Top 5 Features (Importância Clássica) ---")
        print(feature_importance.head())
    
    # 2. ANÁLISE SHAP
    try:
        if type(best_model).__name__ in ['RandomForestClassifier', 'XGBClassifier', 'DecisionTreeClassifier']:
            explainer = shap.TreeExplainer(best_model)
        else:
            explainer = shap.Explainer(best_model, X_test_df) 
        
        shap_values = explainer.shap_values(X_test_df)
        
        if isinstance(shap_values, list):
            # Focamos na classe positiva (diabetes=1)
            shap_values = shap_values[1] 

        # SHAP Summary Plot (Importância e Impacto Global)
        print("\n--- SHAP Summary Plot (Importância e Impacto Global) ---")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_df, feature_names=feat_names, show=False)
        plt.title('SHAP Summary Plot (Impacto Global)')
        plt.tight_layout()
        plt.show()

        # SHAP Bar Plot (Média da Magnitude do Impacto)
        print("\n--- SHAP Bar Plot (Importância Média Absoluta) ---")
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_df, feature_names=feat_names, plot_type="bar", show=False)
        plt.title('SHAP Bar Plot (Importância Média Absoluta)')
        plt.tight_layout()
        plt.show()
        
        print("\nInterpretação SHAP: Pontos vermelhos (alto valor da feature) à direita do zero aumentam a chance de diabetes. Pontos azuis (baixo valor) à direita do zero também aumentam a chance.")
        
    except Exception as e:
        print(f"\nNão foi possível rodar a análise SHAP para este modelo ({type(best_model).__name__}): {e}") 
