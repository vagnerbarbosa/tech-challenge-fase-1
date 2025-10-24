from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def get_models():
    """
    Cria e retorna dicionário com modelos de classificação configurados para melhor recall.
    """    
    return {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

def train_models(models, X_train, y_train):
    for model in models.values():
        model.fit(X_train, y_train)
    return models