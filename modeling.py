from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def get_models():
    return {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

def train_models(models, X_train, y_train):
    for model in models.values():
        model.fit(X_train, y_train)
    return models