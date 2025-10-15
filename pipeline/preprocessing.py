import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def load_data(path):
    """Carrega o dataset a partir de um arquivo CSV."""
    return pd.read_csv(path)

def preprocess_diabetes_data(df):
    """
    Pré-processamento:
    - Remove SkinThickness e Insulin
    - Trata zeros em Glucose, BloodPressure, BMI (converte para NaN e imputa mediana)
    - Mantém zeros em Pregnancies
    - Aplica RobustScaler (exceto na coluna target)
    """
    df = df.drop(['SkinThickness', 'Insulin'], axis=1)
    cols_to_treat = ['Glucose', 'BloodPressure', 'BMI']
    for col in cols_to_treat:
        df[col] = df[col].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df[cols_to_treat] = imputer.fit_transform(df[cols_to_treat])
    X = df.drop('Outcome', axis=1)
    y = df['Outcome'].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, y, scaler