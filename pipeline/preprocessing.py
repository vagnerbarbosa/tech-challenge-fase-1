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
    - Feature engineering: cria variáveis de risco
    - Aplica RobustScaler (exceto na coluna target)
    """
    # 1. Remove colunas
    df = df.drop(['SkinThickness', 'Insulin'], axis=1)
    
    # 2. Trata zeros biologicamente impossíveis
    cols_to_treat = ['Glucose', 'BloodPressure', 'BMI']
    for col in cols_to_treat:
        df[col] = df[col].replace(0, np.nan)
    imputer = SimpleImputer(strategy='median')
    df[cols_to_treat] = imputer.fit_transform(df[cols_to_treat])

    # 3. FEATURE ENGINEERING: Variáveis de risco
    df['idade_maior_45'] = (df['Age'] >= 45).astype(int)
    df['imc_obeso'] = (df['BMI'] >= 30).astype(int)
    df['idade_bmi'] = df['Age'] * df['BMI']  # interação
    df['glucose_bmi'] = df['Glucose'] * df['BMI']  # interação

    # 4. Separa features e target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome'].values

    # 5. Scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X_scaled, y, scaler