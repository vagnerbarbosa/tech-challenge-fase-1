# 1. Módulo de Pré-Processamento
import pandas as pd
import numpy as np  # Adicione esta linha
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        df[col] = df[col].astype(float)  
        df[col] = df[col].replace(0, np.nan)
    df.fillna(df.median(), inplace=True)
    return df

def split_and_scale(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test