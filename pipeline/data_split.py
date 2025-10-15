from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def split_and_oversample(X, y, test_size=0.2, random_state=42, stratify=True, smote_random_state=42):
    """
    Faz split dos dados em treino/teste e aplica SMOTE apenas nos dados de treino.

    Parâmetros:
        X (array ou DataFrame): Features já processadas/escaladas.
        y (array ou Series): Labels/variável alvo.
        test_size (float): Proporção de teste.
        random_state (int): Semente para reprodutibilidade.
        stratify (bool): Se True, faz stratify em y.
        smote_random_state (int): Semente do SMOTE.

    Retorna:
        X_train_res, X_test, y_train_res, y_test
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    smote = SMOTE(random_state=smote_random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test