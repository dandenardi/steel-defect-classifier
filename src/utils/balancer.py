
from imblearn.over_sampling import SMOTE

def balance_data(X_train, y_train):
    """
    Applies balancing in classes using SMOTE
    """

    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("Distribuição de classes após SMOTE:")
    print(y_res.sum(axis=0))
    return X_res, y_res