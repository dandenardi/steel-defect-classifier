import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from utils.data_loader import load_data

DATA_PATH = Path("../data/raw/bootcamp_train.csv")
PROCESSED_PATH = Path("../data/processed/cleaned_data.csv")


def clean_data(df):
    df = df.drop_duplicates()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    

    return df

def clean_failure_column(value):
    """
    Cleans and converts failure values, converting them to 0 or 1
    """

    if value in ['FALSE', 'nao', 'não', '0', 'S', 'FALSE', 'não']:
        return 0
    elif value in ['TRUE', 'Sim', 'y', '1']:
        return 1
    return value

def handle_missing_values(df):
    imputer = SimpleImputer(strategy="mean")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df

def encode_categorical_columns(df):
    label_encoder = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    
    return df

def scale_features(df):
    scaler = StandardScaler()

    # Identificar colunas que não são alvo
    target_cols = [col for col in df.columns if col.startswith("falha_")]
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    feature_cols = [col for col in numeric_cols if col not in target_cols]

    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df

def transform_target(df):
    target_cols = [col for col in df.columns if col.startswith("falha_")]

    # Limpeza dos valores das falhas
    for col in target_cols:
        df[col] = df[col].apply(clean_failure_column)

    # Agora, convertendo para binário (0 ou 1)
    df[target_cols] = df[target_cols].apply(pd.to_numeric, errors='coerce')

    df[target_cols] = df[target_cols].applymap(lambda x: 1 if pd.notnull(x) and (x is True or x > 0) else 0)
    df[target_cols] = df[target_cols].astype(int)

    return df

def save_processed_data(df, path):
    df.to_csv(path, index=False)
    print(f"Datos processados salvos em: {path}")




if __name__ == "__main__":
    df = load_data(DATA_PATH)
    print("Dados carregados com sucesso!")

    df = clean_data(df)
    print("Limpeza dos dados concluída!")

    df = transform_target(df)
    print("Transformação dos alvos (falhas) concluída!")

    df = handle_missing_values(df)
    print("Valores ausentes tratados!")

    df = encode_categorical_columns(df)
    print("Variáveis categóricas codificadas!")

    df = scale_features(df)
    print("Escalonamento das features concluído!")

    save_processed_data(df, PROCESSED_PATH)

   