import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

DATA_PATH = Path("data/raw/bootcamp_train.csv")
PROCESSED_PATH = Path("data/processed/botcamp_train_processed.csv")

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop_duplicates()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    

    return df

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
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

def save_processed_data(df, path):
    df.to_csv(path, index=False)
    print(f"Datos processados salvos em: {path}")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    print("Dados carregados com sucesso!")

    df = clean_data(df)
    print("Limpeza dos dados concluída!")

    df = handle_missing_values(df)
    print("Valores ausentes tratados!")

    df = encode_categorical_columns(df)
    print("Variáveis categóricas codificadas!")

    df = scale_features(df)
    print("Escalonamento das features concluído!")

    save_processed_data(df, PROCESSED_PATH)