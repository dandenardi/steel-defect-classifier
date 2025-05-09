import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/raw/bootcamp_train.csv")

def load_data(path):
    return pd.read_csv(path)

def show_basic_info(df):
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Columns ---")
    print(df.columns)
    print("\n--- Missing values---")
    print(df.isnull().sum())

def plot_target_distribution(df):
    target_cols = [col for col in df.columns if col.startswith("falha_")]
    
    target_df = df[target_cols].apply(pd.to_numeric, errors="coerce")
    
    target_df.sum().sort_values(ascending=False).plot(
        kind="bar", title="Distribuição de Tipos de Falhas"
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df=load_data(DATA_PATH)
    show_basic_info(df)
    plot_target_distribution(df)