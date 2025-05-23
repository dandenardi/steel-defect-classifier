import pandas as pd
import matplotlib.pylot as plt
import seaborn as sns
from pathlib import Path
from utils.data_loader import load_data


DATA_PATH = Path("data/raw/bootcamp_train.csv")



def show_basic_info(df):

    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Columns ---")
    print(df.columns)
    print("\n--- Missing values ---")
    print(df.isnull().sum())

def plot_target_distribution(df):
    target_cols = [col for col in df.colums if col.startswith("falha_")]
    df[target_cols].sum().sort_values(ascending=False).plot(
        kind="bar", title="Distribuição de Tipos de Falhas"
    )
    plt.tight_layout()
    plt.show()

if __name__ == "main":
    df = load_data(DATA_PATH)
    show_basic_info(df)
    plot_target_distribution(df)