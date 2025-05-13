from pathlib import Path
import pandas as pd
from typing import Union

def load_data(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"O arquivo {path} não foi encontrado.")
    return pd.read_csv(path)

def load_model(model_path):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"O arquivo {model_path} não foi encontrado")
    return pd.read_csv(model_path)