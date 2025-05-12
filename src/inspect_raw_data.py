import pandas as pd

# Caminho para a base original (ajuste conforme necessário)
RAW_PATH = "../data/raw/bootcamp_train.csv"

# Lê o CSV original
df = pd.read_csv(RAW_PATH)

print("=== Primeiras linhas ===")
print(df.head())

print("\n=== Colunas disponíveis ===")
print(df.columns)

for col in df.columns:
    print(f"Coluna: {col}")
    print(df[col].unique())
    print()

falha_columns = [col for col in df.columns if col.startswith('falha_')]

# Exibe as primeiras linhas das colunas filtradas
print(df[falha_columns].head())

# Verifica valores únicos nas colunas de falhas (ajuste para suas colunas reais)
possible_label_columns = [col for col in df.columns if 'falha' in col.lower() or 'defect' in col.lower()]
print("\n=== Colunas candidatas a labels ===")
print(possible_label_columns)

# Converte colunas de falhas para 1 (presença) ou 0 (ausência), considerando 'Sim' e 'Não'
for col in possible_label_columns:
    # Substitui 'Sim' por 1 e 'Não' por 0
    df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == 'sim' else (0 if str(x).strip().lower() == 'não' else x))

# Frequência de cada tipo de falha (multi-label possível)
print("\n=== Frequência total por label ===")
for col in possible_label_columns:
    print(f"{col}: {df[col].sum()}")

# Verifica quantos exemplos têm múltiplas falhas
df['num_falhas'] = df[possible_label_columns].sum(axis=1)
print("\n=== Amostras com múltiplas falhas ===")
print(df['num_falhas'].value_counts())

# Verifica se há linhas com labels vazias
print("\n=== Amostras sem falhas marcadas ===")
print(len(df[df['num_falhas'] == 0]))
