import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/processed/cleaned_data.csv")
label_cols = [col for col in df.columns if col.startswith("falha_")]

label_counts = df[label_cols].sum().sort_values(ascending=False)

print(label_counts)

# Visualize
plt.figure(figsize=(10, 6))
label_counts.plot(kind="bar")
plt.title("Distribuição de falhas no dataset")
plt.ylabel("Número de amostras")
plt.xlabel("Falhas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()