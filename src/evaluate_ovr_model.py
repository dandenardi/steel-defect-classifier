import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix


df = pd.read_csv("../data/processed/cleaned_data.csv")
X = df.drop(columns=[col for col in df.columns if col.startswith("falha_")])
y = df[[col for col in df.columns if col.startswith("falha_")]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = {
    "falha": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

for i, col in enumerate(y.columns):
    metrics["falha"].append(col)
    metrics["accuracy"].append(accuracy_score(y_test[col], y_pred[:,i]))
    metrics["precision"].append(precision_score(y_test[col], y_pred[:,i], zero_division=0))
    metrics["recall"].append(recall_score(y_test[col], y_pred[:,1], zero_division=0))
    metrics["f1"].append(f1_score(y_test[col], y_pred[:,1], zero_division=0))

results_df = pd.DataFrame(metrics)
results_df.to_csv("resultados_ovr.cvs", index=False)
print("Resultados salvos em resultados_ovr.csv")

output_dir = Path("figs_ovr")
output_dir.mkdir(exist_ok=True)

for metric in ["accuracy", "precision", "recall", "f1"]:
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["falha"], results_df[metric])
    plt.title(f"Métrica: {metric.upper()} por Falha(OneVsRest)")
    plt.xticks(rotation=45)
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_ovr.png")
    plt.close()
    print(f"Gráfico {metric.upper()} salvo em {output_dir}/{metric}_ovr.png")

conf_matrices = multilabel_confusion_matrix(y_test, y_pred)

for i, col in enumerate(y.columns):
    cm = conf_matrices[i]
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.title(f'Matriz de Confusão - {col}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{col}.png')
    plt.close()
    print(f"Matriz de confusão salva: {output_dir}/confusion_matrix_{col}.png")
