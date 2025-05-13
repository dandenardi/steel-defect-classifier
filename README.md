# Steel Defect Classifier

Este projeto tem como objetivo desenvolver um sistema de detecção automática de falhas em peças de aço, utilizando técnicas de ciência de dados e aprendizado de máquina. O sistema é acessível via uma aplicação interativa construída com Streamlit.

## 🔍 Problema

A inspeção de falhas em processos industriais é uma tarefa crítica. Este projeto busca aplicar modelos supervisionados de aprendizado de máquina para prever diferentes tipos de falhas com base em variáveis do processo produtivo, contribuindo para redução de retrabalho e aumento de eficiência.

## 🛠️ Tecnologias Utilizadas

Python 3.11

Pandas, NumPy

Scikit-learn

Seaborn, Matplotlib

Streamlit

RandomForestClassifier com estratégia One-vs-Rest

## 📁 Estrutura do Projeto

```bash
steel-defect-classifier/
├── data/ # Conjuntos de dados (raw e processed)
├── models/ # Modelos treinados (pkl)
├── results/ # Métricas e gráficos
├── figs_ovr/ # Imagens de métricas e matrizes de confusão
├── src/
│ └── utils/ # Scripts auxiliares
├── app.py # Aplicação principal em Streamlit
├── docs/ # Imagens e materiais para documentação
└── README.md
```

🚀 Como Executar Localmente

Clone este repositório:

```bash
git clone https://github.com/dandenardi/steel-defect-classifier.git
cd steel-defect-classifier
```

Crie e ative o ambiente virtual conda:

```bash
conda env create -f environment.yml
conda activate steell-classfier-env
```

### Execute o app:

```bash
streamlit run app.py
```

## 📈 Resultados

- Utilizou-se Random Forest com estratégia One-vs-Rest para multirrótulo.

- Dados divididos em treino/teste com avaliação por métricas de precisão, recall, F1-score e matriz de confusão.

- Classes com maior representação obtiveram melhor performance.

- Falhas com menos dados mostraram menor desempenho, indicando necessidade de balanceamento.

## ✅ Próximos Passos

- Aumentar a amostra de falhas menos frequentes.

- Testar técnicas de balanceamento como SMOTE.

- Avaliar modelos alternativos como XGBoost e LightGBM.

- Otimizar hiperparâmetros com GridSearch ou RandomSearch.

✍️ Autor
Daniel do Amaral Denardi - @dandenardi
