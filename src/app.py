import os
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from pathlib import Path
from utils.data_loader import load_data, load_model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # sobe da pasta src para a raiz
RESULTS_PATH = BASE_DIR / "results" / "resultados_ovr.csv"

RAW_DATA = Path("../data/raw/bootcamp_train.csv")
DATA_PATH = Path('../data/processed/cleaned_data.csv')
MODEL_PATH = Path("/models/random_forest_model.pkl")
FIGS_DIR = BASE_DIR / "figs_ovr"

image_path = BASE_DIR / "assets" / "unisenai.jpeg"
image = Image.open(image_path)



st.set_page_config(page_title="Classificador de Defeitos em Aço", layout="wide")

st.image(image, use_container_width=True)
st.title("Detecção de falhas em aço através de ferramentas de Ciência de dados e Inteligência Artificial")

st.header("1. Visão geral do problema")
st.markdown("""
    Este projeto tem como objetivo desenvolver um algoritmo, 
            baseado em ciência de dados e inteligência artificial, 
            que permita a identificação de falhas em peças de aço. 
            Tal projeto tem relevância sobretudo no ambiente industrial, no qual
            a identificação prematura de falhas na produção tende a impactar de maneira
            relevante indicadores globais de desempenho, diminuição de retrabalho e melhora
            do retorno financeiro.
            
""")

st.header("2. Análise Exploratória & Visualização")
df = load_data(RAW_DATA)
target_cols = [col for col in df.columns if col.startswith("falha_")]

st.markdown("""
Considerando que o foco do projeto são as colunas "falhas", 
    é importante verificá-las.
""")

st.subheader("Amostra dos dados")
st.dataframe(df.head())


st.subheader("Tipos de dados nas colunas de falhas")
st.write(df[target_cols].dtypes)

st.markdown("""
Percebe-se a presença de tipos "object" o que denota que os dados podem conter tipos diversos como strings.
Dados assim podem ser difíceis de analisar, sem nenhum tratamento.
""")

st.subheader("Valores únicos por coluna de falha")
for col in target_cols:
    unique_vals = df[col].unique()
    st.write(f"{col}: {unique_vals}")

st.markdown("""
Observando-se os valores únicos em cada coluna, fica confirmado o polimorfismo dos dados.
""")

st.subheader("Valores ausentes nas colunas de falha")
st.write(df[target_cols].isna().sum())

st.markdown("""
    
    A análise prévia dos dados demonstrou a necessidade de padronização, já que foram 
    identificadas **inconsistência nos tipos**. A análise de colunas com tipos
    de dados variados dificulta a análise.
    Foram realizados procedimentos para adequar os dados de modo que todas as
    colunas de falha contivessem dados binários.
""")

st.header("3. Limpeza e Pré-processamento")
st.markdown("""
    Após análise, foi delineado um plano de limpeza e pré-processamento.
    As rotinas de pré-processamento incluiram:
            - Remoção de colunas irrelevantes
            - Normalização de valores ausentes
            - Conversão de variáveis categóricas
    
    Após estes procedimentos, obteve-se um novo conjunto de dados mais compatível
            com as análises planejadas. Uma amostra dos dados do novo conjunto demonstra
            que os dados agora podem ser quantificados, permitindo ter uma visão mais
            fidedigna da ocorrência de falhas.
""")
df_clean = load_data(DATA_PATH)
st.subheader("Amostra dos dados tratados")
st.dataframe(df_clean.head())

st.subheader("Distribuição de falhas após limpeza")
target_cols_clean = [col for col in df_clean.columns if col.startswith("falha_")]
fail_counts_clean = df_clean[target_cols_clean].sum().sort_values(ascending=False)
st.bar_chart(fail_counts_clean)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Antes da limpeza**")
    target_cols_raw = [col for col in df.columns if col.startswith("falha_")]

    # Convertemos todas as colunas de falhas para numérico, forçando valores inválidos como NaN
    df[target_cols_raw] = df[target_cols_raw].apply(pd.to_numeric, errors="coerce")
    # Substituímos NaNs por 0 e convertendo para inteiro
    df[target_cols_raw] = df[target_cols_raw].fillna(0).astype(int)

    fail_counts_raw = df[target_cols_raw].sum().sort_values(ascending=False)
    st.bar_chart(fail_counts_raw)


with col2:
    st.markdown("**Depois da limpeza**")
    fail_counts_clean = df_clean[target_cols_clean].sum().sort_values(ascending=False)
    st.bar_chart(fail_counts_clean)


st.header("4. Metodologia de Treinamento e Avaliação")

results_df = pd.read_csv(RESULTS_PATH)


st.markdown("""
Considerando que o projeto tem uma característica de multirrótulo, já que cada peça
    pode conter mais de uma falha, foi escolhido o algoritmo **Random Forrest**.
    Este algoritmo e baseado em árvore de decisão e lida bem com o cenário de 
    rotulação diversa.
    
Os dados foram divididos na proporção de **80% para treino** e **20% para teste**,
    de modo a garantir que dados não vistos durante o aprendizado fossem utilizados para
    avaliação do modelo. Para lidar com o desbalanceamento nas classes de falha, foi utilizada
    a ponderação de classe, aplicando o parâmetro `class_weight=balanced`.
            
Para avaliação de desempenho utilizou-se métricas de:
    - **Precisão (Precision)**: prever corretamente apenas classes verdadeiras.
    - **Revocação (Recall)**: capacidade de capturar todas as ocorrências reais de cada classe.
    - **F1-Score**: média harmônica entre precisão e revocação.
    - **Matriz de Confusão**: Desempenho por classe.

""")

st.subheader("Métricas por classe")
st.dataframe(results_df)

metric_option = st.selectbox("Selecione a métrica para visualização:",["accuracy", "precision", "recall", "f1"])
chart_path = FIGS_DIR / f"{metric_option}_ovr.png"
st.image(chart_path, caption=f"Métrica: {metric_option.upper()} for falha", use_container_width=True)

st.subheader("Matriz de Confusão por Classe")
selected_class = st.selectbox("Selecione uma classe de falha:", results_df["falha"].tolist())
cm_path = f"figs_ovr/confusion_matrix_{selected_class}.png"

if os.path.exists(cm_path):
    st.image(cm_path, caption=f"Matriz de Confusão - {selected_class}", use_container_width=False)
else:
    st.warning("Imagem da matriz não encontrada.")

st.header("5. Considerações Finais e Próximos Passos")

st.subheader("Sobre os resultados")
st.markdown("""
Analisando os resultados, percebe-se que o modelo treinado se sai muito bem nas falhas do tipo 1 e 3. 
    Para as demais, não tanto, visto que exibem métricas baixas de *recall* e *f1-score*.
    Isto pode ter relação com o **desequilíbrio entre as classes**. As falhas em que o sistema tem
    dificuldade em identificar são justamente aquelas com menos amostras no conjunto de dados.
""")

st.subheader("Aprendizados")
st.markdown("""
Durante o desenvolvimento ficou evidenciada a importância de limpeza e tratamento dos
    dados. Houve um momento no desenvolvimento em que os resultados sinalizavam para problemas
    que poderiam ser confundidos com escolha errada de algoritmo. No entanto, uma verificação
    mais cuidadosa mostrou que o problema estava, de fato, nos dados não estarem num formato
    que permitisse a análise. Ou seja, aplicar mais tempo no processamento dos dados é fundamental
    e se justifica na qualidade final do projeto.
Também observou-se o quanto é importante fazer o balanceamento de classes em um problema como este,
    em que se tem o cenário de multirrótulos. Isso também tem impacto na decisão sobre qual modelo utilizar.

""")

st.subheader("Limitações")
st.markdown("""
A solução encontrada, conforme mencionado, possui limitações para identificação de algumas falhas,
    devido a menor representação delas no conjunto. Foram consideradas possibilidades para endereçar
    estas limitações, como SMOTE e ajuste de hiperparâmetros. No entanto, os testes envolvendo estes
    ajustes demonstraram um aumento na complexidade, o que demandaria maior tempo para refinamento da
    solução.
""")

st.subheader("Próximos Passos")
st.markdown("""
Considerando as limitações observadas, um ponto relevante de melhoria seria a coleta
    de mais dados para as falhas com menor representação. Também seria relevante testes
    com modelos alternativos como XGBoost ou LightGBM. O balanceamento sintético também
    seria uma possibilidade para aprimoramento do modelo.
""")