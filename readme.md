# Pipeline de Análise e Previsão de Clientes

## 1. Visão Geral do Projeto

Este projeto implementa um pipeline de dados automatizado para analisar o comportamento de clientes e gerar previsões de compra. O pipeline é construído com **Python**, orquestrado com **Docker Compose** e automatizado via **GitHub Actions**.

O objetivo é:
* Tratar e limpar dados brutos.
* Segmentar clientes em grupos distintos (clusters) com base em seu histórico de compras.
* Construir modelos de Machine Learning para prever a próxima compra de um cliente.

## 2. Fluxo de Trabalho (Pipeline)

O pipeline é executado em sequência e consiste nos seguintes scripts:

1.  **`tratar_dados.py`**: Lê o arquivo de dados brutos (`df_amostragem.csv`), remove os hashes das colunas e salva o resultado limpo em `df_tratado.csv`.
2.  **`problema1.py`**: Lê os dados tratados, calcula as métricas RFM (Recência, Frequência, Valor Monetário), aplica a clusterização K-Means para segmentar os clientes e salva o resultado com a identificação dos clusters em `resultado_p1.csv`.
3.  **`problema2.py`**: Lê os dados segmentados, cria as variáveis-alvo (classificação e regressão) e treina dois modelos de Machine Learning (XGBoost) para prever a probabilidade de uma próxima compra e o número de dias até ela. O resultado é salvo em `resultado_p2.csv`.

## 3. Estrutura do Projeto

A estrutura de pastas e arquivos do repositório é a seguinte:

```
challenge-clickbus/
├── .github/
│   └── workflows/
│       └── pipeline.yml       # Workflow do GitHub Actions
├── data/
│   ├── df_t.csv               # Dados brutos
│   ├── df_amostragem.csv      # Dados para teste rápido
│   ├── df_tratado.csv         # Dados tratados (remoção de hashes e sujeiras)
│   └── resultados/            # Arquivos de saída
│       ├── resultado_p1.csv
│       └── resultado_p2.csv
├── scripts/
│   ├── tratar_dados.py
│   ├── problema1.py
│   └── problema2.py
├── requirements.txt           # Dependências do Python
├── Dockerfile                 # Imagem do contêiner
└── docker-compose.yml         # Orquestração do contêiner
```

## 4. Pré-requisitos

Para rodar o projeto localmente, você precisa ter o **Git** e o **Docker** ou **Podman** instalados em sua máquina.

* Para usar o Podman, instale também o `podman-compose` com `pip install podman-compose`.
* Por possuir arquivos grandes (>100MB), é necessário utilizar o **Git LFS**.

## 5. Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/emissilva/challenge-clickbus
    cd challenge-clickbus
    ```
2.  **Inicie o pipeline:**
    Execute este comando para construir a imagem e rodar o pipeline.
    ```bash
    podman-compose up --build
    ```
3.  **Verifique os resultados:**
    Após a execução, os arquivos de saída (`resultado_p1.csv` e `resultado_p2.csv`) estarão na pasta `data/resultados/`.

## 6. Modelos e Análise

* **Segmentação de Clientes**: O `problema1.py` usa o algoritmo **K-Means** para agrupar clientes.
* **Previsão de Compra**: O `problema2.py` usa dois modelos **XGBoost**:
    * **`XGBClassifier`**: Para prever se a próxima compra ocorrerá. A performance é medida pela **acurácia**.
    * **`XGBRegressor`**: Para prever o número de dias até a próxima compra. A performance é medida pelo **RMSE**.
