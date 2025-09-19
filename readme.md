# Pipeline de Análise e Previsão de Clientes

## 1. Visão Geral do Projeto

Este projeto implementa um pipeline de dados automatizado para analisar o comportamento de clientes e gerar previsões de compra. O pipeline é construído com **Python**, orquestrado com **Docker Compose** e automatizado via **GitHub Actions**.

O objetivo é:
* Tratar e limpar dados brutos.
* Segmentar clientes em grupos distintos (clusters) com base em seu histórico de compras.
* Construir modelos de Machine Learning para prever a próxima compra de um cliente.


## 2. Fluxo de Trabalho (Pipeline)


# Pipeline ClickBus: Tratamento, Clusterização e Previsão de Compras

## Visão Geral

Este projeto implementa um pipeline de dados para análise e previsão do comportamento de clientes de passagens rodoviárias. O fluxo é totalmente automatizado em Python, com scripts para tratamento, clusterização, previsão e análise visual.

### Objetivos
- Tratar e anonimizar dados brutos de compras.
- Gerar features comportamentais e agregadas por cliente.
- Segmentar clientes via clusterização.
- Prever o próximo trecho (origem-destino) provável de cada cliente.
- Gerar outputs e gráficos para análise exploratória e validação dos modelos.

---

## Pipeline e Scripts

O pipeline é composto por etapas independentes, cada uma com seu script e outputs:

### 1. Tratamento de Dados (`scripts/desafio_1.py`)
- Lê o arquivo bruto `dados/raw/df_t.csv`.
- Filtra apenas os últimos 365 dias.
- Limpa, padroniza, anonimiza rodoviárias, empresas e clientes.
- Calcula features agregadas por cliente (recência, frequência, monetário, ticket médio, destinos únicos, etc).
- Salva o resultado em `dados/resultados/desafio_1/df_tratado.csv`.

### 2. Clusterização e Regras Simples (`scripts/desafio_2.py`)
- Lê o arquivo tratado `dados/resultados/desafio_1/df_tratado.csv`.
- Calcula previsões simples de recompra (regras de recência/frequência).
- Salva resultados em `dados/resultados/desafio_2/previsao_simples.csv`.

### 3. Previsão do Próximo Trecho (`scripts/desafio_3.py`)
- Lê o histórico detalhado (`dados/raw/df_t.csv`), filtra 365 dias, anonimiza e gera features.
- Para cada cliente, monta exemplos de previsão do próximo par origem-destino (classificação multi-classe).
- Treina modelo XGBoost multi-classe e, opcionalmente, regressão para dias até próxima compra.
- Salva as previsões de classificação (real x previsto) em `dados/resultados/desafio_3/resultados_classificacao.csv`.

### 4. Análises e Visualização (`scripts/analises/`)
- `analise.ipynb`: Notebook para análise exploratória, PCA, escolha de clusters, validação visual.
- `avaliação dos clusters.py`: Gera gráficos automáticos (distribuição, boxplots, pairplot) para análise dos clusters.

---

## Estrutura do Projeto

```
challenge-clickbus/
├── .github/workflows/pipeline.yml
├── dados/
│   ├── raw/df_t.csv
│   └── resultados/
│       ├── desafio_1/df_tratado.csv
│       ├── desafio_2/previsao_simples.csv
│       ├── desafio_3/resultados_classificacao.csv
│       └── ...
├── scripts/
│   ├── desafio_1.py
│   ├── desafio_2.py
│   ├── desafio_3.py
│   └── analises/
│       ├── analise.ipynb
│       └── avaliação dos clusters.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── readme.md
└── resumo.md
```

---

## Como Executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/emissilva/challenge-clickbus
   cd challenge-clickbus
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute os scripts na ordem desejada (exemplo):
   ```bash
   python scripts/desafio_1.py
   python scripts/desafio_2.py
   python scripts/desafio_3.py
   ```
4. Analise os resultados em `dados/resultados/` e os gráficos/notebooks em `scripts/analises/`.

---

## Observações
- Todos os scripts consideram apenas os últimos 365 dias de dados.
- Os dados e outputs intermediários ficam em `dados/resultados/`.
- O pipeline pode ser adaptado para rodar cada etapa individualmente.
- Os scripts de análise visual são opcionais, mas recomendados para validação dos clusters e previsões.

