
# Pipeline de Análise e Previsão de Clientes

## 1. Visão Geral do Projeto

Este projeto implementa um pipeline de dados automatizado para analisar o comportamento de clientes e gerar previsões de compra. O pipeline é construído com **Python**, orquestrado com **Docker Compose** e automatizado via **GitHub Actions**.

O objetivo é:
* Tratar e limpar dados brutos.
* Segmentar clientes em grupos distintos (clusters) com base em seu histórico de compras.
* Construir modelos de Machine Learning para prever a próxima compra de um cliente.
* Gerar outputs intermediários e finais para análise exploratória e validação dos modelos.

---

## 2. Fluxo de Trabalho (Pipeline)



### 2.1. Estrutura do Projeto

```
challenge-clickbus/
├── dados/
│   ├── raw/df_t.csv
│   └── resultados/
│       ├── desafio_1/
│       │   ├── detalhado_tratado.csv
│       │   └── df_tratado.csv
│       ├── desafio_2/
│       │   └── previsao_simples.csv
│       ├── desafio_3/
│       │   ├── saida_completa.log
│       │   └── resultados_classificacao.csv
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

## 3. Scripts do Pipeline


### 3.1. `desafio_1.py` — Tratamento, Anonimização, Agregação e Clusterização
- **Entrada:** `dados/raw/df_t.csv`
- **Processos:**
   - Limpeza, filtragem dos últimos 365 dias, padronização e anonimização de rodoviárias, empresas e clientes.
   - Geração de dois outputs principais:
      - `detalhado_tratado.csv`: dataset detalhado, com todas as compras tratadas, datas e anonimização.
      - `df_tratado.csv`: dataset agregado por cliente, com métricas comportamentais e cluster.
   - Clusterização dos clientes (KMeans) e identificação de outliers via IQR.
   - Nomeação dos clusters: `cluster_1`, `cluster_2`, ..., `outliers`.


### 3.2. `desafio_2.py` — Previsão Simples de Recompra
- **Entrada:** `detalhado_tratado.csv`
- **Processos:**
   - Cálculo de recência, frequência e monetário por cliente.
   - Aplicação de regras simples:
      - Previsão de recompra em 7 dias: recência ≤ 3 dias ou frequência ≥ 2 compras.
      - Previsão de recompra em 30 dias: recência ≤ 15 dias ou frequência ≥ 2 compras.
   - Estimativa do tempo até a próxima compra (regressão simples baseada em datas).
- **Output:**
   - `previsao_simples.csv`: previsão de recompra e dias até próxima compra por cliente.


### 3.3. `desafio_3.py` — Previsão do Próximo Trecho (Origem-Destino)
- **Entradas:**
   - `df_tratado.csv` (métricas agregadas)
   - `detalhado_tratado.csv` (para calcular o trecho mais frequente)
- **Processos:**
   - Para cada cliente, identifica o trecho (origem-destino) mais frequente no histórico.
   - Remoção de classes raras (trechos pouco frequentes, agrupados como 'outros').
   - Usa apenas métricas agregadas para prever o próximo trecho provável (classificação multi-classe, LogisticRegression balanceada).
   - Baseline: sempre prever o trecho mais comum do dataset.
   - Função para prever para cliente específico.
- **Outputs:**
   - `saida_completa.log`: log completo dos resultados e métricas.
   - `resultados_classificacao.csv`: real vs previsto para cada cliente no teste.

---

## 4. Como Executar

### 4.1. Execução Manual

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
4. Analise os resultados em `dados/resultados/` e os gráficos em `scripts/analises/`.

### 4.2. Como rodar com container (Docker ou Podman)

Você pode executar todo o pipeline em um container, sem precisar instalar dependências Python localmente.

#### Usando Docker Compose

1. Certifique-se de ter o Docker e o Docker Compose instalados.
2. Na raiz do projeto, execute:

```sh
docker-compose up --build
```

#### Usando Podman Compose

1. Certifique-se de ter o Podman e o podman-compose instalados.
2. Na raiz do projeto, execute:

```sh
podman-compose up --build
```

O pipeline será executado automaticamente conforme definido no arquivo `docker-compose.yml`.

Os resultados e arquivos gerados ficarão disponíveis na pasta `dados/resultados/`.

---

## 5. Observações Técnicas
- Todos os scripts consideram apenas os últimos 365 dias de dados.
- Os dados e outputs intermediários ficam em `dados/resultados/`.
- O pipeline pode ser adaptado para rodar cada etapa individualmente.
- Os scripts de análise visual são opcionais, mas recomendados para validação dos clusters e previsões.
- O projeto está preparado para execução local ou em container, facilitando a reprodutibilidade.

---

