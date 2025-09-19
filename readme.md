
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
   - `df_tratado.csv` (métricas agregadas, incluindo cluster do cliente)
   - `detalhado_tratado.csv` (para cálculo de features históricas e contextuais)

- **Processos:**
   - Para cada cliente, constrói um dataset temporal com uma linha por compra, contendo features históricas, contextuais e comportamentais.
   - **Features utilizadas:**
     - recencia, frequencia, monetario, ticket_medio, destinos_unicos, empresas_diferentes, meses_distintos, dias_semana_distintos
     - mês da compra atual, dia da semana da compra atual
     - intervalo médio entre compras, tempo desde a primeira compra
     - trecho mais frequente do cliente até o momento, empresa mais frequente até o momento, último trecho realizado
     - cluster do cliente (obtido do arquivo agregado)
   - **Agrupamento de classes:**
     - Apenas os N (ex: 20) trechos mais frequentes são mantidos como classes. Os demais são agrupados como "outros" e excluídos do treinamento/teste.
   - **Modelagem:**
     - O modelo principal é uma `LogisticRegression` multinomial balanceada, robusta para múltiplas classes e dados tabulares.
     - As features categóricas são codificadas automaticamente.
     - O baseline consiste em prever sempre o trecho mais comum do dataset.
   - **Treinamento:**
     - O dataset é dividido em treino e teste (80/20, estratificado).
     - O modelo é treinado apenas nas classes mais frequentes.
     - Métricas de classificação (accuracy, f1-score, precision, recall) são salvas para o conjunto geral e por cluster de cliente.
     - O script também realiza uma regressão para prever o número de dias até a próxima compra (XGBoostRegressor).
   - **Função para previsão individual:**
     - Permite prever o próximo trecho provável para um cliente específico, usando o histórico mais recente.

- **Outputs:**
   - `saida_completa.log`: log completo dos resultados, métricas globais e por cluster.
   - `resultados_classificacao.csv`: real vs previsto para cada cliente no teste, incluindo cluster.

---

## 4. Como Executar

### 4.1. Execução Manual

> **Atenção:** Para executar o pipeline, é necessário inserir o arquivo de dados brutos no diretório `dados/raw/`.


#### Estrutura esperada dos dados brutos
O arquivo esperado é `df_t.csv` dentro de `dados/raw/`.

> **Importante:** O pipeline utiliza todas as colunas abaixo, que devem obrigatoriamente estar presentes no arquivo. Não remova colunas do arquivo original, pois elas são essenciais para o tratamento, enriquecimento, anonimização e análises.

**Colunas obrigatórias e explicação:**

- `nk_ota_localizer_id`: identificador único da compra (localizador)
- `fk_contact`: identificador do cliente
- `date_purchase`: data da compra (formato YYYY-MM-DD)
- `time_purchase`: hora da compra (formato HH:MM:SS)
- `place_origin_departure`: rodoviária de origem da ida
- `place_destination_departure`: rodoviária de destino da ida
- `place_origin_return`: rodoviária de origem da volta (se houver)
- `place_destination_return`: rodoviária de destino da volta (se houver)
- `fk_departure_ota_bus_company`: empresa de ônibus da ida
- `fk_return_ota_bus_company`: empresa de ônibus da volta (se houver)
- `gmv_success`: valor total da compra (Gross Merchandise Value)
- `total_tickets_quantity_success`: quantidade total de passagens compradas

Essas colunas são utilizadas para:
- Identificação e anonimização de clientes e compras
- Cálculo de datas, valores, métricas e features comportamentais
- Análise de origem/destino, empresas e fluxos de viagem
- Tratamento de ida e volta separadamente

Se possível, mantenha o arquivo bruto completo para garantir máxima flexibilidade e reprodutibilidade do pipeline.

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

