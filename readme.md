
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
│   ├── raw/
│   │   └── df_t.csv
│   └── resultados/
│       ├── desafio_1/
│       │   ├── detalhado_tratado.csv
│       │   └── df_tratado.csv
│       ├── desafio_2/
│       │   └── previsao_simples.csv
│       ├── desafio_3/
│       │   ├── saida_completa.log
│       │   └── resultados_classificacao.csv
│       └── analises/
│           ├── boxplot_destinos_unicos_por_cluster.png
│           ├── boxplot_frequencia_por_cluster.png
│           ├── boxplot_monetario_por_cluster.png
│           ├── boxplot_recencia_por_cluster.png
│           ├── boxplot_ticket_medio_por_cluster.png
│           ├── grafico_clientes_por_cluster.png
│           └── pairplot_clusters.png
├── scripts/
│   ├── desafio_1.py
│   ├── desafio_2.py
│   ├── desafio_3.py
│   └── analises/
│       └── avaliacao_clusters.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── readme.md
└── overview.md
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
   - Nomeação dinâmica dos clusters: os nomes são sugeridos automaticamente com base na análise estatística das principais features (frequência, ticket médio, recência, etc.), podendo variar conforme o perfil dos dados. Exemplos de nomes: "Clientes muito frequentes", "Clientes de ticket alto e baixa frequência", "Clientes de baixo valor". O cluster 'outliers' é mantido para clientes fora do padrão.


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
   - cluster do cliente (obtido do arquivo agregado e nomeado conforme análise estatística)
   - **Agrupamento de classes:**
     - Apenas os N (ex: 20) trechos mais frequentes são mantidos como classes. Os demais são agrupados como "outros" e excluídos do treinamento/teste.
   - **Modelagem:**
     - O modelo principal é uma `LogisticRegression` multinomial balanceada, robusta para múltiplas classes e dados tabulares.
     - As features categóricas são codificadas automaticamente.
   - O baseline foi removido para focar apenas no modelo supervisionado.
   - **Treinamento:**
     - O dataset é dividido em treino e teste (80/20, estratificado).
     - O modelo é treinado apenas nas classes mais frequentes.
     - Métricas de classificação (accuracy, f1-score, precision, recall) são salvas para o conjunto geral e por cluster de cliente.
       - O script também realiza uma regressão para prever o número de dias até a próxima compra (XGBoostRegressor).
    - **Função para previsão individual e top clientes:**
       - Permite prever o próximo trecho provável e o tempo até a próxima compra para qualquer cliente, inclusive automaticamente para os 10 clientes que mais compram.

- **Outputs:**
    - `saida_completa.log`: log completo dos resultados, métricas globais e por cluster.
    - `resultados_classificacao.csv`: real vs previsto para cada cliente no teste, incluindo cluster.
    - Gráficos e análises visuais salvos em `dados/resultados/analises/`.

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
3. Execute os scripts na ordem abaixo (obrigatório):
   > **Importante:** Os scripts de desafio 2 e 3 dependem dos dados gerados pelo desafio 1.
   ```bash
   python scripts/desafio_1.py
   python scripts/desafio_2.py
   python scripts/desafio_3.py
   ```
4. Analise os resultados em `dados/resultados/`.

> **Dica:** Para validação visual dos clusters e das previsões, recomenda-se executar também o script `scripts/analises/avaliacao_clusters.py`, que gera gráficos em `dados/resultados/analises/`. Esta etapa é opcional, mas altamente recomendada para análise exploratória e apresentação dos resultados.

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



## 6. Privacidade e Dados Sensíveis

Todos os dados de clientes, rodoviárias e empresas são anonimizados durante o processamento. Nenhum dado sensível ou identificador real é mantido nos outputs finais, garantindo a privacidade dos clientes e a conformidade com boas práticas de proteção de dados.
- Todos os scripts consideram apenas os últimos 365 dias de dados.
- Os dados e outputs intermediários ficam em `dados/resultados/`.
- O pipeline pode ser adaptado para rodar cada etapa individualmente.
- Os scripts de análise visual são opcionais, mas recomendados para validação dos clusters e previsões. Os gráficos gerados ficam em `dados/resultados/analises/`.
- O projeto está preparado para execução local ou em container, facilitando a reprodutibilidade.
- As mensagens de log dos scripts seguem um padrão para facilitar o acompanhamento da execução e debugging, especialmente em ambientes de produção ou CI.

---

