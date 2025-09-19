


# Overview do Pipeline ClickBus

Este documento detalha o funcionamento dos principais scripts do pipeline de análise e previsão de clientes, alinhado à estrutura real do projeto:
- `scripts/desafio_1.py` – Tratamento, feature engineering, agregação e clusterização de clientes (incluindo detecção de outliers)
- `scripts/desafio_2.py` – Regras simples de recompra baseadas em recência/frequência
- `scripts/desafio_3.py` – Previsão do próximo trecho (origem-destino) via classificação multi-classe e regressão para dias até próxima compra
- Scripts de análise visual: `scripts/analises/analise.ipynb` e `scripts/analises/avaliação dos clusters.py`

Também apresenta os principais conceitos de Machine Learning aplicados: classificação multi-classe, regressão, clusterização (KMeans), métricas e regras heurísticas.

---





## `scripts/desafio_1.py` – Tratamento, Feature Engineering, Agregação e Clusterização

### Objetivo
Preparar, enriquecer e anonimizar os dados brutos de compras, gerar um dataset agregado por cliente e segmentar clientes em clusters, identificando também outliers.

### Principais etapas
1. Leitura do arquivo bruto (`dados/raw/df_t.csv`), concatenação de data/hora e filtro dos últimos 365 dias.
2. Limpeza, padronização e anonimização de rodoviárias, empresas e clientes.
3. Cálculo de features agregadas por cliente: recência, frequência, monetário, ticket médio, destinos únicos, empresas diferentes, sazonalidade, tempo médio entre compras, etc.
4. Remoção de outliers numéricos via método IQR (apenas para clusterização).
5. Clusterização dos clientes via KMeans (4 ou 5 clusters principais, parametrizável), com atribuição de um cluster especial ('outliers') para clientes fora do padrão.
6. Nomeação dos clusters: `cluster_1`, `cluster_2`, ..., `outliers`.
7. Salvamento do dataset tratado e clusterizado em `dados/resultados/desafio_1/df_tratado.csv`.

---





## `scripts/desafio_2.py` – Regras Simples de Recompra

### Objetivo
Gerar previsões simples de recompra para cada cliente usando regras heurísticas baseadas em recência e frequência, a partir dos dados tratados e clusterizados.

### Principais etapas
1. Leitura dos dados tratados e clusterizados de `dados/resultados/desafio_1/df_tratado.csv`.
2. Uso direto das features agregadas (recência, frequência) para aplicar regras simples:
	- Previsão de recompra em 7 dias: recência ≤ 3 dias ou frequência ≥ 2 compras.
	- Previsão de recompra em 30 dias: recência ≤ 15 dias ou frequência ≥ 2 compras.
3. Cálculo opcional do número de dias até a próxima compra (regressão simples baseada em datas).
4. Salvamento dos resultados em `dados/resultados/desafio_2/previsao_simples.csv`.

---





## `scripts/desafio_3.py` – Previsão do Próximo Trecho (Multi-classe)

### Objetivo
Prever, para cada cliente, qual o próximo par origem-destino (trecho) mais provável de ser comprado (classificação multi-classe) e, opcionalmente, em quantos dias essa compra deve ocorrer (regressão).

### Principais etapas
1. Leitura do histórico detalhado (`dados/raw/df_t.csv`), filtro dos últimos 365 dias.
2. Anonimização de rodoviárias, empresas e clientes, e criação da coluna `trecho`.
3. Construção do dataset de treino: para cada cliente, cada linha representa o contexto até uma compra e o target é o próximo trecho comprado.
4. Remoção de classes raras (trechos pouco frequentes, agrupados como 'outros').
5. Treinamento de modelo **XGBoostClassifier** para prever o próximo trecho (multi-classe), com métricas de acurácia e relatório de classificação.
6. (Opcional) Treinamento de **XGBoostRegressor** para prever dias até a próxima compra (RMSE).
7. Função de recomendação do próximo trecho para um cliente específico.
8. Salvamento das previsões de classificação (real x previsto) em `dados/resultados/desafio_3/resultados_classificacao.csv`.

**Observação:** Todo o pipeline considera apenas os últimos 365 dias, garantindo previsões atualizadas e alinhadas com o tratamento de dados.

---




## Modelos e Estratégias Utilizados

### Clusterização (KMeans)
- **Problema:** Agrupar clientes com comportamentos similares.
- **Modelo:** KMeans, que agrupa clientes em clusters baseando-se na similaridade das features agregadas.
- **Uso:** Segmentação de clientes para estratégias de marketing, personalização, etc. Outliers recebem o cluster 'outliers'.

### Regras Heurísticas de Recompra
- **Problema:** Prever recompra em janelas de 7 e 30 dias.
- **Estratégia:** Regras simples baseadas em recência e frequência.

### Classificação Multi-classe (XGBoostClassifier)
- **Problema:** Prever o próximo par origem-destino (trecho) que o cliente irá comprar.
- **Modelo:** XGBoostClassifier, eficiente para múltiplas classes e dados tabulares.
- **Métricas:** Acurácia, precisão, recall, F1-score por classe.

### Regressão (XGBoostRegressor)
- **Problema:** Prever o número de dias até a próxima compra.
- **Modelo:** XGBoostRegressor, baseado em árvores de decisão.
- **Métrica:** RMSE (Root Mean Squared Error).

---




## Resumo do Pipeline

1. **scripts/desafio_1.py:** Limpa, transforma, enriquece, anonimiza, remove outliers (para clusterização) e segmenta clientes em clusters nomeados (incluindo 'outliers'), gerando dataset agregado e clusterizado por cliente em `dados/resultados/desafio_1/df_tratado.csv`.
2. **scripts/desafio_2.py:** Aplica regras simples de recompra (recência/frequência) e salva previsões simples em `dados/resultados/desafio_2/previsao_simples.csv`.
3. **scripts/desafio_3.py:** Prepara targets, balanceia classes, treina modelos de classificação multi-classe e regressão, salva previsões detalhadas (real x previsto) em `dados/resultados/desafio_3/resultados_classificacao.csv`.
4. **scripts/analises/analise.ipynb** e **scripts/analises/avaliação dos clusters.py:** Permitem análise exploratória, validação dos clusters e visualização dos resultados.

---