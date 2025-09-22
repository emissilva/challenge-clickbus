## Análise dos Clusters e Gráficos Gerados

O script `scripts/analises/avaliacao_clusters.py` realiza uma análise detalhada dos clusters de clientes, gerando estatísticas e gráficos salvos em `dados/resultados/analises/`.
---


## Análise dos Clusters e Gráficos Gerados

O script `scripts/analises/avaliacao_clusters.py` realiza uma análise detalhada dos clusters de clientes, gerando estatísticas e gráficos salvos em `dados/resultados/analises/`.

### Resumo estatístico (mediana) das principais features por cluster:

O script imprime no terminal a mediana das principais features (recência, frequência, monetário, ticket médio, destinos únicos) para cada cluster, além de sugerir nomes automáticos para facilitar a comunicação com áreas de negócio.

### Gráficos gerados (em `dados/resultados/analises/`):

- `grafico_clientes_por_cluster.png`: Distribuição de clientes por cluster.
- `boxplot_recencia_por_cluster.png`: Boxplot da recência por cluster.
- `boxplot_frequencia_por_cluster.png`: Boxplot da frequência por cluster.
- `boxplot_monetario_por_cluster.png`: Boxplot do valor monetário por cluster.
- `boxplot_ticket_medio_por_cluster.png`: Boxplot do ticket médio por cluster.
- `boxplot_destinos_unicos_por_cluster.png`: Boxplot do número de destinos únicos por cluster.
- `pairplot_clusters.png`: Matriz de dispersão das principais features por cluster (sem outliers).

Esses gráficos permitem validar visualmente a segmentação dos clientes, identificar perfis de comportamento e analisar a separação entre os grupos. Todos os arquivos são salvos automaticamente pelo script.

---




# Overview do Pipeline ClickBus

Este documento detalha o funcionamento dos principais scripts do pipeline de análise e previsão de clientes, alinhado à estrutura real do projeto e aos outputs gerados nas pastas `dados/resultados/desafio_1/`, `desafio_2/` e `desafio_3/`.

---






## `scripts/desafio_1.py` – Tratamento, Anonimização, Agregação e Clusterização

### Objetivo
Preparar, enriquecer e anonimizar os dados brutos de compras, gerar dois datasets principais:
- Um dataset detalhado, com todas as compras tratadas, padronizadas e anonimização de clientes, rodoviárias e empresas.
- Um dataset agregado por cliente, com métricas comportamentais e clusterização.

### Principais etapas
1. Leitura do arquivo bruto (`dados/raw/df_t.csv`), concatenação de data/hora e filtro dos últimos 365 dias.
2. Limpeza, padronização e anonimização de rodoviárias, empresas e clientes.
3. Cálculo de features detalhadas e agregadas por cliente: recência, frequência, monetário, ticket médio, destinos únicos, empresas diferentes, sazonalidade, tempo médio entre compras, etc.
4. Geração de dois outputs:
	- `dados/resultados/desafio_1/detalhado_tratado.csv`: todas as compras tratadas, com datas e anonimização.
	- `dados/resultados/desafio_1/df_tratado.csv`: dataset agregado por cliente, com métricas e cluster.
5. Clusterização dos clientes via KMeans (4 clusters principais, parametrizável), com atribuição de um cluster especial ('outliers') para clientes fora do padrão (detecção via IQR).
6. Nomeação dinâmica dos clusters: os nomes são sugeridos automaticamente com base na análise estatística das features (frequência, ticket médio, recência), resultando em perfis como "Clientes muito frequentes", "Clientes de ticket alto e baixa frequência", "Clientes de baixo valor", etc. O cluster 'outliers' é mantido para clientes fora do padrão.

**Como e por que os nomes dos clusters são definidos:**
Os nomes dos clusters são atribuídos automaticamente pelo script de análise estatística (`avaliacao_clusters.py`). A lógica utiliza as medianas das principais métricas de cada grupo (como frequência de compra, ticket médio e recência) para identificar padrões de comportamento. Assim, cada cluster recebe um nome descritivo que reflete o perfil predominante dos clientes daquele grupo, facilitando a comunicação com áreas de negócio e a interpretação dos resultados. Essa abordagem torna a segmentação mais transparente e útil para ações estratégicas, pois os nomes são sempre baseados em dados reais e atualizados do próprio projeto.

---

---






## `scripts/desafio_2.py` – Previsão Simples de Recompra

### Objetivo
Gerar previsões simples de recompra para cada cliente usando regras heurísticas baseadas em recência e frequência, a partir do dataset detalhado tratado.

### Principais etapas
1. Leitura dos dados detalhados tratados de `dados/resultados/desafio_1/detalhado_tratado.csv`.
2. Cálculo de recência, frequência e monetário por cliente.
3. Aplicação de regras simples:
   - Previsão de recompra em 7 dias: recência ≤ 3 dias ou frequência ≥ 2 compras.
   - Previsão de recompra em 30 dias: recência ≤ 15 dias ou frequência ≥ 2 compras.
4. Estimativa do número de dias até a próxima compra (regressão simples baseada em datas).
5. Salvamento dos resultados em `dados/resultados/desafio_2/previsao_simples.csv`.

---

---







## `scripts/desafio_3.py` – Previsão do Próximo Trecho (Multi-classe)

### Objetivo
Prever, para cada cliente, qual o próximo par origem-destino (trecho) mais provável de ser comprado (classificação multi-classe), usando métricas históricas, contextuais, comportamentais e de cluster.

### Principais etapas
1. Leitura dos datasets agregados e detalhados:
	- `dados/resultados/desafio_1/df_tratado.csv` (métricas agregadas, incluindo cluster do cliente)
	- `dados/resultados/desafio_1/detalhado_tratado.csv` (para cálculo de features históricas e contextuais)
2. Para cada cliente, constrói um dataset temporal com uma linha por compra, contendo features históricas, contextuais e comportamentais.
3. **Features utilizadas:**
	- recencia, frequencia, monetario, ticket_medio, destinos_unicos, empresas_diferentes, meses_distintos, dias_semana_distintos
	- mês da compra atual, dia da semana da compra atual
	- intervalo médio entre compras, tempo desde a primeira compra
	- trecho mais frequente do cliente até o momento, empresa mais frequente até o momento, último trecho realizado
	- cluster do cliente (obtido do arquivo agregado)
4. **Agrupamento de classes:**
	- Apenas os N (ex: 20) trechos mais frequentes são mantidos como classes. Os demais são agrupados como "outros" e excluídos do treinamento/teste.
5. **Modelagem:**
	- O modelo principal é uma `LogisticRegression` multinomial balanceada, robusta para múltiplas classes e dados tabulares.
	- As features categóricas são codificadas automaticamente.
	- O baseline consiste em prever sempre o trecho mais comum do dataset.
6. **Treinamento:**
	- O dataset é dividido em treino e teste (80/20, estratificado).
	- O modelo é treinado apenas nas classes mais frequentes.
	- Métricas de classificação (accuracy, f1-score, precision, recall) são salvas para o conjunto geral e por cluster de cliente.
	- O script também realiza uma regressão para prever o número de dias até a próxima compra (XGBoostRegressor).
7. **Função para previsão individual:**
	- Permite prever o próximo trecho provável para um cliente específico, usando o histórico mais recente.
8. **Salvamento dos resultados:**
	- `dados/resultados/desafio_3/saida_completa.log`: log completo dos resultados, métricas globais e por cluster.
	- `dados/resultados/desafio_3/resultados_classificacao.csv`: real vs previsto para cada cliente no teste, incluindo cluster.

**Observação:** Todo o pipeline considera apenas os últimos 365 dias, garantindo previsões atualizadas e alinhadas com o tratamento de dados.

---

---






## Modelos e Estratégias Utilizados

### Clusterização (KMeans)
- **Problema:** Agrupar clientes com comportamentos similares.
- **Modelo:** KMeans, que agrupa clientes em clusters baseando-se na similaridade das features agregadas.
- **Uso:** Segmentação de clientes para estratégias de marketing, personalização, etc. Outliers recebem o cluster 'outliers'.
- **Nomeação dos clusters:** Os nomes são sugeridos automaticamente com base na análise estatística das principais features, facilitando a comunicação com áreas de negócio.

### Regras Heurísticas de Recompra
- **Problema:** Prever recompra em janelas de 7 e 30 dias.
- **Estratégia:** Regras simples baseadas em recência e frequência.

### Classificação Multi-classe (LogisticRegression)
- **Problema:** Prever o próximo par origem-destino (trecho) que o cliente irá comprar.
- **Modelo:** LogisticRegression multinomial, balanceada, eficiente para múltiplas classes e dados tabulares.
- **Métricas:** Acurácia, precisão, recall, F1-score por classe.

---

---





## Resumo do Pipeline

1. **scripts/desafio_1.py:** Limpa, transforma, enriquece, anonimiza, remove outliers (para clusterização) e segmenta clientes em clusters nomeados (incluindo 'outliers'), gerando dois outputs principais: dataset detalhado (`detalhado_tratado.csv`) e dataset agregado/clusterizado (`df_tratado.csv`).
2. **scripts/desafio_2.py:** Aplica regras simples de recompra (recência/frequência) e salva previsões simples em `previsao_simples.csv`.
3. **scripts/desafio_3.py:** Calcula o trecho mais frequente por cliente, treina modelo de classificação multi-classe (LogisticRegression), salva logs e previsões detalhadas (real x previsto).
4. **scripts/analises/avaliacao_clusters.py:** Realiza análise estatística detalhada dos clusters, sugere nomes de perfis, gera e salva todos os gráficos de validação em `dados/resultados/analises/`.

---