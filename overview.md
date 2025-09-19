



# Overview do Pipeline ClickBus (Atualizado)

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
6. Nomeação dos clusters: `cluster_1`, `cluster_2`, ..., `outliers`.

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
Prever, para cada cliente, qual o próximo par origem-destino (trecho) mais provável de ser comprado (classificação multi-classe), usando apenas métricas agregadas.

### Principais etapas
1. Leitura dos datasets agregados e detalhados:
	- `dados/resultados/desafio_1/df_tratado.csv` (métricas agregadas)
	- `dados/resultados/desafio_1/detalhado_tratado.csv` (para calcular o trecho mais frequente)
2. Para cada cliente, identificação do trecho (origem-destino) mais frequente no histórico.
3. Remoção de classes raras (trechos pouco frequentes, agrupados como 'outros').
4. Treinamento de modelo de classificação (LogisticRegression multinomial, balanceada) para prever o próximo trecho provável, usando apenas métricas agregadas.
5. Baseline: sempre prever o trecho mais comum do dataset.
6. Função para prever para cliente específico.
7. Salvamento dos resultados:
	- `dados/resultados/desafio_3/saida_completa.log`: log completo dos resultados e métricas.
	- `dados/resultados/desafio_3/resultados_classificacao.csv`: real vs previsto para cada cliente no teste.

**Observação:** Todo o pipeline considera apenas os últimos 365 dias, garantindo previsões atualizadas e alinhadas com o tratamento de dados.

---

---





## Modelos e Estratégias Utilizados

### Clusterização (KMeans)
- **Problema:** Agrupar clientes com comportamentos similares.
- **Modelo:** KMeans, que agrupa clientes em clusters baseando-se na similaridade das features agregadas.
- **Uso:** Segmentação de clientes para estratégias de marketing, personalização, etc. Outliers recebem o cluster 'outliers'.

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
4. **scripts/analises/analise.ipynb** e **scripts/analises/avaliação dos clusters.py:** Permitem análise exploratória, validação dos clusters e visualização dos resultados.

---