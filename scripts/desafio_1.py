import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

# 1. Parâmetros e leitura dos dados
df = pd.read_csv('dados/raw/df_t.csv')

# 2. Conversão de datas e limpeza inicial
df['purchase_datetime'] = pd.to_datetime(df['date_purchase'] + ' ' + df['time_purchase'])
df = df.drop(columns=['date_purchase', 'time_purchase'])

data_max = df['purchase_datetime'].max()
data_min = data_max - pd.Timedelta(days=365)
df = df[(df['purchase_datetime'] > data_min) & (df['purchase_datetime'] <= data_max)]

df = df[df['gmv_success'] > 0]

df['valor_passagem'] = df['gmv_success'] / df['total_tickets_quantity_success']
df['valor_trecho'] = np.where(
    df['place_origin_return'].notna() & (df['place_origin_return'] != '0'),
    df['valor_passagem'] / 2,
    df['valor_passagem']
)

# 3. Criar o DataFrame de retorno já removendo e renomeando as colunas
df_retorno = df[df['place_origin_return'].notna() & (df['place_origin_return'] != '0')].copy()
df_retorno = df_retorno.drop(
    columns=['place_origin_return', 'place_destination_return', 'fk_return_ota_bus_company'],
    errors='ignore'
)
df_retorno = df_retorno.rename(columns={
    'place_origin_return': 'place_origin_departure',
    'place_destination_return': 'place_destination_departure',
    'fk_return_ota_bus_company': 'fk_departure_ota_bus_company'
})

# 4. Remover as colunas de retorno do DataFrame original (ignorar se não existirem)
df = df.drop(
    columns=['place_origin_return', 'place_destination_return', 'fk_return_ota_bus_company'],
    errors='ignore'
)

# 5. Garantir que as colunas e a ordem são idênticas antes de concatenar
df_retorno = df_retorno[df.columns]

# 6. Concatenar ida e retorno
df_unificado = pd.concat([df, df_retorno], ignore_index=True)

df_unificado = df_unificado.sort_values(['fk_contact', 'purchase_datetime'])

# 7. Criar mapping único para rodoviárias (origem e destino, ida e volta)
places_cols = ['place_origin_departure', 'place_destination_departure']
all_places = pd.concat([df_unificado[c] for c in places_cols], ignore_index=True)
unique_places = [v for v in all_places.unique() if v != '0']
place_mapping = {v: f'rodoviaria_{i+1}' for i, v in enumerate(unique_places)}

for c in places_cols:
    df_unificado[c] = df_unificado[c].map(place_mapping).fillna(df_unificado[c])

# 8. Criar mapping único para empresas de ônibus (ida e volta)
bus_cols = ['fk_departure_ota_bus_company']
all_bus = pd.concat([df_unificado[c] for c in bus_cols], ignore_index=True)
unique_bus = [v for v in all_bus.unique() if v != '1' and v != '0']
bus_mapping = {v: f'empresa_{i+1}' for i, v in enumerate(unique_bus)}

for c in bus_cols:
    df_unificado[c] = df_unificado[c].map(bus_mapping).fillna(df_unificado[c])



# 9. Anonimizar clientes individualmente
unique_contacts = [v for v in df_unificado['fk_contact'].unique() if v != '0']
contact_mapping = {v: f'cliente_{i+1}' for i, v in enumerate(unique_contacts)}
df_unificado['fk_contact'] = df_unificado['fk_contact'].map(contact_mapping).fillna(df_unificado['fk_contact'])

# 9.1. Criar df_geral com todas as colunas tratadas para uso em análises temporais
df_geral = df_unificado.copy()

# Salvar arquivo detalhado tratado (com datas, anonimização e padronização) APÓS todos os tratamentos
output_dir = 'dados/resultados/desafio_1'
os.makedirs(output_dir, exist_ok=True)
detalhado_path = os.path.join(output_dir, 'detalhado_tratado.csv')
df_geral.to_csv(detalhado_path, index=False)
print(f"Arquivo detalhado tratado salvo: {detalhado_path}")

# 10. Gerar DataFrame único por cliente e calcular features agregadas
df_clientes = df_unificado.groupby('fk_contact').agg(
    recencia = ('purchase_datetime', lambda x: (df_unificado['purchase_datetime'].max() - x.max()).days),
    frequencia = ('purchase_datetime', 'count'),
    monetario = ('valor_trecho', 'sum'),
    ticket_medio = ('valor_trecho', 'mean'),
    destinos_unicos = ('place_destination_departure', 'nunique'),
    tempo_entre_compras = ('purchase_datetime', lambda x: x.diff().dt.days.mean()),
    std_valor_trecho = ('valor_trecho', 'std'),
    empresas_diferentes = ('fk_departure_ota_bus_company', 'nunique'),
    meses_distintos = ('purchase_datetime', lambda x: x.dt.month.nunique()),
    dias_semana_distintos = ('purchase_datetime', lambda x: x.dt.weekday.nunique())
).reset_index()

# 11. Otimizar tipos e salvar
for col in df_clientes.select_dtypes(include='object').columns:
    df_clientes[col] = df_clientes[col].astype('category')

output_dir = 'dados/resultados/desafio_1'
os.makedirs(output_dir, exist_ok=True)
file_path = os.path.join(output_dir, 'df_tratado.csv')
df_clientes.to_csv(file_path, index=False)

# Parâmetro: número de clusters principais
N_CLUSTERS = 4  # Altere para 4 se desejar

# Seleção de features para clusterização
features = ['recencia', 'frequencia', 'monetario', 'ticket_medio', 'destinos_unicos', 'empresas_diferentes']
X = df_clientes[features].copy()

# Detecção de outliers via IQR
outlier_mask = np.zeros(len(X), dtype=bool)
for col in features:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_mask |= (X[col] < lower) | (X[col] > upper)

# Escalonamento
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X[~outlier_mask])

# Substituir NaN/infs apenas no array escalonado para o KMeans
X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# KMeans (N_CLUSTERS clusters principais)
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
clusters = np.full(len(X), N_CLUSTERS, dtype=int)  # Último cluster = outliers
clusters[~outlier_mask] = kmeans.fit_predict(X_scaled)

# Nomear clusters: 0, 1, ..., N_CLUSTERS-1, 'outliers'
cluster_names = [f'cluster_{i+1}' for i in range(N_CLUSTERS)] + ['outliers']
df_clientes['cluster'] = [cluster_names[c] if c < N_CLUSTERS else 'outliers' for c in clusters]

# Salva novamente com cluster
df_clientes.to_csv(file_path, index=False)
print(f"Arquivo de dados tratados salvo: {file_path}")
print(f"Tratamento de hashes e clusterização concluídos. Clusters: {cluster_names}")