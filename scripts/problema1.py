import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Carregar os dados
df = pd.read_csv('challenge-clickbus/data/df_tratado.csv')


# Pré-processamento e Cálculo das Métricas RFM
df['purchase_datetime'] = pd.to_datetime(df['date_purchase'] + ' ' + df['time_purchase'])
today = df['purchase_datetime'].max() + pd.Timedelta(days=1)
rfm_df = df.groupby('fk_contact').agg(
    recency=('purchase_datetime', lambda x: (today - x.max()).days),
    frequency=('nk_ota_localizer_id', 'count'),
    monetary=('gmv_success', 'sum')
).reset_index()

# Escalar os dados para o K-Means
features = rfm_df[['recency', 'frequency', 'monetary']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Aplicar a clusterização K-Means (com 3 clusters para o MVP)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(scaled_features)
rfm_df['cluster'] = kmeans.labels_

# Salvar o resultado
rfm_df.to_csv('data/resultados/resultado_p1.csv', index=False)

print("Clusterização concluída. O arquivo 'resultado_p1.csv' foi salvo com sucesso.")