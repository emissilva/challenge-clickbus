import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# 1. Obter a lista de arquivos anuais do pipeline anterior
# O script assume que o tratar_dados.py salvou arquivos como 'df_tratado_2023.csv'
anos_disponiveis = [2013,2014,2015,2016,2017,2018, 2019, 2020, 2021, 2022, 2023,2024]

for ano in anos_disponiveis:
    file_path = f'data/data_tratado/df_tratado_{ano}.csv'
    
    # Verifique se o arquivo do ano existe antes de processar
    if not os.path.exists(file_path):
        print(f"Aviso: Arquivo {file_path} não encontrado. Pulando o ano {ano}.")
        continue

    # 2. Carregar os dados do ano
    df = pd.read_csv(file_path)

    # 3. Pré-processamento e Cálculo das Métricas RFM
    # Remove as linhas onde gmv_success é menor ou igual a zero
    initial_rows = len(df)
    df = df[df['gmv_success'] > 0]
    rows_removed = initial_rows - len(df)
    print(f"Linhas removidas por gmv_success <= 0 para o ano {ano}: {rows_removed}")

    df['purchase_datetime'] = pd.to_datetime(df['purchase_datetime'])
    today = df['purchase_datetime'].max() + pd.Timedelta(days=1)
    rfm_df = df.groupby('fk_contact').agg(
        recency=('purchase_datetime', lambda x: (today - x.max()).days),
        frequency=('nk_ota_localizer_id', 'count'),
        monetary=('gmv_success', 'sum')
    ).reset_index()

    # 4. Escalar os dados para o K-Means
    features = rfm_df[['recency', 'frequency', 'monetary']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 5. Aplicar a clusterização K-Means
    kmeans = KMeans(n_clusters=7, random_state=42, n_init='auto')
    kmeans.fit(scaled_features)
    rfm_df['cluster'] = kmeans.labels_

    # 6. Salvar o resultado
    output_file = f'data/resultados/resultado_p1_{ano}.csv'
    rfm_df.to_csv(output_file, index=False)
    print(f"Clusterização para o ano {ano} concluída. O arquivo '{output_file}' foi salvo com sucesso.")