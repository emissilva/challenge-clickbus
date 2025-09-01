import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import argparse

# Configurar o argparse para a janela de previsão
parser = argparse.ArgumentParser(description='Script para gerar RFM e clusters.')
parser.add_argument('prediction_window', type=int, default=30, nargs='?',
                    help='Janela de tempo para a previsão (em dias, ex: 7 ou 30).')
args = parser.parse_args()
PREDICTION_WINDOW_DAYS = args.prediction_window

# 1. Obter a lista de arquivos anuais do pipeline anterior
anos_disponiveis = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

for ano in anos_disponiveis:
    file_path = f'data/data_tratado/df_tratado_{ano}.csv'
    
    if not os.path.exists(file_path):
        print(f"Aviso: Arquivo {file_path} não encontrado. Pulando o ano {ano}.")
        continue

    # 2. Carregar os dados do ano
    df = pd.read_csv(file_path)

    # 3. Pré-processamento e Cálculo das Métricas RFM
    initial_rows = len(df)
    df = df[df['gmv_success'] > 0]
    rows_removed = initial_rows - len(df)
    print(f"Linhas removidas por gmv_success <= 0 para o ano {ano}: {rows_removed}")

    df['purchase_datetime'] = pd.to_datetime(df['purchase_datetime'])

    # --- CORREÇÃO DO VAZAMENTO DE DADOS ---
    # Define a data de corte para o cálculo das features
    cutoff_date = df['purchase_datetime'].max() - pd.Timedelta(days=PREDICTION_WINDOW_DAYS)

    # Filtra o dataframe para incluir apenas transações até a data de corte
    df_filtered = df[df['purchase_datetime'] <= cutoff_date].copy()

    # Verifica se há dados suficientes após o filtro
    if df_filtered.empty:
        print(f"Aviso: Nenhum dado encontrado até a data de corte ({cutoff_date}). Pulando ano {ano}.")
        continue
    
    # Recalcula as métricas RFM usando apenas os dados filtrados
    rfm_df = df_filtered.groupby('fk_contact').agg(
        recency=('purchase_datetime', lambda x: (cutoff_date - x.max()).days),
        frequency=('nk_ota_localizer_id', 'count'),
        monetary=('gmv_success', 'sum')
    ).reset_index()
    # -------------------------------------

    # 4. Escalar os dados para o K-Means
    features = rfm_df[['recency', 'frequency', 'monetary']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 5. Aplicar a clusterização K-Means
    # Aumentar o número de clusters pode ser necessário
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    kmeans.fit(scaled_features)
    rfm_df['cluster'] = kmeans.labels_

    # 6. Salvar o resultado
    output_file = f'data/resultados/problema1/resultado_p1_{ano}.csv'
    rfm_df.to_csv(output_file, index=False)
    print(f"Clusterização para o ano {ano} concluída. O arquivo '{output_file}' foi salvo com sucesso.")