import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
import os
import numpy as np

# Modelo simples: calcula recência e frequência para cada cliente e prevê recompra por regra
janela_previsao = [7, 30]

# 1. Carregar o arquivo de dados tratado
detalhado_path = 'dados/resultados/desafio_1/detalhado_tratado.csv'

if not os.path.exists(detalhado_path):
    print(f"Erro: Arquivo {detalhado_path} não encontrado. Certifique-se de que o primeiro script de tratamento de dados foi executado.")
else:
    # 2. Carregar os dados detalhados
    df = pd.read_csv(detalhado_path)

    # 3. Pré-processamento: garantir tipos corretos
    df['purchase_datetime'] = pd.to_datetime(df['purchase_datetime'])

    id_col = 'fk_contact' if 'fk_contact' in df.columns else 'cliente'
    # Calcular recencia, frequencia e monetario a partir do detalhado
    data_max = df['purchase_datetime'].max()
    resumo = df.groupby(id_col).agg(
        recencia = ('purchase_datetime', lambda x: (data_max - x.max()).days),
        frequencia = ('purchase_datetime', 'count'),
        monetario = ('valor_trecho', 'sum')
    ).reset_index()

    # Previsão simples para 7 e 30 dias
    resumo['preve_comprar_7_dias'] = ((resumo['recencia'] <= 3) | (resumo['frequencia'] >= 2)).astype(int)
    resumo['preve_comprar_30_dias'] = ((resumo['recencia'] <= 15) | (resumo['frequencia'] >= 2)).astype(int)

    # Extra: prever o número de dias até a próxima compra (regressão simples)
    df_sorted = df.sort_values([id_col, 'purchase_datetime'])
    df_sorted['next_purchase'] = df_sorted.groupby(id_col)['purchase_datetime'].shift(-1)
    df_sorted['dias_ate_proxima'] = (df_sorted['next_purchase'] - df_sorted['purchase_datetime']).dt.days
    dias_ate_proxima = df_sorted.groupby(id_col)['dias_ate_proxima'].min().reset_index()
    resumo = pd.merge(resumo, dias_ate_proxima, on=id_col, how='left')

    output_dir = f'dados/resultados/desafio_2'
    os.makedirs(output_dir, exist_ok=True)
    resumo.to_csv(os.path.join(output_dir, 'previsao_simples.csv'), index=False)
    print(f"Previsão simples para todas as janelas salva em 'previsao_simples.csv', incluindo dias até a próxima compra.")