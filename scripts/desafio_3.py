"""
Script para previsão do próximo trecho mais provável do cliente usando métricas agregadas de df_tratado.csv.
Target: trecho mais frequente do cliente, calculado a partir do detalhado_tratado.csv.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
import warnings
import sys
from contextlib import redirect_stdout

warnings.filterwarnings('ignore')

log_path = 'dados/resultados/desafio_3/saida_completa.log'
with open(log_path, 'w') as f, redirect_stdout(f):

    # 1. Gerar coluna de trecho mais frequente por cliente
    agg = pd.read_csv('dados/resultados/desafio_1/df_tratado.csv')
    detalhado = pd.read_csv('dados/resultados/desafio_1/detalhado_tratado.csv')
    detalhado['trecho'] = detalhado['place_origin_departure'].astype(str) + ' - ' + detalhado['place_destination_departure'].astype(str)
    mais_frequente = detalhado.groupby('fk_contact')['trecho'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'outros').reset_index()
    mais_frequente.columns = ['fk_contact', 'trecho_mais_frequente']
    agg = agg.merge(mais_frequente, on='fk_contact', how='left')
    agg['trecho_mais_frequente'] = agg['trecho_mais_frequente'].fillna('outros')

    # Diagnóstico inicial
    def print_diagnostico(df):
        print('Shape:', df.shape)
        print('Colunas:', df.columns.tolist())
        print('Exemplo de linhas:')
        print(df.head())
        print('Clientes únicos:', df['fk_contact'].nunique())
    print_diagnostico(agg)

    # 2. Features e target
    feature_cols = [
        'recencia', 'frequencia', 'monetario', 'ticket_medio',
        'destinos_unicos', 'empresas_diferentes', 'meses_distintos', 'dias_semana_distintos'
    ]
    feature_cols = [c for c in feature_cols if c in agg.columns]
    X = agg[feature_cols]
    y = agg['trecho_mais_frequente']

    # Remover classes raras (com apenas 1 ocorrência)
    counts = agg['trecho_mais_frequente'].value_counts()
    classes_validas = counts[counts > 1].index
    agg = agg[agg['trecho_mais_frequente'].isin(classes_validas)]
    X = agg[feature_cols]
    y = agg['trecho_mais_frequente']

    # Codificar target
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)

    # Padronizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # 3. Baseline: prever trecho mais comum do dataset
    trecho_mais_comum = agg['trecho_mais_frequente'].mode()[0]
    y_pred_baseline = np.full_like(y_test, fill_value=le_target.transform([trecho_mais_comum])[0])
    print("\n--- Baseline: Previsão do trecho mais comum do dataset ---")
    print(classification_report(
        le_target.inverse_transform(y_test),
        le_target.inverse_transform(y_pred_baseline),
        zero_division=0
    ))
    print(f"Acurácia: {accuracy_score(le_target.inverse_transform(y_test), le_target.inverse_transform(y_pred_baseline)):.2f}")

    # 4. Modelo de classificação simples
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n--- Classificação: Previsão do trecho mais frequente (LogisticRegression) ---")
    print(classification_report(
        le_target.inverse_transform(y_test),
        le_target.inverse_transform(y_pred),
        zero_division=0
    ))
    print(f"Acurácia: {accuracy_score(le_target.inverse_transform(y_test), le_target.inverse_transform(y_pred)):.2f}")

    # 5. Função para prever para cliente específico
    def prever_para_cliente(fk_contact):
        row = agg[agg['fk_contact'] == fk_contact]
        if row.empty:
            print('Cliente não encontrado.')
            return
        X_row = scaler.transform(row[feature_cols])
        pred_code = clf.predict(X_row)[0]
        pred_trecho = le_target.inverse_transform([pred_code])[0]
        print(f'Previsão para {fk_contact}: {pred_trecho}')

    # Exemplo de uso:
    # prever_para_cliente('cliente_1')

# Ao final, imprima um resumo no terminal
print(f"\nResumo salvo em {log_path}. Veja o arquivo para detalhes completos.")