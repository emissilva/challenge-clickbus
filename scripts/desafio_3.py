
"""
Script para previsão do próximo trecho mais provável do cliente usando abordagem temporal e classificação multi-classe.
Target: próximo trecho comprado (não o mais frequente), sem vazamento.
Opcional: prever dias até próxima compra (regressão).
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
from contextlib import redirect_stdout
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

log_path = 'dados/resultados/desafio_3/saida_completa.log'
output_csv = 'dados/resultados/desafio_3/resultados_classificacao.csv'
os.makedirs('dados/resultados/desafio_3', exist_ok=True)

with open(log_path, 'w') as f, redirect_stdout(f):
    # 1. Leitura e tratamento dos dados detalhados (garantir 365 dias, anonimização, etc)
    detalhado = pd.read_csv('dados/resultados/desafio_1/detalhado_tratado.csv')
    detalhado['purchase_datetime'] = pd.to_datetime(detalhado['purchase_datetime'])
    data_max = detalhado['purchase_datetime'].max()
    data_min = data_max - pd.Timedelta(days=365)
    detalhado = detalhado[(detalhado['purchase_datetime'] > data_min) & (detalhado['purchase_datetime'] <= data_max)]

    # 3. Criar coluna de trecho
    detalhado['trecho'] = detalhado['place_origin_departure'].astype(str) + ' - ' + detalhado['place_destination_departure'].astype(str)

    # 4. Construção vetorizada do dataset temporal (pandas)
    detalhado = detalhado.sort_values(['fk_contact', 'purchase_datetime'])
    detalhado['target_trecho'] = detalhado.groupby('fk_contact')['trecho'].shift(-1)
    detalhado['purchase_datetime_next'] = detalhado.groupby('fk_contact')['purchase_datetime'].shift(-1)
    detalhado['dias_ate_proxima'] = (detalhado['purchase_datetime_next'] - detalhado['purchase_datetime']).dt.days

    # Features históricas acumuladas (sem vazamento)
    detalhado['frequencia'] = detalhado.groupby('fk_contact').cumcount() + 1
    detalhado['monetario'] = detalhado.groupby('fk_contact')['valor_trecho'].cumsum() - detalhado['valor_trecho']
    detalhado['ticket_medio'] = (
        detalhado.groupby('fk_contact')['valor_trecho']
        .expanding().mean().shift().reset_index(level=0, drop=True)
    )
    detalhado['ticket_medio'] = detalhado['ticket_medio'].fillna(detalhado['valor_trecho'])
    detalhado['destinos_unicos'] = (
        detalhado.groupby('fk_contact')['place_destination_departure']
        .expanding().apply(lambda x: x.nunique()).shift().reset_index(level=0, drop=True)
    )
    detalhado['destinos_unicos'] = detalhado['destinos_unicos'].fillna(1)
    detalhado['empresas_diferentes'] = (
        detalhado.groupby('fk_contact')['fk_departure_ota_bus_company']
        .expanding().apply(lambda x: x.nunique()).shift().reset_index(level=0, drop=True)
    )
    detalhado['empresas_diferentes'] = detalhado['empresas_diferentes'].fillna(1)
    detalhado['meses_distintos'] = (
        detalhado.groupby('fk_contact')['purchase_datetime']
        .expanding().apply(lambda x: x.dt.month.nunique()).shift().reset_index(level=0, drop=True)
    )
    detalhado['meses_distintos'] = detalhado['meses_distintos'].fillna(1)
    detalhado['dias_semana_distintos'] = (
        detalhado.groupby('fk_contact')['purchase_datetime']
        .expanding().apply(lambda x: x.dt.weekday.nunique()).shift().reset_index(level=0, drop=True)
    )
    detalhado['dias_semana_distintos'] = detalhado['dias_semana_distintos'].fillna(1)

    # recencia: diferença para próxima compra
    detalhado['recencia'] = detalhado['dias_ate_proxima']

    # Montar DataFrame final (remover última compra de cada cliente, pois não tem target)
    df = detalhado.dropna(subset=['target_trecho']).copy()

    # 5. Remover trechos pouco frequentes
    counts = df['target_trecho'].value_counts()
    classes_validas = counts[counts > 1].index
    df = df[df['target_trecho'].isin(classes_validas)]

    # 6. Features e target
    feature_cols = [
        'recencia', 'frequencia', 'monetario', 'ticket_medio',
        'destinos_unicos', 'empresas_diferentes', 'meses_distintos', 'dias_semana_distintos'
    ]
    X = df[feature_cols]
    y = df['target_trecho']

    # 7. Codificar target
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)

    # 8. Padronizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 9. Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # 10. Baseline: prever trecho mais comum do dataset
    trecho_mais_comum = df['target_trecho'].mode()[0]
    y_pred_baseline = np.full_like(y_test, fill_value=le_target.transform([trecho_mais_comum])[0])
    print("\n--- Baseline: Previsão do trecho mais comum do dataset ---")
    print(classification_report(
        le_target.inverse_transform(y_test),
        le_target.inverse_transform(y_pred_baseline),
        zero_division=0
    ))
    print(f"Acurácia: {accuracy_score(le_target.inverse_transform(y_test), le_target.inverse_transform(y_pred_baseline)):.2f}")


    # 11. Modelo de classificação multi-classe (XGBoost)
    clf = XGBClassifier(
        objective='multi:softmax',
        num_class=len(le_target.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        n_estimators=50,   # ajuste para acelerar
        max_depth=4        # ajuste para acelerar
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n--- Classificação: Previsão do próximo trecho (XGBoostClassifier) ---")
    print(classification_report(
        le_target.inverse_transform(y_test),
        le_target.inverse_transform(y_pred),
        zero_division=0
    ))
    print(f"Acurácia: {accuracy_score(le_target.inverse_transform(y_test), le_target.inverse_transform(y_pred)):.2f}")

    # 12. Salvar real vs previsto
    resultados = pd.DataFrame({
        'fk_contact': df.iloc[y_test.index]['fk_contact'].values,
        'trecho_real': le_target.inverse_transform(y_test),
        'trecho_previsto': le_target.inverse_transform(y_pred)
    })
    resultados.to_csv(output_csv, index=False)
    print(f"Resultados de classificação salvos em {output_csv}")


    # 13. Regressão para dias até próxima compra (XGBoost)
    reg = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, n_estimators=50, max_depth=4)
    reg.fit(X_train, df.iloc[y_train.index]['dias_ate_proxima'])
    y_pred_reg = reg.predict(X_test)
    rmse = mean_squared_error(df.iloc[y_test.index]['dias_ate_proxima'], y_pred_reg, squared=False)
    print(f"\n--- Regressão: Dias até próxima compra (XGBoostRegressor) ---")
    print(f"RMSE: {rmse:.2f}")


    # 14. Função para prever para cliente específico (última linha do histórico)
    def prever_para_cliente(fk_contact):
        hist = detalhado[detalhado['fk_contact'] == fk_contact].sort_values('purchase_datetime')
        if len(hist) < 2:
            print('Cliente sem histórico suficiente.')
            return
        # Usar última compra como contexto
        i = len(hist) - 2
        linha = {
            'recencia': (hist.iloc[-1]['purchase_datetime'] - hist.iloc[i]['purchase_datetime']).days,
            'frequencia': i+1,
            'monetario': hist.iloc[:i+1]['valor_trecho'].sum(),
            'ticket_medio': hist.iloc[:i+1]['valor_trecho'].mean(),
            'destinos_unicos': hist.iloc[:i+1]['place_destination_departure'].nunique(),
            'empresas_diferentes': hist.iloc[:i+1]['fk_departure_ota_bus_company'].nunique(),
            'meses_distintos': hist.iloc[:i+1]['purchase_datetime'].dt.month.nunique(),
            'dias_semana_distintos': hist.iloc[:i+1]['purchase_datetime'].dt.weekday.nunique()
        }
        X_row = scaler.transform([list(linha.values())])
        pred_code = clf.predict(X_row)[0]
        pred_trecho = le_target.inverse_transform([pred_code])[0]
        print(f'Previsão para {fk_contact}: {pred_trecho}')

    prever_para_cliente('cliente_1')

# Ao final, imprima um resumo no terminal
print(f"\nResumo salvo em {log_path}. Veja o arquivo para detalhes completos.")