
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from math import sqrt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from contextlib import redirect_stdout

log_path = 'dados/resultados/desafio_3/saida_completa.log'
output_csv = 'dados/resultados/desafio_3/resultados_classificacao.csv'
os.makedirs('dados/resultados/desafio_3', exist_ok=True)


with open(log_path, 'w') as f, redirect_stdout(f):
    # 1. Leitura e tratamento dos dados detalhados (amostragem para evitar travamento)
    detalhado = pd.read_csv('dados/resultados/desafio_1/detalhado_tratado.csv')
    if len(detalhado) > 100000:
        detalhado = detalhado.sample(n=100000, random_state=42)
    detalhado['purchase_datetime'] = pd.to_datetime(detalhado['purchase_datetime'])
    data_max = detalhado['purchase_datetime'].max()
    data_min = data_max - pd.Timedelta(days=365)
    detalhado = detalhado[(detalhado['purchase_datetime'] > data_min) & (detalhado['purchase_datetime'] <= data_max)]

    # 2. Criar coluna de trecho
    detalhado['trecho'] = detalhado['place_origin_departure'].astype(str) + ' - ' + detalhado['place_destination_departure'].astype(str)

    # 3. Garantir tipos corretos para operações numéricas
    detalhado['valor_trecho'] = pd.to_numeric(detalhado['valor_trecho'], errors='coerce').fillna(0)
    detalhado['place_destination_departure'] = detalhado['place_destination_departure'].astype(str)
    detalhado['fk_departure_ota_bus_company'] = detalhado['fk_departure_ota_bus_company'].astype(str)

    # 4. Ordenação e features temporais
    detalhado = detalhado.sort_values(['fk_contact', 'purchase_datetime'])
    detalhado['target_trecho'] = detalhado.groupby('fk_contact')['trecho'].shift(-1)
    detalhado['purchase_datetime_next'] = detalhado.groupby('fk_contact')['purchase_datetime'].shift(-1)
    detalhado['dias_ate_proxima'] = (detalhado['purchase_datetime_next'] - detalhado['purchase_datetime']).dt.days

    # 5. Features históricas acumuladas (sem vazamento)
    detalhado['frequencia'] = detalhado.groupby('fk_contact').cumcount() + 1
    detalhado['monetario'] = detalhado.groupby('fk_contact')['valor_trecho'].cumsum() - detalhado['valor_trecho']
    detalhado['ticket_medio'] = (
        detalhado.groupby('fk_contact')['valor_trecho']
        .expanding().mean().shift().reset_index(level=0, drop=True)
    )
    detalhado['ticket_medio'] = detalhado['ticket_medio'].fillna(detalhado['valor_trecho'])


    # 6. Cardinalidade acumulada (otimizada para performance)
    detalhado['place_destination_departure'] = detalhado['place_destination_departure'].astype(str)
    detalhado['fk_departure_ota_bus_company'] = detalhado['fk_departure_ota_bus_company'].astype(str)
    detalhado['purchase_datetime'] = pd.to_datetime(detalhado['purchase_datetime'])

    # Criar colunas auxiliares para mês e dia da semana como inteiros (atuais)
    detalhado['mes_compra'] = detalhado['purchase_datetime'].dt.month.astype(int)
    detalhado['dia_semana_compra'] = detalhado['purchase_datetime'].dt.weekday.astype(int)
    detalhado['mes_atual'] = detalhado['mes_compra']
    detalhado['dia_semana_atual'] = detalhado['dia_semana_compra']
    # Intervalo médio entre compras (rolling mean)
    detalhado['intervalo_entre_compras'] = detalhado.groupby('fk_contact')['purchase_datetime'].diff().dt.days
    detalhado['intervalo_medio'] = detalhado.groupby('fk_contact')['intervalo_entre_compras'].transform(lambda x: x.expanding().mean().shift().fillna(0))

    # Tempo desde a primeira compra
    detalhado['primeira_compra'] = detalhado.groupby('fk_contact')['purchase_datetime'].transform('min')
    detalhado['tempo_desde_primeira'] = (detalhado['purchase_datetime'] - detalhado['primeira_compra']).dt.days

    # Trecho mais frequente do cliente até o momento (rolling mode, sem expanding para string)
    def rolling_mode_fast(s):
        result = []
        counts = {}
        for val in s.shift().fillna(''):  # shift para não usar a linha atual
            if val != '':
                counts[val] = counts.get(val, 0) + 1
            if counts:
                mode_val = max(counts, key=counts.get)
                result.append(mode_val)
            else:
                result.append(np.nan)
        return pd.Series(result, index=s.index)
    detalhado['trecho_mais_frequente'] = detalhado.groupby('fk_contact')['trecho'].transform(rolling_mode_fast)
    detalhado['trecho_mais_frequente'] = detalhado['trecho_mais_frequente'].fillna(detalhado['trecho'])

    # Empresa mais frequente do cliente até o momento (rolling mode, sem expanding para string)
    def rolling_mode_empresa_fast(s):
        result = []
        counts = {}
        for val in s.shift().fillna(''):
            if val != '':
                counts[val] = counts.get(val, 0) + 1
            if counts:
                mode_val = max(counts, key=counts.get)
                result.append(mode_val)
            else:
                result.append(np.nan)
        return pd.Series(result, index=s.index)
    detalhado['empresa_mais_frequente'] = detalhado.groupby('fk_contact')['fk_departure_ota_bus_company'].transform(rolling_mode_empresa_fast)
    detalhado['empresa_mais_frequente'] = detalhado['empresa_mais_frequente'].fillna(detalhado['fk_departure_ota_bus_company'])

    # Último trecho realizado (shift)
    detalhado['ultimo_trecho'] = detalhado.groupby('fk_contact')['trecho'].shift(1).fillna(detalhado['trecho'])

    # Cardinalidade acumulada vetorizada e eficiente
    def acumulada_nunique(grupo):
        return (~grupo.duplicated()).cumsum()

    detalhado['destinos_unicos'] = detalhado.groupby('fk_contact')['place_destination_departure'].transform(acumulada_nunique)
    detalhado['empresas_diferentes'] = detalhado.groupby('fk_contact')['fk_departure_ota_bus_company'].transform(acumulada_nunique)
    detalhado['meses_distintos'] = detalhado.groupby('fk_contact')['mes_compra'].transform(acumulada_nunique)
    detalhado['dias_semana_distintos'] = detalhado.groupby('fk_contact')['dia_semana_compra'].transform(acumulada_nunique)


    # 7. recencia: diferença para próxima compra
    detalhado['recencia'] = detalhado['dias_ate_proxima']

    # 8. Montar DataFrame final (remover última compra de cada cliente, pois não tem target)
    df = detalhado.dropna(subset=['target_trecho']).copy()

    # 9. Agrupar classes raras e limitar para as N mais frequentes
    N_TOP = 20  # top N trechos mais frequentes
    counts = df['target_trecho'].value_counts()
    top_classes = counts.nlargest(N_TOP).index
    df['target_trecho_agrupado'] = np.where(df['target_trecho'].isin(top_classes), df['target_trecho'], 'outros')
    # Remove 'outros' do treino/teste se quiser só as classes mais frequentes:
    df = df[df['target_trecho_agrupado'] != 'outros']

    # 10. Features e target (incluindo novas features)
    feature_cols = [
        'recencia', 'frequencia', 'monetario', 'ticket_medio',
        'destinos_unicos', 'empresas_diferentes', 'meses_distintos', 'dias_semana_distintos',
        'mes_atual', 'dia_semana_atual', 'intervalo_medio', 'tempo_desde_primeira',
        'trecho_mais_frequente', 'empresa_mais_frequente', 'ultimo_trecho'
    ]
    # Adiciona cluster se existir
    if 'cluster' in df.columns:
        feature_cols.append('cluster')
    # Codificar features categóricas
    for col in ['trecho_mais_frequente', 'empresa_mais_frequente', 'ultimo_trecho', 'cluster']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
    X = df[feature_cols]
    y = df['target_trecho_agrupado']
    idx = df.index.values

    # 11. Codificar target
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)

    # 12. Padronizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 13. Split (incluindo índice)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y_enc, idx, test_size=0.2, random_state=42, stratify=y_enc
    )

    # 14. Baseline: prever trecho mais comum do dataset
    trecho_mais_comum = df['target_trecho'].mode()[0]
    y_pred_baseline = np.full_like(y_test, fill_value=le_target.transform([trecho_mais_comum])[0])
    print("\n--- Baseline: Previsão do trecho mais comum do dataset ---")
    print(classification_report(
        le_target.inverse_transform(y_test),
        le_target.inverse_transform(y_pred_baseline),
        zero_division=0
    ))
    print(f"Acurácia: {accuracy_score(le_target.inverse_transform(y_test), le_target.inverse_transform(y_pred_baseline)):.2f}")

    # 15. Modelo de classificação multi-classe (XGBoost ou LogisticRegression)
    # Troque para LogisticRegression se quiser testar modelo mais simples
    from sklearn.linear_model import LogisticRegression
    # clf = XGBClassifier(
    #     objective='multi:softmax',
    #     num_class=len(le_target.classes_),
    #     eval_metric='mlogloss',
    #     random_state=42,
    #     n_jobs=-1,
    #     n_estimators=10,
    #     max_depth=2
    # )
    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n--- Classificação: Previsão do próximo trecho (LogisticRegression) ---")
    print(classification_report(
        le_target.inverse_transform(y_test),
        le_target.inverse_transform(y_pred),
        zero_division=0
    ))
    print(f"Acurácia: {accuracy_score(le_target.inverse_transform(y_test), le_target.inverse_transform(y_pred)):.2f}")

    # 16. Salvar real vs previsto
    resultados = pd.DataFrame({
        'fk_contact': df.loc[idx_test, 'fk_contact'].values,
        'trecho_real': le_target.inverse_transform(y_test),
        'trecho_previsto': le_target.inverse_transform(y_pred),
        'cluster': df.loc[idx_test, 'cluster'].values if 'cluster' in df.columns else None
    })
    resultados.to_csv(output_csv, index=False)
    print(f"Resultados de classificação salvos em {output_csv}")

    # 16.1. Salvar métricas por cluster (se disponível)
    if 'cluster' in df.columns:
        print("\n--- Métricas por cluster ---")
        for cl in resultados['cluster'].unique():
            mask = resultados['cluster'] == cl
            print(f"\nCluster: {cl}")
            print(classification_report(
                resultados.loc[mask, 'trecho_real'],
                resultados.loc[mask, 'trecho_previsto'],
                zero_division=0
            ))

    # 17. Regressão para dias até próxima compra (XGBoost)
    reg = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1, n_estimators=10, max_depth=2)
    reg.fit(X_train, df.loc[idx_train, 'dias_ate_proxima'])
    y_pred_reg = reg.predict(X_test)
    rmse = sqrt(mean_squared_error(df.loc[idx_test, 'dias_ate_proxima'], y_pred_reg))
    print(f"\n--- Regressão: Dias até próxima compra (XGBoostRegressor) ---")
    print(f"RMSE: {rmse:.2f}")



    # 18. Função para prever para cliente específico (última linha do histórico)
    def prever_para_cliente(fk_contact):
        hist = detalhado[detalhado['fk_contact'] == fk_contact].sort_values('purchase_datetime')
        if len(hist) < 2:
            print('Cliente sem histórico suficiente.')
            return
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