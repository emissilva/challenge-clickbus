import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import argparse

# Configurar o argparse para a janela de previsão
parser = argparse.ArgumentParser(description='Script para prever a próxima compra.')
parser.add_argument('prediction_window', type=int, default=30, nargs='?',
                    help='Janela de tempo para a previsão (em dias, ex: 7 ou 30).')
args = parser.parse_args()
PREDICTION_WINDOW_DAYS = args.prediction_window

# 1. Carregar os arquivos de entrada
df_tratado = pd.read_csv('data/df_tratado.csv')
df_clusters_rfm = pd.read_csv('data/resultados/resultado_p1.csv')

# 2. Pré-processamento e criação das features e targets
df_tratado['date_purchase'] = pd.to_datetime(df_tratado['date_purchase'])
df_tratado_sorted = df_tratado.sort_values(['fk_contact', 'date_purchase'])

# Criar o target para a classificação (compra nos próximos X dias)
cutoff_date_class = df_tratado['date_purchase'].max() - pd.Timedelta(days=PREDICTION_WINDOW_DAYS)
df_predict_class = df_tratado[df_tratado['date_purchase'] > cutoff_date_class].copy()
df_target_class = df_predict_class.groupby('fk_contact').agg(
    purchased_in_window=('nk_ota_localizer_id', 'count')
).reset_index()
df_target_class['target_class'] = np.where(df_target_class['purchased_in_window'] > 0, 1, 0)
df_target_class = df_target_class.drop(columns=['purchased_in_window'])

# Criar o target para a regressão (dias até a próxima compra)
df_tratado_sorted['next_purchase_date'] = df_tratado_sorted.groupby('fk_contact')['date_purchase'].shift(-1)
df_tratado_sorted['days_to_next_purchase'] = (df_tratado_sorted['next_purchase_date'] - df_tratado_sorted['date_purchase']).dt.days
df_target_reg = df_tratado_sorted.groupby('fk_contact')['days_to_next_purchase'].min().reset_index()
df_target_reg.rename(columns={'days_to_next_purchase': 'target_reg'}, inplace=True)

# 3. Juntar features e targets
data = pd.merge(df_clusters_rfm, df_target_class, on='fk_contact', how='left')
data = pd.merge(data, df_target_reg, on='fk_contact', how='left')

# Preencher NaN
data['target_class'] = data['target_class'].fillna(0)
data['target_reg'] = data['target_reg'].fillna(data['target_reg'].median()) # Preenche com a mediana
data = data.dropna(subset=['cluster']) # Remove clientes que não foram clusterizados

# 4. Separar features (X) e targets (y)
X = data[['recency', 'frequency', 'monetary', 'cluster']]
y_class = data['target_class']
y_reg = data['target_reg']

# 5. Treinar e avaliar o modelo de CLASSIFICAÇÃO
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
model_class = XGBClassifier(eval_metric='logloss', random_state=42)
model_class.fit(X_train_class, y_train_class)
predictions_class = model_class.predict(X_test_class)
accuracy_class = accuracy_score(y_test_class, predictions_class)
print(f"Acurácia do modelo de Classificação (Janela de {PREDICTION_WINDOW_DAYS} dias): {accuracy_class:.2f}")

# 6. Treinar e avaliar o modelo de REGRESSÃO
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
model_reg = XGBRegressor(eval_metric='rmse', random_state=42)
model_reg.fit(X_train_reg, y_train_reg)
predictions_reg = model_reg.predict(X_test_reg)
rmse_reg = np.sqrt(mean_squared_error(y_test_reg, predictions_reg))
print(f"RMSE do modelo de Regressão: {rmse_reg:.2f} dias")

# 7. Salvar as previsões
predictions_df = X_test_class.copy()
predictions_df['predicted_class'] = predictions_class
predictions_df['predicted_reg'] = predictions_reg
predictions_df.to_csv('data/resultados/resultado_p2.csv', index=False)