import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

# Carregar o DataFrame com o target de regressão
df_tratado = pd.read_csv('challenge-clickbus/data/df_tratado.csv')
df_tratado['date_purchase'] = pd.to_datetime(df_tratado['date_purchase'])
df_tratado_sorted = df_tratado.sort_values(['fk_contact', 'date_purchase'])
df_tratado_sorted['next_purchase_date'] = df_tratado_sorted.groupby('fk_contact')['date_purchase'].shift(-1)
df_tratado_sorted['days_to_next_purchase'] = (df_tratado_sorted['next_purchase_date'] - df_tratado_sorted['date_purchase']).dt.days

# Remover as linhas com NaN antes de calcular o target
df_tratado_sorted.dropna(subset=['days_to_next_purchase'], inplace=True)

df_target_reg = df_tratado_sorted.groupby('fk_contact')['days_to_next_purchase'].min().reset_index()

# Calcular a média do target
mean_days_to_next_purchase = df_target_reg['days_to_next_purchase'].mean()
print(f"Média de dias para a próxima compra: {mean_days_to_next_purchase:.2f} dias")

# Criar a previsão da linha de base
baseline_predictions = np.full(len(df_target_reg), mean_days_to_next_purchase)

# Calcular o RMSE da linha de base
rmse_baseline = np.sqrt(mean_squared_error(df_target_reg['days_to_next_purchase'], baseline_predictions))
print(f"RMSE da linha de base: {rmse_baseline:.2f} dias")