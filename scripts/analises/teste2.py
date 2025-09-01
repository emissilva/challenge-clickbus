import pandas as pd
import numpy as np

# 1. Criar um DataFrame de exemplo (substitua com os seus dados)
data = {
    'valor': [100, 105, 110, 95, 103, 150, 98, 101, 2510, 1002]
}
df = pd.DataFrame(data)

# 2. Calcular a média e o desvio padrão
media = df['valor'].mean()
desvio_padrao = df['valor'].std()

# 3. Definir um limite para outliers (por exemplo, 3 desvios padrão)
limite_superior = media + (3 * desvio_padrao)
limite_inferior = media - (3 * desvio_padrao)

# 4. Identificar e remover os outliers
# Identifica os outliers (valores que estão fora dos limites)
outliers = df[(df['valor'] > limite_superior) | (df['valor'] < limite_inferior)]
df_sem_outliers = df[(df['valor'] <= limite_superior) & (df['valor'] >= limite_inferior)]

print(f"Média: {media:.2f}")
print(f"Desvio Padrão: {desvio_padrao:.2f}")
print(f"Limite Superior: {limite_superior:.2f}")
print("\nDataFrame com Outliers:")
print(outliers.to_markdown(index=False, numalign="left", stralign="left"))
print("\nDataFrame sem Outliers:")
print(df_sem_outliers.to_markdown(index=False, numalign="left", stralign="left"))