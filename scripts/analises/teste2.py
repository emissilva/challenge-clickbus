import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import numpy as np

# 1. Criar um DataFrame de exemplo com features RFM
data = {
    'recency': np.random.randint(1, 1000, 50),
    'frequency': np.random.randint(1, 10, 50),
    'monetary': np.random.randint(10, 1000, 50)
}
df_exemplo = pd.DataFrame(data)

# 2. Escalar as features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_exemplo)

# 3. Encontrar o número ideal de clusters com diferentes métricas
# Visualizador com a métrica de inércia (o Método do Cotovelo que usamos)
model = KMeans(random_state=42, n_init='auto')
visualizer_inertia = KElbowVisualizer(model, k=(2, 10), metric='distortion')
visualizer_inertia.fit(scaled_features)
visualizer_inertia.show()
plt.title('Método do Cotovelo com Inércia')
plt.show()

# Visualizador com a métrica Silhouette
visualizer_silhouette = KElbowVisualizer(model, k=(2, 10), metric='silhouette')
visualizer_silhouette.fit(scaled_features)
visualizer_silhouette.show()
plt.title('Método do Cotovelo com Silhouette Score')
plt.show()

# Visualizador com a métrica Calinski-Harabasz
visualizer_calinski = KElbowVisualizer(model, k=(2, 10), metric='calinski_harabasz')
visualizer_calinski.fit(scaled_features)
visualizer_calinski.show()
plt.title('Método do Cotovelo com Calinski-Harabasz')
plt.show()