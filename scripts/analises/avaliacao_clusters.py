


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tamanho da amostra para o pairplot
amostra_size = 1000

# Leitura direta do arquivo tratado
df = pd.read_csv('dados/resultados/desafio_1/df_tratado.csv')
df['cluster'] = df['cluster'].astype(str)


# Principais features para análise
main_feats = ['recencia', 'frequencia', 'monetario', 'ticket_medio', 'destinos_unicos']

# Resumo estatístico por cluster

print('Resumo estatístico (mediana) das principais features por cluster:')
summary = df.groupby('cluster')[main_feats].median().sort_values(by='frequencia', ascending=False)
print(summary)

# Sugerir nomes automáticos para clusters com base na frequência, ticket_medio e recencia
def sugerir_nome(row):
    if row['frequencia'] >= 4:
        return 'Clientes muito frequentes'
    elif row['frequencia'] == 3:
        return 'Clientes frequentes'
    elif row['frequencia'] == 2:
        return 'Clientes ocasionais'
    elif row['frequencia'] == 1:
        if row['ticket_medio'] > summary['ticket_medio'].median():
            return 'Clientes de ticket alto e baixa frequência'
        else:
            return 'Clientes de baixo valor'
    else:
        return 'Outros'

cluster_sugestoes = summary.apply(sugerir_nome, axis=1)
# Garantir que 'outliers' mantenha o nome 'Outliers'
if 'outliers' in df['cluster'].unique():
    cluster_sugestoes['outliers'] = 'Outliers'

print('\nSugestão de nomes para clusters:')
for cl, nome in cluster_sugestoes.items():
    print(f"{cl}: {nome}")

# Atualizar nomes no DataFrame
cluster_names = cluster_sugestoes.to_dict()
df['cluster_nome'] = df['cluster'].map(cluster_names).fillna(df['cluster'])

# SUGESTÃO DE NOMES (ajuste conforme análise do summary)
cluster_names = {
    'cluster_1': 'Clientes muito frequentes',
    'cluster_2': 'Clientes frequentes',
    'cluster_3': 'Clientes ocasionais',
    'cluster_4': 'Clientes de baixo valor',
    'outliers': 'Outliers'
}
df['cluster_nome'] = df['cluster'].map(cluster_names).fillna(df['cluster'])


sns.countplot(x='cluster_nome', data=df, hue='cluster_nome', palette='tab10', legend=False)
plt.title('Distribuição de clientes por cluster')
plt.xlabel('Perfil do cluster')
plt.close()


# Boxplot de recência por cluster (com nomes) - apenas salvar
sns.boxplot(x='cluster_nome', y='recencia', data=df)
plt.title('Recência por perfil de cluster')
plt.xlabel('Perfil do cluster')
plt.close()


# Pairplot das principais features (sem outliers, com nomes) - AMOSTRA RANDOMICA - apenas salvar
df_no_outliers = df[df['cluster'] != 'outliers']
amostra = df_no_outliers.sample(n=min(amostra_size, len(df_no_outliers)), random_state=None)
sns.pairplot(amostra[main_feats + ['cluster_nome']], hue='cluster_nome')
plt.close()


# Gráfico 1: Distribuição de clientes por cluster (com nomes) - salvar figura
plt.figure(figsize=(8, 5))
sns.countplot(x='cluster_nome', data=df, hue='cluster_nome', palette='tab10', legend=False)
plt.title('Quantidade de clientes por cluster')
plt.xlabel('Perfil do cluster')
plt.ylabel('Quantidade de clientes')
plt.tight_layout()
plt.savefig('dados/resultados/analises/grafico_clientes_por_cluster.png')
plt.close()

# Gráfico 2: Boxplot das features por cluster
features = ['recencia', 'frequencia', 'monetario', 'ticket_medio', 'destinos_unicos']
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='cluster_nome', y=feature, data=df, hue='cluster_nome', palette='tab10', legend=False)
    plt.title(f'{feature.capitalize()} por cluster')
    plt.xlabel('Cluster')
    plt.ylabel(feature.capitalize())
    plt.tight_layout()
    plt.savefig(f'dados/resultados/analises/boxplot_{feature}_por_cluster.png')
    plt.close()

# Gráfico 3: Matriz de dispersão das features principais (exceto outliers)
df_sem_outlier = df[df['cluster_nome'] != 'Outliers']
sns.pairplot(df_sem_outlier, vars=features, hue='cluster_nome', palette='tab10', diag_kind='kde')
plt.suptitle('Matriz de dispersão das features por cluster (sem outliers)', y=1.02)
plt.savefig('dados/resultados/analises/pairplot_clusters.png')
plt.close()

print("Gráficos salvos em dados/analises/graficos/")