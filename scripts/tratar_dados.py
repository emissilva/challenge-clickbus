import pandas as pd

# Carregar os dados
df = pd.read_csv('data/df_amostragem.csv')
print(f"O DataFrame está vazio? {df.empty}")


# Agrupar colunas por tipo de hash
place_geral = [
    'place_origin_departure',
    'place_destination_departure',
    'place_origin_return',
    'place_destination_return'
]
ota_bus_geral = [
    'fk_departure_ota_bus_company',
    'fk_return_ota_bus_company'
]
contact_geral = [
    'fk_contact'
]

# Função para criar e aplicar o mapeamento de hashes para IDs
def create_group_mapping(df, columns, prefix):
    all_hashes = pd.concat([df[col].drop_duplicates() for col in columns]).unique()
    mapping = {
        hash_val: f'{prefix}_{i+1}' for i, hash_val in enumerate(all_hashes) if hash_val != '0'
    }
    return mapping

# Criar e aplicar os mapeamentos para cada grupo
place_mapping = create_group_mapping(df, place_geral, 'rodoviaria')
for col in place_geral:
    df[col] = df[col].map(place_mapping).fillna(df[col])

ota_bus_mapping = create_group_mapping(df, ota_bus_geral, 'empresa')
for col in ota_bus_geral:
    df[col] = df[col].map(ota_bus_mapping).fillna(df[col])

# Criar e aplicar os mapeamentos para colunas únicas
for col in contact_geral:
    unique_hashes = df[col].unique()
    mapping = {
        hash_val: f'cliente_{i+1}' for i, hash_val in enumerate(unique_hashes) if hash_val != '0'
    }
    df[col] = df[col].map(mapping).fillna(df[col])

# Salvar o resultado
df.to_csv('data/df_tratado.csv', index=False)

print("Tratamento de hashes concluído.")