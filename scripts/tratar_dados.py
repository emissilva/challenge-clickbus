import pandas as pd
import sys

try:
    # 1. Carregar os dados
    df = pd.read_csv('data/df_amostragem.csv')
    
    # 2. Agrupar colunas por tipo de hash
    place_columns = [
        'place_origin_departure',
        'place_destination_departure',
        'place_origin_return',
        'place_destination_return'
    ]
    ota_bus_columns = [
        'fk_departure_ota_bus_company',
        'fk_return_ota_bus_company'
    ]
    contact_columns = [
        'fk_contact'
    ]

    # 3. Função para criar e aplicar o mapeamento de hashes para IDs
    def create_group_mapping(df, columns, prefix):
        all_hashes = pd.concat([df[col].drop_duplicates() for col in columns]).unique()
        mapping = {
            hash_val: f'{prefix}_{i+1}' for i, hash_val in enumerate(all_hashes) if hash_val != '0'
        }
        return mapping

    # 4. Criar e aplicar os mapeamentos para cada grupo
    place_mapping = create_group_mapping(df, place_columns, 'rodoviaria')
    for col in place_columns:
        df[col] = df[col].map(place_mapping).fillna(df[col])

    ota_bus_mapping = create_group_mapping(df, ota_bus_columns, 'empresa')
    for col in ota_bus_columns:
        df[col] = df[col].map(ota_bus_mapping).fillna(df[col])

    # 5. Criar e aplicar os mapeamentos para colunas únicas
    for col in contact_columns:
        unique_hashes = df[col].unique()
        mapping = {
            hash_val: f'cliente_{i+1}' for i, hash_val in enumerate(unique_hashes) if hash_val != '0'
        }
        df[col] = df[col].map(mapping).fillna(df[col])

    # 6. Salvar o resultado
    df.to_csv('data/df_tratado.csv', index=False)

    print("Tratamento de hashes concluído.")
    sys.exit(0)  # Código de saída de sucesso

except Exception as e:
    print(f"Erro na execução do script: {e}")
    sys.exit(1) # Código de saída de erro