#Não esqueça de usar usar o comando: pip install -r requirements.txt


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, vstack
import warnings

def carregar_e_preparar_dados(caminho_users, caminho_restaurants):
    """Carrega, limpa e prepara os datasets."""
    try:
        users_df = pd.read_csv(caminho_users)
        restaurants_df = pd.read_csv(caminho_restaurants, sep=';')
    except FileNotFoundError:
        raise Exception("ERRO: Datasets não encontrados na pasta 'data/'.")

    users_df['user_id'] = users_df.index
    restaurants_df['restaurant_id'] = restaurants_df.index

    numeric_cols_rest = ['delivery_fee', 'delivery_time', 'distance', 'minimumOrderValue', 'rating_final']
    for col in numeric_cols_rest:
        restaurants_df[col] = pd.to_numeric(restaurants_df[col], errors='coerce')
    restaurants_df[numeric_cols_rest] = restaurants_df[numeric_cols_rest].fillna(restaurants_df[numeric_cols_rest].median())

    numeric_cols_user = ['Preferencia_Espera', 'Distancia', 'Rating_Padrao']
    for col in numeric_cols_user:
        users_df[col] = pd.to_numeric(users_df[col], errors='coerce')
    users_df[numeric_cols_user] = users_df[numeric_cols_user].fillna(users_df[numeric_cols_user].median())

    restaurants_df['tags_final_list'] = restaurants_df['tags_final'].str.replace(r"[\[\]',]", "", regex=True).str.split().apply(lambda d: d if isinstance(d, list) else [])
    
    return users_df, restaurants_df, numeric_cols_rest

def criar_perfis(users_df, restaurants_df, numeric_cols_rest):
    """Cria os perfis numéricos para usuários e restaurantes."""
    encoder_rest = OneHotEncoder(handle_unknown='ignore')
    encoded_data_rest_sparse = encoder_rest.fit_transform(restaurants_df[['category', 'price_range', 'Cidade']])
    
    mlb_tags = MultiLabelBinarizer(sparse_output=True)
    tags_encoded_sparse = mlb_tags.fit_transform(restaurants_df['tags_final_list'])
    
    scaler_rest = MinMaxScaler()
    numerical_data_rest = scaler_rest.fit_transform(restaurants_df[numeric_cols_rest])
    
    restaurant_profiles_sparse = hstack([numerical_data_rest, encoded_data_rest_sparse, tags_encoded_sparse]).tocsr()
    
    user_vectors_list = []
    for _, user_row in users_df.iterrows():
        temp_cat_df = pd.DataFrame({'category': [user_row['Categoria_Preferida']], 'price_range': [np.nan], 'Cidade': [np.nan]}, dtype='object')
        user_cat_sparse = encoder_rest.transform(temp_cat_df)
        
        temp_numeric_data = [[0, user_row['Preferencia_Espera'], user_row['Distancia'], 0, user_row['Rating_Padrao']]]
        user_num_scaled = scaler_rest.transform(temp_numeric_data)
        
        user_tags_sparse = mlb_tags.transform([[]])
        
        user_vector_sparse = hstack([user_num_scaled, user_cat_sparse, user_tags_sparse]).tocsr()
        user_vectors_list.append(user_vector_sparse)
        
    user_profiles_sparse = vstack(user_vectors_list)
    
    if np.isnan(user_profiles_sparse.data).any():
        user_profiles_sparse.data = np.nan_to_num(user_profiles_sparse.data, nan=0.0)
        
    return user_profiles_sparse, restaurant_profiles_sparse

def gerar_recomendacoes(user_id, restaurants_df, user_profiles_sparse, restaurant_profiles_sparse):
    """Calcula e retorna as 10 melhores recomendações para um usuário."""
    user_vector = user_profiles_sparse[user_id]
    cosine_sim = cosine_similarity(user_vector, restaurant_profiles_sparse)
    sim_scores = sorted(list(enumerate(cosine_sim[0])), key=lambda x: x[1], reverse=True)
    top_scores = sim_scores[:10]
    restaurant_indices = [i[0] for i in top_scores]
    
    recommendations = restaurants_df.iloc[restaurant_indices].copy()
    recommendations['similarity_score'] = [s[1] for s in top_scores]
    return recommendations

# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    warnings.filterwarnings('ignore', message='X does not have valid feature names')

    users, restaurants, numeric_cols = carregar_e_preparar_dados('Data/users_data.csv', 'Data/ifood-restaurants-enriquecido.csv')
    user_profiles, restaurant_profiles = criar_perfis(users, restaurants, numeric_cols)
    
    # Seleciona um usuário aleatório para o teste
    target_user_id = np.random.choice(users.index)
    
    # Pega as informações do usuário selecionado
    user_info = users.loc[target_user_id]
    
    # Gera as recomendações
    recomendacoes = gerar_recomendacoes(target_user_id, restaurants, user_profiles, restaurant_profiles)
    
    # --- Exibição dos Resultados ---
    
    # 1. Mostra o perfil do usuário

    print(f"--- ANÁLISE PARA O USUÁRIO ID: {target_user_id} ---")
    print("PERFIL DO USUÁRIO:")
    user_display_cols = ['Nome', 'Idade', 'Genero', 'Categoria_Preferida', 'Preferencia_Gasto', 
                         'Preferencia_Espera', 'Distancia', 'Rating_Padrao']
    print(user_info[user_display_cols].to_string())

    # 2. Mostra a tabela de recomendações
    print("\nRECOMENDAÇÕES GERADAS:")
    display_cols = ['category', 'rating_final', 'delivery_time', 'price_range', 'distance', 'similarity_score']
    print(recomendacoes[display_cols].to_string())