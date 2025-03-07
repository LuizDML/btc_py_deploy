from fuzzywuzzy import process # lógica difusa
import pandas as pd 
from sklearn.preprocessing import StandardScaler, LabelEncoder # padronização
import joblib # salva objetos
import yaml # ler arquivos de confg
import psycopg2 as ps #postgres

import const # arquivo de constantes

def fetch_data_from_db(sql_query):
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        con = ps.connect(# evitar hardcoding 

            dbname=config['database_config']['dbname'], 
            user=config['database_config']['user'], 
            password=config['database_config']['password'], 
            host=config['database_config']['host']
        )

        cursor = con.cursor()
        cursor.execute(sql_query)

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()

    return df
                                                            
def tratar_nulos(df):
    for coluna in df.columns:
        if df[coluna].dtype == 'object':
            # trocar nulos com a moda em caso de categoricos
            moda = df[coluna].mode()[0] 
            df.loc[:, coluna] = df[coluna].fillna(moda)
            #df[coluna] = df[coluna].fillna(moda) 
            #df[coluna].fillna(moda, inplace=True)
        else:
            # trocar nulos com a mediana em caso de numéricos
            mediana = df[coluna].median()
            df.loc[:, coluna] = df[coluna].fillna(mediana)
            #df[coluna] = df[coluna].fillna(mediana)  
            #df[coluna].fillna(mediana, inplace=True)

def tratar_erros_digitacao(df, coluna, lista_valida):
    for i, valor in enumerate(df[coluna]):
        valor_str = str(valor) if pd.notnull(valor) else valor
        
        if valor_str not in lista_valida and pd.notnull(valor_str):
            correcao = process.extractOne(valor_str, lista_valida)[0]
            df.at[i, coluna] = correcao

def tratar_outliers(df, coluna, minimo, maximo):
    mediana = df[(df[coluna] >= minimo) & (df[coluna] <= maximo)][coluna].median()
    df[coluna] = df[coluna].apply(lambda x: mediana if x <minimo or x > maximo else x)
    
    return df

def save_scalers(df, nome_colunas):
    for nome_coluna in nome_colunas:
        scaler = StandardScaler()
        df[nome_coluna] = scaler.fit_transform(df[[nome_coluna]])
        joblib.dump(scaler, f"./objects/scaler{nome_coluna}.joblib")
        
    return df

def save_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        label_encoder = LabelEncoder()
        df[nome_coluna] = label_encoder.fit_transform(df[nome_coluna])
        joblib.dump(label_encoder, f"./objects/labelencoder{nome_coluna}.joblib")
        
    return df
    
def load_scalers(df, nome_colunas):
    for nome_coluna in nome_colunas:
        nome_arqv_scaler = f"./objects/scaler{nome_coluna}.joblib"
        scaler = joblib.load(nome_arqv_scaler)
        df[nome_coluna] = scaler.transform(df[[nome_coluna]])
    return df


def load_encoders(df, nome_colunas):
    for nome_coluna in nome_colunas:
        nome_arqv_encod = f"./objects/labelencoder{nome_coluna}.joblib"
        scaler = joblib.load(nome_arqv_encod)
        df[nome_coluna] = scaler.transform(df[[nome_coluna]])
    return df