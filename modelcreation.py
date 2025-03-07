import pandas as pd
from datetime import datetime
import numpy as np
import random as python_random
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import tensorflow as tf

from utils import *
import const

# reprodutividade
seed = 41
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

# obter dados do db
df = fetch_data_from_db(const.consulta_sql)

# conversões básicas
df['idade'] = df['idade'].astype(int)
df['valorsolicitado'] = df['valorsolicitado'].astype(float)
df['valortotalbem'] = df['valortotalbem'].astype(float)

# tratar nulos
tratar_nulos(df)

# tratar erros de digitacao
profissioes_validas = ['Advogado', 'Arquiteto', 'Cientista de Dados', 'Contador', 'Dentista', 'Engenheiro', 'Médico', 'Programador']
tratar_erros_digitacao(df, 'profissao', profissioes_validas)

# tratar outliers
df = tratar_outliers(df, 'tempoprofissao', 0, 70)
df = tratar_outliers(df, 'idade', 0, 110)

# criar features
df['propsolicitadototal'] = df['valorsolicitado']/df['valortotalbem']
df['propsolicitadototal'] = df['propsolicitadototal'].astype(float)

# divisão em treino teste e validacao
X = df.drop('classe', axis = 1)
y = df['classe']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# normalização
X_test = save_scalers(X_test, ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem', 'propsolicitadototal'])
X_train = save_scalers(X_train, ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem', 'propsolicitadototal'])

# atribuir valores para variaveis categóricas
mapeamento = {'ruim': 0, 'bom':1}
y_train = np.array([mapeamento[item] for item in y_train])
y_test = np.array([mapeamento[item] for item in y_test])
X_train = save_encoders(X_train, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])
X_test = save_encoders(X_test, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])

# seleção de atributos
model = RandomForestClassifier()

# instancia do RFE
selector = RFE(model, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# transformar os dados
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
joblib.dump(selector, './objects/selector.joblib')

# Modelo
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Otimizando 
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compilar o modelo 
model.compile(optimizer= optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(
    X_train,
    y_train,
    validation_split=0.2, # 20% para validação
    epochs=500,
    batch_size=10,
    verbose=1
)

# Salvando o modelo
model.save('modelo_rnn.keras')

# Previsões
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Avaliando o modelo
print("Avaliação do Modelo com os Dados de Teste:")
model.evaluate(X_test, y_test)

# Métricas de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

