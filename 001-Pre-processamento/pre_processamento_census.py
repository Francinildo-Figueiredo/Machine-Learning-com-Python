# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:42:20 2020

@author: franc
"""

import pandas as pd

base = pd.read_csv('census.csv')
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# Transformação de variáveis categóricas I

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# definindo um objeto do tipo LabelEncoder, que é um codificador de etiquetas
'''labelencoder_previsores = LabelEncoder()
labels = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])'''

# Transformação de variáveis categóricas II

#onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
'''
    A função OneHotEncoder é muito eficiente para codificar bases de dados categóricas
    com diversos atributos e nomes diferentes.
    
    criando variáveis do tipo 'dummy', onde cada categória será transformada em números
    discretos 0 e 1 
'''
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencoder.fit_transform(previsores).toarray()

# A função Labelencoder é mais eficiente em arrays com apenas uma coluna e respostas duplas
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# Escalonando as variáveis

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
previsores = scale.fit_transform(previsores)
