# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:38:27 2020

@author: franc
"""


import pandas as pd
import numpy as np
base = pd.read_csv('credit_data.csv')
base.loc[ base.age < 0, 'age' ] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer = imputer.fit(previsores[:, 1:4])
previsores = imputer.fit_transform(previsores)

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
previsores = scale.fit_transform(previsores)

# Dividindo a base de dados entre treinamento e teste, nenhuma das linhas de teste são
# iguais as do treinamento.

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
