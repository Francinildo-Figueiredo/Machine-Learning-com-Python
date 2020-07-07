# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 16:20:50 2020

@author: franc
"""


import pandas as pd
base = pd.read_csv('risco_credito.csv')

previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 1] = labelencoder.fit_transform(previsores[:, 1])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
# treinamento do algoritimo
classificador.fit(previsores, classe)

# teste do algoritimo
# histórico bom = 0, dívida alta = 0, garantia nenhuma = 1, renda > 35 = 2
# histórico ruim = 2, dívida alta = 0, garantia adequada = 0, renda < 15 = 0
resultado = classificador.predict([[0,0,1,2],[2,0,0,0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)
