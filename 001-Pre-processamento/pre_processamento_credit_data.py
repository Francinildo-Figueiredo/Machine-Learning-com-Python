# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:47:07 2020

@author: franc
"""

import pandas as pd
import numpy as np
base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age'] < 0]
# apagar a coluna
base.drop('age', 1, inplace=True)
# apagar apenas os elementos comprometidos
base.drop(base[base.age < 0].index, inplace=True)
# preencher os valores manualmente
# preencher os valores com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
# para modificação e pesquisa utiliza o loc
base.loc[base.age < 0, 'age'] = 40.92

# Tratamento de valores faltantes
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]
# cria uma copia da base de dados apenas com os valores previsores da coluna 1 até a 3
# comando iloc serve para divisão da base de dados
previsores = base.iloc[:, 1:4].values
# armazenando os dados da classe meta
casse = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
# definindo imputer como um objeto do tipo SimpleImputer, que substitui valores NaN pela média
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# retorna um SimpleImputer que emcaixa o imputer na base de dados
imputer = imputer.fit(previsores[:, 0:3])
# insere todos os valores ausentes transformados em média em previsores
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Escalnamento de atributos

from sklearn.preprocessing import StandardScaler
# definindo o objeto scaler d tipo dardScaler
scaler = StandardScaler()
# ajustando e transformando os valores de previsores para uma mesma escala padronizada
previsores = scaler.fit_transform(previsores)
