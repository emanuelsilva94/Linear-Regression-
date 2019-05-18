import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import skew
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')

dataframe = pd.read_csv('abalone.txt', delimiter=",", header=None)
dataframe.columns=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

dataframe = dataframe.drop(dataframe[dataframe.Height > 0.4].index)
dataframe = dataframe.drop(dataframe[dataframe.Height ==0].index)


dataframe = pd.get_dummies(dataframe)
print(dataframe.describe())

dataframe['Length'] = (dataframe['Length']**2)
dataframe['Diameter'] = (dataframe['Diameter']**2)
dataframe['Height'] = (dataframe['Height']**2)
dataframe['Shell weight'] = (dataframe['Shell weight']**(1/3))
dataframe['Shell weight'] = (dataframe['Shell weight']**2)
dataframe['Viscera weight'] = (dataframe['Viscera weight']**(1/3))
dataframe['Viscera weight'] = (dataframe['Viscera weight']**2)
dataframe['Shucked weight'] = (dataframe['Shucked weight']**(1/3))
dataframe['Shucked weight'] = (dataframe['Shucked weight']**2)
dataframe['Whole weight'] = (dataframe['Whole weight']**(1/3))
dataframe['Whole weight'] = (dataframe['Whole weight']**2)
dataframe['Height'] = (dataframe['Height']**(1/3))
dataframe['Height'] = (dataframe['Height']**2)
dataframe['Shucked weight'] = (dataframe['Shucked weight']**(1/3))
dataframe['Shucked weight'] = (dataframe['Shucked weight']**(2))
dataframe['Shucked weight'] = (dataframe['Shucked weight']**(2))
dataframe['Shucked weight'] = (dataframe['Shucked weight']**(1/3))
dataframe['Shucked weight'] = (dataframe['Shucked weight']**(2))
dataframe['Rings'] = (dataframe['Rings']**(1/3))
dataframe['Rings'] = (dataframe['Rings']**(1/3))
dataframe['Rings'] = (dataframe['Rings']**(2))


dataframe['Rings'] = (dataframe['Rings']**(1/3))
dataframe['Rings'] = (dataframe['Rings']**(1/3))
max_value=dataframe['Rings'].max()
min_value=dataframe['Rings'].min()
mean_value=dataframe['Rings'].mean()
std_value=dataframe['Rings'].std()
#dataframe['Rings'] = (dataframe['Rings']-mean_value)/(std_value)
dataframe['Rings'] = (dataframe['Rings']-min_value)/(max_value-min_value)



max_value=dataframe['Length'].max()
min_value=dataframe['Length'].min()
mean_value=dataframe['Length'].mean()
std_value=dataframe['Length'].std()
#dataframe['Length'] = (dataframe['Length']-mean_value)/(std_value)
dataframe['Length'] = (dataframe['Length']-min_value)/(max_value-min_value)


max_value=dataframe['Diameter'].max()
min_value=dataframe['Diameter'].min()
mean_value=dataframe['Length'].mean()
std_value=dataframe['Length'].std()
#dataframe['Diameter'] = (dataframe['Diameter']-mean_value)/(std_value)
dataframe['Diameter'] = (dataframe['Diameter']-min_value)/(max_value-min_value)



max_value=dataframe['Height'].max()
min_value=dataframe['Height'].min()
mean_value=dataframe['Height'].mean()
std_value=dataframe['Height'].std()
#dataframe['Height'] = (dataframe['Height']-mean_value)/(std_value)
dataframe['Height'] = (dataframe['Height']-min_value)/(max_value-min_value)


max_value=dataframe['Whole weight'].max()
min_value=dataframe['Whole weight'].min()
mean_value=dataframe['Whole weight'].mean()
std_value=dataframe['Whole weight'].std()
#dataframe['Whole weight'] = (dataframe['Whole weight']-mean_value)/(std_value)
dataframe['Whole weight'] = (dataframe['Whole weight']-min_value)/(max_value-min_value)


max_value=dataframe['Shucked weight'].max()
min_value=dataframe['Shucked weight'].min()
mean_value=dataframe['Shucked weight'].mean()
std_value=dataframe['Shucked weight'].std()
#dataframe['Shucked weight'] = (dataframe['Shucked weight']-mean_value)/(std_value)
dataframe['Shucked weight'] = (dataframe['Shucked weight']-min_value)/(max_value-min_value)


max_value=dataframe['Viscera weight'].max()
min_value=dataframe['Viscera weight'].min()
mean_value=dataframe['Viscera weight'].mean()
std_value=dataframe['Viscera weight'].std()
#dataframe['Viscera weight'] = (dataframe['Viscera weight']-mean_value)/(std_value)
dataframe['Viscera weight'] = (dataframe['Viscera weight']-min_value)/(max_value-min_value)

max_value=dataframe['Shell weight'].max()
min_value=dataframe['Shell weight'].min()
mean_value=dataframe['Shell weight'].mean()
std_value=dataframe['Shell weight'].std()
#dataframe['Shell weight'] =(dataframe['Shell weight']-mean_value)/(std_value)
dataframe['Shell weight'] = (dataframe['Shell weight']-min_value)/(max_value-min_value)

dataframe = dataframe.drop(dataframe['Viscera weight'].idxmax())

#dataframe = dataframe.drop(dataframe['Length'].idxmax())
#dataframe = dataframe.drop(dataframe['Diameter'].idxmax())
#dataframe = dataframe.drop(dataframe['Height'].idxmax())
#dataframe = dataframe.drop(dataframe['Whole weight'].idxmax())
#dataframe = dataframe.drop(dataframe['Shell weight'].idxmax())


#dataframe = dataframe.drop(dataframe[dataframe.Length > 0.85].index | dataframe[dataframe.Length < 0.01].index)
#dataframe = dataframe.drop(dataframe[dataframe.Diameter > 0.85].index | dataframe[dataframe.Diameter < 0.01].index)
#dataframe = dataframe.drop(dataframe[dataframe.Height > 0.9].index | dataframe[dataframe.Height < 0.1].index)
#dataframe = dataframe.drop(dataframe[dataframe['Whole weight'] > 0.8].index | dataframe[dataframe['Whole weight'] < 0.01].index)
#dataframe = dataframe.drop(dataframe[dataframe['Viscera weight'] > 0.8].index | dataframe[dataframe['Viscera weight'] < 0.01].index)
#dataframe = dataframe.drop(dataframe[dataframe['Shucked weight'] > 0.75].index | dataframe[dataframe['Shucked weight'] < 0.01].index)
#dataframe = dataframe.drop(dataframe[dataframe['Viscera weight'] > 0.9].index | dataframe[dataframe['Viscera weight'] < 0.01].index )
#dataframe = dataframe.drop(dataframe[dataframe['Shell weight'] > 0.75].index | dataframe[dataframe['Shell weight'] > 0.75].index)
#dataframe = dataframe.drop(dataframe[dataframe.Rings <0.2].index | dataframe[dataframe.Rings ==1].index)






print( 'skewness of normal distribution (should be 0): {}'.format( skew(dataframe) ))

X = dataframe.drop(['Rings'], axis = 1)
y = dataframe['Rings']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.3)
data_teste= pd.concat([X_test,y_test], axis=1)
data_teste.to_csv('teste_data.txt', sep=',')
#np.savetxt("dataset.txt", a, delimiter=",")

regr = linear_model.LinearRegression()
regr= regr.fit(X_train,y_train)
predict = regr.predict(X_train)

print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print('score1:\n',r2_score(y_train, predict))
print('score2:\n',regr.score(X_train,y_train))

print('\n\n\n')

a = regr.coef_
a = np.round(a, 4)
regr.coef_ = a
print(a)
print('\n\n')

linear_model.LinearRegression.coef_=a
regr=linear_model.LinearRegression(normalize=True)
regr.coef_ = a
regr.fit(X_train,y_train)
predict = regr.predict(X_train)


#regr= regr.fit(X_train,y_train)
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)
print('score1:\n', r2_score(y_train, predict))
print('score2:\n', regr.score(X_train,y_train))




#np.savetxt("dataset.txt", a, delimiter=",")