import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from scale_data import scale_data

import warnings
warnings.filterwarnings('ignore')

dataframe = pd.read_csv('abalone.txt', delimiter=",", header=None)
dataframe.columns=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

dataframe = dataframe.drop(dataframe[dataframe.Height > 0.4].index)
dataframe = dataframe.drop(dataframe[dataframe.Height ==0].index)
dataframe = dataframe.drop(dataframe[dataframe['Rings'] < 1].index | dataframe[dataframe['Rings'] > 10].index)

dataframe = pd.get_dummies(dataframe)

dataframe= dataframe.drop(['Sex_F'], axis=1)
dataframe= dataframe.drop(['Sex_M'], axis=1)
dataframe= dataframe.drop(['Viscera weight'], axis=1)
dataframe= dataframe.drop(['Shell weight'], axis=1)
dataframe= dataframe.drop(['Whole weight'], axis=1)


np.savetxt("dataset_reduced.txt", dataframe, delimiter=",")

X = dataframe.drop(['Rings'], axis = 1)
y = dataframe['Rings']

X = scale_data(X, type_of_scale='0')
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)


x_test_frame= pd.DataFrame(X_test)
y_test_frame= pd.DataFrame(y_test)
test_data= pd.concat([x_test_frame, y_test_frame], axis=1)
np.savetxt("test_data.txt", test_data, delimiter=",")

regr = linear_model.LinearRegression()
regr= regr.fit(X_train,y_train)
predict = regr.predict(X_train)

print('Coefficients: \n', regr.coef_)

betas= regr.coef_
betas= np.float16(betas)
print(betas)
np.savetxt("parameters.txt", betas, delimiter=",")

print('Intercept: \n', regr.intercept_)
print('score1:\n',r2_score(y_train, predict))
print('score2:\n',regr.score(X_train,y_train))


#print(dataframe.Rings)

#np.savetxt("dataset.txt", a, delimiter=",")

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
#dataframe=dataframe*10000
#dataframe=np.round(dataframe)


#sns.set()
#cols = ['Length','Diameter','Height', 'Shucked weight','Rings']
#sns.pairplot(dataframe[cols], height = 2.5)
#plt.show()


#predictions = regr.predict(X_test)
#plt.scatter(y_test, predictions)
#plt.xlabel('True Values')
#plt.ylabel('Predictions')
#plt.show()
