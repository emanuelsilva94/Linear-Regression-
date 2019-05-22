import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn import preprocessing
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#%matplotlib inline

import os


#-----------------read the dataset-------------------------#
df = pd.read_csv('abalone.txt', delimiter=",", header=None)
df.columns=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','Rings']

#-----------------head of the dataset----------------------#
print(df.head())


#-----------------information of each column---------------#
print(df.info())


#-----------------information about dataset---------------#
print(df.describe())


#-----------------null values height---------------#
df = df[df.Height > 0]

#-----------------distribution of rings ---------------#
plt.figure(figsize=(12, 10))

plt.subplot(2,2,1)
sns.countplot(df.Rings)

plt.subplot(2,2,2)
sns.distplot(df.Rings)

plt.subplot(2,2,3)
stats.probplot(df.Rings, plot=plt)

plt.subplot(2,2,4)
sns.boxplot(df.Rings)

plt.tight_layout()
#plt.show()

#-----------------distribution of dataset ---------------#
plt.figure(figsize=(12,10))
sns.pairplot(df)


#----------------correlation of dataset ---------------#
df.corr()
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True)
df.corr().Rings.sort_values(ascending=False)


#----------------each column vs rings---------------#
plt.figure(figsize=(15, 15))

plt.subplot(3,3,1)
plt.title('Shell weight vs Rings')
plt.scatter(df['Shell weight'],df['Rings'])

plt.subplot(3,3,2)
plt.title('Diameter vs Rings')
plt.scatter(df['Diameter'],df['Rings'])

plt.subplot(3,3,3)
plt.title('Height vs Rings')
plt.scatter(df['Height'],df['Rings'])

plt.subplot(3,3,4)
plt.title('Length vs Rings')
plt.scatter(df['Length'],df['Rings'])

plt.subplot(3,3,5)
plt.title('Whole weight vs Rings')
plt.scatter(df['Whole weight'],df['Rings'])

plt.subplot(3,3,6)
plt.title('Viscera weight vs Rings')
plt.scatter(df['Viscera weight'],df['Rings'])

plt.tight_layout()
#plt.show()

#----------------limitation of rings between 2-10------------#
new_df = df[df.Rings < 11]
new_df = new_df[new_df.Rings > 2]

#----------------??distrution between rings and lenght??-------#
plt.figure(figsize=(12,6))
sns.violinplot(data=new_df, x='Rings', y='Length')
plt.show()
print(new_df.head())
print(new_df.info())

#---------------- new each column vs rings---------------#
plt.figure(figsize=(12, 10))

plt.subplot(3,3,1)
plt.title('Shell weight vs Rings')
plt.scatter(new_df['Shell weight'],new_df['Rings'])

plt.subplot(3,3,2)
plt.title('Diameter vs Rings')
plt.scatter(new_df['Diameter'],new_df['Rings'])

plt.subplot(3,3,3)
plt.title('Height vs Rings')
plt.scatter(new_df['Height'],new_df['Rings'])

plt.subplot(3,3,4)
plt.title('Length vs Rings')
plt.scatter(new_df['Length'],new_df['Rings'])

plt.subplot(3,3,5)
plt.title('Whole weight vs Rings')
plt.scatter(new_df['Whole weight'],new_df['Rings'])

plt.subplot(3,3,6)
plt.title('Viscera weight vs Rings')
plt.scatter(new_df['Viscera weight'],new_df['Rings'])

plt.tight_layout()
#plt.show()

#---------------- outliars height---------------#

new_df = new_df[new_df.Height < 0.4]


#----------------????---------------#
plt.figure(figsize=(12,10))

plt.subplot(3,2,1)
sns.boxplot(data= new_df, x = 'Rings', y = 'Diameter')

plt.subplot(3,2,2)
sns.boxplot(data= new_df, x = 'Rings', y = 'Length')

plt.subplot(3,2,3)
sns.boxplot(data= new_df, x = 'Rings', y = 'Height')

plt.subplot(3,2,4)
sns.boxplot(data= new_df, x = 'Rings', y = 'Shell weight')

plt.subplot(3,2,5)
sns.boxplot(data= new_df, x = 'Rings', y = 'Whole weight')

plt.subplot(3,2,6)
sns.boxplot(data= new_df, x = 'Rings', y = 'Viscera weight')
plt.tight_layout()
#plt.show()

#---------------- ???---------------#
plt.figure(figsize=(12, 10))

plt.subplot(2,2,1)
sns.countplot(new_df.Rings)

plt.subplot(2,2,2)
sns.distplot(new_df.Rings)

plt.subplot(2,2,3)
stats.probplot(new_df.Rings, plot=plt)

plt.subplot(2,2,4)
sns.boxplot(new_df.Rings)

plt.tight_layout()

#plt.show()
#----------------libraries to train and score---------------#
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

print(new_df.columns)

#----------------alteração de sex para 3 colunas---------------#
new_col = pd.get_dummies(new_df.Sex)
new_df[new_col.columns] = new_col

#----------------drop some columns---------------#
feature = new_df.drop(['Sex', 'Rings','M','F','Viscera weight', 'Shell weight', 'Whole weight'], axis = 1)

label = new_df.Rings

dataset_reduced= pd.concat([feature, label], axis=1)
np.savetxt("dataset_reduced.txt", dataset_reduced, delimiter=",")

#----------------normalization and standardization---------------#
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

convert = StandardScaler()
scaler = MinMaxScaler()

feature = convert.fit_transform(feature)
feature = scaler.fit_transform(feature)


print(feature.shape)
print(label.shape)

#----------------division of dataset to train and test----------#
from sklearn.model_selection import train_test_split
f_train, f_test, l_train, l_test = train_test_split(feature, label, random_state = 23, test_size = 0.2)


#----------------train and test of accuracy----------#
model = linear_model.LinearRegression()
model=model.fit(f_train, l_train)


r2_score(l_test, model.predict(f_test))
x_test_frame= pd.DataFrame(f_test)
y_test_frame= pd.DataFrame(l_test)

np.savetxt("x_test.txt", x_test_frame, delimiter=",")
np.savetxt("y_test.txt", y_test_frame, delimiter=",")

#----------------coefficients----------#
print(model.coef_)
print(model.intercept_)

data= model.coef_
data=np.append(data, model.intercept_)


np.savetxt("parameters_intercept.txt", data, delimiter=",")

#----------------intercept----------#
t_array=model.coef_

t_array = t_array.astype('float16')
aux2=model.intercept_
i_value=aux2.astype('float16')
aux=np.append(t_array,i_value)

print(t_array.dtype)
print(t_array)
np.savetxt("parameters_intercept_float16.txt", aux, delimiter=",")

print(model.intercept_)

#----------------Polynomial regression----------#

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)

feature_train = poly.fit_transform(f_train)

poly_model = linear_model.LinearRegression()
poly_model.fit(feature_train, l_train)
r2_score(l_train, poly_model.predict(feature_train))


print(poly_model.coef_)

print(poly_model.get_params())






