import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'C:\Users\96251\Desktop\ML_code\files\100-Days-Of-ML-Code-master\datasets\50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 4].values

labelencoder_X = LabelEncoder()
X[ : , 3] = labelencoder_X.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features= [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[: , 1: ]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)
regressor.predict(X_train)

pred_Y = regressor.predict(X_test)

print (dataset.columns.tolist())
print (dataset.dtypes)
dataset.columns = dataset.columns.str.strip()
print (dataset.columns.tolist())
print (dataset.dtypes)
sns.lmplot(x="Administration", y="Profit", hue="State", data=dataset)
sns.pairplot(dataset,x_vars=["Administration", "R&D Spend", "Marketing_Spend"], y_vars="Profit", hue="State", kind='reg')
plt.show()