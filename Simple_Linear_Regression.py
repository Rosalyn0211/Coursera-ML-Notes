import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r'C:\Users\96251\Desktop\ML_code\files\100-Days-Of-ML-Code-master\datasets\studentscores.csv')
X = dataset.iloc[ : ,  : 1 ].values
Y = dataset.iloc[ : , 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=1/4,random_state=0)

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

print('x', X_train)
print('y',Y_train)
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.show()

plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()