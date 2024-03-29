import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

dataset = pd.read_csv(r'C:\Users\96251\Desktop\ML_code\files\100-Days-Of-ML-Code-master\datasets\Social_Network_Ads.csv')
X = dataset.iloc[ : , [2,3]].values
Y = dataset.iloc[ : ,4].values

X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test,Y_pred)

X_set ,Y_set = X_train, Y_train
X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))
print('X_set[:,0].min()-1',X_set[:,0].min()-1)
print('X_set[:, 0].max()+1',X_set[:, 0].max()+1)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('LightCoral', 'MintCream')))
print('a',np.array([X1.ravel(),X2.ravel()]))
print('b',classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).shape)
print('c',classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape).shape)

plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(Y_set)):

    plt.scatter(X_set[Y_set==j,0],X_set[Y_set==j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)
    print(X_set[Y_set==j,0])

plt. title(' LOGISTIC(Training set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()

X_set,y_set=sc.inverse_transform(X_test), sc.inverse_transform(Y_test)

X1,X2=np. meshgrid(np. arange(start=X_set[:,0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                   np. arange(start=X_set[:,1].min()-1, stop=X_set[:,1].max()+1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np. unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c = ListedColormap(('red', 'green'))(i), label=j)


plt. title(' LOGISTIC(Test set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()