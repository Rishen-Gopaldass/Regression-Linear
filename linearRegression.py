# Simple Linear Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
#you will know more about this dataset on its documentation. Google it
from sklearn import datasets
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]

X_train = diabetes_X[:-20]
X_test = diabetes_X[-20:]
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

# our own Linear Regression
'''we need to change dimension of X_train, coz it has shape(422,1) = 2D but y_train has shape(422,) = 1D,
so we need to make their dimension same for multiplication '''
X_train = X_train.reshape(y_train.shape)
X_test = X_test.reshape(y_test.shape)
xy = (X_train * y_train)
xx = (X_train * X_train)
m = (np.mean(X_train) * np.mean(y_train) - np.mean(xy))/((np.mean(X_train))**2 - np.mean(xx))
b= np.mean(y_train) - m * np.mean(X_train)

# visualizing our Linear Regression
plt.scatter(X_test, y_test, color='green', label="Test")
plt.scatter(X_train, y_train, color='red', label ='Training')

y_line = b + m*np.array(X_train)
plt.plot(X_train, y_line , color='blue', label='Best fit line')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()