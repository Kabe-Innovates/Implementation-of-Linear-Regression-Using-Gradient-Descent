# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load the dataset from a CSV file and separate the features and target variable, encoding any categorical variables as needed.
2.Scale the features using a standard scaler to normalize the data.
3.Initialize model parameters (theta) and add an intercept term to the feature set.
4.Train the linear regression model using gradient descent by iterating through a specified number of iterations to minimize the cost function.
5.Make predictions on new data by transforming it using the same scaling and encoding applied to the training data.
```

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KABELAN G K
RegisterNumber:  24900985
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def linear_regression(X, y, iters=1000, learning_rate=0.01):
    X = np.hstack((np.ones((X.shape[0], 1)), X))  # Add intercept term
    theta = np.zeros((X.shape[1], 1))
    
    for _ in range(iters):
        predictions = X.dot(theta)
        errors = predictions - y.reshape(-1, 1)
        gradient = (1 / X.shape[0]) * X.T.dot(errors)
        theta -= learning_rate * gradient
    
    return theta

data = pd.read_csv('50_Startups.csv', header=0)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

ct = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(), [3])  
], remainder='passthrough')

X = ct.fit_transform(X)

y = y.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

theta = linear_regression(X_scaled, y, iters=1000, learning_rate=0.01)

new_data = np.array([165349.2, 136897.8, 471784.1, 'New York']).reshape(1, -1)  # Example new data
new_data_scaled = scaler.transform(ct.transform(new_data))

new_prediction = np.dot(np.append(1, new_data_scaled), theta)

print(f"Predicted value: {new_prediction[0]}")
data.head()

```

## Output:
```
Predicted value: 193075.97426510364
```
![Screenshot](https://github.com/user-attachments/assets/ae06e431-21f7-47fe-967b-eaa88d5bf53f)





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
