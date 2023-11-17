# %%
import pandas as pd
# importing the pandas library as pd for data manipulation

# %%
%matplotlib inline

# %%
# importing the matplotlib library as plt for data visualization
import matplotlib.pyplot as plt


# %%
# reading the csv file into a pandas dataframe
nyc=pd.read_csv('ave_hi_nyc_jan_1895-2018.csv')

# %%
# displaying the first five rows of the dataframe
nyc.head()

# %%
# displaying the last five rows of the dataframe
nyc.tail()

# %%
# displaying random 20 rows of the dataframe
nyc.sample(20)

# %%
# declaring the column names of the dataframe
nyc.columns =['Date','Temperature','Anomaly']

# %%
# displaying the first five rows of the dataframe
nyc.head()

# %%
# Takes the values of the Date column in the nyc dataframe and divides them by 100
nyc.Date=nyc.Date.floordiv(100)

# %%
# displaying the first five rows of the dataframe
nyc.head()

# %%
# gets the dimensions of the nyc dataframe
nyc.shape

# %%
# displays the last five rows of the nyc dataframe
nyc.tail()

# %%
# importing the train_test_split function from the sklearn.model_selection library
from sklearn.model_selection import train_test_split

# %%
# This code accesses the Date column of the nyc dataset and returns the shape of the values in that column.
nyc.Date.values.shape

# %%
# splitting the nyc dataframe into training and testing sets. The training set will be used to train the model and the testing set will be used to test the model. 
X_train,X_test,y_train,y_test=train_test_split(nyc.Date.values.reshape(-1,1), nyc.Temperature.values,random_state=11)

# %%
# This code accesses the shape of the X_train variable(dimension of the training set)
X_train.shape

# %%
# This code accesses the shape of the X_test variable(dimension of the testing set)
X_test.shape

# %%

93+31

# %%
93/124*100

# %%
#Training the model

# %%
# importing the LinearRegression class from the sklearn.linear_model library
from sklearn.linear_model import LinearRegression

# %%
# creating an instance of the LinearRegression class
linear_regression=LinearRegression()

# %%
# calling the linear_regression object instance.
linear_regression

# %%
# training the model using the fit method of the linear_regression object instance
linear_regression.fit(X=X_train, y=y_train)

# %%
#Equation M and C

# %%
# This code accesses the slope of the linear regression model(coefficient of the linear regression model)
linear_regression.coef_

# %%
# intercept of the linear regression equation
linear_regression.intercept_

# %% [markdown]
# The equation: $Temperature = 0.01939167 * Date - 0.30779820252656265$

# %%
# Testing the model
(0.01939167 * 2014) - 0.30779820252656265

# %%
# Testing the Model

# %%
# This code accesses the predicted values of the linear regression model on x_test
predicted = linear_regression.predict(X_test)

# %%
# expected values of the linear regression model on the y_test set
expected = y_test

# %%
# This code snippet is iterating over two lists, predicted and expected, and printing out each corresponding pair of values. 
for p,e in zip(predicted[::], expected[::]):
  print(f'Predicted: {p:.2f}, Expected: {e:.2f}')

# %%
# This code snippet is a for loop that iterates over two lists, predicted and expected, and prints the predicted value, expected value, and the difference between them (error) for each corresponding pair of values. 
for p,e in zip(predicted[::], expected[::]):
  print(f'Predicted: {p:.2f}, Expected: {e:.2f}, Error: {e-p:.2f}')

# %%
# Mean Absolute Error (MAE)


# %%
# importing the mean_absolute_error function from the sklearn.metrics library. Used to calculate the mean absolute error of the linear regression model
from sklearn.metrics import mean_absolute_error

# %%
# printing the mean absolute error of the linear regression model
print("MAE", mean_absolute_error(expected,predicted))

# %%
#RMSE - Root Mean Squared Error

# %%
# importing the mean_squared_error function from the sklearn.metrics library. Used to calculate the mean squared error of the linear regression model
from sklearn.metrics import mean_squared_error

# %%
# importing numpy as np for mathematical operations on arrays
import numpy as np


# %%
# printing the root mean squared error of the linear regression model
print(f'RMSE',np.log(np.sqrt(mean_squared_error(expected, predicted)) ))

# %%
# R Squared (R2)

# %%
# importing the r2_score function from the sklearn.metrics library. Used to calculate the r2 score of the linear regression model
from sklearn.metrics import r2_score

# %%
# r2 variable stores the r2 score function of the linear regression model
r2=r2_score(expected,predicted)

# %%
r2

# %%
# predict future model

# %%
# The code you provided defines a lambda function named predict. 
# The lambda function takes a single argument, x, and returns the result of the linear regression model’s coef_ attribute multiplied by x plus the intercept_ attribute.


predict=(lambda x: linear_regression.coef_ *x + linear_regression.intercept_)

# %%
# This code snippet calls the predict function and passes in the value 2014.
predict(2014)

# %%
# This code snippet calls the predict function and passes in the value 2022.
predict(2022)

# %%
# This code snippet calls the predict function and passes in the value 1800.
predict(1800)

# %%
# Visualizing the dataset with a Regression Line

# %%
# importing the seaborn library as sns for data visualization
import seaborn as sns

# %%
# This code snippet creates a scatter plot of the nyc dataframe’s Date and Temperature columns.
axes=sns.scatterplot(data=nyc, x='Date',y='Temperature',
                     hue='Temperature', palette='winter', legend=False)
axes.set_ylim(10,70)
x=np.array([min(nyc.Date.values),max(nyc.Date.values)])
y=predict(x)
line=plt.plot(x,y)

# %%
# The x variable stores the minimum and maximum values of the nyc dataframe’s Date column.
x

# %%
# The y variable stores the predicted values of the linear regression model based on the x variable values.
y

# %%



