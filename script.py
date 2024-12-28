import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

print(df.head())

prod_per_year = df.groupby('year').totalprod.mean().reset_index()

X = prod_per_year['year'] 
y = prod_per_year['totalprod']
X = X.values.reshape(-1, 1)

plt.scatter(X, y)
plt.show()

regr = LinearRegression()
regr.fit(X,y)

print(regr.coef_)
print(regr.intercept_)

y_predict = regr.predict(X)

plt.plot(X, y_predict)
plt.show()

# creating a numpy array
X_future = np.array(range(2013, 2050))

X_future = X_future.reshape(-1,1)

future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)
plt.show()

