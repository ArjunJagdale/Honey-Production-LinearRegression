# Honey Production Linear Regression

This project demonstrates how to apply linear regression to predict honey production over the years. It uses the `honeyproduction.csv` dataset to build a model that predicts the total honey production (`totalprod`) based on the year (`year`). The code leverages popular Python libraries such as `pandas`, `matplotlib`, `numpy`, and `scikit-learn`.

## Libraries Used

- **pandas**: Data manipulation and analysis.
- **matplotlib.pyplot**: Data visualization, especially for plotting graphs.
- **numpy**: Supports large, multi-dimensional arrays and matrices, along with mathematical functions.
- **sklearn.linear_model**: Implements the linear regression model.

## Steps Explained

1. **Loading the Dataset:**
   - The dataset `honeyproduction.csv` is loaded from a URL using `pandas.read_csv()`.
   - The first few rows of the dataset are printed using `df.head()` to inspect the data.

   ```python
   df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")
   print(df.head())
   ```

2. **Data Preparation:**
   - The dataset is grouped by `year`, and the average `totalprod` for each year is calculated using `groupby()` and `mean()`. This creates a new DataFrame, `prod_per_year`, which contains the yearly average honey production.

   ```python
   prod_per_year = df.groupby('year').totalprod.mean().reset_index()
   ```

3. **Data Visualization (Scatter Plot):**
   - A scatter plot is generated to visualize the relationship between the year (`X`) and the total honey production (`y`).
   
   ```python
   X = prod_per_year['year']
   y = prod_per_year['totalprod']
   plt.scatter(X, y)
   plt.show()
   ```

4. **Linear Regression Model:**
   - A linear regression model is instantiated using `LinearRegression()`.
   - The model is trained using the `fit()` method, where `X` is the year and `y` is the total honey production.
   
   ```python
   regr = LinearRegression()
   regr.fit(X.reshape(-1, 1), y)
   ```

   - The model's coefficients (slope) and intercept are printed.

   ```python
   print(regr.coef_)
   print(regr.intercept_)
   ```

5. **Prediction and Line Plot:**
   - The model is used to predict honey production for the years in the dataset.
   - The predicted values are plotted along with the actual scatter plot for comparison.

   ```python
   y_predict = regr.predict(X.reshape(-1, 1))
   plt.plot(X, y_predict)
   plt.show()
   ```

6. **Future Predictions:**
   - Future years from 2013 to 2049 are created using `numpy.array()`, and the model predicts honey production for these years.
   - A line plot is shown to visualize the predicted honey production for these future years.

   ```python
   X_future = np.array(range(2013, 2050)).reshape(-1, 1)
   future_predict = regr.predict(X_future)
   plt.plot(X_future, future_predict)
   plt.show()
   ```

## Output

- **Scatter plot**: Displays the relationship between year and honey production.
- **Line plot**: Shows the linear regression line fitted to the data.
- **Future predictions**: Displays predicted honey production for the years 2013 to 2049.

## Conclusion

This project demonstrates the application of linear regression to predict future values based on historical data. By fitting a linear model to the honey production data, we can make predictions for future years, providing insight into expected trends.

## Requirements

To run the code, the following Python libraries are required:

- pandas
- matplotlib
- numpy
- scikit-learn

You can install them using pip:

```bash
pip install pandas matplotlib numpy scikit-learn
```

## Dataset

The dataset used in this project is the "Honey Production" dataset, which contains information about honey production in the United States from 1998 to 2012. It includes the year and total honey production for each year.
