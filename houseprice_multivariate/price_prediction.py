#multivariate
#step-1: import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#step-2: load the dataset and make dataframe 
df = pd.read_csv("C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Assignments\\SupervisedML\\LinearReg\\houseprice_multivariate\\priceprediction.csv")
print(df.head())
print(df.describe()) # Display the first few rows and summary statistics of the dataset

#step-3: preprocess the data , seperate the input and output variables
df2 = df.drop('price', axis='columns')
price = df.price 

#step-4: Instantiate the Linear Regression model
lr = linear_model.LinearRegression()

#step-5: Train the model
lr.fit(df2 , price)  # Fit the model using 'area'(independent variable) as input and 'price'(dependent variable) as output
# 1. Train-test split + score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(df2, price, test_size=0.2, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# 2. Save output
print(f"Model: price = {lr.intercept_:.0f} + {lr.coef_[0]:.0f}×area + ...")

print(lr.coef_)  # Print the coefficients of the model
print(lr.intercept_)  # Print the intercept of the model

#step-6: Make predictions using the model
print(lr.predict(pd.DataFrame({'area':[1000]})))  

# Plot 1: Actual vs Predicted Prices (BEST for multivariate)
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Multivariate House Price Prediction\nR² = {r2_score(y_test, y_pred):.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

# SAVE IT
plt.savefig('multivariate_house_price.png', dpi=300, bbox_inches='tight')
plt.show()
