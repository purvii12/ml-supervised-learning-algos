#univariate
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Assignments\\SupervisedML\\LinearReg\\salary_prediction\\salary_prediction..csv")

X = df[['Age']] 
y = df['Salary'] 

#create linear regression object
reg = linear_model.LinearRegression()
reg.fit(X, y)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg.fit(X_train, y_train)
print(f"R² Score: {r2_score(y_test, reg.predict(X_test)):.3f}")

# Predict salary for age 27
predicted_salary = reg.predict([[27]])
print("Predicted salary for age 27:", predicted_salary[0])
print("Coefficient (slope):", reg.coef_[0])
print("Intercept (y-intercept):", reg.intercept_)

# plot 
plt.scatter(df['Age'], df['Salary'], color='blue', label='Actual Data')
plt.plot(df['Age'], reg.predict(X), color='red', label='Regression Line')
plt.scatter(27, predicted_salary, color='green', label='Prediction (Age 27)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Salary Prediction Based on Age')
plt.legend()
plt.savefig('salary_age_prediction.png', dpi=300, bbox_inches='tight')
plt.show()
