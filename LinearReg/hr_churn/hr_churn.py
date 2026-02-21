# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# 2. Load the dataset and create a DataFrame
df = pd.read_csv("C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Assignments\\SupervisedML\\LinearReg\\hr_churn\\HR_comma_sep.csv")
print(df.head())
print(df.describe())  # Display the first few rows and summary statistics of the dataset    

# 3. Preprocessing: removing unnecessary columns, exploring attrition

#Get separate DataFrames for those who left and those who stayed
left = df[df.left==1]
print("Left employees shape:", left.shape)

retained = df[df.left==0]
print("Retained employees shape:", retained.shape)

#Compare means of features grouped by attrition
print(df.groupby('left').mean(numeric_only=True))  

# Selecting relevant features for modeling
subdf = df[['satisfaction_level', 'average_montly_hours',
            'promotion_last_5years', 'salary']]
print(subdf.head())

#Encoding categorical variables (salary)
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
print(df_with_dummies.head())

#Drop the original 'salary' column
df_with_dummies.drop('salary', axis='columns', inplace=True)
print(df_with_dummies.head())

# 4. Preparing input and output variables for modeling
X = df_with_dummies
print(X.head())  # Input features
y = df.left
print(y.head())  # Output labels


#Step 5: Split Dataset into Training and Testing Sets (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 6. Instantiate the Linear Regression model
lr = linear_model.LinearRegression()

# 7. Train the model
# Train the model on training data
lr.fit(X_train, y_train)

# 8. calculate model performance
from sklearn.metrics import accuracy_score, r2_score
y_pred = lr.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred.round()):.3f}")


# Print model coefficients and intercept
print(lr.coef_)
print(lr.intercept_)

# Example prediction for a sample input (make sure to match input features structure)
sample = pd.DataFrame([{
    'satisfaction_level': 0.5,
    'average_montly_hours': 200,
    'promotion_last_5years': 0,
    'salary_low': 1,
    'salary_medium': 0,
    'salary_high': 0
}])

# Align columns of sample to training features
sample = sample.reindex(columns=X.columns, fill_value=0)
print(lr.predict(sample))

# 8. Visualize the results- Plot distribution of satisfaction level across all employees
plt.figure(figsize=(8,5))
plt.hist(df.satisfaction_level, bins=20, color='royalblue', edgecolor='white', alpha=0.8)
plt.xlabel('Satisfaction Level', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.title('Distribution of Satisfaction Level', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('satisfaction_hist.png', dpi=300, bbox_inches='tight')
plt.show()

