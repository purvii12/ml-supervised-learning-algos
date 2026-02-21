# 1. Importing necessary packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 2. Reading from dataset
df = pd.read_csv("C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Assignments\\SupervisedML\\SVM\\HR_comma_sep.csv")


# 3. Data exploration
print("Dataset head:\n", df.head())
print("\nGrouped by 'left' (mean values):\n", df.groupby('left').mean())

# 4. Preprocessing
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]

# Convert salary column (categorical) into dummy variables
salary_dummies = pd.get_dummies(subdf.salary, prefix='salary')
df_with_dummies = pd.concat([subdf.drop('salary', axis=1), salary_dummies], axis=1)

# Feature matrix and target vector
X = df_with_dummies
y = df.left

# 5. Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)

# 6. Instantiating & training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Testing & prediction
predictions = model.predict(X_test)

# 8. Measuring model accuracy
print("Model accuracy:", model.score(X_test, y_test))

# 9.Visualizations (uncomment to see charts)

# Impact of salary on employee retention
pd.crosstab(df.salary, df.left).plot(kind='bar', title='Impact of Salary on Retention')
plt.show()

# Department-wise retention rate (optional)
pd.crosstab(df.Department, df.left).plot(kind='bar', title='Department-wise Retention')
plt.show()

