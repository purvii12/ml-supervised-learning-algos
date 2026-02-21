# Employee Attrition Prediction Using SVM

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 2. Load the dataset and create a DataFrame
df = pd.read_csv("C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Assignments\\SupervisedML\\SVM\\HR_comma_sep.csv")
print(df.head())  # Show the first 5 records
print(df.describe())  # Show summary statistics

# 3. Preprocessing: select features and encode categorical variables
# Select relevant features for modeling
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
print(subdf.head())

# One-hot encode the 'salary' categorical variable
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
df_with_dummies.drop('salary', axis='columns', inplace=True)
print(df_with_dummies.head())  # Now all features are numeric

# Prepare final feature set (X) and output variable (y)
X = df_with_dummies
y = df.left
print(X.head())
print(y.head())

# 4. Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Instantiate and train the SVM classifier
svm = SVC(kernel='rbf', random_state=42)  # Radial basis function kernel
svm.fit(X_train, y_train)

# 6. Evaluate model performance on the test set
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))  # Output confusion matrix
print(classification_report(y_test, y_pred))  # Print precision, recall, F1-score

# 7. Make prediction for a new sample (replace values as needed)
sample = pd.DataFrame([{
    'satisfaction_level': 0.5,
    'average_montly_hours': 200,
    'promotion_last_5years': 0,
    'salary_low': 1,
    'salary_medium': 0,
    'salary_high': 0
}])
# Align sample columns with X
sample = sample.reindex(columns=X.columns, fill_value=0)
print(svm.predict(sample))  # Output: [0] or [1], where 1 means predicted to leave

# 8. Visualize the distribution of satisfaction level
plt.figure(figsize=(8,5))
plt.hist(df.satisfaction_level, bins=20, color='royalblue', edgecolor='white', alpha=0.8)
plt.xlabel('Satisfaction Level', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.title('Distribution of Satisfaction Level', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('satisfaction_hist.png', dpi=300, bbox_inches='tight')
plt.show()

