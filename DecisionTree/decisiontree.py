import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Python_basics\\HR_comma_sep.csv")
print(df.head())
print(df.describe())  # Display the first few rows and summary statistics of the dataset

# Check attrition status
left = df[df.left == 1]
print("Left employees shape:", left.shape)

retained = df[df.left == 0]
print("Retained employees shape:", retained.shape)

# Compare means of numeric features grouped by attrition
print(df.groupby('left').mean(numeric_only=True))

# Select features for modeling
subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
print(subdf.head())

# Encode 'salary' categorical variable
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf, salary_dummies], axis='columns')
df_with_dummies.drop('salary', axis='columns', inplace=True)
print(df_with_dummies.head())

# Prepare final feature matrix X and label vector y
X = df_with_dummies
print(X.head())
y = df.left
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
print("Feature importances:", clf.feature_importances_)
print("Decision Tree depth:", clf.get_depth())
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


sample = pd.DataFrame([{
    'satisfaction_level': 0.5,
    'average_montly_hours': 200,
    'promotion_last_5years': 0,
    'salary_low': 1,
    'salary_medium': 0,
    'salary_high': 0
}])
sample = sample.reindex(columns=X.columns, fill_value=0)
print(clf.predict(sample))


plt.figure(figsize=(8,5))
plt.hist(df.satisfaction_level, bins=20, color='royalblue', edgecolor='white', alpha=0.8)
plt.xlabel('Satisfaction Level', fontsize=12)
plt.ylabel('Number of Employees', fontsize=12)
plt.title('Distribution of Satisfaction Level', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15,6))
plot_tree(clf, feature_names=X.columns, class_names=['Not Left', 'Left'], filled=True, max_depth=3)
plt.title('Decision Tree (Top 3 Levels)')
plt.show()
