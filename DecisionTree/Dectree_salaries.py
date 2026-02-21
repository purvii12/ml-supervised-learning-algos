# DECISION TREE ON SALARIES DATA

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load salaries dataset
df = pd.read_csv("C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Assignments\\SupervisedML\\DecisionTree\\salaries.csv")

# Features and target
inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

# Label encoding
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

# Final feature set
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

# Train model
model = DecisionTreeClassifier()
model.fit(inputs_n, target)

# Evaluate model
print("Salaries Model Accuracy:", model.score(inputs_n, target))

# Predictions
print("Prediction for Google, Computer Engineer, Bachelors:", model.predict([[2, 1, 0]]))
print("Prediction for Google, Computer Engineer, Masters:", model.predict([[2, 1, 1]]))



import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(16, 8))
plot_tree(model, 
          feature_names=inputs_n.columns,
          class_names=['<=100k', '>100k'],
          filled=True,
          rounded=True,
          fontsize=12)
plt.title("Decision Tree for Salary Prediction")
plt.show()
