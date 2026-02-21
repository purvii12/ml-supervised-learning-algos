import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv('C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Assignments\\SupervisedML\\DecisionTree\\titanic.csv')

# Select required columns
inputs = df[['Pclass', 'Sex', 'Age', 'Fare']]
target = df['Survived']

# Fill missing values
inputs['Age'] = inputs['Age'].fillna(inputs['Age'].median())
inputs['Fare'] = inputs['Fare'].fillna(inputs['Fare'].median())

# Encode 'Sex'
le_sex = LabelEncoder()
inputs['Sex_n'] = le_sex.fit_transform(inputs['Sex'])

# Prepare feature matrix
inputs_n = inputs.drop('Sex', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(inputs_n, target, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


#visualization
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(model, feature_names=inputs_n.columns, class_names=['Did Not Survive', 'Survived'],
          filled=True, rounded=True, fontsize=12)
plt.title('Decision Tree Visualization for Titanic Survival Prediction')
plt.show()



# Evaluate model
score = model.score(X_test, y_test)
print("Model accuracy score:", score)

