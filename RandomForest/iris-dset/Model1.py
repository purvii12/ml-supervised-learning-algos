# Load datasets and libraries
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Features (X) and Target (y)
X = iris.data
y = iris.target

# Split into train and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Instantiate and train RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy:", accuracy)

# View feature importance (optional)
feature_importances = pd.Series(model.feature_importances_, index=iris.feature_names)
print("\nFeature Importances:\n", feature_importances)

# Feature importance plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
feature_importances.sort_values().plot(kind='barh')
plt.title('Random Forest Feature Importance (Iris Dataset)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
