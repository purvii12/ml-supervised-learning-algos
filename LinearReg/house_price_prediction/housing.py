
#step-1: import libarraies 
import matplotlib
#matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets ,  linear_model 

#step-2: load the dataset
df = pd.read_csv("C:\\Users\\Public\\Downloads\\AML\\FDS\\Unit1\\Assignments\\SupervisedML\\LinearReg\\Housing.csv")

#step-3: data preprocessing

#extract feaatures and taregt varaible
Y = df["price"]
X = df["lotsize"]

#reshape-fit it to sklearn
X = X.to_numpy().reshape(len(X), 1)
Y = Y.to_numpy().reshape(len(Y), 1)

#split the dataset into training and testing sets
X_train = X[:-250]
X_test = X[-250:]
Y_train = Y[:-250]
Y_test = Y[-250:]

#plot the test data
plt.scatter(X_test , Y_test, color = "green")
plt.title("test data")
plt.xlabel("size")
plt.ylabel("Price")
plt.xticks(())
plt.yticks(())  

#step-4: create a linear regression model
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train )

# Calculate R² score (model quality)
from sklearn.metrics import r2_score
r2 = r2_score(Y_test, reg.predict(X_test))
print(f"R² Score: {r2:.3f}")  # Save this number for README!


#plot predictions 
plt.plot(X_test, reg.predict(X_test), color='red', linewidth=2, label='Predicted Line')
plt.savefig('housing_prediction.png', dpi=300, bbox_inches='tight')
plt.show()
