# import required libraries and modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


#load dataset
insurance_data = pd.read_csv(r"D:\Desktop\Prime-cs\.venv-1\SML_pt1_linear regression\insurance.csv")
print(insurance_data)

#visualise the data
sns.scatterplot(x = insurance_data["bmi"], y=insurance_data["charges"], hue=insurance_data["smoker"])
#plt.show()

#input and output features
x = insurance_data.drop(columns=["charges", "region"])
y = insurance_data["charges"]

#map features
x["sex"] = x["sex"].map({"female":1,"male":0})
x["smoker"] = x["smoker"].map({"yes":1,"no":0})

#display our features
print(x.head())
print("x values")
print(y.head())
print("y values")

#define the train test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state =42)

#random state just shuffles the value from the dataset while testing , heres a demostartaion 
print(x_test.head())

#train the model
model = LinearRegression()
model.fit(x_train , y_train)

# predict the values now/ testing
y_pred = (model.predict(x_test))
print(y_pred)
print(y_test)

#evaluation
r2= r2_score(y_test, y_pred)
print("r2 score value is :",r2)

