
# 💰 Salary Prediction by Age (Univariate Linear Regression)

**Predict employee salary using age as the single feature.** Clean visualization with business prediction example.

## What It Does
Input: Employee Age<br>
Output: Predicted Salary ($) <br>
Model: salary = β₀ + β₁ × age <br>

**Business use**: "What salary should we offer a 27-year-old candidate?" <br>


##  Visual Result
Salary Prediction
<br>
Blue dots = real data, Red line = model, Green dot = Age 27 prediction <br>
<img width="1768" height="1361" alt="salary_age_prediction" src="https://github.com/user-attachments/assets/3173b6a1-27ab-4638-aa07-368d17542ca0" />



## 💡 What I Learned
✅ X = df[['Age']] creates 2D DataFrame (sklearn requirement) <br>
✅ [[27]] for single prediction (double brackets!) <br>
✅ Business context visualization (green prediction point) <br>
✅ Model equation: salary = intercept + slope × age <br>

## 🔢 Model Equation
salary = [intercept] + [slope] × age <br>

## 🗂️ Files
-salary_age_prediction.py     # Main script <br>
-salary_age_prediction.png    # Perfect visualization <br>
-salary_prediction.csv        # Dataset (optional) <br>
