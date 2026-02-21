text
# 💰 Salary Prediction by Age (Univariate Linear Regression)

**Predict employee salary using age as the single feature.** Clean visualization with business prediction example.

## What It Does
Input: Employee Age
Output: Predicted Salary ($)
Model: salary = β₀ + β₁ × age

text

**Business use**: "What salary should we offer a 27-year-old candidate?"

## 📊 Results
| Metric | Value |
|--------|-------|
| **Age 27 Prediction** | $[RUN→ADD] |
| **R² Score** | [RUN→ADD] |
| **Slope (coef)** | [RUN→ADD] |
| **Intercept** | [RUN→ADD] |
| **Features** | 1 (Age) |

🛠️ Setup & Run
bash
pip install -r ../../../requirements.txt
python salary_age_prediction.py
Outputs: Console predictions + scatter plot with green prediction dot

📱 Visual Result
Salary Prediction

Blue dots = real data, Red line = model, Green dot = Age 27 prediction

💡 What I Learned
✅ X = df[['Age']] creates 2D DataFrame (sklearn requirement)

✅ [[27]] for single prediction (double brackets!)

✅ Business context visualization (green prediction point)

✅ Model equation: salary = intercept + slope × age

🔢 Model Equation
text
salary = [intercept] + [slope] × age
(Run code to get exact numbers)

🗂️ Files
text
salary_age_prediction.py     # Main script
salary_age_prediction.png    # Perfect visualization
salary_prediction.csv        # Dataset (optional)