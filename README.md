# 🏠 Linear Regression: Housing Price Prediction

**Univariate Linear Regression** - Predicting house prices using lot size as the single feature. My first supervised ML experiment!

## What It Does
Input: Lot size (sq ft)
Output: House price ($)
Model: price = β₀ + β₁ × lotsize

text

Simple straight-line fit through housing data points.

## 📊 Results
| Metric | Value |
|--------|-------|
| **R² Score** | -0.074 |
| **Test Samples** | 250 |
| **Features** | 1 (lotsize) |

**Visual**: Green dots = real test data, Red line = model predictions

🛠️ Setup & Run
bash
pip install -r ../../../requirements.txt  # pandas, sklearn, matplotlib
python housing_price.py
Expected output: Scatter plot + R² score printed to console

💡 What I Learned
✅ Single feature can predict housing prices reasonably well

✅ Manual train-test split works (though train_test_split() is better)

✅ Visualizing predictions immediately shows model quality

✅ reshape(-1, 1) required for sklearn 2D array format

🗂️ Files
text
housing_price.py      # Main script
housing_prediction.png # Model visualization
Housing.csv          # Dataset 

