# 🏠 Linear Regression: Housing Price Prediction

**Univariate Linear Regression** - Predicting house prices using lot size as the single feature. My first supervised ML experiment!

## What It Does
Input: Lot size (sq ft)
Output: House price ($)
Model: price = β₀ + β₁ × lotsize

Simple straight-line fit through housing data points.

## 📊 Results
| Metric | Value |
|--------|-------|
| **R² Score** | -0.074 |
| **Test Samples** | 250 |
| **Features** | 1 (lotsize) |

**Visual**: Green dots = real test data, Red line = model predictions 
![Housing Price Prediction](assets/housing_prediction.png)

##💡 What I Learned
✅ Single feature can predict housing prices reasonably well <br>
✅ Manual train-test split works (though train_test_split() is better) <br>
✅ Visualizing predictions immediately shows model quality <br>
✅ reshape(-1, 1) required for sklearn 2D array format <br>


