
# Support Vector Machine: Employee Attrition Prediction

**SVM Classification** predicting employee turnover using RBF kernel. HR analytics for retention strategy!

## Dataset
**HR Analytics Dataset** (14,999 employees)
Features: satisfaction_level, average_montly_hours, promotion_last_5years, salary (one-hot encoded)
Target: left (0=Stayed, 1=Left company)
Same dataset as KNN - algorithm comparison!


## 📊 Model Results
| Metric | Score |
|--------|-------|
| **Precision (Left)** | [RUN→ADD] |
| **Recall (Left)** | [RUN→ADD] |
| **F1 Score (Left)** | [RUN→ADD] |
| **Kernel** | RBF |
| **Features** | 6 |

## 🔧 Pipeline Highlights
✅ Feature selection: 4 key HR predictors
✅ Salary one-hot encoding
✅ RBF kernel (non-linear decision boundary)
✅ 70/30 train-test split
✅ Business prediction example
✅ Classification report + confusion matrix


## 📈 Key Insights
✅ RBF kernel handles non-linear HR relationships
✅ Compares with KNN (same dataset → algorithm showdown!)
✅ satisfaction_level distribution visualization
✅ Real business prediction: "Will this employee leave?"


## 💼 Business Impact
"Predicted employee will STAY" → Focus retention elsewhere
"Predicted employee will LEAVE" → Salary discussion / promotion offer


## 🛠️ Files
employee_attrition_svm.py # Complete SVM pipeline
satisfaction_hist.png # Employee satisfaction distribution
HR_comma_sep.csv # HR dataset (optional)


## Progression Comparison (Same HR Dataset)
Algorithm	Same Dataset	Key Strength
KNN	HR Churn	Distance-based
SVM	HR Churn	RBF Kernel
Logistic	⏳ Coming	Probability


## Setup & Run
```bash
pip install -r ../../requirements.txt
python employee_attrition_svm.py

Previous: KNN Diabetes → SVM HR → Next: Logistic Regression
Skills: SVM RBF Kernel | Categorical Encoding | HR Analytics | Classification Metrics
Built withPythonScikit-learn

