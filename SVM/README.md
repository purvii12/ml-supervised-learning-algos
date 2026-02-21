<img width="2364" height="1464" alt="satisfaction_hist" src="https://github.com/user-attachments/assets/edad299e-399c-43ec-afa0-3c0a90556433" />
# Support Vector Machine: Employee Attrition Prediction

**SVM Classification** predicting employee turnover using RBF kernel. HR analytics for retention strategy!

## Dataset
**HR Analytics Dataset** (14,999 employees)<br>
Features: satisfaction_level, average_montly_hours, promotion_last_5years, salary (one-hot encoded)<br>
Target: left (0=Stayed, 1=Left company)<br>
Same dataset as KNN - algorithm comparison!<br>



## 📊 Model Results  
| Metric | Score |
|--------|-------|
| **Accuracy** | **78%** |
| **Precision (Left)** | **1.00** |
| **Recall (Left)** | **0.07** ⚠️ |
| **F1 Score (Left)** | **0.14** ⚠️ |
| **Kernel** | RBF |
| **Features** | 6 |

## Visulaisation 
<img width="2364" height="1464" alt="satisfaction_hist" src="https://github.com/user-attachments/assets/640d89b2-9147-42e5-ab6e-2a1e1aea2025" />



## 📈 Confusion Matrix Breakdown
[[3428 0] ← PERFECT on "Stay" predictions (100% recall)<br>
[ 994 78]] ← Poor on "Leave" (only 7% recall)<br>

## 🔧 Pipeline Highlights
✅ Feature selection: 4 key HR predictors<br>
✅ Salary one-hot encoding<br>
✅ RBF kernel (non-linear decision boundary)<br>
✅ 70/30 train-test split<br>
✅ Business prediction example<br>
✅ Classification report + confusion matrix<br>


## 📈 Key Insights
✅ RBF kernel handles non-linear HR relationships<br>
✅ Compares with KNN (same dataset → algorithm showdown!)<br>
✅ satisfaction_level distribution visualization<br>
✅ Real business prediction: "Will this employee leave?"<br>


## 💼 Business Impact
"Predicted employee will STAY" → Focus retention elsewhere<br>
"Predicted employee will LEAVE" → Salary discussion / promotion offer<br>


## 🛠️ Files
employee_attrition_svm.py # Complete SVM pipeline<br>
satisfaction_hist.png # Employee satisfaction distribution<br>
HR_comma_sep.csv # HR dataset <br>


## Setup & Run
```bash
pip install -r ../../requirements.txt
python employee_attrition_svm.py

Previous: KNN Diabetes → SVM HR → Next: Logistic Regression
Skills: SVM RBF Kernel | Categorical Encoding | HR Analytics | Classification Metrics
Built withPythonScikit-learn

