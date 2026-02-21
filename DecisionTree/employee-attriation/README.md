# 🌳 Decision Tree: Employee Attrition Prediction

**Decision Tree Classification** predicting employee turnover with interpretable tree visualization!

## Dataset  
**HR Analytics Dataset** (14,999 employees) - **SAME as SVM for algorithm comparison**
Features: satisfaction_level, average_montly_hours, promotion_last_5years, salary (one-hot)
Target: left (0=Stayed, 1=Left)
Rich EDA: Left vs Retained employee analysis

## 📊 Model Results  
| Metric | Score |
|--------|-------|
| **Accuracy** | **92%**  |
| **F1 Left** | **0.83** |
| **Precision Left** | **0.83** |
| **Recall Left** | **0.84** |
| **Tree Depth** | **31** |
| **Key Feature** | **satisfaction_level (69%)** |


## 🌿 Tree Visualization
Top 3 levels shown - fully interpretable! <br>
feature_importances_ shows key predictors <br>
satisfaction_level histogram for context <br>


![Decision Tree Visualization](DecisionTree/employee-attrition/tree_visualization.png) <br>
![Employee Satisfaction Histogram](DecisionTree/employee-attrition/satisfaction_hist.png)



## 🔧 Pipeline Highlights
✅ Complete EDA: Left vs Retained comparison <br>
✅ Smart feature selection (4 key HR factors) <br>
✅ Salary one-hot encoding <br>
✅ Decision tree visualization (plot_tree) <br>
✅ Feature importance ranking <br>
✅ Business prediction example <br>


## 💼 Business Value
"IF satisfaction_level < 0.5 AND hours > 220 → HIGH attrition risk" <br>
→ Direct actionable HR rules from tree! <br>

## 🛠️ Files
-employee_attrition_tree.py # Complete decision tree pipeline<br>
-satisfaction_hist.png # Employee distribution <br>
-tree_visualization.png # Decision tree plot <br>
-HR_comma_sep.csv # Dataset (optional) <br>



## Algorithm Showdown (SAME HR Dataset)
Algorithm	Dataset	Strength <br>
KNN	Diabetes	Distance-based <br>
SVM (RBF)	HR	Non-linear boundary <br>
Decision Tree	HR	Interpretable rules <br>
Logistic Reg	⏳ Coming	Probabilities <br>


## Key Decision Tree Advantages
✅ Human-readable rules ("IF satisfaction < X AND hours > Y...") <br>
✅ Feature importance ranking <br>
✅ Handles categorical + numeric features <br>
✅ No feature scaling needed <br>
✅ Tree visualization for stakeholders <br>


## Setup & Run
```bash
pip install -r ../../requirements.txt
python employee_attrition_tree.py
Progression

Previous: [svm/](../svm/) → Decision Tree → Next: Logistic Regression
KNN → SVM → Decision Tree = Classification mastery!

**Skills**: Tree Visualization | Feature Importance | Interpretable ML | HR Analytics
**Built with**  ![Python](https://img.shields.io/badge/Python-3.11-green) ![Scikit-learn](https://img.shields.io/badge/scikit-learn-1.5-blue)

