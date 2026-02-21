
# Decision Tree: Salary Prediction 

**Decision Tree Classification** predicting high salaries - **PERFECT 100% accuracy** on categorical data!

## Dataset
**Salaries Dataset** (company, job, degree → salary binary)
Features: company, job_title, degree → LabelEncoded
Target: salary_more_then_100k (0=≤100k, 1=>100k)

## 📊 Model Results  
| Metric | Score | 
|--------|-------|
| **Accuracy** | **100%**  | 
| **Google+Eng+Bachelor** | **≤100k** `[0]` |
| **Google+Eng+Masters**  | **>100k** `[1]` |
| **Features** | company_n, job_n, degree_n |

## 🌳 Tree Visualization
<img width="3780" height="1971" alt="tree_visualization" src="https://github.com/user-attachments/assets/046e447a-ffa9-4daa-ab47-e0c7563a6471" />

Perfect rules: "Masters = >100k salary!" <br>

## 🔥 Key Insights
✅ 100% accuracy = Decision Trees SHINE on categorical data <br>
✅ No overfitting (small, clean dataset) <br>
✅ LabelEncoder perfect for company/job/degree <br>
✅ Clear business rules from tree visualization <br>

## 💼 Business Rules
✅ "Google Engineer + Masters = >100k" ✓ <br>
✅ "Google Engineer + Bachelors = ≤100k" ✓ <br>
✅ Recruiters/HR can read tree directly! <br>

## 🛠️ Files
-salary_prediction_tree.py <br>
-tree_visualization.png <br>
-salaries.csv (optional) <br>

## Decision Tree
Example	Dataset	Accuracy	Features <br>
HR Attrition	14,999 employees	92%	Numeric + salary <br>
Salary	Jobs	100%	Categorical <br>

**Progression**
HR Trees(92%) → Salary Trees(100%) → Logistic Regression ⏳ <br>

**Skills**: Categorical Trees | Label Encoding | Perfect Accuracy | Business Rules <br>
<br>
**Built with** ![Python](https://img.shields.io/badge/Python-3.11-green) <br>
<br>
![Scikit-learn](https://img.shields.io/badge/scikit-learn-1.5-blue) <br>

