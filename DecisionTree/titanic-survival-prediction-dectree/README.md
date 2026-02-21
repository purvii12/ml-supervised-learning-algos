# 🚢 Decision Tree: Titanic Survival Prediction

**Decision Tree Classification** predicting Titanic survival - **73.7% accuracy**

## Dataset
**Titanic Dataset** (Kaggle classic) <br>
Features: Pclass, Sex(Encoded), Age(Median-filled), Fare(Median-filled) <br>
Target: Survived (0=Died, 1=Survived) <br>
Test accuracy: 73.7% <br>

## 📊 Model Results  
| Metric | Score |
|--------|-------|
| **Accuracy** | **73.7%** |
| **Test Split** | 80/20 |
| **Key Features** | Sex_n, Pclass, Age, Fare |

## 🌳 Tree Visualization
![Titanic Decision Tree](titanic_tree.png) <br>
Captures "women and children first" policy perfectly! <br>

## 🔧 Production Pipeline
✅ Median imputation → Age/Fare missing values <br>
✅ Sex LabelEncoder → male=1, female=0 <br>
✅ Train/test split → Proper evaluation <br>

## Decision Tree Mastery 
Example	Dataset	Accuracy	Domain <br>
HR Attrition	14,999 employees	92%	Business <br>
Salary Prediction	Jobs	100%	HR <br>
Titanic	Historical	73.7%	Kaggle <br>

##  Historical Accuracy
✅ 73.7% = Solid baseline for Titanic <br>
✅ Trees naturally capture Pclass/Sex priority <br>
✅ Median imputation handles missing data <br>
✅ Ready for ensemble methods (Random Forest) <br>


## 🛠️ Files
titanic_survival_tree.py
titanic_tree.png
titanic.csv (optional)

**Built with** ![Python](https://img.shields.io/badge/Python-3.11-green) ![Scikit-learn](https://img.shields.io/badge/scikit-learn-1.5-blue)
