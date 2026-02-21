# 🌲 Random Forest: Iris Classification

**Random Forest Ensemble** achieving **97.8% accuracy** on classic Iris dataset! <br>

## Dataset<br>
**Iris Dataset** (150 flowers, 3 species - ML benchmark)<br>
Features: sepal length, sepal width, petal length, petal width<br>
Target: setosa, versicolor, virginica<br>
Test set: 30% (45 samples)<br>

## 📊 Model Results
| Metric | Score |
|--------|-------|
| **Accuracy** | **97.8%** |
| **Trees** | 100 |
| **Test Split** | 70/30 |
| **Correct** | 44/45 predictions |

## 📈 Feature Importance<br>
petal length (cm) 47.98% ← DOMINANT<br>
petal width (cm) 39.41%<br>
sepal length (cm) 10.24%<br>
sepal width (cm) 2.36%<br>

![Feature Importance](feature_importance.png)

## Random forest vs Single Decision Tree<br>
Algorithm	Iris Accuracy<br>
Decision Tree	~95% (typical)<br>
Random Forest	97.8%<br>

**✅ 100 trees > 1 tree!**<br>


## Algorithm	Dataset	Accuracy
Linear Reg	Housing/HR	R² scores<br>
KNN	Diabetes	81.8%<br>
SVM	HR Attrition	78%<br>
Decision Tree	HR	92%<br>
Decision Tree	Salary	100%<br>
Decision Tree	Titanic	73.7%<br>
Random Forest	Iris	97.8% 👑<br>

## 🛠️ Files
-iris_classification_rf.py<br>
feature_importance.png<br>

## Key Ensemble Learnings
✅ 100 trees voting = 97.8% (vs single tree ~95%)<br>
✅ Petal features dominate <br>
✅ Automatic feature ranking<br>
✅ No overfitting<br>

## Progression<br>
Trees → Random Forest = Ensemble upgrade complete!<br>

---
**Skills:** Ensemble Methods | Feature Importance | Iris Benchmark | Production ML  <br>
**Built with** ![Python](https://img.shields.io/badge/Python-3.11-green)<br> ![Scikit-learn](https://img.shields.io/badge/scikit-learn-1.5-blue)
