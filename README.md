# Task 5 - Decision Trees and Random Forests

## Objective
Train and compare tree-based models (Decision Tree and Random Forest) for classification on the Heart Disease dataset. Analyze overfitting, visualize the decision tree, interpret feature importances, and evaluate using cross-validation.

## Tools & Libraries
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Dataset
- **Name:** Heart Disease Dataset
- **Samples:** 1025
- **Features:** 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target:** 1 = Heart Disease, 0 = No Heart Disease
- **Missing Values:** None

## What I Did
1. Loaded and explored the Heart Disease dataset
2. Trained a Decision Tree with no depth limit to observe overfitting
3. Trained a pruned Decision Tree with `max_depth=4` to control overfitting
4. Visualized the pruned Decision Tree
5. Trained a Random Forest with 100 trees and compared accuracy
6. Plotted feature importances from the Random Forest
7. Evaluated both models using 5-fold cross-validation

## Results

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Decision Tree (No Limit) | 1.0000 | 0.9854 |
| Decision Tree (max_depth=4) | 0.8829 | 0.8000 |
| Random Forest (100 trees) | 1.0000 | 0.9854 |

### Cross-Validation (5-Fold)

| Model | CV Mean Accuracy | Std Dev |
|---|---|---|
| Decision Tree (max_depth=4) | 0.8341 | ± 0.0239 |
| Random Forest | 0.9971 | ± 0.0059 |

## Plots
- **decision_tree_visualization.png** — Visual structure of the pruned Decision Tree (max_depth=4)
- **overfitting_analysis.png** — Train vs Test accuracy across different tree depths
- **feature_importances.png** — Feature importance scores from Random Forest
- **model_comparison.png** — Side-by-side accuracy comparison of all 3 models
