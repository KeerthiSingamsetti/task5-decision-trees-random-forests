import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv('heart.csv')
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# ─────────────────────────────────────────────
# 2. PREPARE FEATURES AND TARGET
# ─────────────────────────────────────────────
# Target column: 'target' (1 = heart disease, 0 = no heart disease)
X = df.drop('target', axis=1)
y = df['target']

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. DECISION TREE — FULL DEPTH (Overfitting)
# ─────────────────────────────────────────────
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
test_acc_full  = accuracy_score(y_test,  dt_full.predict(X_test))

print("\n── Decision Tree (No Depth Limit) ──")
print(f"Train Accuracy: {train_acc_full:.4f}")
print(f"Test  Accuracy: {test_acc_full:.4f}")
print(f"Tree Depth:     {dt_full.get_depth()}")

# ─────────────────────────────────────────────
# 4. DECISION TREE — CONTROLLED DEPTH
# ─────────────────────────────────────────────
dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_pruned.fit(X_train, y_train)

train_acc_pruned = accuracy_score(y_train, dt_pruned.predict(X_train))
test_acc_pruned  = accuracy_score(y_test,  dt_pruned.predict(X_test))

print("\n── Decision Tree (max_depth=4) ──")
print(f"Train Accuracy: {train_acc_pruned:.4f}")
print(f"Test  Accuracy: {test_acc_pruned:.4f}")

# ─────────────────────────────────────────────
# 5. RANDOM FOREST
# ─────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

train_acc_rf = accuracy_score(y_train, rf.predict(X_train))
test_acc_rf  = accuracy_score(y_test,  rf.predict(X_test))

print("\n── Random Forest (100 trees) ──")
print(f"Train Accuracy: {train_acc_rf:.4f}")
print(f"Test  Accuracy: {test_acc_rf:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, rf.predict(X_test)))

# ─────────────────────────────────────────────
# 6. CROSS-VALIDATION
# ─────────────────────────────────────────────
cv_dt  = cross_val_score(dt_pruned, X, y, cv=5, scoring='accuracy')
cv_rf  = cross_val_score(rf,        X, y, cv=5, scoring='accuracy')

print("── Cross-Validation (5-fold) ──")
print(f"Decision Tree CV: {cv_dt.mean():.4f} ± {cv_dt.std():.4f}")
print(f"Random Forest CV: {cv_rf.mean():.4f} ± {cv_rf.std():.4f}")

# ─────────────────────────────────────────────
# PLOT 1: Decision Tree Visualization
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    dt_pruned,
    feature_names=feature_names,
    class_names=['No Disease', 'Disease'],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax
)
ax.set_title('Decision Tree (max_depth=4)', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: decision_tree_visualization.png")

# ─────────────────────────────────────────────
# PLOT 2: Overfitting Analysis — Depth vs Accuracy
# ─────────────────────────────────────────────
depths = range(1, 15)
train_scores, test_scores = [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, dt.predict(X_train)))
    test_scores.append(accuracy_score(y_test,  dt.predict(X_test)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths, train_scores, 'o-', color='#e74c3c', linewidth=2, label='Train Accuracy')
ax.plot(depths, test_scores,  's-', color='#2ecc71', linewidth=2, label='Test Accuracy')
ax.axvline(x=4, color='#3498db', linestyle='--', linewidth=1.5, label='Chosen depth (4)')
ax.set_xlabel('Tree Depth', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Overfitting Analysis: Tree Depth vs Accuracy', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: overfitting_analysis.png")

# ─────────────────────────────────────────────
# PLOT 3: Feature Importances (Random Forest)
# ─────────────────────────────────────────────
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(sorted_features[::-1], sorted_importances[::-1],
               color='#3498db', edgecolor='white')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importances — Random Forest', fontsize=14, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
for bar, val in zip(bars, sorted_importances[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importances.png")

# ─────────────────────────────────────────────
# PLOT 4: Model Comparison Bar Chart
# ─────────────────────────────────────────────
models      = ['DT (Full)', 'DT (depth=4)', 'Random Forest']
train_accs  = [train_acc_full, train_acc_pruned, train_acc_rf]
test_accs   = [test_acc_full,  test_acc_pruned,  test_acc_rf]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', color='#e74c3c', alpha=0.85)
b2 = ax.bar(x + width/2, test_accs,  width, label='Test Accuracy',  color='#2ecc71', alpha=0.85)

ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Comparison: Train vs Test Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0.7, 1.05)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")
