"""
OptiCrop - Machine Learning Model Training
Trains multiple ML models on crop recommendation data and saves the best model.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Explore Data
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Crop_recommendation.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'images', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("     OptiCrop - Model Training Pipeline")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"    Crops: {df['label'].nunique()} unique labels")
print(f"    Missing values: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features].values
y = df['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = le.classes_

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n[2] Train/Test split: {len(X_train)} / {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Train & Evaluate Models
# ─────────────────────────────────────────────────────────────────────────────
models = {
    'K-Nearest Neighbors':   KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression':   LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':         DecisionTreeClassifier(random_state=42, max_depth=15),
    'Random Forest':         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

results = {}
print("\n[3] Model Training & Evaluation")
print("-" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5, scoring='accuracy')
    results[name] = {
        'model': model,
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std':  cv_scores.std(),
        'y_pred':  y_pred,
    }
    print(f"  {name:25s} | Test Acc: {acc:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. K-Means Clustering (unsupervised insight)
# ─────────────────────────────────────────────────────────────────────────────
kmeans = KMeans(n_clusters=len(classes), random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
print(f"\n[4] K-Means Clustering complete: {len(classes)} clusters formed")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Select Best Model
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['accuracy'])
best_result = results[best_name]
best_model = best_result['model']
print(f"\n[5] Best Model: {best_name} (Accuracy: {best_result['accuracy']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Classification Report
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6] Classification Report — {best_name}")
print("-" * 60)
print(classification_report(y_test, best_result['y_pred'],
                             target_names=classes))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Plots
# ─────────────────────────────────────────────────────────────────────────────
# 7a. Model Accuracy Comparison
fig, ax = plt.subplots(figsize=(9, 5))
model_names = list(results.keys())
accuracies  = [results[n]['accuracy'] for n in model_names]
cv_means    = [results[n]['cv_mean']  for n in model_names]
x = np.arange(len(model_names))
bars1 = ax.bar(x - 0.2, accuracies, 0.35, label='Test Accuracy',
               color=['#2ecc71','#3498db','#e67e22','#9b59b6'])
bars2 = ax.bar(x + 0.2, cv_means,   0.35, label='CV Mean Accuracy',
               color=['#27ae60','#2980b9','#d35400','#8e44ad'], alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.legend()
for bar in bars1: ax.text(bar.get_x()+bar.get_width()/2,
                          bar.get_height()+0.005,
                          f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'), dpi=120, bbox_inches='tight')
plt.close()
print("\n[7] Saved: model_comparison.png")

# 7b. Feature Importance (Random Forest)
if 'Random Forest' in results:
    rf = results['Random Forest']['model']
    importances = rf.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(importances)[::-1]
    colors = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db','#9b59b6','#1abc9c']
    ax.bar([features[i] for i in sorted_idx], importances[sorted_idx], color=colors)
    ax.set_title('Feature Importance (Random Forest)', fontweight='bold')
    ax.set_ylabel('Importance Score')
    ax.set_xlabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("       Saved: feature_importance.png")

# 7c. Crop Distribution
fig, ax = plt.subplots(figsize=(12, 5))
crop_counts = df['label'].value_counts()
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(crop_counts)))
ax.bar(crop_counts.index, crop_counts.values, color=colors_pie)
ax.set_title('Crop Sample Distribution in Dataset', fontweight='bold')
ax.set_xlabel('Crop')
ax.set_ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'crop_distribution.png'), dpi=120, bbox_inches='tight')
plt.close()
print("       Saved: crop_distribution.png")

# 7d. Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, square=True)
ax.set_title('Feature Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'correlation_heatmap.png'), dpi=120, bbox_inches='tight')
plt.close()
print("       Saved: correlation_heatmap.png")

# 7e. Confusion Matrix for Best Model
fig, ax = plt.subplots(figsize=(14, 12))
cm = confusion_matrix(y_test, best_result['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'), dpi=120, bbox_inches='tight')
plt.close()
print("       Saved: confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Save Model Artifacts
# ─────────────────────────────────────────────────────────────────────────────
with open(os.path.join(MODEL_DIR, 'model.pkl'),   'wb') as f: pickle.dump(best_model, f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'),  'wb') as f: pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, 'encoder.pkl'), 'wb') as f: pickle.dump(le, f)

# Save all model results as summary JSON
import json
summary = {
    'best_model': best_name,
    'best_accuracy': round(best_result['accuracy'], 4),
    'features': features,
    'crops': list(classes),
    'results': {
        name: {
            'accuracy': round(v['accuracy'], 4),
            'cv_mean':  round(v['cv_mean'],  4),
            'cv_std':   round(v['cv_std'],   4),
        }
        for name, v in results.items()
    }
}
with open(os.path.join(MODEL_DIR, 'model_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[8] Saved model artifacts to {MODEL_DIR}/")
print("    model.pkl | scaler.pkl | encoder.pkl | model_summary.json")
print("\n" + "=" * 60)
print("     Training Complete!")
print("=" * 60)
