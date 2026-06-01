import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# LOAD PREPROCESSED DATA
print("=" * 60)
print("LOAD PREPROCESSED DATA")
print("=" * 60)

df = pd.read_csv("EV_preprocessed_standard.csv")
print(f"Data shape: {df.shape[0]} rows x {df.shape[1]} columns")


# DEFINE TARGET LABEL (BEV / PHEV)
print("\n" + "=" * 60)
print("DEFINE TARGET LABEL")
print("=" * 60)

# We use the BEV column directly as our binary target
# True = BEV (Battery Electric Vehicle)
# False = PHEV (Plug-in Hybrid Electric Vehicle)
y = df['Electric Vehicle Type_Battery Electric Vehicle (BEV)']

print("[Target Label Distribution]")
print(y.value_counts().rename({True: 'BEV', False: 'PHEV'}))
print("\n[Target Label Ratio]")
print(y.value_counts(normalize=True).rename({True: 'BEV', False: 'PHEV'}).map('{:.1%}'.format))

# Visualize class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = y.value_counts()
bars = ax.bar(['BEV', 'PHEV'], counts.values,
              color=['steelblue', 'salmon'], edgecolor='white')
ax.set_title('BEV vs PHEV Distribution')
ax.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{val:,}', ha='center', fontsize=11)
# ax.text: Places a text string at specific coordinates.
# bar.get_x() [left edge] + bar.get_width()/2 [half width]: centers the text
# bar.get_height() [top edge] + 500: adds a padding of 500 units 
# f'{val:,}': Formats the integer with thousands separators
# ha='center': Ensures the string alignment is centered
plt.tight_layout()
plt.savefig('bev_phev_distribution.png', dpi=150)
plt.show()
print("→ 'bev_phev_distribution.png' saved")


# PREPARE X
print("\n" + "=" * 60)
print("PREPARE X")
print("=" * 60)

# Columns to exclude:
# Make, Model: knowing the brand already reveals the type (e.g. Tesla = always BEV)
# Electric Range: directly correlated with vehicle type, would leak the answer
# Electric Vehicle Type (both one-hot cols): these ARE the label
# CAFV Eligibility: determined by battery range, indirectly leaks BEV/PHEV info
ev_type_cols = [c for c in df.columns if 'Electric Vehicle Type' in c]
cafv_cols = [c for c in df.columns if 'Eligibility' in c]
drop_for_X = ['Make', 'Model', 'Electric Range'] + ev_type_cols + cafv_cols
drop_for_X = [c for c in drop_for_X if c in df.columns]

X = df.drop(columns=drop_for_X)

print(f"Dropped columns : {drop_for_X}")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeatures used for training:")
for col in X.columns:
    print(f"  - {col}")


# CLASSIFICATION WITH K-FOLD CROSS VALIDATION
print("\n" + "=" * 60)
print("STEP 4. CLASSIFICATION WITH K-FOLD CROSS VALIDATION")
print("=" * 60)

# 5-fold CV gives a more reliable performance estimate than a single split
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# SVM was ruled out due to excessive training time on 270k+ rows
# Logistic Regression is used as a faster linear baseline instead
models = {
    'KNN (k=5)'          : KNeighborsClassifier(n_neighbors=5),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

for name, model in models.items():
    print(f"\n[{name}] Training...")
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    results[name] = scores
    print(f"  Fold Scores : {scores.round(4)}")
    print(f"  Mean        : {scores.mean():.4f}")
    print(f"  Std         : {scores.std():.4f}")


# EVALUATION
print("\n" + "=" * 60)
print("STEP 5. DETAILED EVALUATION")
print("=" * 60)

fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))

for ax, (name, model) in zip(axes, models.items()):
    # cross_val_predict collects out-of-fold predictions across all 5 folds
    # so every sample gets predicted exactly once without data leakage
    y_pred = cross_val_predict(model, X, y, cv=kf)

    # Convert True/False back to BEV/PHEV for readability
    y_display = y.map({True: 'BEV', False: 'PHEV'})
    y_pred_display = ['BEV' if p else 'PHEV' for p in y_pred]

    print(f"\n[{name}] Classification Report")
    print(classification_report(y_display, y_pred_display))

    cm = confusion_matrix(y_display, y_pred_display, labels=['BEV', 'PHEV'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BEV', 'PHEV'],
                yticklabels=['BEV', 'PHEV'],
                ax=ax)
    ax.set_title(f'{name}\nConfusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print("→ 'confusion_matrices.png' saved")


# MODEL COMPARISON
print("\n" + "=" * 60)
print("STEP 6. MODEL COMPARISON")
print("=" * 60)

means = {name: scores.mean() for name, scores in results.items()}
best_model = max(means, key=means.get)

print(f"\n{'Model':<25} {'Mean Accuracy':>15} {'Std':>10}")
print("-" * 52)
for name, scores in results.items():
    print(f"{name:<25} {scores.mean():>15.4f} {scores.std():>10.4f}")

print(f"\n★ Best Model: {best_model} (Accuracy: {means[best_model]:.4f})")

# Boxplot shows both average accuracy and variance across folds for each model
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(results.values(), labels=results.keys(), patch_artist=True,
           boxprops=dict(facecolor='lightblue'))
# - results.values() & results.keys(): Extracts the data values and maps them to their corresponding dictionary keys
# - patch_artist=True: converts so it can be filled with color.
# - boxprops=dict(facecolor='lightblue'): Passes a dictionary of properties to style the box
#   'facecolor': fills the interior with a light blue colo
ax.set_title('Model Accuracy Comparison (5-Fold CV)')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("→ 'model_comparison.png' saved")


# SUMMARY
print("\n" + "=" * 60)
print("CLASSIFICATION COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Dataset       : {len(df)} rows")
print(f"  Target        : BEV (True) / PHEV (False)")
print(f"  Features      : {X.shape[1]} columns")
print(f"  Excluded      : Make, Model (brand leakage)")
print(f"                  Electric Range (answer leakage)")
print(f"                  EV Type columns (label itself)")
print(f"                  CAFV columns (indirect leakage)")
print(f"  Evaluation    : 5-Fold Cross Validation")
print(f"  Models tested : {', '.join(models.keys())}")
print(f"  Best Model    : {best_model} (Accuracy: {means[best_model]:.4f})")