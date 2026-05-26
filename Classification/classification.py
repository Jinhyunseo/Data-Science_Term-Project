import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# STEP 1. LOAD PREPROCESSED DATA
# ============================================================
print("=" * 60)
print("STEP 1. LOAD PREPROCESSED DATA")
print("=" * 60)

df = pd.read_csv("EV_preprocessed_standard.csv")
print(f"Data shape: {df.shape[0]} rows x {df.shape[1]} columns")


# ============================================================
# STEP 2. CREATE TARGET LABEL (BEV / PHEV)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2. CREATE TARGET LABEL")
print("=" * 60)

# One-Hot 인코딩된 경우 컬럼명 확인
ev_type_cols = [c for c in df.columns if 'Electric Vehicle Type' in c or 'EV_Type' in c]
print(f"EV Type related columns: {ev_type_cols}")

# Electric Range로 BEV/PHEV 구분
# BEV: Electric Range > 0
# PHEV: Electric Range == 0
df['EV_Label'] = df['Electric Range'].apply(lambda x: 'BEV' if x > 0 else 'PHEV')

print("\n[Target Label Distribution]")
print(df['EV_Label'].value_counts())
print("\n[Target Label Ratio]")
print(df['EV_Label'].value_counts(normalize=True).map('{:.1%}'.format))

# 분포 시각화
fig, ax = plt.subplots(figsize=(6, 4))
counts = df['EV_Label'].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=['steelblue', 'salmon'], edgecolor='white')
ax.set_title('BEV vs PHEV Distribution')
ax.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{val:,}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('bev_phev_distribution.png', dpi=150)
plt.show()
print("→ 'bev_phev_distribution.png' saved")


# ============================================================
# STEP 3. PREPARE X, y
# ============================================================
print("\n" + "=" * 60)
print("STEP 3. PREPARE X, y")
print("=" * 60)

# y: BEV / PHEV 라벨
y = df['EV_Label']

# X에서 제거할 컬럼들
# 1) 브랜드/모델명 치트키 방지
# 2) Electric Range → 답을 직접 포함
# 3) EV Type One-Hot 컬럼 → 답 그 자체
# 4) CAFV 컬럼 → 배터리 주행거리 기반이라 BEV/PHEV 간접적으로 노출
drop_for_X = ['Make', 'Model', 'Electric Range', 'EV_Label']
drop_for_X += ev_type_cols

cafv_cols = [c for c in df.columns if 'CAFV' in c or 'Eligibility' in c or 'Fuel Vehicle' in c]
drop_for_X += cafv_cols
drop_for_X = [c for c in drop_for_X if c in df.columns]

X = df.drop(columns=drop_for_X)

print(f"Dropped columns (answer-leaking): {drop_for_X}")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeatures used for training:")
for col in X.columns:
    print(f"  - {col}")


# ============================================================
# STEP 4. CLASSIFICATION WITH K-FOLD CROSS VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("STEP 4. CLASSIFICATION WITH K-FOLD CROSS VALIDATION")
print("=" * 60)

# K-Fold 설정 (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 사용할 모델들 (수업에서 배운 것만)
# ※ SVM → 데이터 27만개에서 학습 시간 과도하게 소요
#          Logistic Regression으로 대체
models = {
    'KNN (k=5)'          : KNeighborsClassifier(n_neighbors=5),
    'Decision Tree'      : DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

results = {}

for name, model in models.items():
    print(f"\n[{name}] 학습 중...")
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    results[name] = scores
    print(f"  Fold Scores : {scores.round(4)}")
    print(f"  Mean        : {scores.mean():.4f}")
    print(f"  Std         : {scores.std():.4f}")


# ============================================================
# STEP 5. EVALUATION - CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 60)
print("STEP 5. DETAILED EVALUATION")
print("=" * 60)

fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))

for ax, (name, model) in zip(axes, models.items()):
    y_pred = cross_val_predict(model, X, y, cv=kf)

    print(f"\n[{name}] Classification Report")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred, labels=['BEV', 'PHEV'])
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


# ============================================================
# STEP 6. MODEL COMPARISON
# ============================================================
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

# 모델 비교 박스플롯
fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(results.values(), labels=results.keys(), patch_artist=True,
           boxprops=dict(facecolor='lightblue'))
ax.set_title('Model Accuracy Comparison (5-Fold CV)')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("→ 'model_comparison.png' saved")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("CLASSIFICATION COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Dataset       : {len(df)} rows")
print(f"  Target        : EV_Label (BEV / PHEV)")
print(f"  Features      : {X.shape[1]} columns")
print(f"  Excluded      : Make, Model, Electric Range (answer-leaking)")
print(f"                  EV Type columns (answer itself)")
print(f"                  CAFV columns (indirect answer-leaking)")
print(f"  Evaluation    : 5-Fold Cross Validation")
print(f"  Models tested : {', '.join(models.keys())}")
print(f"  Best Model    : {best_model} (Accuracy: {means[best_model]:.4f})")