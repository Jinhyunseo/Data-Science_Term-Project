# ============================================================
# Term Project - EDA & Preprocessing
# Dataset: Electric Vehicle Population Data
# ============================================================
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1. LOAD DATA
# ============================================================
print("=" * 60)
print("STEP 1. LOAD DATA")
print("=" * 60)

df = pd.read_csv("Electric_Vehicle_Population_Data.csv")

print(f"Data shape: {df.shape[0]} rows x {df.shape[1]} columns")
print("\n[Column List]")
print(df.columns.tolist())
print("\n[First 5 Rows]")
print(df.head())


# ============================================================
# STEP 2. BASIC INFO CHECK (EDA - 1)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2. BASIC INFO CHECK")
print("=" * 60)

print("\n[Data Types and Non-Null Counts]")
print(df.info())

print("\n[Descriptive Statistics - Numeric Variables]")
print(df.describe())

print("\n[Descriptive Statistics - Categorical Variables]")
print(df.describe(include='object'))


# ============================================================
# STEP 3. MISSING VALUE CHECK & HANDLING (EDA - 2 / Preprocessing - 1)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3. MISSING VALUE CHECK & HANDLING")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing Rate (%)': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Rate (%)', ascending=False)

print("\n[Missing Value Summary]")
print(missing_df)

# Visualize missing values
if not missing_df.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(missing_df.index, missing_df['Missing Rate (%)'], color='salmon')
    ax.set_xlabel('Missing Rate (%)')
    ax.set_title('Missing Rate by Column')
    plt.tight_layout()
    plt.savefig('missing_values.png', dpi=150)
    plt.show()
    print("→ 'missing_values.png' saved")

# Handle missing values
# Numeric: fill with median / Categorical: fill with mode
df_clean = df.copy()

for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype in ['float64', 'int64']:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"  [Numeric] '{col}' → filled with median ({median_val})")
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"  [Categorical] '{col}' → filled with mode ('{mode_val}')")

print(f"\nTotal missing values after handling: {df_clean.isnull().sum().sum()}")


# ============================================================
# STEP 4. OUTLIER CHECK (EDA - 3)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4. NUMERIC VARIABLE DISTRIBUTION & OUTLIER CHECK")
print("=" * 60)

# Select meaningful numeric columns
numeric_cols = ['Model Year', 'Electric Range', 'Legislative District']
numeric_cols = [c for c in numeric_cols if c in df_clean.columns]

# Histograms
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))
if len(numeric_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, numeric_cols):
    ax.hist(df_clean[col].dropna(), bins=30, color='steelblue', edgecolor='white')
    ax.set_title(f'{col} Distribution')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('numeric_distribution.png', dpi=150)
plt.show()
print("→ 'numeric_distribution.png' saved")

# Boxplots (outlier visualization)
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))
if len(numeric_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, numeric_cols):
    ax.boxplot(df_clean[col].dropna(), patch_artist=True,
               boxprops=dict(facecolor='lightblue'))
    ax.set_title(f'{col} Boxplot')
    ax.set_ylabel(col)
plt.tight_layout()
plt.savefig('boxplots.png', dpi=150)
plt.show()
print("→ 'boxplots.png' saved")

# Count outliers using IQR method
print("\n[Outlier Count (IQR Method)]")
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_clean[(df_clean[col] < Q1 - 1.5 * IQR) |
                        (df_clean[col] > Q3 + 1.5 * IQR)]
    print(f"  '{col}': {len(outliers)} outliers")


# ============================================================
# STEP 5. CATEGORICAL VARIABLE DISTRIBUTION (EDA - 4)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5. CATEGORICAL VARIABLE DISTRIBUTION")
print("=" * 60)

cat_cols = ['Make', 'Electric Vehicle Type',
            'Clean Alternative Fuel Vehicle (CAFV) Eligibility', 'State']
cat_cols = [c for c in cat_cols if c in df_clean.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    top10 = df_clean[col].value_counts().head(10)
    axes[i].barh(top10.index[::-1], top10.values[::-1], color='mediumseagreen')
    axes[i].set_title(f'{col} (Top 10)')
    axes[i].set_xlabel('Frequency')

# Hide unused subplots
for j in range(len(cat_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('categorical_distribution.png', dpi=150)
plt.show()
print("→ 'categorical_distribution.png' saved")

# Annual EV registration trend
print("\n[Annual EV Registration Trend]")
year_counts = df_clean['Model Year'].value_counts().sort_index()
print(year_counts)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(year_counts.index, year_counts.values, marker='o', color='darkorange', linewidth=2)
ax.set_title('Annual EV Registration Trend')
ax.set_xlabel('Model Year')
ax.set_ylabel('Registration Count')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('yearly_trend.png', dpi=150)
plt.show()
print("→ 'yearly_trend.png' saved")


# ============================================================
# STEP 6. CORRELATION ANALYSIS (EDA - 5)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6. NUMERIC VARIABLE CORRELATION ANALYSIS")
print("=" * 60)

corr_matrix = df_clean[numeric_cols].corr()
print(corr_matrix)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, ax=ax)
ax.set_title('Numeric Variable Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print("→ 'correlation_heatmap.png' saved")


# ============================================================
# STEP 7. ENCODING (Preprocessing - 2)
# ============================================================
print("\n" + "=" * 60)
print("STEP 7. CATEGORICAL VARIABLE ENCODING")
print("=" * 60)

df_encoded = df_clean.copy()

# --- Label Encoding (high-cardinality columns without ordinal meaning) ---
label_enc_cols = ['County', 'City', 'Make', 'Model', 'Electric Utility']
label_enc_cols = [c for c in label_enc_cols if c in df_encoded.columns]

le = LabelEncoder()
for col in label_enc_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    print(f"  Label Encoding applied: '{col}'")

# --- One-Hot Encoding (low-cardinality categorical columns) ---
onehot_cols = ['Electric Vehicle Type',
               'Clean Alternative Fuel Vehicle (CAFV) Eligibility']
onehot_cols = [c for c in onehot_cols if c in df_encoded.columns]

df_encoded = pd.get_dummies(df_encoded, columns=onehot_cols, drop_first=False)
print(f"\n  One-Hot Encoding applied: {onehot_cols}")
print(f"  Column count after encoding: {df_encoded.shape[1]}")

# Drop non-informative columns (ID-like, coordinates, etc.)
drop_cols = ['VIN (1-10)', 'Vehicle Location', '2020 Census Tract',
             'DOL Vehicle ID', 'Postal Code', 'State']
drop_cols = [c for c in drop_cols if c in df_encoded.columns]
df_encoded.drop(columns=drop_cols, inplace=True)
print(f"\n  Dropped columns: {drop_cols}")
print(f"  Final column count: {df_encoded.shape[1]}")


# ============================================================
# STEP 8. SCALING (Preprocessing - 3)
# ============================================================
# ※ Electric Range는 Classification 팀에서 구간 라벨(y)로 사용하므로
#    스케일링에서 제외하고 원본값 유지
print("\n" + "=" * 60)
print("STEP 8. NUMERIC VARIABLE SCALING")
print("=" * 60)

# Electric Range 제외하고 스케일링
scale_cols = ['Model Year', 'Legislative District']
scale_cols = [c for c in scale_cols if c in df_encoded.columns]

# StandardScaler (mean=0, std=1)
scaler_std = StandardScaler()
df_scaled_std = df_encoded.copy()
df_scaled_std[scale_cols] = scaler_std.fit_transform(df_encoded[scale_cols])

# MinMaxScaler (range 0–1)
scaler_minmax = MinMaxScaler()
df_scaled_minmax = df_encoded.copy()
df_scaled_minmax[scale_cols] = scaler_minmax.fit_transform(df_encoded[scale_cols])

print(f"  Standard Scaling applied to: {scale_cols}")
print(f"  MinMax Scaling applied to: {scale_cols}")
print(f"  ※ 'Electric Range' excluded from scaling (used as Classification label)")

print("\n[Statistics After Standard Scaling]")
print(df_scaled_std[scale_cols].describe().round(3))

print("\n[Statistics After MinMax Scaling]")
print(df_scaled_minmax[scale_cols].describe().round(3))

print("\n[Electric Range - Original Value Confirmed]")
print(df_scaled_std['Electric Range'].describe().round(3))


# ============================================================
# STEP 9. SAVE PREPROCESSED DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 9. SAVE PREPROCESSED DATA")
print("=" * 60)

df_scaled_std.to_csv("EV_preprocessed_standard.csv", index=False)
df_scaled_minmax.to_csv("EV_preprocessed_minmax.csv", index=False)

print(f"  'EV_preprocessed_standard.csv' saved → for Classification team")
print(f"  'EV_preprocessed_minmax.csv' saved → for Regression/Clustering team")
print(f"\n  Final data shape: {df_scaled_std.shape[0]} rows x {df_scaled_std.shape[1]} columns")
print(f"\n  Column list:")
for col in df_scaled_std.columns:
    print(f"    - {col}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EDA / PREPROCESSING COMPLETE — SUMMARY")
print("=" * 60)
print(f"  Original data:      {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Preprocessed data:  {df_scaled_std.shape[0]} rows x {df_scaled_std.shape[1]} columns")
print(f"  Missing values:     Numeric=median, Categorical=mode")
print(f"  Encoding:           Label Encoding + One-Hot Encoding")
print(f"  Scaling:            Standard Scaler / MinMax Scaler")
print(f"  ※ Electric Range:  Scaling excluded (original values preserved)")
print(f"  Output files:       EV_preprocessed_standard.csv")
print(f"                      EV_preprocessed_minmax.csv")
print(f"  Visualization files: missing_values.png")
print(f"                       numeric_distribution.png")
print(f"                       boxplots.png")
print(f"                       categorical_distribution.png")
print(f"                       yearly_trend.png")
print(f"                       correlation_heatmap.png")