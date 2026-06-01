import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

def run_ev_clustering_optimization(filepath, max_k=6, sample_size=15000, random_state=42):
    print("=" * 60)
    print("STARTING CLUSTERING OPTIMIZATION PIPELINE")
    print("=" * 60)

    # 1. Load Data
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find the file at {filepath}. Please check the directory.")

    # 2. Feature Selection
    base_drop_cols = ['County', 'City', 'Make', 'Model', 'Legislative District', 'Electric Utility', 'Cluster']
    leakage_cols = [c for c in df.columns if 'Electric Vehicle Type' in c or 'EV_Type' in c or 'CAFV' in c or 'Eligibility' in c or 'Fuel Vehicle' in c]
    
    columns_to_drop = list(set(base_drop_cols + leakage_cols))
    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    X = df.drop(columns=existing_cols_to_drop)
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)
    X = X.fillna(X.mean())

    # [Core] Apply sampling to prevent memory overflow (O(N^2)) in Hierarchical Clustering
    print(f"\nSampling {sample_size} rows for fair algorithm comparison (Memory constraint)...")
    X_eval = X.sample(n=sample_size, random_state=random_state)
    df_eval = df.loc[X_eval.index].copy()

    # 3. Model Setup (K-Means + Hierarchical Clustering)
    algorithms = {
        'KMeans (k-means++)': KMeans(init='k-means++', random_state=random_state, n_init=10),
        'KMeans (random)': KMeans(init='random', random_state=random_state, n_init=10),
        'Agglomerative (Average)': AgglomerativeClustering(linkage='average'),
        'Agglomerative (Complete)': AgglomerativeClustering(linkage='complete')
    }
    
    k_values = range(2, max_k + 1)
    results = []

    print("\n" + "=" * 60)
    print("EVALUATING MODELS AND PARAMETERS")
    print("=" * 60)

    # 4. Training and Evaluating
    for algo_name, model in algorithms.items():
        for k in k_values:
            model.set_params(n_clusters=k)
            
            # Use fit_predict which works for both KMeans and AgglomerativeClustering
            eval_labels = model.fit_predict(X_eval)
            
            sil_score = silhouette_score(X_eval, eval_labels, random_state=random_state)
            
            # Inertia is only available for K-Means models
            if hasattr(model, 'inertia_'):
                inertia_val = round(model.inertia_, 2)
                inertia_str = f"{inertia_val:.2f}"
            else:
                inertia_val = 999999  # Assign an arbitrarily large value for sorting purposes
                inertia_str = "-"
            
            results.append({
                'Algorithm': algo_name,
                'n_clusters (K)': k,
                'Silhouette_Score': round(sil_score, 4),
                'Inertia': inertia_val,
                'Inertia_Display': inertia_str
            })
            
            print(f"Evaluated: {algo_name:25} | K={k} | Silhouette: {sil_score:.4f} | Inertia: {inertia_str}")

    # 5. Extract Top 5
    results_df = pd.DataFrame(results)
    
    # Sort in descending order based on Silhouette Score
    results_df = results_df.sort_values(by=['Silhouette_Score'], ascending=False).reset_index(drop=True)
    
    top_5_df = results_df.head(5)
    best_combination = top_5_df.iloc[0].to_dict()

    display_df = top_5_df[['Algorithm', 'n_clusters (K)', 'Silhouette_Score', 'Inertia_Display']]
    display_df.to_csv("top_5_clustering_results.csv", index=False)

    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(display_df.to_string(index=False))

    # 6. Final Result Saving & Visualization
    best_k = int(best_combination['n_clusters (K)'])
    best_algo_name = best_combination['Algorithm']
    best_model = algorithms[best_algo_name]
    best_model.set_params(n_clusters=best_k)
    
    final_labels = best_model.fit_predict(X_eval)
    
    result_df = df_eval.copy()
    result_df["Cluster"] = final_labels
    result_df.to_csv("best_clustering_result.csv", index=False)
    
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_eval)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, s=10, alpha=0.6)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"EV Clustering PCA Visualization (Algorithm: {best_algo_name}, K={best_k})")
    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    plt.savefig("best_clustering_pca.png", dpi=300)
    plt.show()

    return best_combination, display_df

if __name__ == "__main__":
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass 

    run_ev_clustering_optimization("EV_preprocessed_minmax.csv", max_k=6)
