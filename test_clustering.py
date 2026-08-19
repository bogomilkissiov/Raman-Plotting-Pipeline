import os
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from spectra_class import spectra
import preprocessing as pp
import spectra_plotters as sp
import clustering as cl

# 1. Ensure output folder exists
output_dir = os.path.join("test_plots", "Clustering")
os.makedirs(output_dir, exist_ok=True)

# 2. Load dataset
data_filepath = os.path.join("test_data", "2s_100lp_map-2 (1).wdf")
print(f"Loading 2s_100lp dataset from {data_filepath} ...")
two_100lp_raw = spectra.from_wdf(data_filepath)
print(f"Loaded {two_100lp_raw.intensities.shape[0]} spectra with {two_100lp_raw.intensities.shape[1]} wavenumber points.")

print("Running preprocessing pipeline...")
dataset = pp.preprocess_pipeline(two_100lp_raw)
print("Preprocessing complete.\n")

# 3. Determine optimal number of clusters
# We'll use PCA-reduced features (e.g. 5 PCs) to evaluate silhouette scores for KMeans and BIC for GMM
print("Determining optimal number of clusters...")
X_pca, _, _ = cl.pca_reduction(dataset, n_components=5)

k_range = range(2, 8)
silhouette_scores = {}
bic_scores = {}

for k in k_range:
    # KMeans Silhouette Score (higher is better)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, km_labels)
    silhouette_scores[k] = sil

    # GMM BIC Score (lower is better)
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_pca)
    bic = gmm.bic(X_pca)
    bic_scores[k] = bic

    print(f"  k={k} -> Silhouette Score (KMeans): {sil:.4f} | BIC (GMM): {bic:.1f}")

best_k_sil = max(silhouette_scores, key=silhouette_scores.get)
best_k_bic = min(bic_scores, key=bic_scores.get)

print(f"\nOptimal k by Silhouette Score: {best_k_sil}")
print(f"Optimal k by GMM BIC: {best_k_bic}")

# Use optimal k
m_clusters = best_k_sil
print(f"Selected m_clusters = {m_clusters} for plotting and representative spectra extraction.\n")

# 4. Run and save all 4 clustering plot functions
print("1/6: Generating PCA K-Means plot...")
cl.pca_kmeans_plot(
    dataset,
    m_clusters=m_clusters,
    n_components=5,
    random_state=42,
    show=False,
    savepath=os.path.join(output_dir, "pca_kmeans.png")
)

print("2/6: Generating PCA GMM plot (soft)...")
cl.pca_gmm_plot(
    dataset,
    m_clusters=m_clusters,
    n_components=5,
    soft=True,
    random_state=42,
    show=False,
    savepath=os.path.join(output_dir, "pca_gmm_soft.png")
)

print("3/6: Generating t-SNE K-Means plot...")
cl.tSNE_kmeans_plot(
    dataset,
    m_clusters=m_clusters,
    n_components=5,
    perplexity=30.0,
    random_state=42,
    show=False,
    savepath=os.path.join(output_dir, "tsne_kmeans.png")
)

print("4/6: Generating t-SNE GMM plot (soft)...")
cl.tSNE_gmm_plot(
    dataset,
    m_clusters=m_clusters,
    n_components=5,
    soft=True,
    perplexity=30.0,
    random_state=42,
    show=False,
    savepath=os.path.join(output_dir, "tsne_gmm_soft.png")
)

# 5. Generate spatial cluster maps
print("5/8: Generating spatial K-Means cluster map...")
cl.cluster_map(
    dataset,
    m_clusters=m_clusters,
    clustering_algo="kmeans",
    n_components=5,
    grid_dimensions=[100, 100],
    random_state=42,
    show=False,
    filepath=os.path.join(output_dir, "cluster_map_kmeans.png")
)

print("6/8: Generating spatial GMM cluster map...")
cl.cluster_map(
    dataset,
    m_clusters=m_clusters,
    clustering_algo="gmm",
    n_components=5,
    grid_dimensions=[100, 100],
    random_state=42,
    show=False,
    filepath=os.path.join(output_dir, "cluster_map_gmm.png")
)

# 6. Extract representative spectra and plot composite images using plot_spectra
print("7/8: Computing & plotting representative KMeans spectra...")
km_rep_spectra = cl.avg_kmeans_spectra(dataset, m_clusters=m_clusters, n_components=5, random_state=42)
km_rep_packed = spectra(km_rep_spectra)
sp.plot_spectra(
    km_rep_packed,
    index_range=[0, m_clusters],
    filepath=os.path.join(output_dir, "avg_kmeans_spectra_composite.png"),
    show=False
)

print("8/8: Computing & plotting representative GMM spectra (soft)...")
gmm_rep_spectra = cl.avg_gmm_spectra(dataset, m_clusters=m_clusters, n_components=5, soft=True, random_state=42)
gmm_rep_packed = spectra(gmm_rep_spectra)
sp.plot_spectra(
    gmm_rep_packed,
    index_range=[0, m_clusters],
    filepath=os.path.join(output_dir, "avg_gmm_spectra_composite.png"),
    show=False
)

print(f"\nAll clustering plots, maps, and representative spectra generated and saved successfully to '{output_dir}/'!")
