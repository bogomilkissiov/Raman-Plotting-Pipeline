import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap, BoundaryNorm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from spectra_class import spectrum, spectra
from gruvbox_theme import GRUVBOX, GRUVBOX_CYCLE, apply_gruvbox_theme
# this program was almost entirely vibecoded

def pca_reduction(spectral_data, n_components: int = 5, print_variance: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Takes in a spectra object or list of spectrum objects and returns the dataset
    projected and reduced to n principal components, the principal components, and what 
    percentage of the variance this reduction preserves."""
    
    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])
    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")
    
    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")
    
    X = spectral_data.intensities.astype(float)
    
    pca = PCA(n_components=n_components)
    projected_data = pca.fit_transform(X)
    selected_components = pca.components_
    variance_preserved = float(np.sum(pca.explained_variance_ratio_) * 100.0)

    if print_variance:
        print(f"Kept {n_components} Principal Components, preserving {variance_preserved:.2f}% of the total variance.")

    return projected_data, selected_components, variance_preserved

def kmeans_labels(spectral_data, m_clusters: int, n_components: int = None, random_state: int = None) -> np.ndarray:
    """Takes in either a numpy array, spectra object, or a list of spectrum objects, 
    in addition to m_clusters and optional n_components, and optionally reduces the data to n_components, 
    assigns each data point a cluster label from 0 to m_clusters - 1. Returns the labels.
    NOTE: Labels are a flat numpy array whereas spectra data is row vectors.
    NOTE: It is up to the user whether or not to run pca_reduction on the data prior to clustering.
    """
    if isinstance(spectral_data, np.ndarray):
        X = np.atleast_2d(spectral_data).astype(float)
        if X.size == 0:
            raise ValueError("Provided numpy array is empty.")
    else:
        if isinstance(spectral_data, spectrum):
            spectral_data = spectra([spectral_data])
        elif isinstance(spectral_data, list):
            if not spectral_data:
                raise ValueError("List is empty.")
            if all(isinstance(s, spectrum) for s in spectral_data):
                spectral_data = spectra(spectral_data)
            else:
                raise TypeError("List must contain spectrum objects.")
        elif not isinstance(spectral_data, spectra):
            raise TypeError("spectral_data must be a numpy array, spectrum, spectra object, or list of spectrum objects.")
        
        if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
            raise ValueError("No intensity or wavenumber data provided.")
        
        X = spectral_data.intensities.astype(float)

    if n_components is not None:
        if isinstance(spectral_data, spectra):
            X, _, _ = pca_reduction(spectral_data, n_components=n_components)
        else:
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=m_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X)

    return labels

def gmm_labels(spectral_data, m_clusters: int, n_components: int = None, soft: bool = True, random_state: int = None) -> np.ndarray:
    """Takes in either a numpy array, spectra object, or a list of spectrum objects, 
    and m_clusters, n_components, and soft. Returns either the soft probabilities (n_samples, m_clusters)
    or hard cluster labels (n_samples,) depending on the value of soft.
    NOTE: Each row of the soft probabilities corresponds to one spectrum vector (also rows!).
    NOTE: It is up to the user whether or not to run pca_reduction on the data prior to clustering.
    """
    if isinstance(spectral_data, np.ndarray):
        X = np.atleast_2d(spectral_data).astype(float)
        if X.size == 0:
            raise ValueError("Provided numpy array is empty.")
    else:
        if isinstance(spectral_data, spectrum):
            spectral_data = spectra([spectral_data])
        elif isinstance(spectral_data, list):
            if not spectral_data:
                raise ValueError("List is empty.")
            if all(isinstance(s, spectrum) for s in spectral_data):
                spectral_data = spectra(spectral_data)
            else:
                raise TypeError("List must contain spectrum objects.")
        elif not isinstance(spectral_data, spectra):
            raise TypeError("spectral_data must be a numpy array, spectrum, spectra object, or list of spectrum objects.")
        
        if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
            raise ValueError("No intensity or wavenumber data provided.")
        
        X = spectral_data.intensities.astype(float)

    if n_components is not None:
        if isinstance(spectral_data, spectra):
            X, _, _ = pca_reduction(spectral_data, n_components=n_components)
        else:
            pca = PCA(n_components=n_components)
            X = pca.fit_transform(X)

    gmm = GaussianMixture(n_components=m_clusters, random_state=random_state)
    gmm.fit(X)

    if soft:
        return gmm.predict_proba(X)
    else:
        return gmm.predict(X)

def avg_kmeans_spectra(spectral_data, m_clusters: int, n_components: int = None, random_state: int = None) -> list[spectrum]:
    """Takes in either a numpy array, spectra object, or a list of spectrum objects,
    in addition to m_clusters, optional n_components, and optional random_state.
    Clusters the data using KMeans and returns a list of representative `spectrum` objects
    for each cluster, where the index in the list corresponds to the cluster group (0 to m_clusters - 1).
    """
    labels = kmeans_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, random_state=random_state)
    
    if isinstance(spectral_data, np.ndarray):
        X_raw = np.atleast_2d(spectral_data).astype(float)
        wavenumbers = np.arange(X_raw.shape[1], dtype=float)
    elif isinstance(spectral_data, (spectrum, list, spectra)):
        if isinstance(spectral_data, spectrum):
            spectral_data = spectra([spectral_data])
        elif isinstance(spectral_data, list):
            spectral_data = spectra(spectral_data)
        X_raw = spectral_data.intensities.astype(float)
        wavenumbers = np.mean(spectral_data.wavenumbers.astype(float), axis=0)
    else:
        raise TypeError("spectral_data must be a numpy array, spectrum, spectra object, or list of spectrum objects.")

    representative_spectra = []
    for k in range(m_clusters):
        mask = (labels == k)
        if np.any(mask):
            mean_intensity = np.mean(X_raw[mask], axis=0)
        else:
            mean_intensity = np.zeros(X_raw.shape[1], dtype=float)
        representative_spectra.append(spectrum(wavenumbers, mean_intensity))

    return representative_spectra

def avg_gmm_spectra(spectral_data, m_clusters: int, n_components: int = None, soft: bool = True, random_state: int = None) -> list[spectrum]:
    """Takes in either a numpy array, spectra object, or a list of spectrum objects,
    in addition to m_clusters, optional n_components, soft, and optional random_state.
    Clusters the data using Gaussian Mixture Models (GMM) and returns a list of representative
    `spectrum` objects for each cluster, where the index in the list corresponds to the cluster group (0 to m_clusters - 1).
    If soft=True, calculates the probability-weighted average spectrum for each cluster.
    If soft=False, calculates the unweighted mean spectrum of all spectra assigned to each cluster.
    """
    if isinstance(spectral_data, np.ndarray):
        X_raw = np.atleast_2d(spectral_data).astype(float)
        wavenumbers = np.arange(X_raw.shape[1], dtype=float)
    elif isinstance(spectral_data, (spectrum, list, spectra)):
        if isinstance(spectral_data, spectrum):
            spectral_data = spectra([spectral_data])
        elif isinstance(spectral_data, list):
            spectral_data = spectra(spectral_data)
        X_raw = spectral_data.intensities.astype(float)
        wavenumbers = np.mean(spectral_data.wavenumbers.astype(float), axis=0)
    else:
        raise TypeError("spectral_data must be a numpy array, spectrum, spectra object, or list of spectrum objects.")

    representative_spectra = []
    if soft:
        probs = gmm_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, soft=True, random_state=random_state)
        for k in range(m_clusters):
            weights = probs[:, k]
            total_weight = np.sum(weights)
            if total_weight > 0:
                mean_intensity = np.sum(X_raw * weights[:, np.newaxis], axis=0) / total_weight
            else:
                mean_intensity = np.zeros(X_raw.shape[1], dtype=float)
            representative_spectra.append(spectrum(wavenumbers, mean_intensity))
    else:
        labels = gmm_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, soft=False, random_state=random_state)
        for k in range(m_clusters):
            mask = (labels == k)
            if np.any(mask):
                mean_intensity = np.mean(X_raw[mask], axis=0)
            else:
                mean_intensity = np.zeros(X_raw.shape[1], dtype=float)
            representative_spectra.append(spectrum(wavenumbers, mean_intensity))

    return representative_spectra

def _extract_intensities(spectral_data) -> np.ndarray:
    """Helper to convert input spectral data into a 2D numpy intensity matrix."""
    if isinstance(spectral_data, np.ndarray):
        return np.atleast_2d(spectral_data).astype(float)
    elif isinstance(spectral_data, (spectrum, list, spectra)):
        if isinstance(spectral_data, spectrum):
            spectral_data = spectra([spectral_data])
        elif isinstance(spectral_data, list):
            spectral_data = spectra(spectral_data)
        return spectral_data.intensities.astype(float)
    else:
        raise TypeError("spectral_data must be a numpy array, spectrum, spectra object, or list of spectrum objects.")

def pca_kmeans_plot(
    spectral_data,
    m_clusters: int,
    n_components: int = None,
    point_size: float = 15,
    alpha: float = 0.95,
    random_state: int = None,
    show: bool = True,
    savepath: str = None):
    """Generates a 2D PCA projection scatter plot of the spectral data, color-coded
    by K-Means cluster assignments using the Gruvbox theme.

    Args:
        spectral_data: np.ndarray, spectrum, spectra object, or list of spectrum objects.
        m_clusters: Number of clusters to partition the data into.
        n_components: Optional number of components for initial PCA reduction before clustering.
        point_size: Size of data points (default: 15).
        alpha: Opacity of data points (default: 0.95).
        random_state: Optional random seed for reproducible clustering and PCA.
        show: Whether to display the plot (default: True).
        savepath: Optional filepath to save the generated plot image.
    """
    apply_gruvbox_theme()

    labels = kmeans_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, random_state=random_state)
    X = _extract_intensities(spectral_data)

    pca_2d = PCA(n_components=2, random_state=random_state)
    X_proj = pca_2d.fit_transform(X)
    var_exp = pca_2d.explained_variance_ratio_ * 100.0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(GRUVBOX["bg0"])
    ax.set_facecolor(GRUVBOX["bg0"])

    cluster_colors = [GRUVBOX_CYCLE[i % len(GRUVBOX_CYCLE)] for i in range(m_clusters)]

    for k in range(m_clusters):
        mask = (labels == k)
        cluster_size = np.sum(mask)
        ax.scatter(
            X_proj[mask, 0],
            X_proj[mask, 1],
            color=cluster_colors[k],
            label=f"Cluster {k} (n={cluster_size})",
            alpha=alpha,
            edgecolors="none",
            linewidths=0,
            s=point_size
        )

    pca_info = f" (Pre-reduced to {n_components} PCs)" if n_components is not None else ""
    ax.set_title(f"K-Means Clustering (k={m_clusters}){pca_info}", fontsize=13, fontweight="bold", pad=12, color=GRUVBOX["fg0"])
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% Variance)", fontsize=11, color=GRUVBOX["fg"])
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% Variance)", fontsize=11, color=GRUVBOX["fg"])

    ax.legend(loc="best", framealpha=0.85, edgecolor=GRUVBOX["bg3"])
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def pca_gmm_plot(
    spectral_data,
    m_clusters: int,
    n_components: int = None,
    point_size: float = 15,
    alpha: float = 0.95,
    soft: bool = False,
    random_state: int = None,
    show: bool = True,
    savepath: str = None):
    """Generates a 2D PCA projection scatter plot of the spectral data, color-coded
    by Gaussian Mixture Model (GMM) cluster assignments using the Gruvbox theme.
    If soft=True, point opacity (alpha) is scaled by classification certainty.

    Args:
        spectral_data: np.ndarray, spectrum, spectra object, or list of spectrum objects.
        m_clusters: Number of clusters to partition the data into.
        n_components: Optional number of components for initial PCA reduction before clustering.
        point_size: Size of data points (default: 15).
        alpha: Base opacity of data points (default: 0.95).
        soft: Whether to use soft clustering confidence for point opacity (default: False).
        random_state: Optional random seed for reproducible clustering and PCA.
        show: Whether to display the plot (default: True).
        savepath: Optional filepath to save the generated plot image.
    """
    apply_gruvbox_theme()

    X = _extract_intensities(spectral_data)

    pca_2d = PCA(n_components=2, random_state=random_state)
    X_proj = pca_2d.fit_transform(X)
    var_exp = pca_2d.explained_variance_ratio_ * 100.0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(GRUVBOX["bg0"])
    ax.set_facecolor(GRUVBOX["bg0"])

    cluster_colors = [GRUVBOX_CYCLE[i % len(GRUVBOX_CYCLE)] for i in range(m_clusters)]

    if soft:
        probs = gmm_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, soft=True, random_state=random_state)
        labels = np.argmax(probs, axis=1)
        certainty = np.max(probs, axis=1)
        # Normalize alpha range from 0.15 to the specified alpha limit
        min_p = 1.0 / m_clusters
        norm_certainty = np.clip((certainty - min_p) / (1.0 - min_p + 1e-9), 0.0, 1.0)
        alphas = 0.15 + (alpha - 0.15) * norm_certainty

        for k in range(m_clusters):
            mask = (labels == k)
            cluster_size = np.sum(mask)
            if cluster_size > 0:
                base_color = cluster_colors[k]
                point_rgba = np.array([to_rgba(base_color, alpha=a) for a in alphas[mask]])
                ax.scatter(
                    X_proj[mask, 0],
                    X_proj[mask, 1],
                    c=point_rgba,
                    label=f"Cluster {k} (n={cluster_size})",
                    edgecolors="none",
                    linewidths=0,
                    s=point_size
                )
    else:
        labels = gmm_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, soft=False, random_state=random_state)
        for k in range(m_clusters):
            mask = (labels == k)
            cluster_size = np.sum(mask)
            ax.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                color=cluster_colors[k],
                label=f"Cluster {k} (n={cluster_size})",
                alpha=alpha,
                edgecolors="none",
                linewidths=0,
                s=point_size
            )

    pca_info = f" (Pre-reduced to {n_components} PCs)" if n_components is not None else ""
    soft_info = " [Soft Fading]" if soft else ""
    ax.set_title(f"GMM Clustering (k={m_clusters}){pca_info}{soft_info}", fontsize=13, fontweight="bold", pad=12, color=GRUVBOX["fg0"])
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% Variance)", fontsize=11, color=GRUVBOX["fg"])
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% Variance)", fontsize=11, color=GRUVBOX["fg"])

    ax.legend(loc="best", framealpha=0.85, edgecolor=GRUVBOX["bg3"])
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def tSNE_kmeans_plot(
    spectral_data,
    m_clusters: int,
    n_components: int = None,
    point_size: float = 15,
    alpha: float = 0.95,
    perplexity: float = 30.0,
    random_state: int = None,
    show: bool = True,
    savepath: str = None):
    """Generates a 2D t-SNE projection scatter plot of the spectral data, color-coded
    by K-Means cluster assignments using the Gruvbox theme.

    Args:
        spectral_data: np.ndarray, spectrum, spectra object, or list of spectrum objects.
        m_clusters: Number of clusters to partition the data into.
        n_components: Optional number of components for initial PCA reduction before clustering.
        point_size: Size of data points (default: 15).
        alpha: Opacity of data points (default: 0.95).
        perplexity: Perplexity parameter for t-SNE (related to number of nearest neighbors).
        random_state: Optional random seed for reproducible clustering and t-SNE.
        show: Whether to display the plot (default: True).
        savepath: Optional filepath to save the generated plot image.
    """
    apply_gruvbox_theme()

    labels = kmeans_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, random_state=random_state)
    X = _extract_intensities(spectral_data)

    n_samples = X.shape[0]
    effective_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))

    tsne_2d = TSNE(n_components=2, perplexity=effective_perplexity, random_state=random_state)
    X_proj = tsne_2d.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(GRUVBOX["bg0"])
    ax.set_facecolor(GRUVBOX["bg0"])

    cluster_colors = [GRUVBOX_CYCLE[i % len(GRUVBOX_CYCLE)] for i in range(m_clusters)]

    for k in range(m_clusters):
        mask = (labels == k)
        cluster_size = np.sum(mask)
        ax.scatter(
            X_proj[mask, 0],
            X_proj[mask, 1],
            color=cluster_colors[k],
            label=f"Cluster {k} (n={cluster_size})",
            alpha=alpha,
            edgecolors="none",
            linewidths=0,
            s=point_size
        )

    pca_info = f" (Pre-reduced to {n_components} PCs)" if n_components is not None else ""
    ax.set_title(f"K-Means Clustering (k={m_clusters}){pca_info}", fontsize=13, fontweight="bold", pad=12, color=GRUVBOX["fg0"])
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, color=GRUVBOX["fg"])
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, color=GRUVBOX["fg"])

    ax.legend(loc="best", framealpha=0.85, edgecolor=GRUVBOX["bg3"])
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def tSNE_gmm_plot(
    spectral_data,
    m_clusters: int,
    n_components: int = None,
    point_size: float = 15,
    alpha: float = 0.95,
    soft: bool = False,
    perplexity: float = 30.0,
    random_state: int = None,
    show: bool = True,
    savepath: str = None):
    """Generates a 2D t-SNE projection scatter plot of the spectral data, color-coded
    by Gaussian Mixture Model (GMM) cluster assignments using the Gruvbox theme.
    If soft=True, point opacity (alpha) is scaled by classification certainty.

    Args:
        spectral_data: np.ndarray, spectrum, spectra object, or list of spectrum objects.
        m_clusters: Number of clusters to partition the data into.
        n_components: Optional number of components for initial PCA reduction before clustering.
        point_size: Size of data points (default: 15).
        alpha: Base opacity of data points (default: 0.95).
        soft: Whether to use soft clustering confidence for point opacity (default: False).
        perplexity: Perplexity parameter for t-SNE (related to number of nearest neighbors).
        random_state: Optional random seed for reproducible clustering and t-SNE.
        show: Whether to display the plot (default: True).
        savepath: Optional filepath to save the generated plot image.
    """
    apply_gruvbox_theme()

    X = _extract_intensities(spectral_data)

    n_samples = X.shape[0]
    effective_perplexity = min(perplexity, max(1.0, float(n_samples - 1)))

    tsne_2d = TSNE(n_components=2, perplexity=effective_perplexity, random_state=random_state)
    X_proj = tsne_2d.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(GRUVBOX["bg0"])
    ax.set_facecolor(GRUVBOX["bg0"])

    cluster_colors = [GRUVBOX_CYCLE[i % len(GRUVBOX_CYCLE)] for i in range(m_clusters)]

    if soft:
        probs = gmm_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, soft=True, random_state=random_state)
        labels = np.argmax(probs, axis=1)
        certainty = np.max(probs, axis=1)
        min_p = 1.0 / m_clusters
        norm_certainty = np.clip((certainty - min_p) / (1.0 - min_p + 1e-9), 0.0, 1.0)
        alphas = 0.15 + (alpha - 0.15) * norm_certainty

        for k in range(m_clusters):
            mask = (labels == k)
            cluster_size = np.sum(mask)
            if cluster_size > 0:
                base_color = cluster_colors[k]
                point_rgba = np.array([to_rgba(base_color, alpha=a) for a in alphas[mask]])
                ax.scatter(
                    X_proj[mask, 0],
                    X_proj[mask, 1],
                    c=point_rgba,
                    label=f"Cluster {k} (n={cluster_size})",
                    edgecolors="none",
                    linewidths=0,
                    s=point_size
                )
    else:
        labels = gmm_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, soft=False, random_state=random_state)
        for k in range(m_clusters):
            mask = (labels == k)
            cluster_size = np.sum(mask)
            ax.scatter(
                X_proj[mask, 0],
                X_proj[mask, 1],
                color=cluster_colors[k],
                label=f"Cluster {k} (n={cluster_size})",
                alpha=alpha,
                edgecolors="none",
                linewidths=0,
                s=point_size
            )

    pca_info = f" (Pre-reduced to {n_components} PCs)" if n_components is not None else ""
    soft_info = " [Soft Fading]" if soft else ""
    ax.set_title(f"GMM Clustering (k={m_clusters}){pca_info}{soft_info}", fontsize=13, fontweight="bold", pad=12, color=GRUVBOX["fg0"])
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, color=GRUVBOX["fg"])
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, color=GRUVBOX["fg"])

    ax.legend(loc="best", framealpha=0.85, edgecolor=GRUVBOX["bg3"])
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def cluster_map(
    spectral_data,
    m_clusters: int,
    clustering_algo: str = "kmeans",
    n_components: int = None,
    grid_dimensions: list = None,
    title: str = None,
    filepath: str = None,
    show: bool = True,
    show_spectra_positions: bool = True,
    random_state: int = None):
    """Creates a spatial map of cluster assignments on a binned rectangular grid of square cells,
    matching the grid fitting and aspect ratio logic of `intensity_heatmap`.
    If multiple spectra fall into the same grid cell, the cell takes the majority cluster group.
    Empty cells with no spectra are rendered in the dark background color.

    Args:
        spectral_data: `spectra` object, `spectrum` object, or list of `spectrum` objects.
        m_clusters: Number of clusters.
        clustering_algo: Clustering algorithm ('kmeans' or 'gmm', default: 'kmeans').
        n_components: Optional number of components for PCA reduction before clustering.
        grid_dimensions: Optional list or tuple [nx, ny] of grid bins (default: [20, 20]).
        title: Optional title for the plot.
        filepath: Optional path to save the generated figure.
        show: Whether to display the plot (default: True).
        show_spectra_positions: Whether to plot acquisition (x, y) coordinates as subtle black dots (default: True).
        random_state: Optional random seed for reproducible clustering and PCA.
    """
    apply_gruvbox_theme()

    if isinstance(spectral_data, spectrum):
        spectral_data = spectra([spectral_data])
    elif isinstance(spectral_data, list):
        if not spectral_data:
            raise ValueError("List is empty.")
        if all(isinstance(s, spectrum) for s in spectral_data):
            spectral_data = spectra(spectral_data)
        else:
            raise TypeError("List must contain spectrum objects.")
    elif not isinstance(spectral_data, spectra):
        raise TypeError("spectral_data must be a spectrum, spectra object, or list of spectrum objects.")

    if spectral_data.positions is None:
        raise ValueError("No spatial position data (X/Y) available in the provided spectra object.")

    if spectral_data.intensities.size == 0 or spectral_data.wavenumbers.size == 0:
        raise ValueError("No intensity or wavenumber data provided.")

    x_coords = spectral_data.x
    y_coords = spectral_data.y

    if (np.all(x_coords == 0) and np.all(y_coords == 0)) or (np.all(x_coords == x_coords[0]) and np.all(y_coords == y_coords[0])):
        raise ValueError(
            "No spatial distribution data found in dataset (all spectra coordinates are identical or (0, 0)). "
            "Note: Renishaw WiRE exports to .spc format do not include stage position metadata. Use .wdf or .txt files for spatial heatmaps."
        )

    algo = clustering_algo.lower().strip()
    if algo == "kmeans":
        labels = kmeans_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, random_state=random_state)
    elif algo == "gmm":
        labels = gmm_labels(spectral_data, m_clusters=m_clusters, n_components=n_components, soft=False, random_state=random_state)
    else:
        raise ValueError(f"Unsupported clustering_algo '{clustering_algo}'. Choose either 'kmeans' or 'gmm'.")

    if grid_dimensions is None:
        grid_dimensions = [20, 20]
    elif len(grid_dimensions) != 2:
        raise ValueError("grid_dimensions must be a list or tuple of 2 numbers: [nx, ny].")

    nx, ny = int(grid_dimensions[0]), int(grid_dimensions[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("grid_dimensions values must be positive integers.")

    span_x = float(np.nanmax(x_coords) - np.nanmin(x_coords))
    span_y = float(np.nanmax(y_coords) - np.nanmin(y_coords))

    if span_x == 0 and span_y == 0:
        s = 1.0
    elif span_x == 0:
        s = span_y / ny
    elif span_y == 0:
        s = span_x / nx
    else:
        s = max(span_x / nx, span_y / ny)

    x_c = float(np.nanmin(x_coords) + np.nanmax(x_coords)) / 2.0
    y_c = float(np.nanmin(y_coords) + np.nanmax(y_coords)) / 2.0

    w_grid = nx * s
    h_grid = ny * s

    x_min_grid = x_c - w_grid / 2.0
    x_max_grid = x_c + w_grid / 2.0
    y_min_grid = y_c - h_grid / 2.0
    y_max_grid = y_c + h_grid / 2.0

    x_edges = np.linspace(x_min_grid, x_max_grid, nx + 1)
    y_edges = np.linspace(y_min_grid, y_max_grid, ny + 1)

    # Bin each point into the grid
    ix = np.digitize(x_coords, x_edges) - 1
    iy = np.digitize(y_coords, y_edges) - 1

    # Clamp edge cases where points fall exactly on maximum edge
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # 2D grid of cluster labels, initialized with NaN for unpopulated cells
    grid_clusters = np.full((nx, ny), np.nan, dtype=float)

    # Group points by grid cell and assign majority cluster
    cell_bins = {}
    for pt_idx, (bx, by) in enumerate(zip(ix, iy)):
        cell_bins.setdefault((bx, by), []).append(labels[pt_idx])

    for (bx, by), cell_labels in cell_bins.items():
        # Find the majority cluster (most frequent label)
        counts = np.bincount(cell_labels, minlength=m_clusters)
        majority_cluster = np.argmax(counts)
        grid_clusters[bx, by] = majority_cluster

    # Create discrete Gruvbox colormap for m_clusters
    cluster_colors = [GRUVBOX_CYCLE[i % len(GRUVBOX_CYCLE)] for i in range(m_clusters)]
    cmap_obj = ListedColormap(cluster_colors)
    cmap_obj.set_bad(color=GRUVBOX["bg0"])

    # Discrete boundaries for cluster indices 0, 1, ..., m_clusters - 1
    bounds = np.arange(m_clusters + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap_obj.N)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    fig.patch.set_facecolor(GRUVBOX["bg0"])
    ax.set_facecolor(GRUVBOX["bg0"])

    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        grid_clusters.T,
        shading='flat',
        cmap=cmap_obj,
        norm=norm
    )
    ax.set_aspect('equal')

    if show_spectra_positions:
        ax.scatter(x_coords, y_coords, color="black", s=2, alpha=0.7, edgecolors="none", zorder=4)

    # Colorbar with discrete cluster ticks
    cbar = fig.colorbar(mesh, ax=ax, pad=0.04, ticks=np.arange(m_clusters))
    cbar.set_label("Cluster Group", color=GRUVBOX["fg"])
    cbar.set_ticklabels([f"Cluster {i}" for i in range(m_clusters)])

    ax.set_xlabel("X Position", color=GRUVBOX["fg"])
    ax.set_ylabel("Y Position", color=GRUVBOX["fg"])

    algo_name = "K-Means" if algo == "kmeans" else "GMM"
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color=GRUVBOX["fg0"])
    else:
        pca_info = f" ({n_components} PCs)" if n_components is not None else ""
        ax.set_title(f"{algo_name} Cluster Map (k={m_clusters}){pca_info}", fontsize=13, fontweight="bold", pad=12, color=GRUVBOX["fg0"])

    fig.tight_layout()

    if filepath:
        plt.savefig(filepath, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())

    if show:
        plt.show()
    else:
        plt.close(fig)

