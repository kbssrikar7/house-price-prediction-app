"""Unsupervised learning functionality."""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ml_portfolio")


def perform_kmeans_clustering(
    X: np.ndarray,
    k_range: List[int] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform KMeans clustering with elbow method analysis.

    Args:
        X: Feature matrix
        k_range: Range of k values to try
        random_state: Random seed

    Returns:
        Dictionary with clustering results including models,
        inertias, silhouette scores, and optimal k
    """
    if k_range is None:
        k_range = list(range(2, 11))

    inertias = []
    silhouette_scores_list = []
    models = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        inertias.append(kmeans.inertia_)
        silhouette_scores_list.append(silhouette_score(X, labels))
        models[k] = kmeans

    # Find optimal k using silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores_list)]

    logger.info(
        f"KMeans: Optimal k={optimal_k} with "
        f"silhouette={max(silhouette_scores_list):.4f}"
    )

    return {
        'k_range': k_range,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores_list,
        'optimal_k': optimal_k,
        'best_model': models[optimal_k],
        'labels': models[optimal_k].labels_
    }


def perform_dbscan_clustering(
    X: np.ndarray,
    eps_range: List[float] = None,
    min_samples_range: List[int] = None
) -> Dict[str, Any]:
    """
    Perform DBSCAN clustering with parameter search.

    Args:
        X: Feature matrix
        eps_range: Range of eps values to try
        min_samples_range: Range of min_samples values to try

    Returns:
        Dictionary with clustering results and best parameters
    """
    if eps_range is None:
        eps_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if min_samples_range is None:
        min_samples_range = [3, 5, 10, 15]

    best_score = -1
    best_params = None
    best_labels = None
    results = []

    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()

            # Only compute silhouette if we have valid clusters
            if n_clusters >= 2 and n_clusters < len(X) - 1:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    score = silhouette_score(X[mask], labels[mask])
                else:
                    score = -1
            else:
                score = -1

            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': score
            })

            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_labels = labels

    logger.info(f"DBSCAN: Best params={best_params} with silhouette={best_score:.4f}")

    return {
        'results': pd.DataFrame(results),
        'best_params': best_params,
        'best_score': best_score,
        'labels': best_labels
    }


def perform_pca(
    X: np.ndarray,
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95
) -> Tuple[PCA, np.ndarray]:
    """
    Perform PCA for dimensionality reduction.

    Args:
        X: Feature matrix
        n_components: Number of components (None for auto based on variance)
        variance_threshold: Variance threshold for auto component selection

    Returns:
        Tuple of (fitted PCA, transformed data)
    """
    # First fit to find optimal components
    if n_components is None:
        pca_full = PCA()
        pca_full.fit(X)

        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= variance_threshold) + 1
        n_components = max(2, min(n_components, X.shape[1]))

    # Final PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    logger.info(
        f"PCA: Reduced to {n_components} components, "
        f"explaining {pca.explained_variance_ratio_.sum():.2%} variance"
    )

    return pca, X_pca


def perform_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> np.ndarray:
    """
    Perform t-SNE for visualization.

    Args:
        X: Feature matrix
        n_components: Number of output dimensions
        perplexity: t-SNE perplexity parameter
        random_state: Random seed

    Returns:
        Transformed data array
    """
    # Subsample if dataset is large
    if len(X) > 5000:
        logger.warning("Dataset large, subsampling to 5000 for t-SNE")
        indices = np.random.RandomState(random_state).choice(
            len(X), 5000, replace=False
        )
        X = X[indices]

    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(X) - 1),
        random_state=random_state,
        max_iter=1000
    )

    X_tsne = tsne.fit_transform(X)

    logger.info(f"t-SNE completed with perplexity={perplexity}")

    return X_tsne


def interpret_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str]
) -> Tuple[pd.DataFrame, Dict]:
    """
    Interpret cluster characteristics by analyzing feature statistics.

    Args:
        X: Feature data (numpy array or DataFrame)
        labels: Cluster labels
        feature_names: Names of features

    Returns:
        Tuple of (cluster_stats DataFrame, distinguishing features dict)
    """
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['Cluster'] = labels

    # Compute statistics per cluster
    cluster_stats = X_df.groupby('Cluster').agg(['mean', 'std', 'median'])

    # Overall statistics for comparison
    overall_stats = X_df.drop('Cluster', axis=1).agg(['mean', 'std', 'median'])

    # Identify distinguishing features per cluster
    distinguishing = {}
    for cluster in X_df['Cluster'].unique():
        if cluster == -1:  # Skip noise cluster from DBSCAN
            continue

        cluster_data = X_df[X_df['Cluster'] == cluster].drop('Cluster', axis=1)

        # Z-score vs overall mean
        overall_std = overall_stats.loc['std']
        non_zero_std = overall_std.replace(0, np.nan)
        z_scores = (cluster_data.mean() - overall_stats.loc['mean']) / non_zero_std
        z_scores = z_scores.abs().sort_values(ascending=False).dropna()

        distinguishing[cluster] = z_scores.head(5).to_dict()

    logger.info(f"Analyzed {len(set(labels) - {-1})} clusters")

    return cluster_stats, distinguishing


def plot_elbow_and_silhouette(
    kmeans_results: Dict[str, Any],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot elbow curve and silhouette scores side by side.

    Args:
        kmeans_results: Results from perform_kmeans_clustering
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    k_range = kmeans_results['k_range']

    # Elbow plot
    axes[0].plot(k_range, kmeans_results['inertias'], 'bo-', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inertia', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14)
    axes[0].axvline(x=kmeans_results['optimal_k'], color='r', linestyle='--',
                   label=f'Optimal k={kmeans_results["optimal_k"]}')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Silhouette plot
    axes[1].plot(k_range, kmeans_results['silhouette_scores'], 'go-', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Analysis', fontsize=14)
    axes[1].axvline(x=kmeans_results['optimal_k'], color='r', linestyle='--',
                   label=f'Optimal k={kmeans_results["optimal_k"]}')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
