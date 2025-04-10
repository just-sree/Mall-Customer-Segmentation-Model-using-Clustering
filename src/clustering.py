from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from exceptions import ClusteringError
from logger import get_logger

logger = get_logger(__name__)

def optimal_clusters(data, max_clusters=10):
    inertias = []
    silhouettes = []
    try:
        for k in range(2, max_clusters+1):
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(data)
            inertia = model.inertia_
            silhouette_avg = silhouette_score(data, labels)
            inertias.append(inertia)
            silhouettes.append(silhouette_avg)
            logger.info(f"Calculated KMeans for k={k}, inertia={inertia}, silhouette={silhouette_avg}")
        return inertias, silhouettes
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        raise ClusteringError(f"Clustering process failed. Error: {e}")

def perform_clustering(data, n_clusters):
    try:
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(data)
        logger.info(f"KMeans clustering performed with {n_clusters} clusters.")
        return labels, model
    except Exception as e:
        logger.error(f"Error performing clustering: {e}")
        raise ClusteringError(f"Clustering failed for {n_clusters} clusters. Error: {e}")
