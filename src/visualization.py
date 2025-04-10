import matplotlib.pyplot as plt
import seaborn as sns
from logger import get_logger

logger = get_logger(__name__)

def plot_elbow(inertias, silhouettes):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(2, len(inertias)+2), inertias, 'b-', marker='o', label='Inertia')
    ax2.plot(range(2, len(silhouettes)+2), silhouettes, 'r-', marker='x', label='Silhouette')

    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia', color='b')
    ax2.set_ylabel('Silhouette Score', color='r')
    plt.title('Elbow and Silhouette Method')
    plt.show()
    logger.info("Elbow and silhouette plots generated.")

def plot_clusters(data, labels, model):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=data.iloc[:,0], y=data.iloc[:,1], hue=labels, palette='viridis', s=50, alpha=0.7)
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='red', marker='X', s=200)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title('Clusters Visualization')
    plt.legend(title='Cluster')
    plt.show()
    logger.info("Cluster visualization generated.")
