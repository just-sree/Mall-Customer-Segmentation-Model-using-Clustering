from src.data_loader import load_data
from src.clustering import optimal_clusters, perform_clustering
from src.visualization import plot_elbow, plot_clusters

def main():
    data = load_data(r"D:\Personal Projects\Mall-Customer-Segmentation-Model-using-Clustering\mall_customers.csv")

    features = data[['Annual_Income', 'Spending_Score']]
    inertias, silhouettes = optimal_clusters(features)

    plot_elbow(inertias, silhouettes)

    # Choose optimal clusters (e.g., 5 based on elbow)
    labels, model = perform_clustering(features, n_clusters=5)
    plot_clusters(features, labels, model)

if __name__ == "__main__":
    main()
