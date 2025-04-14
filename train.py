import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

def train_kmeans_model():
    # Load data
    df = pd.read_csv('data/mall_customers.csv')
    
    # Select features for clustering
    features = ['Age', 'Annual_Income', 'Spending_Score'] 
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KMeans model
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled)
    
    # Save model and scaler
    with open('models/kmeans_model.pkl', 'wb') as f:
        pickle.dump((kmeans, scaler), f)
        pickle.dump({'model': kmeans, 'scaler': scaler}, f)
if __name__ == '__main__':
    train_kmeans_model()