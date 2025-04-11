import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocessing import load_and_preprocess
from utils.clustering import run_kmeans, visualize_clusters
from sklearn.metrics import silhouette_score

# Page setup
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Sidebar
st.sidebar.title("📊 Analysis Options")
analysis_type = st.sidebar.radio(
    "Choose Analysis Type",
    ["Data Overview", "Clustering Analysis", "Customer Insights"]
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\Personal Projects\Mall-Customer-Segmentation-Model-using-Clustering\data\mall_customers.csv")

df = load_data()

if analysis_type == "Data Overview":
    st.title("🛍️ Mall Customer Data Overview")
    
    # Display basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", len(df))
    with col2:
        st.metric("Average Age", f"{df['Age'].mean():.1f}")
    with col3:
        st.metric("Average Income", f"${df['Annual_Income'].mean():.2f}k")
    
    # Data distribution
    st.subheader("📈 Data Distribution")
    feature = st.selectbox("Select Feature", ["Age", "Annual_Income", "Spending_Score"])
    fig = px.histogram(df, x=feature, nbins=30)
    st.plotly_chart(fig)
    
    # Correlation heatmap
    st.subheader("🔗 Feature Correlations")
    numeric_cols = ['Age', 'Annual_Income', 'Spending_Score']
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif analysis_type == "Clustering Analysis":
    st.title("🎯 Customer Segmentation Analysis")
    
    # Clustering options
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Select number of clusters (K):", 2, 10, 5)
    with col2:
        features = st.multiselect(
            "Select Features for Clustering",
            ["Age", "Annual_Income", "Spending_Score"],
            default=["Annual_Income", "Spending_Score"]
        )
    
    if len(features) < 2:
        st.warning("Please select at least 2 features for clustering.")
    else:
        # Preprocessing
        processed_data = df[features]
        processed_data, scaled_data = load_and_preprocess(processed_data)
        
        # Run clustering
        labels, model = run_kmeans(scaled_data, n_clusters)
        silhouette = silhouette_score(scaled_data, labels)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Silhouette Score", f"{silhouette:.3f}")
        with col2:
            st.metric("Inertia", f"{model.inertia_:.1f}")
        
        # Visualize clusters
        st.subheader("📌 Cluster Visualization")
        if len(features) == 2:
            fig = px.scatter(
                processed_data,
                x=features[0],
                y=features[1],
                color=labels,
                title=f"Customer Segments based on {features[0]} and {features[1]}"
            )
            st.plotly_chart(fig)
        else:
            visualize_clusters(scaled_data, labels)

elif analysis_type == "Customer Insights":
    st.title("💡 Customer Insights")
    
    # Preprocessing and clustering
    processed_data, scaled_data = load_and_preprocess(df[["Age", "Annual_Income", "Spending_Score"]])
    labels, model = run_kmeans(scaled_data, 5)  # Default 5 clusters
    df['Cluster'] = labels
    
    # Cluster profiles
    st.subheader("📊 Cluster Profiles")
    cluster_profiles = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual_Income': 'mean',
        'Spending_Score': 'mean',
        'Customer_ID': 'count'
    }).round(2)
    cluster_profiles.columns = ['Avg Age', 'Avg Income', 'Avg Spending', 'Count']
    st.dataframe(cluster_profiles)
    
    # Cluster distribution
    st.subheader("📈 Cluster Size Distribution")
    fig = px.pie(
        values=cluster_profiles['Count'],
        names=cluster_profiles.index,
        title="Customer Distribution across Clusters"
    )
    st.plotly_chart(fig)
    
    # Feature distribution by cluster
    st.subheader("🎯 Feature Distribution by Cluster")
    feature = st.selectbox("Select Feature", ["Age", "Annual_Income", "Spending_Score"])
    fig = px.box(df, x='Cluster', y=feature)
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")