import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from utils.preprocessing import load_and_preprocess
from utils.clustering import run_kmeans, visualize_clusters
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # Import KMeans

# Page setup
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Sidebar
st.sidebar.title("📊 Analysis Options")
analysis_type = st.sidebar.radio(
    "Choose Analysis Type",
    ["Data Overview", "Clustering Analysis", "Customer Insights", "Predict Cluster"]
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\Personal Projects\Mall-Customer-Segmentation-Model-using-Clustering\data\mall_customers.csv")

df = load_data()

def perform_clustering(df, n_clusters=5):
    """
    Performs clustering on the given DataFrame and returns the labels and model.
    """
    processed_data, scaled_data = load_and_preprocess(df[["Age", "Annual_Income", "Spending_Score"]])
    labels, model = run_kmeans(scaled_data, n_clusters)
    return labels, model, processed_data, scaled_data

match analysis_type:
    case "Data Overview":
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

    case "Clustering Analysis":
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
                    color=labels.astype(str),  # Convert labels to string for Plotly
                    title=f"Customer Segments based on {features[0]} and {features[1]}"
                )
                st.plotly_chart(fig)
            else:
                st.warning("Cluster visualization is only available for 2 features.")

    case "Customer Insights":
        st.title("💡 Customer Insights")
        
        # Preprocessing and clustering
        labels, model, processed_data, scaled_data = perform_clustering(df)
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
            names=cluster_profiles.index.astype(str),  # Convert index to string for Plotly
            title="Customer Distribution across Clusters"
        )
        st.plotly_chart(fig)
        
        # Feature distribution by cluster
        st.subheader("🎯 Feature Distribution by Cluster")
        feature = st.selectbox("Select Feature", ["Age", "Annual_Income", "Spending_Score"])
        fig = px.box(df, x='Cluster', y=feature)
        st.plotly_chart(fig)

    case "Predict Cluster":
        st.title("🎯 Predict Customer Cluster")
        st.write("Enter customer information to predict their segment")

        # Create columns for input fields
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
        with col2:
            annual_income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
        with col3:
            spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

        # Create a prediction button
        if st.button("Predict Cluster"):
            # Prepare the input data
            input_data = pd.DataFrame({
                'Age': [age],
                'Annual_Income': [annual_income],
                'Spending_Score': [spending_score]
            })

            # Scale the input data using the same scaler used for training
            scaler = StandardScaler()
            train_data = df[['Age', 'Annual_Income', 'Spending_Score']]
            scaled_train_data = scaler.fit_transform(train_data)
            scaled_input = scaler.transform(input_data)

            # Train the model and get predictions
            labels, model, processed_data, scaled_data = perform_clustering(df)
            kmeans = KMeans(n_clusters=5, random_state=42)  # Initialize KMeans
            kmeans.fit(scaled_data)  # Fit the model
            cluster = kmeans.predict(scaled_input)[0]  # Predict the cluster
            
            # Display the prediction
            st.success(f"🎯 Predicted Customer Segment: Cluster {cluster}")

            # Show cluster characteristics
            st.subheader("📊 Cluster Characteristics")
            
            # Add the new point to the existing data for visualization
            df_temp = df.copy()
            df_temp['Cluster'] = labels  # Use existing labels
            df_temp.loc[len(df_temp)] = [len(df_temp)+1, 'Unknown', age, annual_income, spending_score, cluster]  # Add new row
            
            # Scale all data
            all_data = scaler.fit_transform(df_temp[['Age', 'Annual_Income', 'Spending_Score']])
            
            # Create visualization with the new point highlighted
            fig = px.scatter(
                df_temp,
                x='Annual_Income',
                y='Spending_Score',
                color='Cluster',
                title="Customer Segments with New Customer Highlighted",
                hover_data=['Age']
            )
            
            # Highlight the new point
            fig.add_scatter(
                x=[annual_income],
                y=[spending_score],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='star',
                    color='yellow',
                    line=dict(color='black', width=2)
                ),
                name='New Customer',
                hovertext=f'Age: {age}'
            )
            
            st.plotly_chart(fig)

            # Show cluster profile
            cluster_profile = df.groupby('Cluster').agg({
                'Age': 'mean',
                'Annual_Income': 'mean',
                'Spending_Score': 'mean',
                'Customer_ID': 'count'
            }).round(2)
            
            cluster_profile.columns = ['Avg Age', 'Avg Income', 'Avg Spending', 'Count']
            st.write("Your customer belongs to a segment with these average characteristics:")
            st.dataframe(cluster_profile.iloc[[cluster]])

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit")