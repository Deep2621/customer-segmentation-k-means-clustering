import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set Streamlit page config
st.set_page_config(page_title="Customer Segmentation", page_icon="📊", layout="wide")

# Custom CSS for modern, light-themed UI
st.markdown("""
    <style>
        .main {
            background-color: #ffffff;
        }
        .stApp {
            background: linear-gradient(to right, #f0f4f8, #dfe9f3);
            color: black;
        }
        .css-1d391kg {
            color: black !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            padding: 10px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stFileUploader {
            border-radius: 8px;
            background-color: white;
            padding: 10px;
            border: 1px solid #ccc;
        }
        .stSlider {
            background-color: #f8f9fa;
        }
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.title("📊 Customer Segmentation using K-Means Clustering")
    st.write("Upload your customer dataset to perform clustering analysis.")
    
    # File upload section
    uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"], help="Upload a CSV file containing customer data")
    
    # Check if a file has been uploaded
    if uploaded_file is not None:
        
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("### 🔍 Preview of Uploaded Data")
        st.dataframe(df.head())
        
        # Feature selection section
        st.write("### 🛠 Select Features for Clustering")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.write("Select 'Spending Score' first, then 'Annual Income'")
        selected_features = st.multiselect("Select features", numeric_cols, default=[])
        
        # Check if at least two features are selected
        if len(selected_features) > 1:
            
            # Selecting the number of clusters (K)
            st.write("### 🔢 Choose Number of Clusters (K)")
            k = st.slider("Select K", min_value=3, max_value=5, value=3, step=1)
            
            # Data Preprocessing - Standardization
            X = df[selected_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Applying K-Means Clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Dynamically classifying clusters based on feature means
            cluster_means = df.groupby('Cluster')[selected_features].mean()
            cluster_interpretation = {}
            
            for cluster in range(k):
                mean_values = cluster_means.loc[cluster]
                if mean_values[0] > cluster_means[selected_features[0]].mean() and mean_values[1] < cluster_means[selected_features[1]].mean():
                    cluster_interpretation[cluster] = "High Spend - Low Income"
                elif mean_values[0] < cluster_means[selected_features[0]].mean() and mean_values[1] > cluster_means[selected_features[1]].mean():
                    cluster_interpretation[cluster] = "Low Spend - High Income"
                elif mean_values[0] > cluster_means[selected_features[0]].mean() and mean_values[1] > cluster_means[selected_features[1]].mean():
                    cluster_interpretation[cluster] = "High Spend - High Income"
                else:
                    cluster_interpretation[cluster] = "Low Spend - Low Income"
            
            df['Cluster Label'] = df['Cluster'].map(lambda x: cluster_interpretation.get(x, f"Segment {x}"))
            
            # Display the clustered data
            st.write("### 📊 Clustered Data")
            st.dataframe(df.head())
            
            # Display categorized customers based on cluster labels
            st.write("### 🏷 Customer Categories by Cluster")
            for label in df['Cluster Label'].unique():
                st.write(f"#### 🎯 {label}")
                st.dataframe(df[df['Cluster Label'] == label])
            
            # Visualization of Clusters with Centroids (only for two selected features)
            st.write("### 🎨 Cluster Visualization")
            if len(selected_features) >= 2:
                plt.figure(figsize=(10, 6))
                scatter = sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]], hue=df['Cluster Label'], palette=sns.color_palette("husl", k), edgecolor='black')
                
                # Plot cluster centroids
                centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
                
                plt.xlabel(selected_features[0])
                plt.ylabel(selected_features[1])
                plt.title("Customer Segmentation with Centroids")
                plt.legend(title="Cluster Interpretation", bbox_to_anchor=(1.05, 1), loc='upper left')
                st.pyplot(plt)
            
            # Display Cluster Centers
            st.write("### 🔢 Cluster Centers")
            st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=selected_features))
        
        else:
            st.warning("⚠️ Please select at least two features for clustering.")

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()
