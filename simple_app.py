import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import os

# Set page configuration
st.set_page_config(page_title="Sales Pitch Analyzer", layout="wide")

# Define directories
DATA_DIR = "data"

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Cache model loading for better performance
@st.cache_resource
def load_model():
    """Load and cache the sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(phrases, model):
    """Generate embeddings for phrases"""
    with st.spinner("Generating phrase embeddings..."):
        return model.encode(phrases)

def cluster_phrases(embeddings, n_clusters):
    """Cluster phrases based on their embeddings"""
    with st.spinner(f"Clustering into {n_clusters} groups..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)

def find_best_phrase(group_df):
    """Find the best phrase in a group based on frequency and success rate"""
    if len(group_df) == 0:
        return "", 0, 0, 0
    
    # Calculate score combining success rate and frequency
    group_df['score'] = group_df['success_rate'] * group_df['freq']
    
    # Find phrase with highest score
    best_idx = group_df['score'].idxmax()
    return (group_df.loc[best_idx, 'phrase'], 
            group_df.loc[best_idx, 'score'],
            group_df.loc[best_idx, 'freq'],
            group_df.loc[best_idx, 'success_rate'])

def main():
    st.title("Sales Pitch Semantic Analysis - Simplified")
    
    # UI for file upload
    st.subheader("Step 1: Upload your sales pitch data")
    
    # Add test data toggle
    use_test_data = st.checkbox("Use test dataset (first 250 records only)", value=False)
    
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Load data
        try:
            full_df = pd.read_excel(uploaded_file)
            
            # If test mode is enabled, use only first 250 records
            if use_test_data:
                df = full_df.head(250).copy()
                st.info(f"TEST MODE: Using first 250 records out of {len(full_df)}")
            else:
                df = full_df.copy()
            
            # Check required columns
            required_columns = ['phrase', 'freq', 'success', 'success_rate']
            
            if not all(col in df.columns for col in required_columns):
                st.error(f"Excel file must contain columns: {', '.join(required_columns)}")
            else:
                st.success(f"Successfully loaded {len(df)} records")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Show dataset stats
                with st.expander("Dataset Statistics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Records", len(df))
                    with col2:
                        st.metric("Avg. Frequency", f"{df['freq'].mean():.2f}")
                    with col3:
                        st.metric("Avg. Success Rate", f"{df['success_rate'].mean():.4f}")
                
                # Clustering options
                st.subheader("Step 2: Set clustering parameters")
                
                # Simple options for clustering
                num_clusters = st.slider("Number of groups", min_value=2, max_value=10, value=5)
                
                # Run clustering
                if st.button("Run Analysis"):
                    with st.spinner("Analyzing phrases..."):
                        # Load model and get embeddings
                        model = load_model()
                        embeddings = get_embeddings(df['phrase'].tolist(), model)
                        
                        # Perform clustering
                        clusters = cluster_phrases(embeddings, num_clusters)
                        df['cluster'] = clusters
                        
                        # Save in session state for potential reuse
                        st.session_state.clustered_df = df
                        st.session_state.num_clusters = num_clusters
                        
                        # Display results
                        st.subheader("Step 3: Results")
                        
                        # For each cluster
                        for cluster_id in range(num_clusters):
                            cluster_df = df[df['cluster'] == cluster_id]
                            
                            if len(cluster_df) > 0:
                                # Find best phrase
                                best_phrase, score, freq, success_rate = find_best_phrase(cluster_df)
                                
                                # Display cluster
                                with st.expander(f"Group {cluster_id + 1} - {len(cluster_df)} phrases"):
                                    st.markdown(f"**Best Phrase:** {best_phrase}")
                                    st.markdown(f"**Frequency:** {freq}")
                                    st.markdown(f"**Success Rate:** {success_rate:.4f}")
                                    
                                    # Display all phrases in cluster
                                    st.dataframe(
                                        cluster_df[['phrase', 'freq', 'success', 'success_rate']]
                                        .sort_values('success_rate', ascending=False)
                                    )
                
                # If results available, show option to download
                if 'clustered_df' in st.session_state:
                    st.subheader("Download Results")
                    csv = st.session_state.clustered_df.to_csv(index=False)
                    st.download_button(
                        "Download clustered data as CSV",
                        csv,
                        "clustered_sales_pitches.csv",
                        "text/csv",
                        key='download-csv'
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main() 