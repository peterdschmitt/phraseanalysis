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

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'clustered_df' not in st.session_state:
        st.session_state.clustered_df = None
    if 'num_clusters' not in st.session_state:
        st.session_state.num_clusters = 5
    if 'manual_groups' not in st.session_state:
        st.session_state.manual_groups = {}
    if 'custom_group_counter' not in st.session_state:
        st.session_state.custom_group_counter = 0

def main():
    st.title("Sales Pitch Semantic Analysis - With Manual Grouping")
    
    # Initialize session state
    initialize_session_state()
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Data Analysis", "Manual Group Management"])
    
    with tab1:
        st.header("Data Analysis")
        
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
                            
                            # Reset manual groups when running new analysis
                            st.session_state.manual_groups = {}
                            
                            # Initialize manual groups based on clusters
                            for cluster_id in range(num_clusters):
                                group_name = f"Group {cluster_id + 1}"
                                phrases = df[df['cluster'] == cluster_id]['phrase'].tolist()
                                st.session_state.manual_groups[group_name] = phrases
                            
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
                    if st.session_state.clustered_df is not None:
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
    
    with tab2:
        st.header("Manual Group Management")
        
        if st.session_state.clustered_df is None:
            st.warning("Please upload data and run analysis first in the Data Analysis tab")
        else:
            df = st.session_state.clustered_df
            
            # Create columns for the layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Group Operations")
                
                # Create new group
                new_group_name = st.text_input(
                    "Create new group:", 
                    value=f"Custom Group {st.session_state.custom_group_counter + 1}"
                )
                
                if st.button("Add Group") and new_group_name:
                    if new_group_name in st.session_state.manual_groups:
                        st.warning(f"Group '{new_group_name}' already exists!")
                    else:
                        st.session_state.manual_groups[new_group_name] = []
                        st.session_state.custom_group_counter += 1
                        st.success(f"Created new group: {new_group_name}")
                        st.experimental_rerun()
                
                # Search feature
                st.subheader("Search Phrases")
                search_term = st.text_input("Search for phrase:")
                
                if search_term and len(search_term) > 2:
                    matches = df[df['phrase'].str.contains(search_term, case=False)]
                    
                    if len(matches) == 0:
                        st.info(f"No phrases found containing '{search_term}'")
                    else:
                        st.write(f"Found {len(matches)} matching phrase(s):")
                        for _, row in matches.iterrows():
                            phrase = row['phrase']
                            
                            # Find which group this phrase is in
                            current_group = None
                            for group_name, phrases in st.session_state.manual_groups.items():
                                if phrase in phrases:
                                    current_group = group_name
                                    break
                            
                            st.write(f"**{phrase}**")
                            if current_group:
                                st.write(f"Currently in: {current_group}")
            
            with col2:
                st.subheader("Manage Groups")
                
                # Display each group
                for group_name, phrases in st.session_state.manual_groups.items():
                    with st.expander(f"{group_name} - {len(phrases)} phrases"):
                        # If the group has phrases, find the best one
                        if phrases:
                            group_df = df[df['phrase'].isin(phrases)]
                            if not group_df.empty:
                                best_phrase, score, freq, success_rate = find_best_phrase(group_df)
                                st.markdown(f"**Best Phrase:** {best_phrase}")
                                st.markdown(f"**Success Rate:** {success_rate:.4f}")
                        
                        # List each phrase with option to move
                        for phrase in phrases:
                            cols = st.columns([3, 2])
                            with cols[0]:
                                st.write(phrase)
                            with cols[1]:
                                other_groups = [g for g in st.session_state.manual_groups.keys() if g != group_name]
                                if other_groups:
                                    target_group = st.selectbox(
                                        "Move to:", 
                                        options=other_groups,
                                        key=f"select_{group_name}_{phrase}"
                                    )
                                    
                                    if st.button("Move", key=f"move_{group_name}_{phrase}"):
                                        # Remove from current group
                                        st.session_state.manual_groups[group_name].remove(phrase)
                                        
                                        # Add to target group
                                        st.session_state.manual_groups[target_group].append(phrase)
                                        
                                        st.success(f"Moved to {target_group}")
                                        st.experimental_rerun()

if __name__ == "__main__":
    main() 