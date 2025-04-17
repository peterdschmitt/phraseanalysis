import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
from typing import List, Dict, Tuple, Any
import time

# Cache the model loading to improve performance
@st.cache_resource
def load_model(model_name='all-MiniLM-L6-v2'):
    """Load and cache the sentence transformer model"""
    st.write(f"Loading sentence transformer model: {model_name}")
    return SentenceTransformer(model_name)

def get_sentence_embeddings(phrases: List[str], model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for a list of phrases"""
    st.write(f"Generating embeddings for {len(phrases)} phrases...")
    start_time = time.time()
    embeddings = model.encode(phrases)
    elapsed = time.time() - start_time
    st.write(f"Embeddings generated in {elapsed:.2f} seconds")
    return embeddings

def find_optimal_clusters(embeddings: np.ndarray, max_clusters: int, min_clusters: int = 2) -> int:
    """Find the optimal number of clusters using silhouette score"""
    st.write(f"Finding optimal number of clusters between {min_clusters} and {max_clusters}...")
    progress_bar = st.progress(0)
    silhouette_scores = []
    
    # Try different cluster counts and measure silhouette score
    total_iterations = min(max_clusters + 1, len(embeddings)) - min_clusters
    
    for i, n_clusters in enumerate(range(min_clusters, min(max_clusters + 1, len(embeddings)))):
        # Update progress
        progress = (i + 1) / total_iterations
        progress_bar.progress(progress)
        
        st.write(f"Testing {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        try:
            score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append((n_clusters, score))
            st.write(f"Clusters: {n_clusters}, Silhouette score: {score:.4f}")
        except Exception as e:
            st.write(f"Error with {n_clusters} clusters: {str(e)}")
            continue
    
    # Find the cluster count with the highest score
    if silhouette_scores:
        optimal_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        st.write(f"Optimal number of clusters: {optimal_clusters}")
        return optimal_clusters
    
    # Default to minimum if no scores available
    st.write(f"Could not find optimal clusters, using minimum: {min_clusters}")
    return min_clusters

def cluster_phrases(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster phrases based on their embeddings"""
    st.write(f"Clustering phrases into {n_clusters} groups...")
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    elapsed = time.time() - start_time
    st.write(f"Clustering completed in {elapsed:.2f} seconds")
    
    # Count phrases per cluster
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    # Show distribution
    for cluster_id, count in cluster_counts.items():
        st.write(f"Cluster {cluster_id}: {count} phrases")
        
    return clusters

def recalculate_clusters(df: pd.DataFrame, phrase_column: str = 'phrase', 
                        n_clusters: int = None, max_phrases_per_group: int = 50,
                        model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """
    Recalculate clusters for phrases
    
    Args:
        df: DataFrame with phrases
        phrase_column: Column name containing phrases
        n_clusters: Number of clusters (calculated automatically if None)
        max_phrases_per_group: Maximum phrases per group
        model_name: Name of the sentence transformer model to use
        
    Returns:
        DataFrame with added cluster column
    """
    # Show a status message
    status_container = st.empty()
    status_container.info("Starting semantic analysis...")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get phrases from dataframe
    phrases = result_df[phrase_column].tolist()
    status_container.info(f"Processing {len(phrases)} phrases...")
    
    # Load model and get embeddings
    model = load_model(model_name)
    embeddings = get_sentence_embeddings(phrases, model)
    
    # Determine optimal number of clusters if not specified
    if n_clusters is None:
        status_container.info("Determining optimal number of clusters...")
        # Calculate minimum clusters needed to keep groups under max size
        min_required_clusters = max(2, len(phrases) // max_phrases_per_group + 
                                 (1 if len(phrases) % max_phrases_per_group > 0 else 0))
        
        # Find optimal clusters with silhouette score
        max_possible_clusters = len(phrases) // 2  # Upper limit
        n_clusters = find_optimal_clusters(
            embeddings, 
            max_clusters=min(20, max_possible_clusters),  # Reasonable upper limit
            min_clusters=min_required_clusters
        )
    
    # Perform clustering
    clusters = cluster_phrases(embeddings, n_clusters)
    result_df['cluster'] = clusters
    
    # Check if any cluster exceeds max size
    cluster_sizes = result_df['cluster'].value_counts()
    oversized_clusters = cluster_sizes[cluster_sizes > max_phrases_per_group].index.tolist()
    
    # If any cluster is too large, split it further
    if oversized_clusters:
        status_container.warning(f"Found {len(oversized_clusters)} oversized clusters. Splitting...")
        for cluster_id in oversized_clusters:
            # Get subset of oversized cluster
            cluster_mask = result_df['cluster'] == cluster_id
            cluster_df = result_df[cluster_mask].copy()
            cluster_phrases = cluster_df[phrase_column].tolist()
            
            # Skip if too few phrases
            if len(cluster_phrases) < 3:
                continue
                
            # Get embeddings for this cluster
            cluster_embeddings = get_sentence_embeddings(cluster_phrases, model)
            
            # Determine subclusters needed for this group
            subclusters_needed = len(cluster_phrases) // max_phrases_per_group + 1
            st.write(f"Splitting cluster {cluster_id} ({len(cluster_phrases)} phrases) into {subclusters_needed} subclusters")
            
            # Perform sub-clustering
            subclusters = cluster_phrases(cluster_embeddings, subclusters_needed)
            
            # Update original cluster IDs by adding offset
            # Use max cluster ID as offset to avoid overlap
            offset = result_df['cluster'].max() + 1
            result_df.loc[cluster_mask, 'cluster'] = subclusters + offset
    
    status_container.success("Semantic analysis complete!")
    
    return result_df

def find_best_phrase(group_df: pd.DataFrame, 
                   phrase_col: str = 'phrase',
                   freq_col: str = 'freq',
                   success_col: str = 'success',
                   success_rate_col: str = 'success_rate') -> Dict[str, Any]:
    """Find the best phrase in a group based on frequency and success rate"""
    if len(group_df) == 0:
        return {"phrase": "", "score": 0}
        
    # Create a score combining frequency and success rate
    # Normalize frequency to 0-1 range
    max_freq = group_df[freq_col].max()
    min_freq = group_df[freq_col].min()
    
    # Calculate normalized frequency
    if max_freq == min_freq:
        group_df['norm_freq'] = 1.0
    else:
        group_df['norm_freq'] = (group_df[freq_col] - min_freq) / (max_freq - min_freq)
    
    # Weight success rate more heavily than frequency
    group_df['score'] = (0.7 * group_df[success_rate_col]) + (0.3 * group_df['norm_freq'])
    
    # Find phrase with highest score
    best_idx = group_df['score'].idxmax()
    best_phrase = group_df.loc[best_idx, phrase_col]
    best_score = group_df.loc[best_idx, 'score']
    best_freq = group_df.loc[best_idx, freq_col]
    best_success_rate = group_df.loc[best_idx, success_rate_col]
    
    return {
        "phrase": best_phrase,
        "score": best_score,
        "frequency": best_freq,
        "success_rate": best_success_rate
    } 