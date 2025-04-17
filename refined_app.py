import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import os
import plotly.express as px
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

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

# Download NLTK resources
@st.cache_resource
def load_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass  # Handle offline cases gracefully

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
    # Use .loc to avoid SettingWithCopyWarning
    group_df_copy = group_df.copy()
    group_df_copy.loc[:, 'score'] = group_df_copy['success_rate'] * group_df_copy['freq']
    
    # Find phrase with highest score
    best_idx = group_df_copy['score'].idxmax()
    return (group_df.loc[best_idx, 'phrase'], 
            group_df_copy.loc[best_idx, 'score'],
            group_df.loc[best_idx, 'freq'],
            group_df.loc[best_idx, 'success_rate'])

def find_top_phrases(group_df, top_n=3):
    """
    Find the top N phrases in a group based on different criteria
    
    Returns:
        List of tuples (phrase, score, freq, success_rate, reason)
    """
    if len(group_df) < top_n:
        top_n = len(group_df)
        
    if len(group_df) == 0:
        return []
    
    result = []
    
    # Copy to avoid warnings
    df_copy = group_df.copy()
    
    # 1. Best overall score (balanced approach)
    df_copy.loc[:, 'balanced_score'] = df_copy['success_rate'] * df_copy['freq']
    best_balanced_idx = df_copy['balanced_score'].idxmax()
    
    result.append((
        df_copy.loc[best_balanced_idx, 'phrase'],
        df_copy.loc[best_balanced_idx, 'balanced_score'],
        df_copy.loc[best_balanced_idx, 'freq'],
        df_copy.loc[best_balanced_idx, 'success_rate'],
        "Best overall performance (balances frequency and success rate)"
    ))
    
    # 2. Highest success rate (quality-focused)
    if top_n > 1:
        # Only consider phrases with reasonable frequency (above 25th percentile)
        freq_threshold = df_copy['freq'].quantile(0.25)
        quality_df = df_copy[df_copy['freq'] >= freq_threshold]
        
        if not quality_df.empty:
            best_quality_idx = quality_df['success_rate'].idxmax()
            
            result.append((
                df_copy.loc[best_quality_idx, 'phrase'],
                df_copy.loc[best_quality_idx, 'balanced_score'],
                df_copy.loc[best_quality_idx, 'freq'],
                df_copy.loc[best_quality_idx, 'success_rate'],
                "Highest success rate with reasonable frequency"
            ))
        else:
            # Fallback to highest success rate overall
            best_quality_idx = df_copy['success_rate'].idxmax()
            
            result.append((
                df_copy.loc[best_quality_idx, 'phrase'],
                df_copy.loc[best_quality_idx, 'balanced_score'],
                df_copy.loc[best_quality_idx, 'freq'],
                df_copy.loc[best_quality_idx, 'success_rate'],
                "Highest success rate overall"
            ))
    
    # 3. Most tested (frequency-focused for statistical significance)
    if top_n > 2:
        # Find most frequently used with good success rate
        # Only consider phrases with success rate above average
        avg_success = df_copy['success_rate'].mean()
        frequent_df = df_copy[df_copy['success_rate'] >= avg_success]
        
        if not frequent_df.empty:
            most_frequent_idx = frequent_df['freq'].idxmax()
            
            result.append((
                df_copy.loc[most_frequent_idx, 'phrase'],
                df_copy.loc[most_frequent_idx, 'balanced_score'],
                df_copy.loc[most_frequent_idx, 'freq'],
                df_copy.loc[most_frequent_idx, 'success_rate'],
                "Most frequently used with above-average success rate"
            ))
        else:
            # Fallback to most frequent overall
            most_frequent_idx = df_copy['freq'].idxmax()
            
            result.append((
                df_copy.loc[most_frequent_idx, 'phrase'],
                df_copy.loc[most_frequent_idx, 'balanced_score'],
                df_copy.loc[most_frequent_idx, 'freq'],
                df_copy.loc[most_frequent_idx, 'success_rate'],
                "Most frequently used overall"
            ))
    
    return result

def reclassify_based_on_centroids(df, centroid_phrases, model):
    """
    Reclassify all phrases based on similarity to centroid phrases
    
    Args:
        df: DataFrame with phrases
        centroid_phrases: List of phrases to use as centroids
        model: SentenceTransformer model
        
    Returns:
        DataFrame with updated cluster assignments
    """
    # Get embeddings for all phrases
    all_phrases = df['phrase'].tolist()
    all_embeddings = model.encode(all_phrases)
    
    # Get embeddings for centroid phrases
    centroid_embeddings = model.encode(centroid_phrases)
    
    # Calculate similarity between each phrase and each centroid
    similarities = cosine_similarity(all_embeddings, centroid_embeddings)
    
    # Assign each phrase to the most similar centroid
    new_clusters = np.argmax(similarities, axis=1)
    
    # Update dataframe
    result_df = df.copy()
    result_df['cluster'] = new_clusters
    
    return result_df

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'clustered_df' not in st.session_state:
        st.session_state.clustered_df = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'num_clusters' not in st.session_state:
        st.session_state.num_clusters = 5
    if 'manual_groups' not in st.session_state:
        st.session_state.manual_groups = {}
    if 'custom_group_counter' not in st.session_state:
        st.session_state.custom_group_counter = 0
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'centroid_phrases' not in st.session_state:
        st.session_state.centroid_phrases = []

def generate_group_label(group_df, top_n=20, label_length=5):
    """
    Generate a concise label for a group based on the most important words in top phrases
    
    Args:
        group_df: DataFrame containing phrases from a group
        top_n: Number of top phrases to consider
        label_length: Maximum number of words in the label
    
    Returns:
        A string containing a concise label (3-5 words)
    """
    if len(group_df) == 0:
        return "Empty Group"
    
    # Get top phrases based on success_rate and frequency
    group_df_copy = group_df.copy()
    group_df_copy['score'] = group_df_copy['success_rate'] * group_df_copy['freq']
    top_phrases = group_df_copy.nlargest(top_n, 'score')['phrase'].tolist()
    
    # Preprocess phrases: lowercase, remove punctuation
    processed_phrases = []
    for phrase in top_phrases:
        # Convert to lowercase
        phrase = phrase.lower()
        # Remove punctuation except hyphens (to keep compound words)
        phrase = re.sub(r'[^\w\s-]', ' ', phrase)
        # Replace multiple spaces with single space
        phrase = re.sub(r'\s+', ' ', phrase).strip()
        processed_phrases.append(phrase)
    
    # Tokenize phrases and create a list of all words
    load_nltk_resources()
    stop_words = set(stopwords.words('english'))
    all_words = []
    for phrase in processed_phrases:
        words = phrase.split()
        # Filter out stop words and very short words
        words = [word for word in words if word not in stop_words and len(word) > 2]
        all_words.extend(words)
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Use TF-IDF to find important words
    if len(processed_phrases) > 1:
        try:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(processed_phrases)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF score for each word
            word_importance = {}
            for i, phrase in enumerate(processed_phrases):
                feature_index = tfidf_matrix[i,:].nonzero()[1]
                tfidf_scores = zip(feature_names[feature_index], [tfidf_matrix[i, x] for x in feature_index])
                for word, score in tfidf_scores:
                    if word in word_importance:
                        word_importance[word] += score
                    else:
                        word_importance[word] = score
            
            # Normalize by number of phrases
            for word in word_importance:
                word_importance[word] /= len(processed_phrases)
            
            # Combine TF-IDF scores with word frequency
            for word in word_importance:
                word_importance[word] *= word_counts.get(word, 1)
            
            # Get top words based on importance
            top_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
            top_words = [word for word, _ in top_words[:label_length]]
        except:
            # Fallback if TF-IDF fails
            top_words = [word for word, _ in word_counts.most_common(label_length)]
    else:
        # If only one phrase, just use the most frequent words
        top_words = [word for word, _ in word_counts.most_common(label_length)]
    
    # Create label from top words (max 5 words)
    label = " ".join(top_words[:label_length])
    
    # Capitalize first letter of each word
    label = " ".join(word.capitalize() for word in label.split())
    
    return label

def main():
    st.title("Sales Pitch Semantic Analysis")
    
    # Initialize session state
    initialize_session_state()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Analysis", "Manual Group Management", "Create New Groups", "Group Labels", "Final Recommendations"])
    
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
                
                # Save original df
                st.session_state.original_df = df.copy()
                
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
                    num_clusters = st.slider("Number of groups", min_value=2, max_value=15, value=5)
                    
                    # Run clustering
                    if st.button("Run Analysis"):
                        with st.spinner("Analyzing phrases..."):
                            # Load model and get embeddings
                            model = load_model()
                            st.session_state.model = model
                            embeddings = get_embeddings(df['phrase'].tolist(), model)
                            
                            # Perform clustering
                            clusters = cluster_phrases(embeddings, num_clusters)
                            df['cluster'] = clusters
                            
                            # Save in session state for potential reuse
                            st.session_state.clustered_df = df
                            st.session_state.num_clusters = num_clusters
                            
                            # Reset manual groups when running new analysis
                            st.session_state.manual_groups = {}
                            st.session_state.centroid_phrases = []
                            
                            # Initialize manual groups based on clusters
                            for cluster_id in range(num_clusters):
                                group_name = f"Group {cluster_id + 1}"
                                phrases = df[df['cluster'] == cluster_id]['phrase'].tolist()
                                st.session_state.manual_groups[group_name] = phrases
                                
                                # Find best phrase for each cluster as centroid
                                if len(phrases) > 0:
                                    cluster_df = df[df['cluster'] == cluster_id]
                                    best_phrase, _, _, _ = find_best_phrase(cluster_df)
                                    st.session_state.centroid_phrases.append(best_phrase)
                            
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
    
    with tab3:
        st.header("Create New Groups & Reclassify")
        
        if st.session_state.clustered_df is None:
            st.warning("Please upload data and run analysis first in the Data Analysis tab")
        else:
            df = st.session_state.clustered_df
            
            st.write("""
            This tab lets you create a new group using a seed phrase and reclassify all phrases.
            
            Steps:
            1. Select a phrase you'd like to use as the seed for a new group
            2. Give the new group a name
            3. Run reclassification to update all groups based on similarity
            """)
            
            # Create columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Select seed phrase from existing data
                st.subheader("Step 1: Select Seed Phrase")
                
                # Let user search for a phrase
                search_query = st.text_input("Search for a phrase:", key="new_group_search")
                
                if search_query and len(search_query) > 2:
                    # Find matching phrases
                    matches = df[df['phrase'].str.contains(search_query, case=False)]
                    
                    if len(matches) == 0:
                        st.info(f"No phrases found containing '{search_query}'")
                    else:
                        # Display matches and allow selection
                        st.write(f"Select from {len(matches)} matching phrases:")
                        phrases_list = matches['phrase'].tolist()
                        selected_phrase = st.selectbox("Choose a phrase:", phrases_list, key="seed_phrase_select")
                        
                        # Show current group for selected phrase
                        if selected_phrase:
                            current_group = None
                            for group_name, phrases in st.session_state.manual_groups.items():
                                if selected_phrase in phrases:
                                    current_group = group_name
                                    break
                            
                            if current_group:
                                st.write(f"Currently in: **{current_group}**")
                                
                            # Allow setting group name
                            new_group_name = st.text_input(
                                "New group name:", 
                                value=f"Custom Group {st.session_state.custom_group_counter + 1}",
                                key="new_seed_group_name"
                            )
                            
                            # Button to create new group and reclassify
                            if st.button("Create New Group & Reclassify"):
                                if st.session_state.model is None:
                                    st.error("Model not loaded. Please run analysis first.")
                                else:
                                    with st.spinner("Reclassifying phrases..."):
                                        # Add new group with the selected phrase
                                        if new_group_name in st.session_state.manual_groups:
                                            # If group exists already, add the phrase
                                            if selected_phrase not in st.session_state.manual_groups[new_group_name]:
                                                # Remove from current group
                                                if current_group:
                                                    st.session_state.manual_groups[current_group].remove(selected_phrase)
                                                # Add to new group
                                                st.session_state.manual_groups[new_group_name].append(selected_phrase)
                                        else:
                                            # Create new group
                                            st.session_state.manual_groups[new_group_name] = [selected_phrase]
                                            st.session_state.custom_group_counter += 1
                                            
                                            # Remove from current group
                                            if current_group:
                                                st.session_state.manual_groups[current_group].remove(selected_phrase)
                                        
                                        # Add to centroid phrases
                                        if selected_phrase not in st.session_state.centroid_phrases:
                                            st.session_state.centroid_phrases.append(selected_phrase)
                                        
                                        # Reclassify all phrases based on centroids
                                        model = st.session_state.model
                                        
                                        # Get best phrases from each group as centroids
                                        centroids = st.session_state.centroid_phrases
                                        
                                        # Reclassify
                                        new_df = reclassify_based_on_centroids(
                                            st.session_state.original_df,
                                            centroids,
                                            model
                                        )
                                        
                                        # Update clustered_df
                                        st.session_state.clustered_df = new_df
                                        
                                        # Update manual groups based on new classification
                                        st.session_state.manual_groups = {}
                                        for i, centroid in enumerate(centroids):
                                            group_name = f"Group {i + 1}" if i < len(centroids) - 1 else new_group_name
                                            st.session_state.manual_groups[group_name] = new_df[new_df['cluster'] == i]['phrase'].tolist()
                                        
                                        st.success(f"Created new group '{new_group_name}' and reclassified all phrases!")
                                        st.experimental_rerun()
            
            with col2:
                st.subheader("Current Groups")
                
                # Show the current groups and their sizes
                if st.session_state.manual_groups:
                    group_sizes = {group: len(phrases) for group, phrases in st.session_state.manual_groups.items()}
                    
                    # Display as bar chart
                    group_names = list(group_sizes.keys())
                    group_counts = list(group_sizes.values())
                    
                    st.bar_chart({
                        'Group': group_names,
                        'Phrases': group_counts
                    })
                    
                    # Also show as text
                    for group, count in group_sizes.items():
                        st.write(f"**{group}**: {count} phrases")
                else:
                    st.info("No groups available yet.")
    
    with tab4:
        st.header("Group Labels Analysis")
        
        if st.session_state.clustered_df is None:
            st.warning("Please upload data and run analysis first in the Data Analysis tab")
        else:
            df = st.session_state.clustered_df
            
            st.write("""
            This tab analyzes the top phrases in each group and generates a concise label (3-5 words) 
            that best represents the semantic meaning of the group.
            
            The labels are generated by:
            1. Finding the most important words in the top phrases of each group
            2. Filtering out common stop words and short words
            3. Combining the most significant words into a concise label
            """)
            
            st.subheader("Group Labels")
            
            # Settings for label generation
            col1, col2 = st.columns(2)
            with col1:
                top_phrases = st.slider("Number of top phrases to consider", 5, 50, 20)
            with col2:
                label_length = st.slider("Maximum words in label", 2, 6, 4)
            
            # Generate and display labels for each group
            if st.session_state.manual_groups:
                # Create a dataframe to store group information
                label_data = []
                
                for group_name, phrases in st.session_state.manual_groups.items():
                    if phrases:
                        group_df = df[df['phrase'].isin(phrases)]
                        if not group_df.empty:
                            # Generate label
                            label = generate_group_label(group_df, top_n=top_phrases, label_length=label_length)
                            
                            # Get best phrase and stats
                            best_phrase, _, freq, success_rate = find_best_phrase(group_df)
                            
                            # Add to data
                            label_data.append({
                                'Group': group_name,
                                'Generated Label': label,
                                'Phrases Count': len(phrases),
                                'Best Phrase': best_phrase,
                                'Success Rate': success_rate,
                                'Usage': freq
                            })
                
                # Create and display dataframe
                if label_data:
                    label_df = pd.DataFrame(label_data)
                    st.dataframe(label_df)
                    
                    # Expandable sections for each group with more details
                    st.subheader("Detailed Group Analysis")
                    
                    for group_name, phrases in st.session_state.manual_groups.items():
                        if not phrases:
                            continue
                            
                        group_df = df[df['phrase'].isin(phrases)]
                        if group_df.empty:
                            continue
                        
                        # Get label
                        label = generate_group_label(group_df, top_n=top_phrases, label_length=label_length)
                        
                        with st.expander(f"{group_name} - Label: '{label}'"):
                            # Show word cloud if available
                            try:
                                from wordcloud import WordCloud
                                import matplotlib.pyplot as plt
                                
                                # Preprocess and join all phrases
                                processed_text = " ".join(group_df['phrase'].str.lower().tolist())
                                processed_text = re.sub(r'[^\w\s-]', ' ', processed_text)
                                
                                # Generate word cloud
                                wordcloud = WordCloud(width=800, height=400, 
                                                    background_color='white', 
                                                    max_words=100,
                                                    collocations=False).generate(processed_text)
                                
                                # Display word cloud
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                            except:
                                st.info("Word cloud visualization is not available. Install wordcloud package if needed.")
                            
                            # Show top phrases in this group
                            st.write("### Top Phrases in This Group")
                            
                            # Sort by the combined score
                            group_df_copy = group_df.copy()
                            group_df_copy['score'] = group_df_copy['success_rate'] * group_df_copy['freq']
                            top_group_phrases = group_df_copy.nlargest(10, 'score')
                            
                            st.dataframe(
                                top_group_phrases[['phrase', 'freq', 'success_rate', 'score']]
                            )
                            
                            # Allow manual label edit
                            new_label = st.text_input("Edit label manually:", label, key=f"edit_label_{group_name}")
                            
                            if st.button("Update Label", key=f"update_label_{group_name}"):
                                # Here you would save the manual label, but we can't persist it between sessions
                                # without a database, so just show a success message
                                st.success(f"Label for {group_name} updated to: {new_label}")
            else:
                st.info("No groups available yet.")
    
    with tab5:
        st.header("Final Recommendations")
        
        if st.session_state.clustered_df is None:
            st.warning("Please upload data and run analysis first in the Data Analysis tab")
        else:
            df = st.session_state.clustered_df
            
            st.write("""
            This tab provides final recommendations for each group based on your data analysis and manual refinements.
            
            For each group, we recommend the top 3 phrases to use based on:
            1. **Best Overall Performance**: Balances both frequency and success rate
            2. **Highest Success Rate**: Focus on quality with reasonable frequency
            3. **Statistical Reliability**: Focus on most-tested phrases with good success rates
            """)
            
            # Group summary statistics
            st.subheader("Group Overview")
            
            # Create summary dataframe
            if st.session_state.manual_groups:
                summary_data = []
                
                for group_name, phrases in st.session_state.manual_groups.items():
                    if phrases:
                        group_df = df[df['phrase'].isin(phrases)]
                        if not group_df.empty:
                            avg_success = group_df['success_rate'].mean()
                            avg_freq = group_df['freq'].mean()
                            total_freq = group_df['freq'].sum()
                            best_phrase, _, _, best_rate = find_best_phrase(group_df)
                            
                            summary_data.append({
                                'Group': group_name,
                                'Phrases': len(phrases),
                                'Best Phrase': best_phrase,
                                'Best Success Rate': best_rate,
                                'Avg Success Rate': avg_success,
                                'Total Usage': total_freq
                            })
                
                # Create summary dataframe
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df)
                    
                    # Also show group distribution chart
                    fig = px.bar(
                        summary_df, 
                        x='Group', 
                        y='Phrases',
                        color='Avg Success Rate',
                        hover_data=['Best Phrase', 'Best Success Rate', 'Total Usage'],
                        title="Groups Overview",
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig)
                    
                # Detailed recommendations for each group
                st.subheader("Detailed Recommendations by Group")
                
                for group_name, phrases in st.session_state.manual_groups.items():
                    if not phrases:
                        continue
                        
                    group_df = df[df['phrase'].isin(phrases)]
                    if group_df.empty:
                        continue
                    
                    with st.expander(f"{group_name} - {len(phrases)} phrases"):
                        # Get top phrases
                        top_phrases = find_top_phrases(group_df, top_n=3)
                        
                        if top_phrases:
                            st.write("### Top Recommended Phrases")
                            
                            for i, (phrase, score, freq, success_rate, reason) in enumerate(top_phrases):
                                st.markdown(f"#### {i+1}. {phrase}")
                                st.markdown(f"**Success Rate:** {success_rate:.4f} | **Frequency:** {freq}")
                                st.markdown(f"**Reason:** {reason}")
                                st.markdown("---")
                        
                        # Show distribution of success rates in this group
                        st.write("### Group Statistics")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Success rate distribution
                            fig = px.histogram(
                                group_df, 
                                x='success_rate',
                                nbins=10,
                                title='Success Rate Distribution'
                            )
                            st.plotly_chart(fig)
                            
                        with col2:
                            # Frequency distribution
                            fig = px.histogram(
                                group_df, 
                                x='freq',
                                nbins=10,
                                title='Frequency Distribution'
                            )
                            st.plotly_chart(fig)
                            
                        # Show actual data table
                        st.write("### All Phrases in this Group")
                        st.dataframe(
                            group_df[['phrase', 'freq', 'success', 'success_rate']]
                            .sort_values('success_rate', ascending=False)
                        )
            else:
                st.info("No groups available yet.")

if __name__ == "__main__":
    main() 