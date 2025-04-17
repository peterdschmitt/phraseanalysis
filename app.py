import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from data_importer import (
    import_excel_to_dataframe, 
    save_dataframe_to_permanent_storage, 
    load_dataframe_from_permanent_storage,
    list_saved_datasets,
    setup_data_directory
)
from semantic_analyzer import (
    recalculate_clusters,
    find_best_phrase,
    get_sentence_embeddings,
    load_model
)
import time

def update_clusters_from_manual_groups(df, manual_groups):
    """
    Update cluster assignments based on manual groupings
    
    Args:
        df: DataFrame with the original data
        manual_groups: Dictionary mapping group names to lists of phrases
        
    Returns:
        Updated DataFrame with new cluster assignments
    """
    # Create a copy of the dataframe
    updated_df = df.copy()
    
    # Create a mapping of phrases to their group IDs
    phrase_to_group = {}
    for group_id, (group_name, phrases) in enumerate(manual_groups.items()):
        for phrase in phrases:
            phrase_to_group[phrase] = group_id
    
    # Update cluster assignments in the dataframe
    updated_df['manual_cluster'] = updated_df['phrase'].map(phrase_to_group)
    
    # If any phrases weren't mapped (shouldn't happen), keep their original cluster
    if updated_df['manual_cluster'].isnull().any():
        # Fill NaN with original cluster if available
        if 'cluster' in updated_df.columns:
            updated_df.loc[updated_df['manual_cluster'].isnull(), 'manual_cluster'] = \
                updated_df.loc[updated_df['manual_cluster'].isnull(), 'cluster']
        else:
            # Assign to a special "uncategorized" cluster
            max_cluster = updated_df['manual_cluster'].max()
            updated_df['manual_cluster'] = updated_df['manual_cluster'].fillna(max_cluster + 1)
    
    # Convert to integer
    updated_df['manual_cluster'] = updated_df['manual_cluster'].astype(int)
    
    # Replace the original cluster column with the manual one
    if 'cluster' in updated_df.columns:
        updated_df['cluster'] = updated_df['manual_cluster']
    else:
        updated_df['cluster'] = updated_df['manual_cluster']
    
    # Drop the temporary column
    updated_df = updated_df.drop(columns=['manual_cluster'])
    
    return updated_df

def main():
    # Configure Streamlit page
    st.set_page_config(
        page_title="Sales Pitch Semantic Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Sales Pitch Semantic Analysis")
    
    # Initialize logging section in sidebar
    st.sidebar.title("Navigation")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
    
    if debug_mode:
        st.sidebar.warning("Debug Mode Enabled")
        log_container = st.sidebar.container()
        
        # Function for logging in debug mode
        def debug_log(message):
            timestamp = time.strftime("%H:%M:%S")
            log_container.write(f"[{timestamp}] {message}")
    else:
        # No-op function when debug is disabled
        def debug_log(message):
            pass
            
    # Set up the sidebar for navigation
    page = st.sidebar.radio("Go to", ["Data Import", "Semantic Analysis", "Group Management"])
    
    # Initialize session state for dataframe if not exists
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'clustered_df' not in st.session_state:
        st.session_state.clustered_df = None
    if 'manual_groups' not in st.session_state:
        st.session_state.manual_groups = {}
    if 'custom_group_counter' not in st.session_state:
        st.session_state.custom_group_counter = 0
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False
    if 'last_action_time' not in st.session_state:
        st.session_state.last_action_time = time.time()
    
    # Help toggle in sidebar
    if st.sidebar.button("Toggle Help"):
        st.session_state.show_help = not st.session_state.show_help
    
    if st.session_state.show_help:
        st.sidebar.info("""
        ## How to use this app
        
        1. **Data Import**: Upload your Excel file with sales pitches
        2. **Semantic Analysis**: Group phrases semantically 
        3. **Group Management**: Fine-tune groups and reanalyze
        
        Your Excel file should have columns:
        - phrase: The sales pitch text
        - freq: How often it's used
        - success: Number of successful uses
        - success_rate: Success probability (0-1)
        """)
    
    # Data Import Page
    if page == "Data Import":
        st.header("Import Sales Pitch Data")
        debug_log("Entered Data Import page")
        
        # Create tabs for different import methods
        tab1, tab2 = st.tabs(["Upload New File", "Use Existing Dataset"])
        
        with tab1:
            st.write("Upload your Excel file containing sales pitches")
            uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
            
            if uploaded_file is not None:
                debug_log(f"File uploaded: {uploaded_file.name}")
                # Save filename for later use
                original_filename = uploaded_file.name
                
                # Import the uploaded file to a dataframe
                with st.spinner("Processing uploaded file..."):
                    df = import_excel_to_dataframe(uploaded_file)
                
                if df is not None:
                    debug_log(f"Successfully imported {len(df)} records")
                    st.success(f"Successfully imported {len(df)} records!")
                    
                    # Display preview of the data
                    st.write("Preview of imported data:")
                    st.dataframe(df.head())
                    
                    # Show column information
                    st.write("Column information:")
                    info_df = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes,
                        'Non-Null Count': df.count(),
                        'Example Values': [str(df[col].iloc[0]) if len(df) > 0 else "" for col in df.columns]
                    })
                    st.dataframe(info_df)
                    
                    # Option to save permanently
                    save_as = st.text_input("Save dataset as (filename):", 
                                          value=os.path.splitext(original_filename)[0])
                    
                    if save_as and st.button("Save Dataset"):
                        with st.spinner("Saving dataset..."):
                            filename = f"{save_as}.csv"
                            file_path = save_dataframe_to_permanent_storage(df, filename)
                            st.session_state.current_df = df
                            st.session_state.dataset_name = filename
                            debug_log(f"Dataset saved as {file_path}")
                            st.success(f"Dataset saved as {file_path}")
                        
                    # Option to use without saving
                    if st.button("Use Without Saving"):
                        st.session_state.current_df = df
                        st.session_state.dataset_name = "Temporary Dataset"
                        debug_log("Using dataset without saving")
                        st.success("Dataset loaded for analysis (not saved permanently)")
        
        with tab2:
            # List available datasets
            saved_datasets = list_saved_datasets()
            debug_log(f"Found {len(saved_datasets)} saved datasets")
            
            if not saved_datasets:
                st.info("No saved datasets found. Please upload a file first.")
            else:
                st.write("Select a previously saved dataset:")
                selected_dataset = st.selectbox("Choose dataset", saved_datasets)
                
                if selected_dataset and st.button("Load Selected Dataset"):
                    with st.spinner(f"Loading dataset {selected_dataset}..."):
                        df = load_dataframe_from_permanent_storage(selected_dataset)
                        if df is not None:
                            st.session_state.current_df = df
                            st.session_state.dataset_name = selected_dataset
                            debug_log(f"Loaded dataset: {selected_dataset} with {len(df)} records")
                            st.success(f"Loaded dataset: {selected_dataset}")
                            
                            # Display preview of the data
                            st.write("Preview of loaded data:")
                            st.dataframe(df.head())
    
    # Semantic Analysis page 
    elif page == "Semantic Analysis":
        st.header("Semantic Analysis")
        debug_log("Entered Semantic Analysis page")
        
        if st.session_state.current_df is None:
            st.warning("Please import a dataset first in the Data Import page")
            return
            
        st.write(f"Current dataset: {st.session_state.dataset_name}")
        
        # Get the dataframe from session state
        df = st.session_state.current_df
        
        # Dataset summary
        with st.expander("Dataset Summary"):
            st.write(f"Total phrases: {len(df)}")
            if 'freq' in df.columns:
                st.write(f"Total usage frequency: {df['freq'].sum()}")
            if 'success' in df.columns:
                st.write(f"Total successes: {df['success'].sum()}")
            if 'success_rate' in df.columns:
                st.write(f"Average success rate: {df['success_rate'].mean():.4f}")
                
            # Show data distribution charts
            if 'freq' in df.columns:
                fig_freq = px.histogram(df, x='freq', title='Frequency Distribution', nbins=20)
                st.plotly_chart(fig_freq)
            
            if 'success_rate' in df.columns:
                fig_success = px.histogram(df, x='success_rate', title='Success Rate Distribution', nbins=20)
                st.plotly_chart(fig_success)
            
        # Cluster settings
        st.subheader("Clustering Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            max_phrases_per_group = st.slider(
                "Maximum phrases per group", 
                min_value=5, 
                max_value=50, 
                value=50
            )
        
        with col2:
            # Calculate min and max clusters based on data size
            min_clusters = max(2, len(df) // max_phrases_per_group)
            max_clusters = min(20, len(df) // 2)
            
            # Allow user to specify clusters or choose automatic
            use_auto_clusters = st.checkbox("Auto-determine optimal number of clusters", value=True)
            
            if use_auto_clusters:
                n_clusters = None
                st.write(f"Will determine optimal clusters (minimum: {min_clusters})")
            else:
                n_clusters = st.slider(
                    "Number of semantic groups", 
                    min_value=min_clusters, 
                    max_value=max_clusters,
                    value=min_clusters
                )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            model_name = st.selectbox(
                "Sentence Transformer Model", 
                ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"],
                index=0
            )
            st.info(f"Using model: {model_name}. Larger models may be more accurate but slower.")
        
        # Run clustering 
        if st.button("Run Semantic Analysis"):
            debug_log(f"Starting semantic analysis with {n_clusters if n_clusters else 'auto'} clusters using model {model_name}")
            
            # Record start time for benchmarking
            start_time = time.time()
            
            # Create a progress container
            progress_container = st.empty()
            progress_container.info("Starting semantic analysis...")
            
            with st.spinner("Analyzing phrases semantically..."):
                try:
                    clustered_df = recalculate_clusters(
                        df,
                        phrase_column='phrase',
                        n_clusters=n_clusters,
                        max_phrases_per_group=max_phrases_per_group,
                        model_name=model_name
                    )
                    st.session_state.clustered_df = clustered_df
                    
                    # Reset manual groups when running new analysis
                    st.session_state.manual_groups = {}
                    
                    # Count actual clusters
                    actual_clusters = clustered_df['cluster'].nunique()
                    elapsed = time.time() - start_time
                    
                    debug_log(f"Analysis complete in {elapsed:.2f} seconds with {actual_clusters} clusters")
                    progress_container.success(f"Analysis complete! Phrases have been grouped into {actual_clusters} semantic clusters in {elapsed:.2f} seconds.")
                except Exception as e:
                    debug_log(f"Error during analysis: {str(e)}")
                    progress_container.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
                
        # Display clusters if available
        if st.session_state.clustered_df is not None:
            st.subheader("Semantic Groups")
            
            clustered_df = st.session_state.clustered_df
            
            # Get cluster IDs sorted by size
            cluster_counts = clustered_df['cluster'].value_counts().sort_values(ascending=False)
            cluster_ids = cluster_counts.index.tolist()
            
            # Summary of clusters
            st.write(f"Found {len(cluster_ids)} clusters with average of {len(clustered_df) / len(cluster_ids):.1f} phrases per cluster")
            
            # Visual representation of cluster sizes
            fig = px.bar(
                x=[f"Group {i+1}" for i in range(len(cluster_ids))], 
                y=[cluster_counts[cluster_id] for cluster_id in cluster_ids],
                title="Phrases per Group",
                labels={"x": "Group", "y": "Number of Phrases"}
            )
            st.plotly_chart(fig)
            
            # Display each cluster
            for i, cluster_id in enumerate(cluster_ids):
                cluster_df = clustered_df[clustered_df['cluster'] == cluster_id]
                cluster_size = len(cluster_df)
                
                # Find best phrase in cluster
                best = find_best_phrase(
                    cluster_df, 
                    phrase_col='phrase',
                    freq_col='freq',
                    success_col='success',
                    success_rate_col='success_rate'
                )
                
                # Create expander for each cluster
                with st.expander(f"Group {i+1}: {cluster_size} phrases (Best: {best['phrase']})"):
                    # Show best phrase stats
                    st.markdown(f"**Best Phrase:** {best['phrase']}")
                    st.markdown(f"**Score:** {best['score']:.4f} | **Frequency:** {best['frequency']} | **Success Rate:** {best['success_rate']:.4f}")
                    
                    # Display phrases in this cluster
                    st.dataframe(
                        cluster_df[['phrase', 'freq', 'success', 'success_rate']]
                        .sort_values('success_rate', ascending=False)
                    )
                    
                    # Visualization of success rates
                    fig = px.histogram(
                        cluster_df, 
                        x='success_rate',
                        nbins=10,
                        title=f'Success Rate Distribution - Group {i+1}'
                    )
                    st.plotly_chart(fig)
    
    # Group Management page 
    elif page == "Group Management":
        st.header("Group Management")
        debug_log("Entered Group Management page")
        
        if st.session_state.current_df is None or st.session_state.clustered_df is None:
            st.warning("Please complete the Semantic Analysis first")
            return
            
        st.write(f"Current dataset: {st.session_state.dataset_name}")
        
        # Get the clustered dataframe from session state
        df = st.session_state.clustered_df.copy()
        
        # First time initialization of custom groups if not done yet
        if not st.session_state.manual_groups and 'cluster' in df.columns:
            debug_log("Initializing manual groups from automatic clusters")
            # Initialize manual groups based on automatic clusters
            for cluster_id in df['cluster'].unique():
                group_name = f"Group {cluster_id + 1}"
                phrases = df[df['cluster'] == cluster_id]['phrase'].tolist()
                st.session_state.manual_groups[group_name] = phrases
        
        # Create tabs for different operations
        tab1, tab2, tab3 = st.tabs(["Manage Groups", "Create New Group", "Run Reanalysis"])
        
        with tab1:
            st.subheader("Manage Existing Groups")
            
            if not st.session_state.manual_groups:
                st.info("No groups available. Please run the Semantic Analysis first.")
            else:
                # Display summary
                st.write(f"You have {len(st.session_state.manual_groups)} groups")
                
                # Display groups
                for group_name, phrases in st.session_state.manual_groups.items():
                    with st.expander(f"{group_name} - {len(phrases)} phrases"):
                        # Get statistics for this group
                        group_df = df[df['phrase'].isin(phrases)]
                        
                        if not group_df.empty:
                            # Find best phrase
                            best = find_best_phrase(
                                group_df, 
                                phrase_col='phrase',
                                freq_col='freq',
                                success_col='success',
                                success_rate_col='success_rate'
                            )
                            
                            # Show best phrase
                            st.markdown(f"**Best Phrase:** {best['phrase']}")
                            st.markdown(f"**Score:** {best['score']:.4f}")
                            
                            # Show metrics visualization if debug mode enabled
                            if debug_mode:
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig = px.histogram(group_df, x='freq', title='Frequency Distribution')
                                    st.plotly_chart(fig)
                                with col2:
                                    fig = px.histogram(group_df, x='success_rate', title='Success Rate Distribution')
                                    st.plotly_chart(fig)
                        
                        # Show all phrases with option to move to another group
                        for phrase in phrases:
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(phrase)
                            
                            with col2:
                                # Create dropdown with other groups
                                other_groups = [g for g in st.session_state.manual_groups.keys() 
                                             if g != group_name]
                                
                                # Only show dropdown if there are other groups
                                if other_groups:
                                    target_group = st.selectbox(
                                        "Move to:", 
                                        options=other_groups,
                                        key=f"move_{group_name}_{phrase}"
                                    )
                                    
                                    if st.button("Move", key=f"btn_{group_name}_{phrase}"):
                                        debug_log(f"Moving '{phrase}' from '{group_name}' to '{target_group}'")
                                        # Remove from current group
                                        st.session_state.manual_groups[group_name].remove(phrase)
                                        
                                        # Add to target group
                                        st.session_state.manual_groups[target_group].append(phrase)
                                        
                                        st.success(f"Moved '{phrase}' to {target_group}")
                                        st.session_state.last_action_time = time.time()
                                        st.experimental_rerun()
        
        with tab2:
            st.subheader("Create New Group")
            
            # Create new group
            new_group_name = st.text_input("Enter new group name:", 
                                         value=f"Custom Group {st.session_state.custom_group_counter + 1}")
            
            if st.button("Create Group") and new_group_name:
                if new_group_name in st.session_state.manual_groups:
                    st.warning(f"Group '{new_group_name}' already exists!")
                else:
                    debug_log(f"Creating new group: {new_group_name}")
                    st.session_state.manual_groups[new_group_name] = []
                    st.session_state.custom_group_counter += 1
                    st.success(f"Created new group: {new_group_name}")
                    st.session_state.last_action_time = time.time()
                    st.experimental_rerun()
            
            # Optional: Search for phrases to add to the new group
            st.write("Search for phrases to add to groups:")
            search_term = st.text_input("Search phrase:")
            
            if search_term:
                # Find matching phrases
                matches = df[df['phrase'].str.contains(search_term, case=False)]
                
                if matches.empty:
                    st.info(f"No phrases found containing '{search_term}'")
                else:
                    st.write(f"Found {len(matches)} matching phrases:")
                    for _, row in matches.iterrows():
                        phrase = row['phrase']
                        current_group = None
                        
                        # Find which group the phrase is in
                        for group_name, phrases in st.session_state.manual_groups.items():
                            if phrase in phrases:
                                current_group = group_name
                                break
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.write(phrase)
                        with col2:
                            if current_group:
                                st.write(f"Currently in: {current_group}")
                        with col3:
                            target_groups = list(st.session_state.manual_groups.keys())
                            if current_group:
                                target_groups = [g for g in target_groups if g != current_group]
                                
                            if target_groups:
                                target_group = st.selectbox(
                                    "Move to:", 
                                    options=target_groups,
                                    key=f"search_move_{phrase}"
                                )
                                
                                if st.button("Move", key=f"search_btn_{phrase}"):
                                    debug_log(f"Moving '{phrase}' from '{current_group}' to '{target_group}'")
                                    # Remove from current group if needed
                                    if current_group:
                                        st.session_state.manual_groups[current_group].remove(phrase)
                                    
                                    # Add to target group
                                    st.session_state.manual_groups[target_group].append(phrase)
                                    
                                    st.success(f"Moved '{phrase}' to {target_group}")
                                    st.session_state.last_action_time = time.time()
                                    st.experimental_rerun()
        
        with tab3:
            st.subheader("Run Reanalysis")
            st.write("Rerun the semantic analysis incorporating your manual groupings")
            
            # Options for reanalysis
            reanalysis_method = st.radio(
                "Reanalysis Method:",
                ["Use Current Manual Groups", "Refine with Semantic Analysis"]
            )
            
            if reanalysis_method == "Use Current Manual Groups":
                st.write("This will update the clusters based solely on your manual groupings.")
            else:
                st.write("This will run semantic analysis again, but use your manual groupings as a starting point.")
                
                # Additional parameters for semantic refinement
                max_phrases = st.slider("Maximum phrases per group", min_value=5, max_value=50, value=50)
                
                # Select model for reanalysis
                model_name = st.selectbox(
                    "Sentence Transformer Model", 
                    ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"],
                    index=0
                )
            
            if st.button("Reanalyze"):
                debug_log(f"Starting reanalysis with method: {reanalysis_method}")
                start_time = time.time()
                
                if reanalysis_method == "Use Current Manual Groups":
                    # Simple update based on manual groups
                    with st.spinner("Updating groups..."):
                        # Update the clustered dataframe with manual group assignments
                        updated_df = update_clusters_from_manual_groups(
                            st.session_state.current_df, 
                            st.session_state.manual_groups
                        )
                        
                        # Update the clustered dataframe in session state
                        st.session_state.clustered_df = updated_df
                        
                        elapsed = time.time() - start_time
                        debug_log(f"Clusters updated in {elapsed:.2f} seconds")
                        st.success(f"Clusters updated based on manual groupings in {elapsed:.2f} seconds!")
                        
                else:
                    # Run semantic analysis with manual groups as seed
                    with st.spinner("Running semantic analysis with manual groupings as seed..."):
                        try:
                            # First update clusters based on manual groups
                            updated_df = update_clusters_from_manual_groups(
                                st.session_state.current_df, 
                                st.session_state.manual_groups
                            )
                            
                            # Then refine with semantic analysis
                            n_clusters = len(st.session_state.manual_groups)
                            refined_df = recalculate_clusters(
                                updated_df,
                                phrase_column='phrase',
                                n_clusters=n_clusters,
                                max_phrases_per_group=max_phrases,
                                model_name=model_name
                            )
                            
                            # Update session state
                            st.session_state.clustered_df = refined_df
                            
                            # Clear manual groups to rebuild them
                            st.session_state.manual_groups = {}
                            
                            elapsed = time.time() - start_time
                            debug_log(f"Semantic analysis complete in {elapsed:.2f} seconds")
                            st.success(f"Semantic analysis complete with {n_clusters} groups as a starting point in {elapsed:.2f} seconds!")
                        except Exception as e:
                            debug_log(f"Error during reanalysis: {str(e)}")
                            st.error(f"Error during reanalysis: {str(e)}")
                            st.exception(e)
                
                # Update last action time
                st.session_state.last_action_time = time.time()
                
                # Suggest going to Semantic Analysis page to see results
                st.info("Visit the Semantic Analysis page to see the updated groupings.")

if __name__ == "__main__":
    main() 