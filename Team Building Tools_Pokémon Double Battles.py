import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from collections import defaultdict
from itertools import combinations

def load_data(uploaded_file):
    """Load and preprocess the uploaded CSV file"""
    df = pd.read_csv(uploaded_file)
    
    # Add default columns if not present
    default_columns = {
        'PrimaryRole': '',
        'SecondaryRole': '',
        'Win Condition': '',
        'Early Game': '',
        'Mid Game': '',
        'Late Game': '',
        'Counters': ''
    }
    
    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    return df

def calculate_synergy_scores(team_df):
    """Calculate synergy scores between team members"""
    # Select only numeric stats for similarity calculation
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    stats = team_df[numeric_cols].values
    
    # Normalize the stats
    scaler = StandardScaler()
    normalized_stats = scaler.fit_transform(stats)
    
    # Calculate cosine similarity between all pairs
    synergy_matrix = cosine_similarity(normalized_stats)
    
    return synergy_matrix

def visualize_type_coverage(team_df):
    """Create type coverage visualization"""
    # This would be expanded with actual type data
    type_counts = team_df['Type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    
    fig = px.bar(type_counts, x='Type', y='Count', title="Team Type Distribution")
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer Pro")
    
    st.title("⚔️ Pokémon Competitive Team Analyzer Pro")
    st.write("""
    Advanced toolkit for competitive Pokémon team analysis with enhanced team building tools and meta analysis.
    """)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Pokémon Data CSV", type=["csv"])
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()
    
    df = load_data(uploaded_file)
    
    # Ensure required columns exist
    required_cols = ['Pokemon', 'Team', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        st.stop()
    
    # Main tabs
    tab_names = [
        "🏆 Team Overview", 
        "🔍 Pokémon Analysis", 
        "📊 Team Comparison", 
        "🤖 ML Recommendations",
        "🛡️ Type Coverage",
        "🔄 Team Synergy",
        "⚔️ Team Matchup",
        "🧩 Enhanced Team Building",
        "📈 Meta Analysis"
    ]
    
    tabs = st.tabs(tab_names)
    
    with tabs[0]:  # Team Overview
        st.header("🏆 Team Overview")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()))
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            st.dataframe(team_df)
            
            # Basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Team Size", len(team_df))
            with col2:
                avg_speed = team_df['Speed'].mean()
                st.metric("Average Speed", f"{avg_speed:.1f}")
            with col3:
                total_attack = team_df['Attack'].sum() + team_df['Sp. Atk'].sum()
                st.metric("Total Offensive Power", f"{total_attack:.1f}")
    
    with tabs[1]:  # Pokémon Analysis
        st.header("🔍 Pokémon Analysis")
        selected_pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
        pokemon_data = df[df['Pokemon'] == selected_pokemon].iloc[0]
        
        st.subheader("Base Stats")
        stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        values = [pokemon_data[stat] for stat in stats]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=stats,
            fill='toself'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=False,
            title=f"{selected_pokemon} Base Stats"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:  # Team Synergy
        st.header("🔄 Team Synergy Analysis")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='synergy_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            synergy_matrix = calculate_synergy_scores(team_df)
            
            fig = px.imshow(
                synergy_matrix,
                labels=dict(x="Pokémon", y="Pokémon", color="Synergy"),
                x=team_df['Pokemon'].tolist(),
                y=team_df['Pokemon'].tolist(),
                title="Team Synergy Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[7]:  # Enhanced Team Building
        st.header("🧩 Enhanced Team Building Tools")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='enhanced_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            st.subheader("🎭 Role Synergy Compatibility Matrix")
            roles = team_df['PrimaryRole'].unique()
            compatibility = pd.DataFrame(index=roles, columns=roles)
            
            ROLE_SYNERGY = {
                'Sweeper': {'Support': 2, 'Wallbreaker': 1, 'Disruptor': 1},
                'Wallbreaker': {'Support': 1, 'Tank': 1, 'Sweeper': 1},
                'Support': {'Sweeper': 2, 'Tank': 1, 'Disruptor': 1},
                'Tank': {'Support': 1, 'Wallbreaker': 1},
                'Disruptor': {'Sweeper': 1, 'Wallbreaker': 1}
            }
            
            for r1 in roles:
                for r2 in roles:
                    if r1 == r2:
                        compatibility.loc[r1, r2] = 0
                    else:
                        score = ROLE_SYNERGY.get(r1, {}).get(r2, 0) + ROLE_SYNERGY.get(r2, {}).get(r1, 0)
                        compatibility.loc[r1, r2] = score
            
            fig = px.imshow(compatibility, 
                           labels=dict(x="Role", y="Role", color="Synergy"),
                           x=compatibility.columns,
                           y=compatibility.index,
                           title="Role Synergy Compatibility Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[8]:  # Meta Analysis
        st.header("📈 Meta Analysis Tools")
        st.subheader("📊 Usage Statistics")
        
        usage_stats = df['Pokemon'].value_counts().reset_index()
        usage_stats.columns = ['Pokemon', 'Usage Count']
        
        fig = px.bar(usage_stats.head(20), 
                     x='Pokemon', y='Usage Count',
                     title="Top 20 Most Used Pokémon")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
