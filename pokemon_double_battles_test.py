import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# Load data with caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Machine learning similarity functions
def calculate_pokemon_similarity(df, selected_pokemon):
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    numeric_df = df[stats].fillna(0)
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(numeric_df)
    
    pokemon_index = df[df['Pokemon'] == selected_pokemon].index[0]
    similarities = cosine_similarity([scaled_stats[pokemon_index]], scaled_stats)[0]
    
    df['Similarity'] = similarities
    return df.sort_values('Similarity', ascending=False)

def calculate_team_similarity(df, selected_team):
    team_stats = df.groupby('Team').agg({
        'HP': 'mean',
        'Attack': 'mean',
        'Defense': 'mean',
        'Sp. Atk': 'mean',
        'Sp. Def': 'mean',
        'Speed': 'mean'
    }).reset_index()
    
    numeric_df = team_stats.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(numeric_df)
    
    team_index = team_stats[team_stats['Team'] == selected_team].index[0]
    similarities = cosine_similarity([scaled_stats[team_index]], scaled_stats)[0]
    
    team_stats['Similarity'] = similarities
    return team_stats.sort_values('Similarity', ascending=False)

# Radar chart function
def create_radar_chart(df, team_name):
    team_df = df[df['Team'] == team_name]
    avg_stats = team_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=avg_stats.values,
        theta=avg_stats.index,
        fill='toself',
        name=team_name
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(avg_stats)*1.2]
            )),
        showlegend=True,
        title=f'Team {team_name} Average Stats Radar Chart'
    )
    
    return fig

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer")
    
    st.title("⚔️ Pokémon Competitive Team Analyzer")
    st.write("""
    Analyze competitive Pokémon teams, compare individual Pokémon and team stats, 
    and find similar Pokémon/teams using machine learning.
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Team Overview", 
        "🔍 Pokémon Analysis", 
        "📊 Team Comparison", 
        "🤖 ML Recommendations"
    ])
    
    with tab1:
        st.header("Team Overview")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()))
        
        # Team radar chart
        st.plotly_chart(create_radar_chart(df, selected_team), use_container_width=True)
        
        # Team composition
        st.subheader("Team Composition")
        team_df = df[df['Team'] == selected_team]
        role_dist = team_df['PrimaryRole'].value_counts().reset_index()
        fig = px.pie(role_dist, names='PrimaryRole', values='count', title='Role Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Team members table
        st.dataframe(team_df[['Pokemon', 'PrimaryRole', 'Item', 'Ability']], hide_index=True)
    
    with tab2:
        st.header("Pokémon Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
        
        with col2:
            comparison_type = st.radio("Comparison Type", ["Same Role", "All Pokémon"])
        
        # Pokémon details
        pokemon_data = df[df['Pokemon'] == selected_pokemon].iloc[0]
        st.subheader(f"🧬 {selected_pokemon} Details")
        
        # Stats comparison
        st.subheader("📈 Stats Comparison")
        similar_pokemon = calculate_pokemon_similarity(
            df[df['PrimaryRole'] == pokemon_data['PrimaryRole']] if comparison_type == "Same Role" else df,
            selected_pokemon
        ).head(10)
        
        fig = px.bar(
            similar_pokemon,
            x='Pokemon',
            y=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
            barmode='group',
            title=f"Stats Comparison (Top 10 Similar Pokémon)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Team Comparison")
        selected_team = st.selectbox("Select Team to Compare", sorted(df['Team'].unique()), key='team_compare')
        
        # Team similarity
        similar_teams = calculate_team_similarity(df, selected_team).head(10)
        st.subheader("Most Similar Teams (ML-based)")
        st.dataframe(similar_teams, hide_index=True)
        
        # Radar chart comparison
        st.subheader("Team Stats Comparison")
        teams_to_compare = st.multiselect(
            "Select teams to compare",
            options=df['Team'].unique(),
            default=[selected_team, similar_teams.iloc[1]['Team']]
        )
        
        if len(teams_to_compare) >= 1:
            fig = go.Figure()
            for team in teams_to_compare:
                team_stats = df[df['Team'] == team][['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
                fig.add_trace(go.Scatterpolar(
                    r=team_stats.values,
                    theta=team_stats.index,
                    fill='toself',
                    name=team
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title="Team Stats Radar Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Machine Learning Recommendations")
        st.write("""
        ### Pokémon Replacement Suggestions
        Find alternative Pokémon that could fill similar roles in your team
        based on statistical similarity.
        """)
        
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='ml_team')
        selected_role = st.selectbox("Select Role to Replace", sorted(df['PrimaryRole'].unique()))
        
        # Filter by role first
        role_df = df[df['PrimaryRole'] == selected_role]
        
        # Get team Pokémon with this role
        team_pokemon = role_df[role_df['Team'] == selected_team]
        
        if not team_pokemon.empty:
            target_pokemon = team_pokemon.iloc[0]['Pokemon']
            similar_options = calculate_pokemon_similarity(role_df, target_pokemon)
            
            if not similar_options.empty:
                st.subheader(f"Top 5 Alternatives for {target_pokemon} ({selected_role})")
                
                # Exclude the target Pokémon itself and show next 5
                alternatives = similar_options[similar_options['Pokemon'] != target_pokemon].head(5)
                
                if not alternatives.empty:
                    st.dataframe(
                        alternatives[['Pokemon', 'Item', 'Ability', 'Similarity']],
                        hide_index=True
                    )
                    
                    # Visual comparison
                    st.subheader("Statistical Comparison")
                    comparison_df = pd.concat([team_pokemon.head(1), alternatives.head(5)])
                    fig = px.radar(
                        comparison_df,
                        r=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
                        theta=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
                        color='Pokemon',
                        title=f"Stats Comparison: {target_pokemon} vs Alternatives"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No similar Pokémon found for {target_pokemon} in this role")
            else:
                st.warning("Could not calculate similarity for this Pokémon")
        else:
            st.warning(f"No Pokémon in {selected_team} with {selected_role} role")

if __name__ == "__main__":
    main()
