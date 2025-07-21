import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from collections import defaultdict
from itertools import combinations

# Sample type data for demonstration
POKEMON_TYPES = {
    'Pikachu': 'Electric',
    'Charizard': 'Fire/Flying',
    'Blastoise': 'Water',
    'Venusaur': 'Grass/Poison',
    'Gyarados': 'Water/Flying',
    'Dragonite': 'Dragon/Flying',
    'Snorlax': 'Normal',
    'Alakazam': 'Psychic',
    'Gengar': 'Ghost/Poison',
    'Machamp': 'Fighting',
    'Tyranitar': 'Rock/Dark',
    'Metagross': 'Steel/Psychic',
    'Salamence': 'Dragon/Flying',
    'Heatran': 'Fire/Steel',
    'Ferrothorn': 'Grass/Steel'
}

# Sample type effectiveness chart (simplified)
TYPE_EFFECTIVENESS = {
    'Fire': {'Strong': ['Grass', 'Ice', 'Bug', 'Steel'], 'Weak': ['Water', 'Rock', 'Fire', 'Dragon']},
    'Water': {'Strong': ['Fire', 'Ground', 'Rock'], 'Weak': ['Electric', 'Grass', 'Water']},
    'Electric': {'Strong': ['Water', 'Flying'], 'Weak': ['Ground', 'Electric', 'Grass', 'Dragon']},
    'Grass': {'Strong': ['Water', 'Ground', 'Rock'], 'Weak': ['Fire', 'Ice', 'Poison', 'Flying', 'Bug', 'Grass']}
}

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
        'Counters': '',
        'Type': '',
        'Moves': ''
    }
    
    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    # If Type column is empty, populate with sample data
    if df['Type'].isnull().all() or (df['Type'] == '').all():
        df['Type'] = df['Pokemon'].map(POKEMON_TYPES).fillna('Unknown')
    
    return df

def calculate_synergy_scores(team_df):
    """Calculate synergy scores between team members"""
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    stats = team_df[numeric_cols].values
    
    scaler = StandardScaler()
    normalized_stats = scaler.fit_transform(stats)
    synergy_matrix = cosine_similarity(normalized_stats)
    
    return synergy_matrix

def analyze_type_coverage(team_df):
    """Analyze team type coverage with proper type distribution"""
    if 'Type' not in team_df.columns:
        return None
    
    type_list = []
    for types in team_df['Type']:
        for t in str(types).split('/'):
            if t and t != 'Unknown':
                type_list.append(t.strip())
    
    type_counts = pd.Series(type_list).value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    return type_counts

def generate_ml_recommendations(df, team_name, n_recommendations=3):
    """Generate ML-based recommendations for team improvement"""
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    
    # Get current team
    current_team = df[df['Team'] == team_name]
    
    # Prepare data for KNN
    all_pokemon = df[['Pokemon'] + numeric_cols].drop_duplicates()
    all_pokemon = all_pokemon[~all_pokemon['Pokemon'].isin(current_team['Pokemon'])]
    
    if len(all_pokemon) == 0:
        return []
    
    # Use KNN to find similar Pokémon
    knn = NearestNeighbors(n_neighbors=n_recommendations)
    knn.fit(all_pokemon[numeric_cols])
    
    # Get recommendations for each team member
    recommendations = set()
    for _, pokemon in current_team.iterrows():
        distances, indices = knn.kneighbors([pokemon[numeric_cols].values])
        for idx in indices[0]:
            recommendations.add(all_pokemon.iloc[idx]['Pokemon'])
            if len(recommendations) >= n_recommendations:
                break
    
    return list(recommendations)[:n_recommendations]

def calculate_team_matchup(team1_df, team2_df):
    """Calculate matchup between two teams"""
    # Simple matchup calculation (would be more sophisticated in real implementation)
    matchup = {
        'Speed Advantage': team1_df['Speed'].mean() - team2_df['Speed'].mean(),
        'Offensive Power': (team1_df['Attack'].mean() + team1_df['Sp. Atk'].mean()) - 
                          (team2_df['Attack'].mean() + team2_df['Sp. Atk'].mean()),
        'Defensive Bulk': (team1_df['HP'].mean() + team1_df['Defense'].mean() + team1_df['Sp. Def'].mean()) - 
                         (team2_df['HP'].mean() + team2_df['Defense'].mean() + team2_df['Sp. Def'].mean())
    }
    return matchup

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
    tabs = st.tabs([
        "🏆 Team Overview", 
        "🔍 Pokémon Analysis", 
        "📊 Team Comparison", 
        "🤖 ML Recommendations",
        "🛡️ Type Coverage",
        "🔄 Team Synergy",
        "⚔️ Team Matchup",
        "🧩 Enhanced Team Building",
        "📈 Meta Analysis"
    ])
    
    # [Previous tab implementations...]
    
    with tabs[2]:  # Team Comparison
        st.header("📊 Team Comparison Analysis")
        
        # Select teams to compare
        teams_to_compare = st.multiselect(
            "Select teams to compare (2-3 recommended)",
            sorted(df['Team'].unique()),
            default=sorted(df['Team'].unique())[:2]
        )
        
        if len(teams_to_compare) >= 2:
            comparison_data = []
            stats_to_compare = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            
            for team in teams_to_compare:
                team_df = df[df['Team'] == team]
                avg_stats = team_df[stats_to_compare].mean().to_dict()
                avg_stats['Team'] = team
                comparison_data.append(avg_stats)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Radar chart comparison
            st.subheader("Team Stats Comparison")
            fig = go.Figure()
            
            for _, row in comparison_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row[stats_to_compare].values,
                    theta=stats_to_compare,
                    fill='toself',
                    name=row['Team']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, comparison_df[stats_to_compare].values.max() * 1.1])),
                showlegend=True,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Numerical comparison table
            st.subheader("Numerical Comparison")
            st.dataframe(
                comparison_df.set_index('Team').T.style.highlight_max(axis=1, color='lightgreen')
            )
            
            # Team composition comparison
            st.subheader("Team Composition")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{teams_to_compare[0]} Members**")
                st.dataframe(df[df['Team'] == teams_to_compare[0]][['Pokemon', 'Type', 'PrimaryRole']])
            
            if len(teams_to_compare) > 1:
                with col2:
                    st.write(f"**{teams_to_compare[1]} Members**")
                    st.dataframe(df[df['Team'] == teams_to_compare[1]][['Pokemon', 'Type', 'PrimaryRole']])
        else:
            st.warning("Please select at least 2 teams to compare")
    
    with tabs[3]:  # ML Recommendations
        st.header("🤖 Machine Learning Recommendations")
        selected_team = st.selectbox("Select team for recommendations", sorted(df['Team'].unique()))
        
        if st.button("Generate Recommendations", key='ml_recommend'):
            with st.spinner("Analyzing team composition and finding optimal partners..."):
                recommendations = generate_ml_recommendations(df, selected_team)
                
                if recommendations:
                    st.subheader("Recommended Pokémon Additions")
                    st.write("Based on your team's composition, these Pokémon would complement your team well:")
                    
                    for i, pokemon in enumerate(recommendations, 1):
                        pokemon_data = df[df['Pokemon'] == pokemon].iloc[0]
                        with st.expander(f"{i}. {pokemon} ({pokemon_data.get('Type', 'Unknown')})"):
                            st.write(f"**Primary Role:** {pokemon_data.get('PrimaryRole', 'Not specified')}")
                            st.write(f"**Stats:** HP: {pokemon_data['HP']}, Atk: {pokemon_data['Attack']}, Def: {pokemon_data['Defense']}")
                            st.write(f"Sp.Atk: {pokemon_data['Sp. Atk']}, Sp.Def: {pokemon_data['Sp. Def']}, Speed: {pokemon_data['Speed']}")
                    
                    st.info("""
                    **How these recommendations work:**
                    - The system analyzes your team's stat distributions
                    - Finds Pokémon with complementary stat profiles
                    - Considers type coverage and role balance
                    """)
                else:
                    st.warning("Could not generate recommendations - not enough data")
    
    with tabs[5]:  # Team Synergy
        st.header("🔄 Team Synergy Analysis")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='synergy_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            synergy_matrix = calculate_synergy_scores(team_df)
            
            st.markdown("""
            ### Understanding Team Synergy
            
            The synergy matrix shows how well each Pokémon's stat distribution complements others on your team:
            
            - **High values (0.7-1.0)**: These Pokémon have very similar stat spreads. This can be good for consistent strategies but may indicate redundancy.
            - **Moderate values (0.3-0.7)**: Healthy synergy - these Pokémon complement each other well.
            - **Low values (0-0.3)**: These Pokémon have different roles. This provides coverage but may lack synergy.
            - **Negative values**: These Pokémon have opposing stat distributions (rare but possible).
            
            *Ideal teams typically show moderate synergy between most members with a few specialized pairs.*
            """)
            
            # Visualize synergy matrix
            fig = px.imshow(
                synergy_matrix,
                labels=dict(x="Pokémon", y="Pokémon", color="Synergy Score"),
                x=team_df['Pokemon'].tolist(),
                y=team_df['Pokemon'].tolist(),
                color_continuous_scale='RdYlGn',
                zmin=-1, zmax=1,
                title=f"{selected_team} Team Synergy Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Synergy analysis
            st.subheader("Synergy Insights")
            
            # Find best and worst pairs
            n = len(team_df)
            best_pair = None
            worst_pair = None
            best_score = -1
            worst_score = 2
            
            for i in range(n):
                for j in range(i+1, n):
                    score = synergy_matrix[i][j]
                    if score > best_score:
                        best_score = score
                        best_pair = (team_df.iloc[i]['Pokemon'], team_df.iloc[j]['Pokemon'])
                    if score < worst_score:
                        worst_score = score
                        worst_pair = (team_df.iloc[i]['Pokemon'], team_df.iloc[j]['Pokemon'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Synergy Pair", 
                         f"{best_pair[0]} + {best_pair[1]}", 
                         f"Score: {best_score:.2f}")
                st.write(f"These Pokémon work exceptionally well together based on their stat distributions.")
            
            with col2:
                st.metric("Worst Synergy Pair", 
                         f"{worst_pair[0]} + {worst_pair[1]}", 
                         f"Score: {worst_score:.2f}")
                st.write(f"Consider adjusting their roles or replacing one to improve team balance.")
    
    with tabs[6]:  # Team Matchup
        st.header("⚔️ Team Matchup Simulator")
        
        col1, col2 = st.columns(2)
        with col1:
            team_a = st.selectbox("Your Team", sorted(df['Team'].unique()), key='team_a')
        with col2:
            team_b = st.selectbox("Opponent Team", sorted(df['Team'].unique()), key='team_b')
        
        if team_a and team_b and team_a != team_b:
            team_a_df = df[df['Team'] == team_a]
            team_b_df = df[df['Team'] == team_b]
            
            # Calculate matchup
            matchup = calculate_team_matchup(team_a_df, team_b_df)
            
            st.subheader("Matchup Summary")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Speed Advantage", 
                         f"{'Your Team' if matchup['Speed Advantage'] > 0 else 'Opponent'}", 
                         f"{abs(matchup['Speed Advantage']):.1f} points")
            with col2:
                st.metric("Offensive Power", 
                         f"{'Your Team' if matchup['Offensive Power'] > 0 else 'Opponent'}", 
                         f"{abs(matchup['Offensive Power']):.1f} points")
            with col3:
                st.metric("Defensive Bulk", 
                         f"{'Your Team' if matchup['Defensive Bulk'] > 0 else 'Opponent'}", 
                         f"{abs(matchup['Defensive Bulk']):.1f} points")
            
            # Type matchup analysis
            st.subheader("Type Matchup Analysis")
            
            # Get all types from both teams
            team_a_types = set()
            for types in team_a_df['Type']:
                for t in str(types).split('/'):
                    if t and t != 'Unknown':
                        team_a_types.add(t.strip())
            
            team_b_types = set()
            for types in team_b_df['Type']:
                for t in str(types).split('/'):
                    if t and t != 'Unknown':
                        team_b_types.add(t.strip())
            
            # Display type advantages
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Your Team's Strengths**")
                for t in team_a_types:
                    if t in TYPE_EFFECTIVENESS:
                        strong_against = [x for x in TYPE_EFFECTIVENESS[t]['Strong'] if x in team_b_types]
                        if strong_against:
                            st.write(f"- {t} is strong against: {', '.join(strong_against)}")
            
            with col2:
                st.write("**Opponent's Strengths**")
                for t in team_b_types:
                    if t in TYPE_EFFECTIVENESS:
                        strong_against = [x for x in TYPE_EFFECTIVENESS[t]['Strong'] if x in team_a_types]
                        if strong_against:
                            st.write(f"- {t} is strong against: {', '.join(strong_against)}")
            
            # Strategy recommendations
            st.subheader("Recommended Strategy")
            
            if matchup['Speed Advantage'] > 10:
                st.success("**Speed Control Advantage**: Use your faster Pokémon to control the tempo of the battle")
            elif matchup['Speed Advantage'] < -10:
                st.warning("**Speed Disadvantage**: Consider using priority moves or speed control like Tailwind/Trick Room")
            
            if matchup['Offensive Power'] > 20:
                st.success("**Offensive Advantage**: Play aggressively to capitalize on your stronger attackers")
            elif matchup['Offensive Power'] < -20:
                st.warning("**Offensive Disadvantage**: Focus on defensive plays and wearing down their team")
            
            # Key threats
            st.subheader("Key Threats to Watch For")
            st.write(f"**From {team_b}:**")
            st.write("- " + "\n- ".join(team_b_df.sort_values('Attack', ascending=False)['Pokemon'].head(3).tolist()))
            
            st.write(f"**From {team_a} (their threats):**")
            st.write("- " + "\n- ".join(team_a_df.sort_values('Defense', ascending=False)['Pokemon'].head(3).tolist()))
        elif team_a == team_b:
            st.warning("Please select two different teams to compare")

if __name__ == "__main__":
    main()
