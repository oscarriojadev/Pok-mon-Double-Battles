import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from collections import defaultdict

# Complete type effectiveness chart
TYPE_CHART = {
    'Normal': {
        'weak': ['Fighting'],
        'resist': [],
        'immune': ['Ghost'],
        'strong_against': []
    },
    'Fire': {
        'weak': ['Water', 'Ground', 'Rock'],
        'resist': ['Fire', 'Grass', 'Ice', 'Bug', 'Steel', 'Fairy'],
        'immune': [],
        'strong_against': ['Grass', 'Ice', 'Bug', 'Steel']
    },
    'Water': {
        'weak': ['Electric', 'Grass'],
        'resist': ['Fire', 'Water', 'Ice', 'Steel'],
        'immune': [],
        'strong_against': ['Fire', 'Ground', 'Rock']
    },
    'Electric': {
        'weak': ['Ground'],
        'resist': ['Electric', 'Flying', 'Steel'],
        'immune': [],
        'strong_against': ['Water', 'Flying']
    },
    'Grass': {
        'weak': ['Fire', 'Ice', 'Poison', 'Flying', 'Bug'],
        'resist': ['Water', 'Electric', 'Grass', 'Ground'],
        'immune': [],
        'strong_against': ['Water', 'Ground', 'Rock']
    },
    'Ice': {
        'weak': ['Fire', 'Fighting', 'Rock', 'Steel'],
        'resist': ['Ice'],
        'immune': [],
        'strong_against': ['Grass', 'Ground', 'Flying', 'Dragon']
    },
    'Fighting': {
        'weak': ['Flying', 'Psychic', 'Fairy'],
        'resist': ['Bug', 'Rock', 'Dark'],
        'immune': [],
        'strong_against': ['Normal', 'Ice', 'Rock', 'Dark', 'Steel']
    },
    'Poison': {
        'weak': ['Ground', 'Psychic'],
        'resist': ['Grass', 'Fighting', 'Poison', 'Bug', 'Fairy'],
        'immune': [],
        'strong_against': ['Grass', 'Fairy']
    },
    'Ground': {
        'weak': ['Water', 'Grass', 'Ice'],
        'resist': ['Poison', 'Rock'],
        'immune': ['Electric'],
        'strong_against': ['Fire', 'Electric', 'Poison', 'Rock', 'Steel']
    },
    'Flying': {
        'weak': ['Electric', 'Ice', 'Rock'],
        'resist': ['Grass', 'Fighting', 'Bug'],
        'immune': ['Ground'],
        'strong_against': ['Grass', 'Fighting', 'Bug']
    },
    'Psychic': {
        'weak': ['Bug', 'Ghost', 'Dark'],
        'resist': ['Fighting', 'Psychic'],
        'immune': [],
        'strong_against': ['Fighting', 'Poison']
    },
    'Bug': {
        'weak': ['Fire', 'Flying', 'Rock'],
        'resist': ['Grass', 'Fighting', 'Ground'],
        'immune': [],
        'strong_against': ['Grass', 'Psychic', 'Dark']
    },
    'Rock': {
        'weak': ['Water', 'Grass', 'Fighting', 'Ground', 'Steel'],
        'resist': ['Normal', 'Fire', 'Poison', 'Flying'],
        'immune': [],
        'strong_against': ['Fire', 'Ice', 'Flying', 'Bug']
    },
    'Ghost': {
        'weak': ['Ghost', 'Dark'],
        'resist': ['Poison', 'Bug'],
        'immune': ['Normal', 'Fighting'],
        'strong_against': ['Psychic', 'Ghost']
    },
    'Dragon': {
        'weak': ['Ice', 'Dragon', 'Fairy'],
        'resist': ['Fire', 'Water', 'Electric', 'Grass'],
        'immune': [],
        'strong_against': ['Dragon']
    },
    'Dark': {
        'weak': ['Fighting', 'Bug', 'Fairy'],
        'resist': ['Ghost', 'Dark'],
        'immune': ['Psychic'],
        'strong_against': ['Psychic', 'Ghost']
    },
    'Steel': {
        'weak': ['Fire', 'Fighting', 'Ground'],
        'resist': ['Normal', 'Grass', 'Ice', 'Flying', 'Psychic', 'Bug', 'Rock', 'Dragon', 'Steel', 'Fairy'],
        'immune': ['Poison'],
        'strong_against': ['Ice', 'Rock', 'Fairy']
    },
    'Fairy': {
        'weak': ['Poison', 'Steel'],
        'resist': ['Fighting', 'Bug', 'Dark'],
        'immune': ['Dragon'],
        'strong_against': ['Fighting', 'Dragon', 'Dark']
    }
}

ALL_TYPES = sorted(TYPE_CHART.keys())

# Load data with caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Fixed machine learning similarity function
def calculate_pokemon_similarity(df, selected_pokemon):
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    
    # Create a temporary DataFrame with reset index
    temp_df = df.reset_index(drop=True)
    
    # Check if Pokémon exists in DataFrame
    if selected_pokemon not in temp_df['Pokemon'].values:
        return pd.DataFrame(columns=df.columns.tolist() + ['Similarity'])
    
    # Get numeric stats and scale
    numeric_df = temp_df[stats].fillna(0)
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(numeric_df)
    
    # Find index in the reset DataFrame
    try:
        pokemon_index = temp_df.index[temp_df['Pokemon'] == selected_pokemon][0]
        similarities = cosine_similarity([scaled_stats[pokemon_index]], scaled_stats)[0]
    except IndexError:
        return pd.DataFrame(columns=df.columns.tolist() + ['Similarity'])
    
    # Return results with original DataFrame structure
    result_df = df.copy()
    result_df['Similarity'] = similarities
    return result_df.sort_values('Similarity', ascending=False)

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
    
    try:
        team_index = team_stats[team_stats['Team'] == selected_team].index[0]
        similarities = cosine_similarity([scaled_stats[team_index]], scaled_stats)[0]
    except IndexError:
        return pd.DataFrame(columns=team_stats.columns.tolist() + ['Similarity'])
    
    team_stats['Similarity'] = similarities
    return team_stats.sort_values('Similarity', ascending=False)

# Radar chart function
def create_radar_chart(df, team_name):
    team_df = df[df['Team'] == team_name]
    if team_df.empty:
        return go.Figure()
    
    avg_stats = team_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_stats.values,
        theta=avg_stats.index,
        fill='toself',
        name=team_name
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(avg_stats)*1.2])),
        showlegend=True,
        title=f'Team {team_name} Average Stats Radar Chart'
    )
    
    return fig

def calculate_team_coverage(team_types):
    """Calculate all coverage aspects for the team"""
    resistances = set()
    immunities = set()
    offensive_coverage = defaultdict(int)
    
    # Collect all defensive properties
    for t in team_types:
        resistances.update(TYPE_CHART[t]['resist'])
        immunities.update(TYPE_CHART[t]['immune'])
        
        # Count offensive coverage
        for target in TYPE_CHART[t]['strong_against']:
            offensive_coverage[target] += 1
    
    # Find uncovered weaknesses
    all_attack_types = set()
    for t in ALL_TYPES:
        all_attack_types.update(TYPE_CHART[t]['weak'])
    
    uncovered_weaknesses = []
    for attack_type in sorted(all_attack_types):
        if attack_type not in resistances and attack_type not in immunities:
            uncovered_weaknesses.append(attack_type)
    
    # Get resisted types (not including immunities)
    resisted_types = [t for t in sorted(resistances) if t not in immunities]
    
    # Get super effective coverage (types hit by at least 2 team members)
    good_coverage = [t for t, count in offensive_coverage.items() if count >= 2]
    excellent_coverage = [t for t, count in offensive_coverage.items() if count >= 3]
    
    return {
        'uncovered_weaknesses': uncovered_weaknesses,
        'resisted_types': resisted_types,
        'immune_types': sorted(immunities),
        'offensive_coverage': dict(offensive_coverage),
        'good_coverage': sorted(good_coverage),
        'excellent_coverage': sorted(excellent_coverage)
    }

def display_type_info(attack_type):
    """Display detailed information about a specific type"""
    st.write(f"### {attack_type} Attacks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Super Effective Against:**")
        super_effective = [t for t in ALL_TYPES 
                         if attack_type in TYPE_CHART[t]['weak']]
        for t in super_effective:
            st.write(f"- {t}")
    
    with col2:
        st.write("**Resisted By:**")
        resisted_by = [t for t in ALL_TYPES 
                     if attack_type in TYPE_CHART[t]['resist']]
        if not resisted_by:
            st.write("(No types resist this)")
        else:
            for t in resisted_by:
                st.write(f"- {t}")
    
    st.write("**Ineffective Against:**")
    immune = [t for t in ALL_TYPES 
             if attack_type in TYPE_CHART[t]['immune']]
    if not immune:
        st.write("(No types are immune to this)")
    else:
        for t in immune:
            st.write(f"- {t}")

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
    
    # Add Type1 and Type2 columns if not present (for type coverage tab)
    if 'Type1' not in df.columns:
        df['Type1'] = 'Unknown'
    if 'Type2' not in df.columns:
        df['Type2'] = ''
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Team Overview", 
        "🔍 Pokémon Analysis", 
        "📊 Team Comparison", 
        "🤖 ML Recommendations",
        "🛡️ Type Coverage"
    ])
    
    with tab1:
        st.header("Team Overview")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()))
        
        # Team radar chart
        st.plotly_chart(create_radar_chart(df, selected_team), use_container_width=True)
        
        # Team composition
        st.subheader("Team Composition")
        team_df = df[df['Team'] == selected_team]
        if not team_df.empty:
            role_dist = team_df['PrimaryRole'].value_counts().reset_index()
            fig = px.pie(role_dist, names='PrimaryRole', values='count', title='Role Distribution')
            st.plotly_chart(fig, use_container_width=True)
            
            # Team members table
            st.dataframe(team_df[['Pokemon', 'PrimaryRole', 'Item', 'Ability']], hide_index=True)
        else:
            st.warning("No data available for selected team")

    with tab2:
        st.header("Pokémon Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            selected_pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
        
        with col2:
            comparison_type = st.radio("Comparison Type", ["Same Role", "All Pokémon"])
        
        # Pokémon details
        pokemon_data = df[df['Pokemon'] == selected_pokemon]
        if not pokemon_data.empty:
            pokemon_data = pokemon_data.iloc[0]
            st.subheader(f"🧬 {selected_pokemon} Details")
            
            # Stats comparison
            st.subheader("📈 Stats Comparison")
            comparison_df = df[df['PrimaryRole'] == pokemon_data['PrimaryRole']] if comparison_type == "Same Role" else df
            similar_pokemon = calculate_pokemon_similarity(comparison_df, selected_pokemon)
            
            if not similar_pokemon.empty:
                similar_pokemon = similar_pokemon.head(10)
                fig = px.bar(
                    similar_pokemon,
                    x='Pokemon',
                    y=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
                    barmode='group',
                    title=f"Stats Comparison (Top 10 Similar Pokémon)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No similar Pokémon found")
        else:
            st.warning("Pokémon data not available")

    with tab3:
        st.header("Team Comparison")
        selected_team = st.selectbox("Select Team to Compare", sorted(df['Team'].unique()), key='team_compare')
        
        # Team similarity
        similar_teams = calculate_team_similarity(df, selected_team)
        if not similar_teams.empty:
            st.subheader("Most Similar Teams (ML-based)")
            st.dataframe(similar_teams.head(10), hide_index=True)
            
            # Radar chart comparison
            st.subheader("Team Stats Comparison")
            teams_to_compare = st.multiselect(
                "Select teams to compare",
                options=df['Team'].unique(),
                default=[selected_team, similar_teams.iloc[1]['Team']] if len(similar_teams) > 1 else [selected_team]
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
        else:
            st.warning("No similar teams found")

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
                    
                    # Visual comparison - using go.Figure instead of px.radar
                    st.subheader("Statistical Comparison")
                    comparison_df = pd.concat([team_pokemon.head(1), alternatives.head(5)])
                    
                    fig = go.Figure()
                    for pokemon in comparison_df['Pokemon'].unique():
                        pokemon_stats = comparison_df[comparison_df['Pokemon'] == pokemon].iloc[0]
                        fig.add_trace(go.Scatterpolar(
                            r=[
                                pokemon_stats['HP'],
                                pokemon_stats['Attack'],
                                pokemon_stats['Defense'],
                                pokemon_stats['Sp. Atk'],
                                pokemon_stats['Sp. Def'],
                                pokemon_stats['Speed']
                            ],
                            theta=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],
                            fill='toself',
                            name=pokemon
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        title=f"Stats Comparison: {target_pokemon} vs Alternatives"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No similar Pokémon found for {target_pokemon} in this role")
            else:
                st.warning("Could not calculate similarity for this Pokémon")
        else:
            st.warning(f"No Pokémon in {selected_team} with {selected_role} role")

    with tab5:
        st.header("Team Type Coverage Analysis")
        st.write("Analyze your team's type weaknesses and resistances")
        
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='type_team')
        
        team_df = df[df['Team'] == selected_team]
        if not team_df.empty:
            # Get all types present on the team
            team_types = []
            for _, row in team_df.iterrows():
                team_types.append(row['Type1'])
                if pd.notna(row['Type2']) and row['Type2'] != '':
                    team_types.append(row['Type2'])
            
            coverage = calculate_team_coverage(team_types)
            
            # Display defensive coverage
            st.header("Defensive Coverage")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("❌ Uncovered Weaknesses")
                if coverage['uncovered_weaknesses']:
                    for t in coverage['uncovered_weaknesses']:
                        st.error(t)
                else:
                    st.success("All attack types are covered!")
            
            with col2:
                st.subheader("🛡️ Resisted Types")
                if coverage['resisted_types']:
                    for t in coverage['resisted_types']:
                        st.info(t)
                else:
                    st.warning("No resisted types")
            
            with col3:
                st.subheader("✅ Immune Types")
                if coverage['immune_types']:
                    for t in coverage['immune_types']:
                        st.success(t)
                else:
                    st.warning("No immunities")
            
            # Display offensive coverage
            st.header("Offensive Coverage")
            
            st.subheader("⚔️ Super Effective Against:")
            if coverage['offensive_coverage']:
                cols = st.columns(3)
                for i, (t, count) in enumerate(sorted(coverage['offensive_coverage'].items())):
                    with cols[i%3]:
                        st.write(f"{t}: {'⭐' * count}")
            else:
                st.warning("No notable offensive coverage")
            
            st.subheader("✨ Good Coverage (2+ members):")
            if coverage['good_coverage']:
                st.write(", ".join(coverage['good_coverage']))
            else:
                st.warning("No types with good coverage")
            
            st.subheader("💫 Excellent Coverage (3+ members):")
            if coverage['excellent_coverage']:
                st.write(", ".join(coverage['excellent_coverage']))
            else:
                st.warning("No types with excellent coverage")
            
            # Show details for uncovered weaknesses
            if coverage['uncovered_weaknesses']:
                st.header("Details of Uncovered Weaknesses")
                for attack_type in coverage['uncovered_weaknesses']:
                    display_type_info(attack_type)
                    st.write("---")
        else:
            st.warning("No data available for selected team")

if __name__ == "__main__":
    main()
