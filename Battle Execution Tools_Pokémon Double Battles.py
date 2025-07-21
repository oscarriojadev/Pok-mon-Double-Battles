import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from collections import defaultdict
import random
import time

# Complete type effectiveness chart
TYPE_CHART = {
    'Normal': {'weak': ['Fighting'], 'resist': [], 'immune': ['Ghost'], 'strong_against': []},
    'Fire': {'weak': ['Water', 'Ground', 'Rock'], 'resist': ['Fire', 'Grass', 'Ice', 'Bug', 'Steel', 'Fairy'], 'immune': [], 'strong_against': ['Grass', 'Ice', 'Bug', 'Steel']},
    'Water': {'weak': ['Electric', 'Grass'], 'resist': ['Fire', 'Water', 'Ice', 'Steel'], 'immune': [], 'strong_against': ['Fire', 'Ground', 'Rock']},
    'Electric': {'weak': ['Ground'], 'resist': ['Electric', 'Flying', 'Steel'], 'immune': [], 'strong_against': ['Water', 'Flying']},
    'Grass': {'weak': ['Fire', 'Ice', 'Poison', 'Flying', 'Bug'], 'resist': ['Water', 'Electric', 'Grass', 'Ground'], 'immune': [], 'strong_against': ['Water', 'Ground', 'Rock']},
    'Ice': {'weak': ['Fire', 'Fighting', 'Rock', 'Steel'], 'resist': ['Ice'], 'immune': [], 'strong_against': ['Grass', 'Ground', 'Flying', 'Dragon']},
    'Fighting': {'weak': ['Flying', 'Psychic', 'Fairy'], 'resist': ['Bug', 'Rock', 'Dark'], 'immune': [], 'strong_against': ['Normal', 'Ice', 'Rock', 'Dark', 'Steel']},
    'Poison': {'weak': ['Ground', 'Psychic'], 'resist': ['Grass', 'Fighting', 'Poison', 'Bug', 'Fairy'], 'immune': [], 'strong_against': ['Grass', 'Fairy']},
    'Ground': {'weak': ['Water', 'Grass', 'Ice'], 'resist': ['Poison', 'Rock'], 'immune': ['Electric'], 'strong_against': ['Fire', 'Electric', 'Poison', 'Rock', 'Steel']},
    'Flying': {'weak': ['Electric', 'Ice', 'Rock'], 'resist': ['Grass', 'Fighting', 'Bug'], 'immune': ['Ground'], 'strong_against': ['Grass', 'Fighting', 'Bug']},
    'Psychic': {'weak': ['Bug', 'Ghost', 'Dark'], 'resist': ['Fighting', 'Psychic'], 'immune': [], 'strong_against': ['Fighting', 'Poison']},
    'Bug': {'weak': ['Fire', 'Flying', 'Rock'], 'resist': ['Grass', 'Fighting', 'Ground'], 'immune': [], 'strong_against': ['Grass', 'Psychic', 'Dark']},
    'Rock': {'weak': ['Water', 'Grass', 'Fighting', 'Ground', 'Steel'], 'resist': ['Normal', 'Fire', 'Poison', 'Flying'], 'immune': [], 'strong_against': ['Fire', 'Ice', 'Flying', 'Bug']},
    'Ghost': {'weak': ['Ghost', 'Dark'], 'resist': ['Poison', 'Bug'], 'immune': ['Normal', 'Fighting'], 'strong_against': ['Psychic', 'Ghost']},
    'Dragon': {'weak': ['Ice', 'Dragon', 'Fairy'], 'resist': ['Fire', 'Water', 'Electric', 'Grass'], 'immune': [], 'strong_against': ['Dragon']},
    'Dark': {'weak': ['Fighting', 'Bug', 'Fairy'], 'resist': ['Ghost', 'Dark'], 'immune': ['Psychic'], 'strong_against': ['Psychic', 'Ghost']},
    'Steel': {'weak': ['Fire', 'Fighting', 'Ground'], 'resist': ['Normal', 'Grass', 'Ice', 'Flying', 'Psychic', 'Bug', 'Rock', 'Dragon', 'Steel', 'Fairy'], 'immune': ['Poison'], 'strong_against': ['Ice', 'Rock', 'Fairy']},
    'Fairy': {'weak': ['Poison', 'Steel'], 'resist': ['Fighting', 'Bug', 'Dark'], 'immune': ['Dragon'], 'strong_against': ['Fighting', 'Dragon', 'Dark']}
}

ALL_TYPES = sorted(TYPE_CHART.keys())

# Load data with caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

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

def predict_lead(team1, team2, df):
    """Predict likely leads based on team composition"""
    # Simple prediction based on speed and role
    team1_members = df[df['Team'] == team1]
    team2_members = df[df['Team'] == team2]
    
    # Find fastest Pokémon with lead potential
    def get_lead_candidates(team_df):
        candidates = []
        for _, row in team_df.iterrows():
            score = 0
            # Speed is important
            score += row['Speed'] * 0.5
            # Lead moves add points
            moves = [row['Move 1'], row['Move 2'], row['Move 3'], row['Move 4']]
            if any(move in ['Fake Out', 'Tailwind', 'Stealth Rock', 'Spikes'] for move in moves if pd.notna(move)):
                score += 50
            if row['PrimaryRole'] == 'Lead':
                score += 30
            candidates.append((row['Pokemon'], score))
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:3]
    
    team1_leads = get_lead_candidates(team1_members)
    team2_leads = get_lead_candidates(team2_members)
    
    return {
        'team1_leads': team1_leads,
        'team2_leads': team2_leads
    }

def generate_lead_options(team_df):
    """Generate flexible lead options based on team composition"""
    options = []
    
    # Identify potential leads
    potential_leads = []
    for _, row in team_df.iterrows():
        lead_score = 0
        moves = [row['Move 1'], row['Move 2'], row['Move 3'], row['Move 4']]
        
        # Check for lead moves
        lead_moves = ['Fake Out', 'Tailwind', 'Stealth Rock', 'Spikes', 'Sticky Web']
        if any(move in lead_moves for move in moves if pd.notna(move)):
            lead_score += 30
        
        # Check for speed control
        speed_control = ['Electroweb', 'Icy Wind', 'Rock Tomb', 'Thunder Wave']
        if any(move in speed_control for move in moves if pd.notna(move)):
            lead_score += 20
            
        # Check for setup potential
        setup_moves = ['Swords Dance', 'Nasty Plot', 'Dragon Dance']
        if any(move in setup_moves for move in moves if pd.notna(move)):
            lead_score += 15
            
        if lead_score > 0:
            potential_leads.append((row['Pokemon'], lead_score))
    
    # Sort by lead score
    potential_leads.sort(key=lambda x: x[1], reverse=True)
    
    # Generate lead pairings
    if len(potential_leads) >= 2:
        for i in range(min(3, len(potential_leads))):
            for j in range(i+1, min(5, len(potential_leads))):
                options.append(f"{potential_leads[i][0]} + {potential_leads[j][0]}")
    
    return options[:5] if options else ["No clear lead options identified"]

def calculate_spread_damage(attacker, defender, move, spread_factor=0.75):
    """Calculate spread move damage in doubles"""
    # Simplified damage calculation
    base_power = 100  # Placeholder for actual move power
    attack = attacker['Attack'] if move['category'] == 'Physical' else attacker['Sp. Atk']
    defense = defender['Defense'] if move['category'] == 'Physical' else defender['Sp. Def']
    
    damage = ((2 * 100 / 5 + 2) * base_power * attack / defense / 50 + 2) * spread_factor
    return damage

def track_redirection(team_df):
    """Track redirection abilities in the team"""
    redirection_pokemon = []
    for _, row in team_df.iterrows():
        if row['Ability'] in ['Lightning Rod', 'Storm Drain', 'Follow Me', 'Rage Powder']:
            redirection_pokemon.append(row['Pokemon'])
    return redirection_pokemon

def optimize_switch_positions(team1, team2, df):
    """Suggest optimal switch positions based on type matchups"""
    team1_members = df[df['Team'] == team1]
    team2_members = df[df['Team'] == team2]
    
    suggestions = []
    
    # Check for safe switch opportunities
    for _, pokemon1 in team1_members.iterrows():
        for _, pokemon2 in team2_members.iterrows():
            # Check if pokemon1 resists pokemon2's common moves
            resists = False
            type1 = pokemon1['Type1']
            type2 = pokemon1['Type2'] if pd.notna(pokemon1['Type2']) else None
            
            # Check against common moves (simplified)
            common_moves = ['Water', 'Fire', 'Grass', 'Electric']  # Placeholder for actual move types
            for move_type in common_moves:
                if (move_type in TYPE_CHART[type1]['resist'] or 
                    (type2 and move_type in TYPE_CHART[type2]['resist'])):
                    resists = True
                    break
            
            if resists:
                suggestions.append(f"{pokemon1['Pokemon']} can safely switch into {pokemon2['Pokemon']}")
    
    return suggestions[:5] if suggestions else ["No clear switch advantages identified"]

def main():
    st.set_page_config(layout="wide", page_title="Pokémon Battle Tools")
    
    st.title("⚔️ Pokémon Battle Execution Tools")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Pokémon Data CSV", type=["csv"])
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()
    
    df = load_data(uploaded_file)
    
    # Ensure required columns exist
    required_cols = ['Pokemon', 'Team', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Type1', 'Type2']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        st.stop()
    
    # Add default columns if not present
    if 'Ability' not in df.columns:
        df['Ability'] = ''
    if 'Move 1' not in df.columns:
        df['Move 1'] = ''
    if 'Move 2' not in df.columns:
        df['Move 2'] = ''
    if 'Move 3' not in df.columns:
        df['Move 3'] = ''
    if 'Move 4' not in df.columns:
        df['Move 4'] = ''
    
    # Main tabs for new features
    tab1, tab2 = st.tabs([
        "🎯 Team Preview Analyzer", 
        "🔄 Double Battle Mechanics"
    ])
    
    with tab1:
        st.header("Team Preview Analysis Tools")
        
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select Your Team", sorted(df['Team'].unique()), key='your_team')
        with col2:
            team2 = st.selectbox("Select Opponent Team", sorted(df['Team'].unique()), key='opp_team')
        
        st.subheader("90-Second Team Preview Trainer")
        if st.button("Start Preview Simulation"):
            with st.spinner("Analyzing teams..."):
                time.sleep(2)
                
                # Team coverage analysis
                team1_types = []
                team1_df = df[df['Team'] == team1]
                for _, row in team1_df.iterrows():
                    team1_types.append(row['Type1'])
                    if pd.notna(row['Type2']):
                        team1_types.append(row['Type2'])
                
                team2_types = []
                team2_df = df[df['Team'] == team2]
                for _, row in team2_df.iterrows():
                    team2_types.append(row['Type1'])
                    if pd.notna(row['Type2']):
                        team2_types.append(row['Type2'])
                
                team1_coverage = calculate_team_coverage(team1_types)
                team2_coverage = calculate_team_coverage(team2_types)
                
                # Lead prediction
                leads = predict_lead(team1, team2, df)
                
                st.success("Analysis complete! Here's your game plan:")
                
                st.subheader("Lead Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Your likely leads:**")
                    for pokemon, score in leads['team1_leads']:
                        st.write(f"- {pokemon} (confidence: {score/100:.0%})")
                with col2:
                    st.write(f"**Opponent likely leads:**")
                    for pokemon, score in leads['team2_leads']:
                        st.write(f"- {pokemon} (confidence: {score/100:.0%})")
                
                st.subheader("Game Plan Adaptation Suggestions")
                
                # Check for opponent's uncovered weaknesses
                if team2_coverage['uncovered_weaknesses']:
                    st.write("**Exploit opponent's weaknesses:**")
                    for weakness in team2_coverage['uncovered_weaknesses']:
                        st.write(f"- Bring Pokémon with {weakness} attacks")
                
                # Check for opponent's strong coverage
                if team2_coverage['excellent_coverage']:
                    st.write("**Watch out for opponent's strong coverage:**")
                    for coverage in team2_coverage['excellent_coverage']:
                        st.write(f"- They have multiple Pokémon with {coverage} attacks")
                
                st.subheader("Flexible Lead Options")
                lead_options = generate_lead_options(team1_df)
                for option in lead_options:
                    st.write(f"- {option}")
                
                # Countdown timer
                st.write("---")
                st.write("⏱️ You have 90 seconds to make your lead decision!")
                with st.empty():
                    for sec in range(90, 0, -1):
                        st.write(f"Time remaining: {sec} seconds")
                        time.sleep(1)
                    st.write("⏰ Time's up! Make your decision.")
    
    with tab2:
        st.header("Double Battle Mechanics")
        
        col1, col2 = st.columns(2)
        with col1:
            your_team = st.selectbox("Select Your Team", sorted(df['Team'].unique()), key='your_double_team')
            your_pokemon = st.multiselect("Select Your Pokémon", 
                                         df[df['Team'] == your_team]['Pokemon'].unique(),
                                         max_selections=2)
        with col2:
            opp_team = st.selectbox("Select Opponent Team", sorted(df['Team'].unique()), key='opp_double_team')
            opp_pokemon = st.multiselect("Select Opponent Pokémon", 
                                       df[df['Team'] == opp_team]['Pokemon'].unique(),
                                       max_selections=2)
        
        if len(your_pokemon) == 2 and len(opp_pokemon) == 2:
            st.subheader("Spread Move Damage Calculator")
            
            # Get Pokémon data
            your_poke1 = df[df['Pokemon'] == your_pokemon[0]].iloc[0]
            your_poke2 = df[df['Pokemon'] == your_pokemon[1]].iloc[0]
            opp_poke1 = df[df['Pokemon'] == opp_pokemon[0]].iloc[0]
            opp_poke2 = df[df['Pokemon'] == opp_pokemon[1]].iloc[0]
            
            # Simplified move selection
            move_type = st.selectbox("Select Move Type", ALL_TYPES)
            move_category = st.selectbox("Select Move Category", ['Physical', 'Special'])
            
            if st.button("Calculate Damage"):
                move = {'type': move_type, 'category': move_category}
                
                # Calculate damage to each opponent
                damage1 = calculate_spread_damage(your_poke1, opp_poke1, move)
                damage2 = calculate_spread_damage(your_poke1, opp_poke2, move)
                
                st.write(f"**{your_poke1['Pokemon']}'s spread move against:**")
                st.write(f"- {opp_poke1['Pokemon']}: ~{damage1:.0f}% damage")
                st.write(f"- {opp_poke2['Pokemon']}: ~{damage2:.0f}% damage")
            
            st.subheader("Redirection Ability Tracker")
            your_redirection = track_redirection(df[df['Team'] == your_team])
            opp_redirection = track_redirection(df[df['Team'] == opp_team])
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Your redirection:**")
                if your_redirection:
                    for pokemon in your_redirection:
                        st.write(f"- {pokemon}")
                else:
                    st.write("No redirection abilities")
            with col2:
                st.write("**Opponent redirection:**")
                if opp_redirection:
                    for pokemon in opp_redirection:
                        st.write(f"- {pokemon}")
                else:
                    st.write("No redirection abilities")
            
            st.subheader("Partner Protection Timing Guide")
            st.write("""
            - **Turn 1:** Identify opponent's threat patterns
            - **Turn 2:** Use Protect on vulnerable Pokémon
            - **Turn 3:** Switch out threatened Pokémon
            - **Turn 4:** Set up with your other Pokémon
            """)
            
            st.subheader("Switch Positioning Optimizer")
            switch_suggestions = optimize_switch_positions(your_team, opp_team, df)
            st.write("**Recommended switches:**")
            for suggestion in switch_suggestions:
                st.write(f"- {suggestion}")

if __name__ == "__main__":
    main()
