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

def analyze_team_synergy(df, team_name):
    """Analyze how items, abilities, and natures mitigate weaknesses and reinforce the team"""
    team_df = df[df['Team'] == team_name]
    if team_df.empty:
        return {}
    
    # Collect all team types
    team_types = []
    for _, row in team_df.iterrows():
        team_types.append(row['Type1'])
        if pd.notna(row['Type2']) and row['Type2'] != '':
            team_types.append(row['Type2'])
    
    # Calculate coverage
    coverage = calculate_team_coverage(team_types)
    
    # Analyze mitigation strategies
    mitigation = {
        'item_mitigations': defaultdict(list),
        'ability_mitigations': defaultdict(list),
        'nature_mitigations': defaultdict(list),
        'move_mitigations': defaultdict(list),
        'role_mitigations': defaultdict(list),
        'strategy_synergies': defaultdict(list)
    }
    
    # Check each Pokémon's mitigation strategies
    for _, pokemon in team_df.iterrows():
        # Item mitigations (all berry types)
        item = pokemon['Item']
        if 'Berry' in str(item):
            if 'Coba' in str(item):  # Reduces Flying damage
                mitigation['item_mitigations']['Flying'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Shuca' in str(item):  # Reduces Ground damage
                mitigation['item_mitigations']['Ground'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Yache' in str(item):  # Reduces Ice damage
                mitigation['item_mitigations']['Ice'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Occa' in str(item):  # Reduces Fire damage
                mitigation['item_mitigations']['Fire'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Passho' in str(item):  # Reduces Water damage
                mitigation['item_mitigations']['Water'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Rindo' in str(item):  # Reduces Grass damage
                mitigation['item_mitigations']['Grass'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Wacan' in str(item):  # Reduces Electric damage
                mitigation['item_mitigations']['Electric'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Tanga' in str(item):  # Reduces Bug damage
                mitigation['item_mitigations']['Bug'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Charti' in str(item):  # Reduces Rock damage
                mitigation['item_mitigations']['Rock'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Kasib' in str(item):  # Reduces Ghost damage
                mitigation['item_mitigations']['Ghost'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Haban' in str(item):  # Reduces Dragon damage
                mitigation['item_mitigations']['Dragon'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Colbur' in str(item):  # Reduces Dark damage
                mitigation['item_mitigations']['Dark'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Babiri' in str(item):  # Reduces Steel damage
                mitigation['item_mitigations']['Steel'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Chilan' in str(item):  # Reduces Normal damage
                mitigation['item_mitigations']['Normal'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Chople' in str(item):  # Reduces Fighting damage
                mitigation['item_mitigations']['Fighting'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Kebia' in str(item):  # Reduces Poison damage
                mitigation['item_mitigations']['Poison'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Payapa' in str(item):  # Reduces Psychic damage
                mitigation['item_mitigations']['Psychic'].append(f"{pokemon['Pokemon']} ({item})")
            elif 'Roseli' in str(item):  # Reduces Fairy damage
                mitigation['item_mitigations']['Fairy'].append(f"{pokemon['Pokemon']} ({item})")
        
        # Ability mitigations (all relevant abilities)
        ability = pokemon['Ability']
        if ability == 'Levitate':
            mitigation['ability_mitigations']['Ground'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Flash Fire':
            mitigation['ability_mitigations']['Fire'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Storm Drain':
            mitigation['ability_mitigations']['Water'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Volt Absorb' or ability == 'Lightning Rod' or ability == 'Motor Drive':
            mitigation['ability_mitigations']['Electric'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Sap Sipper':
            mitigation['ability_mitigations']['Grass'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Thick Fat':
            mitigation['ability_mitigations']['Fire'].append(f"{pokemon['Pokemon']} ({ability})")
            mitigation['ability_mitigations']['Ice'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Heatproof':
            mitigation['ability_mitigations']['Fire'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Water Absorb' or ability == 'Dry Skin':
            mitigation['ability_mitigations']['Water'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Sand Veil':
            mitigation['ability_mitigations']['Rock'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Wonder Guard':
            mitigation['ability_mitigations']['All'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Filter' or ability == 'Solid Rock':
            mitigation['ability_mitigations']['Super Effective'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Fluffy':
            mitigation['ability_mitigations']['Fire'].append(f"{pokemon['Pokemon']} ({ability})")
            mitigation['ability_mitigations']['Contact'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Bulletproof':
            mitigation['ability_mitigations']['Ball'].append(f"{pokemon['Pokemon']} ({ability})")
        elif ability == 'Soundproof':
            mitigation['ability_mitigations']['Sound'].append(f"{pokemon['Pokemon']} ({ability})")
        
        # Nature mitigations (handles nature strings like "Relaxed (+Def -Spe)")
        nature = pokemon['Nature']
        if nature and isinstance(nature, str):
            if '+' in nature and '-' in nature:
                # Extract the parts after + and before -
                plus_part = nature.split('+')[1].split(')')[0].split(' ')[0]
                minus_part = nature.split('-')[1].split(')')[0].split(' ')[0]
                
                # Map to stat names used in your dataset
                stat_map = {
                    'Atk': 'Attack',
                    'Def': 'Defense',
                    'SpA': 'Sp. Atk',
                    'SpD': 'Sp. Def',
                    'Spe': 'Speed'
                }
                
                plus_stat = stat_map.get(plus_part, plus_part)
                minus_stat = stat_map.get(minus_part, minus_part)
                
                mitigation['nature_mitigations'][f"+{plus_stat}"].append(pokemon['Pokemon'])
                mitigation['nature_mitigations'][f"-{minus_stat}"].append(pokemon['Pokemon'])
        
        # Role-based analysis
        primary_role = pokemon['PrimaryRole']
        secondary_role = pokemon['SecondaryRole']
        strategy_synergy = pokemon['StrategySynergy']
        
        # Add to role-specific mitigation tracking
        mitigation['role_mitigations'][primary_role].append(pokemon['Pokemon'])
        if pd.notna(secondary_role) and secondary_role != '':
            mitigation['role_mitigations'][secondary_role].append(pokemon['Pokemon'])
        
        # Strategy synergy analysis
        if pd.notna(strategy_synergy) and strategy_synergy != '':
            mitigation['strategy_synergies'][strategy_synergy].append(pokemon['Pokemon'])
        
        # Comprehensive move categorization
        moves = [pokemon['Move 1'], pokemon['Move 2'], pokemon['Move 3'], pokemon['Move 4']]
        for move in moves:
            if pd.isna(move) or move == '':
                continue
                
            # Handle moves with slashes (like "Iron Defense/Bulk Up")
            if '/' in move:
                sub_moves = move.split('/')
                for sub_move in sub_moves:
                    categorize_move(sub_move, pokemon, mitigation)
            else:
                categorize_move(move, pokemon, mitigation)
    
    return {
        'coverage': coverage,
        'mitigation': mitigation,
        'team_df': team_df
    }

def categorize_move(move, pokemon, mitigation):
    """Helper function to categorize individual moves"""
    # Status moves
    status_moves = [
        'Will-O-Wisp', 'Thunder Wave', 'Spore', 'Sleep Powder', 'Yawn',
        'Hypnosis', 'Toxic', 'Glare', 'Stun Spore', 'Poison Powder'
    ]
    
    # Speed control moves
    speed_control = [
        'Tailwind', 'Electroweb', 'Icy Wind', 'Rock Tomb', 'Bulldoze',
        'String Shot', 'Cotton Spore', 'Scary Face'
    ]
    
    # Disruption/redirection moves
    disruption_moves = [
        'Fake Out', 'Follow Me', 'Rage Powder', 'Ally Switch', 'Taunt',
        'Encore', 'Torment', 'Disable', 'Encore', 'Protect', 'Detect',
        'Quick Guard', 'Wide Guard', 'Crafty Shield'
    ]
    
    # Recovery moves
    recovery_moves = [
        'Recover', 'Roost', 'Moonlight', 'Morning Sun', 'Synthesis',
        'Wish', 'Heal Pulse', 'Pain Split', 'Leech Seed', 'Strength Sap'
    ]
    
    # Setup moves
    setup_moves = [
        'Swords Dance', 'Nasty Plot', 'Bulk Up', 'Calm Mind', 'Dragon Dance',
        'Iron Defense', 'Agility', 'Growth', 'Hone Claws', 'Shell Smash',
        'Work Up', 'Coil', 'Quiver Dance', 'Geomancy'
    ]
    
    # Field effect moves
    field_effect_moves = [
        'Perish Song', 'Trick Room', 'Rain Dance', 'Sunny Day', 'Sandstorm',
        'Hail', 'Gravity', 'Magic Room', 'Wonder Room', 'Electric Terrain',
        'Psychic Terrain', 'Grassy Terrain', 'Misty Terrain', 'Stealth Rock',
        'Toxic Spikes', 'Spikes', 'Sticky Web', 'Aurora Veil', 'Reflect',
        'Light Screen', 'Safeguard', 'Lucky Chant'
    ]
    
    # Hazard control
    hazard_control = [
        'Rapid Spin', 'Defog', 'Mortal Spin', 'Court Change'
    ]
    
    # Offensive utility
    offensive_utility = [
        'U-turn', 'Volt Switch', 'Parting Shot', 'Baton Pass', 'Pursuit',
        'Knock Off', 'Trick', 'Switcheroo'
    ]
    
    # Healing/support moves
    healing_moves = [
        'Helping Hand', 'Heal Bell', 'Aromatherapy', 'Life Dew', 'Floral Healing',
        'Jungle Healing', 'Pollen Puff', 'Lunar Blessing'
    ]
    
    # Priority moves
    priority_moves = [
        'Aqua Jet', 'Bullet Punch', 'Ice Shard', 'Mach Punch', 'Quick Attack',
        'Shadow Sneak', 'Sucker Punch', 'Extreme Speed', 'Vacuum Wave',
        'Water Shuriken', 'First Impression', 'Fake Out'
    ]
    
    # Weather moves
    weather_moves = [
        'Rain Dance', 'Sunny Day', 'Sandstorm', 'Hail'
    ]
    
    # Move categorization logic
    if move == 'Protect':
        mitigation['move_mitigations']['General'].append(f"{pokemon['Pokemon']} (Protect)")
    elif move in status_moves:
        mitigation['move_mitigations']['Status'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in speed_control:
        mitigation['move_mitigations']['Speed Control'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in disruption_moves:
        mitigation['move_mitigations']['Disruption'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in recovery_moves:
        mitigation['move_mitigations']['Recovery'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in setup_moves:
        mitigation['move_mitigations']['Setup'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in field_effect_moves:
        mitigation['move_mitigations']['Field Effect'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in hazard_control:
        mitigation['move_mitigations']['Hazard Control'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in offensive_utility:
        mitigation['move_mitigations']['Offensive Utility'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in healing_moves:
        mitigation['move_mitigations']['Healing'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in priority_moves:
        mitigation['move_mitigations']['Priority'].append(f"{pokemon['Pokemon']} ({move})")
    elif move in weather_moves:
        mitigation['move_mitigations']['Weather'].append(f"{pokemon['Pokemon']} ({move})")
    else:
        # Default category for offensive moves
        mitigation['move_mitigations']['Offensive'].append(f"{pokemon['Pokemon']} ({move})")

def analyze_move_synergy(team_df):
    """Analyze how moves align with team roles and win conditions"""
    role_move_synergy = defaultdict(list)
    win_condition_contributions = defaultdict(list)
    
    for _, pokemon in team_df.iterrows():
        role = pokemon['PrimaryRole']
        moves = [pokemon['Move 1'], pokemon['Move 2'], pokemon['Move 3'], pokemon['Move 4']]
        win_condition = pokemon.get('Win Condition', '')
        
        # Analyze moves for role alignment
        for move in moves:
            if role == 'Sweeper' and ('Swords Dance' in move or 'Nasty Plot' in move):
                role_move_synergy['Sweeper Setup'].append(f"{pokemon['Pokemon']}: {move}")
            elif role == 'Support' and ('Tailwind' in move or 'Helping Hand' in move):
                role_move_synergy['Support Utility'].append(f"{pokemon['Pokemon']}: {move}")
            elif role == 'Wallbreaker' and ('Choice Band' in str(pokemon['Item']) or 'Choice Specs' in str(pokemon['Item'])):
                role_move_synergy['Wallbreaker Power'].append(f"{pokemon['Pokemon']}: {move} (with {pokemon['Item']})")
        
        # Analyze contributions to win condition
        if win_condition:
            if 'Perish Song' in win_condition and 'Perish Song' in moves:
                win_condition_contributions['Perish Song'].append(pokemon['Pokemon'])
            elif 'Sand' in win_condition and ('Sandstorm' in moves or 'Sand Stream' in str(pokemon['Ability'])):
                win_condition_contributions['Sand'].append(pokemon['Pokemon'])
            elif 'Tailwind' in win_condition and 'Tailwind' in moves:
                win_condition_contributions['Tailwind'].append(pokemon['Pokemon'])
    
    return {
        'role_move_synergy': dict(role_move_synergy),
        'win_condition_contributions': dict(win_condition_contributions)
    }

def compare_teams_threats(team1_name, team2_name, df):
    """Compare two teams to identify threats and opportunities"""
    team1 = df[df['Team'] == team1_name]
    team2 = df[df['Team'] == team2_name]
    
    if team1.empty or team2.empty:
        return {}
    
    # Get all types for each team
    team1_types = []
    for _, row in team1.iterrows():
        team1_types.append(row['Type1'])
        if pd.notna(row['Type2']) and row['Type2'] != '':
            team1_types.append(row['Type2'])
    
    team2_types = []
    for _, row in team2.iterrows():
        team2_types.append(row['Type1'])
        if pd.notna(row['Type2']) and row['Type2'] != '':
            team2_types.append(row['Type2'])
    
    # Calculate coverage for both teams
    team1_coverage = calculate_team_coverage(team1_types)
    team2_coverage = calculate_team_coverage(team2_types)
    
    # Find threats (team1's strengths vs team2's weaknesses)
    threats = []
    for t in team1_coverage['offensive_coverage']:
        if t in team2_coverage['uncovered_weaknesses']:
            threats.append(f"Team {team1_name}'s {t} attacks vs Team {team2_name}'s weakness")
    
    # Find opportunities (team2's uncovered weaknesses)
    opportunities = []
    for t in team2_coverage['uncovered_weaknesses']:
        if t not in team1_coverage['offensive_coverage']:
            opportunities.append(f"Team {team1_name} could exploit Team {team2_name}'s {t} weakness")
    
    return {
        'threats': threats,
        'opportunities': opportunities,
        'team1_coverage': team1_coverage,
        'team2_coverage': team2_coverage
    }

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
    
    # Add default columns if not present
    if 'PrimaryRole' not in df.columns:
        df['PrimaryRole'] = 'Unknown'
    if 'SecondaryRole' not in df.columns:
        df['SecondaryRole'] = ''
    if 'StrategySynergy' not in df.columns:
        df['StrategySynergy'] = ''
    if 'Win Condition' not in df.columns:
        df['Win Condition'] = ''
    if 'Item' not in df.columns:
        df['Item'] = ''
    if 'Ability' not in df.columns:
        df['Ability'] = ''
    if 'Nature' not in df.columns:
        df['Nature'] = ''
    if 'Move 1' not in df.columns:
        df['Move 1'] = ''
    if 'Move 2' not in df.columns:
        df['Move 2'] = ''
    if 'Move 3' not in df.columns:
        df['Move 3'] = ''
    if 'Move 4' not in df.columns:
        df['Move 4'] = ''
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏆 Team Overview", 
        "🔍 Pokémon Analysis", 
        "📊 Team Comparison", 
        "🤖 ML Recommendations",
        "🛡️ Type Coverage",
        "🔄 Team Synergy",
        "⚔️ Team Matchup"
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

    with tab6:
        st.header("Team Synergy Analysis")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='synergy_team')
        
        synergy_data = analyze_team_synergy(df, selected_team)
        if synergy_data:
            team_df = synergy_data['team_df']
            
            # Display team composition overview
            st.subheader("🧩 Team Composition")
            col1, col2 = st.columns(2)
            
            with col1:
                # Role distribution pie chart
                role_dist = team_df['PrimaryRole'].value_counts().reset_index()
                fig = px.pie(role_dist, names='PrimaryRole', values='count', 
                             title='Primary Role Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show all team members with roles
                st.write("**Team Members:**")
                for _, row in team_df.iterrows():
                    role_info = f"{row['Pokemon']} ({row['PrimaryRole']}"
                    if pd.notna(row['SecondaryRole']) and row['SecondaryRole'] != '':
                        role_info += f" / {row['SecondaryRole']}"
                    st.write(f"- {role_info}")
            
            st.divider()
            
            # Enhanced Role Analysis
            st.subheader("🎯 Role-Specific Analysis")
            
            # Define expected moves/abilities for each role
            ROLE_EXPECTATIONS = {
                'Sweeper': ['Swords Dance', 'Nasty Plot', 'Dragon Dance', 'Bulk Up'],
                'Wallbreaker': ['Choice Band', 'Choice Specs', 'Life Orb'],
                'Support': ['Tailwind', 'Helping Hand', 'Light Screen', 'Reflect'],
                'Tank': ['Recover', 'Protect', 'Iron Defense', 'Amnesia'],
                'Disruptor': ['Taunt', 'Encore', 'Will-O-Wisp', 'Thunder Wave']
            }
            
            for role in team_df['PrimaryRole'].unique():
                role_pokemon = team_df[team_df['PrimaryRole'] == role]
                st.write(f"#### {role} Role Analysis")
                
                for _, pokemon in role_pokemon.iterrows():
                    # Check for expected moves/items
                    missing_components = []
                    moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                    
                    if role in ROLE_EXPECTATIONS:
                        for expected in ROLE_EXPECTATIONS[role]:
                            if expected not in ' '.join(str(m) for m in moves) and expected not in str(pokemon['Item']):
                                missing_components.append(expected)
                    
                    # Display Pokémon info
                    if missing_components:
                        st.warning(f"{pokemon['Pokemon']} is missing common {role} elements: {', '.join(missing_components)}")
                    else:
                        st.success(f"{pokemon['Pokemon']} has good {role} setup")
                    
                    # Show key moves
                    st.write(f"**Key Moves:** {', '.join([m for m in moves if pd.notna(m)])}")
                    st.write(f"**Item:** {pokemon['Item']}")
                    st.write(f"**Ability:** {pokemon['Ability']}")
                    st.write("---")
            
            st.divider()
            
            # Enhanced Win Condition Analysis
            st.subheader("🏆 Win Condition Analysis")
            
            # Get unique win conditions (some teams might have multiple)
            win_conditions = team_df['Win Condition'].dropna().unique()
            
            if len(win_conditions) > 0:
                for wc in win_conditions:
                    st.write(f"#### Win Condition: {wc}")
                    
                    # Find Pokémon that directly contribute
                    contributors = []
                    for _, pokemon in team_df.iterrows():
                        moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                        if any(keyword in ' '.join(str(m) for m in moves) for keyword in wc.split()) or \
                           any(keyword in str(pokemon['Ability']) for keyword in wc.split()):
                            contributors.append(pokemon['Pokemon'])
                    
                    if contributors:
                        st.success(f"Primary Contributors: {', '.join(contributors)}")
                    else:
                        st.warning("No clear contributors identified - check team strategy")
                    
                    # Show execution phases
                    st.write("**Execution Phases:**")
                    try:
                        phases = {
                            'Early Game': team_df['Early Game'].iloc[0],
                            'Mid Game': team_df['Mid Game'].iloc[0],
                            'Late Game': team_df['Late Game'].iloc[0]
                        }
                        for phase, desc in phases.items():
                            if pd.notna(desc):
                                st.write(f"- **{phase}:** {desc}")
                    except:
                        st.warning("Phase information not available")
            
            else:
                st.warning("No explicit win condition defined for this team")
            
            st.divider()
            
            # Move Synergy Analysis (existing code)
            st.subheader("🔄 Move Synergy")
            move_synergy = analyze_move_synergy(team_df)
            
            if move_synergy['role_move_synergy']:
                for category, moves in move_synergy['role_move_synergy'].items():
                    st.write(f"**{category}:**")
                    for move in moves:
                        st.write(f"- {move}")
            else:
                st.warning("No notable move-role synergies found")
            
            st.divider()
            
            # Setup Move Tracking
            st.subheader("📈 Setup Moves")
            setup_moves = ['Swords Dance', 'Nasty Plot', 'Bulk Up', 'Calm Mind', 'Dragon Dance']
            setup_users = []
            
            for _, pokemon in team_df.iterrows():
                moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                for move in moves:
                    if any(sm in str(move) for sm in setup_moves):
                        setup_users.append(f"{pokemon['Pokemon']} ({move})")
            
            if setup_users:
                st.success("Setup Move Users:")
                for user in setup_users:
                    st.write(f"- {user}")
            else:
                st.warning("No setup moves detected - team may lack sweeping potential")

    with tab7:
        st.header("Team Matchup Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Team 1", sorted(df['Team'].unique()), key='team1')
        
        with col2:
            team2 = st.selectbox("Select Team 2", sorted(df['Team'].unique()), key='team2')
        
        if team1 and team2:
            comparison = compare_teams_threats(team1, team2, df)
            
            if comparison:
                st.subheader(f"Threats for Team {team2} from Team {team1}")
                if comparison['threats']:
                    for threat in comparison['threats']:
                        st.warning(threat)
                else:
                    st.info(f"Team {team1} has no clear type advantages against Team {team2}")
                
                st.divider()
                
                st.subheader(f"Opportunities for Team {team1} against Team {team2}")
                if comparison['opportunities']:
                    for opportunity in comparison['opportunities']:
                        st.success(opportunity)
                else:
                    st.info(f"Team {team1} is already exploiting all of Team {team2}'s weaknesses")
                
                st.divider()
                
                st.subheader("Detailed Coverage Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"### Team {team1}")
                    st.write("**Uncovered Weaknesses:**")
                    if comparison['team1_coverage']['uncovered_weaknesses']:
                        for weakness in comparison['team1_coverage']['uncovered_weaknesses']:
                            st.error(weakness)
                    else:
                        st.success("All weaknesses covered!")
                    
                    st.write("**Excellent Coverage (3+):**")
                    if comparison['team1_coverage']['excellent_coverage']:
                        for coverage in comparison['team1_coverage']['excellent_coverage']:
                            st.success(coverage)
                    else:
                        st.warning("No excellent coverage")
                
                with col2:
                    st.write(f"### Team {team2}")
                    st.write("**Uncovered Weaknesses:**")
                    if comparison['team2_coverage']['uncovered_weaknesses']:
                        for weakness in comparison['team2_coverage']['uncovered_weaknesses']:
                            st.error(weakness)
                    else:
                        st.success("All weaknesses covered!")
                    
                    st.write("**Excellent Coverage (3+):**")
                    if comparison['team2_coverage']['excellent_coverage']:
                        for coverage in comparison['team2_coverage']['excellent_coverage']:
                            st.success(coverage)
                    else:
                        st.warning("No excellent coverage")

if __name__ == "__main__":
    main()
