import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import math
import re

# ========================
# 1. DATA PREPROCESSING FUNCTIONS
# ========================

def find_column(df, possible_names):
    """Helper to find the right column from possible names (case-insensitive)"""
    df_columns_lower = [str(col).lower() for col in df.columns]
    for name in possible_names:
        name_lower = str(name).lower()
        if name_lower in df_columns_lower:
            return df.columns[df_columns_lower.index(name_lower)]
    return None

def preprocess_all_pokemon_data_for_threats(df):
    """Process threats dataset with extensive column name flexibility"""
    if df.empty:
        st.error("Empty DataFrame provided to preprocess_all_pokemon_data_for_threats")
        return None

    processed = pd.DataFrame()
    
    # Column name mappings with many variations
    name_col = find_column(df, ['Name', 'Pokemon', 'Pokémon', 'Pokémon Name', 'pokemon', 'name'])
    type1_col = find_column(df, ['Typing (Primary)', 'Type1', 'Primary Type', 'Type 1', 'type1', 'primary type'])
    type2_col = find_column(df, ['Typing (Secondary)', 'Type2', 'Secondary Type', 'Type 2', 'type2', 'secondary type'])
    usage_col = find_column(df, ['Meta Usage (%)', 'Usage', 'Usage %', 'Meta Usage', 'usage'])
    
    # Basic info with validation
    if not name_col:
        st.error("Could not find Pokémon name column in threats data")
        st.write("Available columns:", df.columns.tolist())
        return None
    
    processed['Name'] = df[name_col].astype(str).str.strip()
    
    if not type1_col:
        st.error(f"Could not find primary type column in threats data. Available columns: {df.columns.tolist()}")
        return None
    
    processed['Type1'] = (
        df[type1_col]
        .astype(str)
        .str.strip()
        .str.title()
        .replace(['Na', 'Nan', 'None', 'N/A', ''], None)
    )
    
    if type2_col:
        processed['Type2'] = (
            df[type2_col]
            .astype(str)
            .str.strip()
            .str.title()
            .replace(['Na', 'Nan', 'None', 'N/A', ''], None)
        )
    else:
        processed['Type2'] = None
    
    # Base stats with flexible column names
    stat_mapping = {
        'HP': ['Base Stats: HP', 'HP', 'Base HP'],
        'Atk': ['Base Stats: Atk', 'Attack', 'Atk', 'Base Atk'],
        'Def': ['Base Stats: Def', 'Defense', 'Def', 'Base Def'],
        'SpA': ['Base Stats: SpA', 'Sp. Atk', 'SpA', 'Base SpA'],
        'SpD': ['Base Stats: SpD', 'Sp. Def', 'SpD', 'Base SpD'],
        'Spe': ['Base Stats: Spe', 'Speed', 'Spe', 'Base Spe']
    }
    
    for stat, possible_names in stat_mapping.items():
        col = find_column(df, possible_names)
        if col:
            processed[stat] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            st.warning(f"Could not find {stat} column, using 0 as default")
            processed[stat] = 0
    
    # Usage data
    if usage_col:
        try:
            processed['Usage'] = pd.to_numeric(df[usage_col], errors='coerce').fillna(0.0)
        except Exception as e:
            st.warning(f"Could not parse usage column: {e}")
            processed['Usage'] = 0.0
    else:
        st.warning("No usage column found, defaulting to 0")
        processed['Usage'] = 0.0
    
    # Tier assignment
    processed['Tier'] = processed['Usage'].apply(
        lambda x: 'S' if x >= 80 else 'A' if x >= 60 else 'B' if x >= 30 else 'C'
    )
    
    return processed

def preprocess_team_data_for_analyzer(df):
    """Process team dataset with flexible column names"""
    processed = pd.DataFrame()
    
    # Flexible column name handling
    name_col = find_column(df, ['Pokemon', 'Pokémon', 'Name', 'Pokémon Name'])
    type1_col = find_column(df, ['Typing (Primary)', 'Type1', 'Primary Type'])
    type2_col = find_column(df, ['Typing (Secondary)', 'Type2', 'Secondary Type'])
    evs_col = find_column(df, ['EVs', 'EV Spread', 'Effort Values'])
    
    # Basic info
    if not name_col:
        st.error("Could not find Pokémon name column in team data")
        return None
    processed['Name'] = df[name_col]
    
    if not type1_col:
        st.error("Could not find primary type column in team data")
        return None
    processed['Type1'] = df[type1_col]
    
    processed['Type2'] = df[type2_col].replace('NA', None) if type2_col else None
    
    # Base stats
    stat_mapping = {
        'HP': ['Base Stats: HP', 'HP'],
        'Atk': ['Base Stats: Atk', 'Attack'],
        'Def': ['Base Stats: Def', 'Defense'],
        'SpA': ['Base Stats: SpA', 'Sp. Atk'],
        'SpD': ['Base Stats: SpD', 'Sp. Def'],
        'Spe': ['Base Stats: Spe', 'Speed']
    }
    
    for stat, possible_names in stat_mapping.items():
        col = find_column(df, possible_names)
        if col:
            processed[stat] = df[col]
        else:
            st.warning(f"Could not find {stat} column, using 0 as default")
            processed[stat] = 0
    
    # Parse EVs
    def parse_evs(ev_text):
        ev_dict = {'HP': 0, 'Atk': 0, 'Def': 0, 'SpA': 0, 'SpD': 0, 'Spe': 0}
        if pd.isna(ev_text) or ev_text == 'NA':
            return ev_dict
        parts = str(ev_text).split('/')
        for part in parts:
            match = re.match(r'(\d+)\s*(\w+)', part.strip())
            if match:
                value, stat = match.groups()
                if stat in ev_dict:
                    ev_dict[stat] = int(value)
        return ev_dict
    
    if evs_col:
        ev_data = df[evs_col].apply(parse_evs)
        processed['EV_HP'] = [ev['HP'] for ev in ev_data]
        processed['EV_Atk'] = [ev['Atk'] for ev in ev_data]
        processed['EV_Def'] = [ev['Def'] for ev in ev_data]
        processed['EV_SpA'] = [ev['SpA'] for ev in ev_data]
        processed['EV_SpD'] = [ev['SpD'] for ev in ev_data]
        processed['EV_Spe'] = [ev['Spe'] for ev in ev_data]
    else:
        st.warning("No EVs column found, using 0 for all EVs")
        processed['EV_HP'] = 0
        processed['EV_Atk'] = 0
        processed['EV_Def'] = 0
        processed['EV_SpA'] = 0
        processed['EV_SpD'] = 0
        processed['EV_Spe'] = 0
    
    # Other info
    item_col = find_column(df, ['Item', 'Held Item', 'Equipment'])
    ability_col = find_column(df, ['Ability', 'Abilities'])
    
    processed['Item'] = df[item_col] if item_col else None
    processed['Ability'] = df[ability_col] if ability_col else None
    
    # Combine moves
    moves = []
    for _, row in df.iterrows():
        move_list = []
        for i in range(1, 5):
            move_col = find_column(df, [f'Move {i}', f'Move{i}', f'Attack {i}'])
            if move_col and pd.notna(row[move_col]) and str(row[move_col]).strip() != 'NA':
                move_list.append(str(row[move_col]).strip())
        moves.append(','.join(move_list))
    
    processed['Moves'] = moves
    
    return processed

# ========================
# 2. TYPE CHART AND ANALYSIS FUNCTIONS
# ========================

@st.cache_data
def load_type_chart():
    """Complete type effectiveness chart"""
    return {
        'Normal': {'weak': ['Fighting'], 'resist': [], 'immune': ['Ghost']},
        'Fire': {'weak': ['Water', 'Rock', 'Ground'], 'resist': ['Fire', 'Grass', 'Ice', 'Bug', 'Steel', 'Fairy'], 'immune': []},
        'Water': {'weak': ['Electric', 'Grass'], 'resist': ['Fire', 'Water', 'Ice', 'Steel'], 'immune': []},
        'Electric': {'weak': ['Ground'], 'resist': ['Electric', 'Flying', 'Steel'], 'immune': []},
        'Grass': {'weak': ['Fire', 'Ice', 'Poison', 'Flying', 'Bug'], 'resist': ['Water', 'Electric', 'Grass', 'Ground'], 'immune': []},
        'Ice': {'weak': ['Fire', 'Fighting', 'Rock', 'Steel'], 'resist': ['Ice'], 'immune': []},
        'Fighting': {'weak': ['Flying', 'Psychic', 'Fairy'], 'resist': ['Bug', 'Rock', 'Dark'], 'immune': []},
        'Poison': {'weak': ['Ground', 'Psychic'], 'resist': ['Grass', 'Fighting', 'Poison', 'Bug', 'Fairy'], 'immune': []},
        'Ground': {'weak': ['Water', 'Grass', 'Ice'], 'resist': ['Poison', 'Rock'], 'immune': ['Electric']},
        'Flying': {'weak': ['Electric', 'Ice', 'Rock'], 'resist': ['Grass', 'Fighting', 'Bug'], 'immune': ['Ground']},
        'Psychic': {'weak': ['Bug', 'Ghost', 'Dark'], 'resist': ['Fighting', 'Psychic'], 'immune': []},
        'Bug': {'weak': ['Fire', 'Flying', 'Rock'], 'resist': ['Grass', 'Fighting', 'Ground'], 'immune': []},
        'Rock': {'weak': ['Water', 'Grass', 'Fighting', 'Ground', 'Steel'], 'resist': ['Normal', 'Fire', 'Poison', 'Flying'], 'immune': []},
        'Ghost': {'weak': ['Ghost', 'Dark'], 'resist': ['Poison', 'Bug'], 'immune': ['Normal', 'Fighting']},
        'Dragon': {'weak': ['Ice', 'Dragon', 'Fairy'], 'resist': ['Fire', 'Water', 'Electric', 'Grass'], 'immune': []},
        'Dark': {'weak': ['Fighting', 'Bug', 'Fairy'], 'resist': ['Ghost', 'Dark'], 'immune': ['Psychic']},
        'Steel': {'weak': ['Fire', 'Fighting', 'Ground'], 'resist': ['Normal', 'Grass', 'Ice', 'Flying', 'Psychic', 'Bug', 'Rock', 'Dragon', 'Steel', 'Fairy'], 'immune': ['Poison']},
        'Fairy': {'weak': ['Poison', 'Steel'], 'resist': ['Fighting', 'Bug', 'Dark'], 'immune': ['Dragon']}
    }

def calculate_type_effectiveness(attacker_type: str, defender_types: List[str], type_chart: Dict) -> float:
    """Calculate type effectiveness multiplier"""
    effectiveness = 1.0
    for def_type in defender_types:
        if def_type and def_type in type_chart:
            if attacker_type in type_chart[def_type]['weak']:
                effectiveness *= 2.0
            elif attacker_type in type_chart[def_type]['resist']:
                effectiveness *= 0.5
            elif attacker_type in type_chart[def_type]['immune']:
                return 0.0
    return effectiveness

def calculate_stats_with_evs(base_stat: int, ev: int, level: int = 50) -> int:
    """Calculate actual stat with EVs at level 50"""
    return math.floor(((2 * base_stat + 31 + (ev // 4)) * level) // 100) + 5

def calculate_hp_with_evs(base_hp: int, ev_hp: int, level: int = 50) -> int:
    """Calculate HP with EVs at level 50"""
    return math.floor(((2 * base_hp + 31 + (ev_hp // 4)) * level) // 100) + level + 10

def analyze_team_weaknesses(team_types: List[List[str]], type_chart: Dict) -> Dict[str, int]:
    """Analyze team-wide type weaknesses"""
    weaknesses = {}
    all_types = list(type_chart.keys())
    
    for attack_type in all_types:
        weak_count = 0
        for pokemon_types in team_types:
            effectiveness = calculate_type_effectiveness(attack_type, pokemon_types, type_chart)
            if effectiveness > 1.0:
                weak_count += 1
        if weak_count > 0:
            weaknesses[attack_type] = weak_count
    
    return dict(sorted(weaknesses.items(), key=lambda x: x[1], reverse=True))

def get_speed_tier(speed: int) -> str:
    """Categorize speed into tiers"""
    if speed <= 70:
        return 'Trick Room (≤70)'
    elif speed <= 100:
        return 'Average (71-100)'
    elif speed <= 120:
        return 'Fast (101-120)'
    else:
        return 'Very Fast (121+)'

def analyze_speed_tiers(team_df: pd.DataFrame) -> List[Dict]:
    """Analyze team speed distribution"""
    speed_data = []
    for _, row in team_df.iterrows():
        speed_data.append({
            'Pokemon': row['Name'],
            'Speed': row['Spe'],
            'Tier': get_speed_tier(row['Spe'])
        })
    return speed_data

def estimate_move_powers(pokemon):
    """Estimate possible moves a Pokémon might have based on its stats and typing"""
    moves = {
        'Physical': [],
        'Special': []
    }
    
    # Physical moves
    if pokemon['Atk'] >= pokemon['SpA']:
        # STAB moves
        moves['Physical'].append((
            f"{pokemon['Type1']} Move",
            [80, 100, 120],  # Typical BP range
            pokemon['Type1']
        ))
        if pd.notna(pokemon.get('Type2')):
            moves['Physical'].append((
                f"{pokemon['Type2']} Move",
                [80, 100, 120],
                pokemon['Type2']
            ))
        
        # Coverage moves
        moves['Physical'].append((
            "Coverage Move",
            [70, 90],
            "Normal"  # Neutral coverage
        ))
    
    # Special moves
    if pokemon['SpA'] >= pokemon['Atk'] * 0.8:  # If special attack is at least 80% of physical
        # STAB moves
        moves['Special'].append((
            f"{pokemon['Type1']} Move",
            [80, 100, 120],
            pokemon['Type1']
        ))
        if pd.notna(pokemon.get('Type2')):
            moves['Special'].append((
                f"{pokemon['Type2']} Move",
                [80, 100, 120],
                pokemon['Type2']
            ))
        
        # Coverage moves
        moves['Special'].append((
            "Coverage Move",
            [70, 90],
            "Normal"
        ))
    
    return moves

def simulate_damage_range(attacker, defender, power, move_type, category, type_chart):
    """Simulate damage range for a move"""
    # Calculate stats with EVs
    if category == 'Physical':
        attack_stat = calculate_stats_with_evs(attacker['Atk'], attacker.get('EV_Atk', 0))
        defense_stat = calculate_stats_with_evs(defender['Def'], defender.get('EV_Def', 0))
    else:  # Special
        attack_stat = calculate_stats_with_evs(attacker['SpA'], attacker.get('EV_SpA', 0))
        defense_stat = calculate_stats_with_evs(defender['SpD'], defender.get('EV_SpD', 0))
    
    # Calculate HP
    defender_hp = calculate_hp_with_evs(defender['HP'], defender.get('EV_HP', 0))
    
    # Type effectiveness
    defender_types = [defender['Type1']]
    if pd.notna(defender.get('Type2')):
        defender_types.append(defender['Type2'])
    effectiveness = calculate_type_effectiveness(move_type, defender_types, type_chart)
    
    # Damage calculation (simplified)
    damage = (((2 * 50 / 5 + 2) * power * attack_stat / defense_stat) / 50 + 2)
    
    # Apply modifiers
    min_dmg = math.floor(damage * 0.85 * effectiveness)  # Minimum roll
    max_dmg = math.floor(damage * 1.0 * effectiveness)   # Maximum roll
    
    return min_dmg, max_dmg, defender_hp

def analyze_comprehensive_threats(team_df, threats_df, type_chart):
    """Analyze threats against the team with comprehensive move estimation"""
    results = []
    
    # Validate inputs
    if team_df.empty or threats_df.empty:
        st.error("Empty DataFrame provided to analyze_comprehensive_threats")
        return pd.DataFrame()
    
    # Check required columns
    required_columns = ['Name', 'Type1']
    for col in required_columns:
        if col not in threats_df.columns:
            st.error(f"Required column '{col}' missing from threats data")
            st.write("Available columns:", threats_df.columns.tolist())
            return pd.DataFrame()
    
    for _, threat in threats_df.iterrows():
        try:
            # Safely get threat attributes with defaults
            threat_name = threat.get('Name', 'Unknown')
            type1 = threat.get('Type1')
            
            if pd.isna(type1):
                st.warning(f"Skipping threat {threat_name} - missing Type1")
                continue
                
            type2 = threat.get('Type2', None)
            types_display = f"{type1}{f'/{type2}' if type2 else ''}"
            
            threat_data = {
                'Threat': threat_name,
                'Types': types_display,
                'Usage %': threat.get('Usage', 0),
                'OHKO Count': 0,
                '2HKO Count': 0,
                'Max Damage %': 0,
                'Most Dangerous Move': '',
                'Worst Matchup': ''
            }
            
            estimated_moves = estimate_move_powers(threat)
            max_damage_pct = 0
            worst_move = ''
            worst_matchup = ''
            
            for category, moves in estimated_moves.items():
                for move_name, power_range, move_type in moves:
                    for power in power_range:
                        for _, member in team_df.iterrows():
                            min_dmg, max_dmg, defender_hp = simulate_damage_range(
                                threat, member, power, move_type, category, type_chart
                            )
                            
                            damage_pct = (max_dmg / defender_hp) * 100
                            
                            # Track worst case scenario
                            if damage_pct > max_damage_pct:
                                max_damage_pct = damage_pct
                                worst_move = f"{move_name} ({power}BP)"
                                worst_matchup = member['Name']
                            
                            # Count OHKOs and 2HKOs
                            if max_dmg >= defender_hp:
                                threat_data['OHKO Count'] += 1
                            elif max_dmg >= defender_hp * 0.5:
                                threat_data['2HKO Count'] += 1
            
            threat_data['Max Damage %'] = max_damage_pct
            threat_data['Most Dangerous Move'] = worst_move
            threat_data['Worst Matchup'] = worst_matchup
            
            results.append(threat_data)
            
        except Exception as e:
            st.warning(f"Error processing threat {threat.get('Name', 'unknown')}: {str(e)}")
            continue
    
    return pd.DataFrame(results)

def get_move_example(type_name):
    """Get example move for a given type"""
    examples = {
        'Normal': 'Body Slam',
        'Fire': 'Flamethrower',
        'Water': 'Hydro Pump',
        'Electric': 'Thunderbolt',
        'Grass': 'Energy Ball',
        'Ice': 'Ice Beam',
        'Fighting': 'Close Combat',
        'Poison': 'Sludge Bomb',
        'Ground': 'Earthquake',
        'Flying': 'Brave Bird',
        'Psychic': 'Psychic',
        'Bug': 'Bug Buzz',
        'Rock': 'Rock Slide',
        'Ghost': 'Shadow Ball',
        'Dragon': 'Draco Meteor',
        'Dark': 'Dark Pulse',
        'Steel': 'Flash Cannon',
        'Fairy': 'Moonblast'
    }
    return examples.get(type_name, 'Standard Move')

# ========================
# 3. STREAMLIT APP
# ========================

def main():
    st.set_page_config(page_title="VGC Team Analyzer", layout="wide")
    
    # Sidebar for data upload
    with st.sidebar:
        st.header("📁 Data Upload")
        team_file = st.file_uploader("Upload Team Data (CSV)", type=["csv"])
        threats_file = st.file_uploader("Upload Threats Data (CSV)", type=["csv"])
        
        if team_file:
            team_df = pd.read_csv(team_file)
            st.write("Team columns:", team_df.columns.tolist())
            processed_team = preprocess_team_data_for_analyzer(team_df)
        else:
            st.warning("Using sample team data")
            sample_team = [{
                'Pokemon': 'Iron Hands', 'Typing (Primary)': 'Fighting', 'Typing (Secondary)': 'Electric',
                'Base Stats: HP': 154, 'Base Stats: Atk': 140, 'Base Stats: Def': 108,
                'Base Stats: SpA': 50, 'Base Stats: SpD': 68, 'Base Stats: Spe': 50,
                'EVs': '252 HP/252 Atk/4 SpD', 'Item': 'Assault Vest', 'Ability': 'Quark Drive',
                'Move 1': 'Drain Punch', 'Move 2': 'Thunder Punch', 'Move 3': 'Fake Out', 'Move 4': 'Wild Charge'
            }]
            team_df = pd.DataFrame(sample_team)
            processed_team = preprocess_team_data_for_analyzer(team_df)
        
        if threats_file:
            threats_df = pd.read_csv(threats_file)
            st.write("Raw threats columns:", threats_df.columns.tolist())
            processed_threats = preprocess_all_pokemon_data_for_threats(threats_df)
            if processed_threats is not None:
                st.write("Processed threats columns:", processed_threats.columns.tolist())
        else:
            st.warning("Using sample threats data")
            sample_threats = [{
                'Name': 'Flutter Mane', 'Typing (Primary)': 'Ghost', 'Typing (Secondary)': 'Fairy',
                'Base Stats: HP': 55, 'Base Stats: Atk': 55, 'Base Stats: Def': 55,
                'Base Stats: SpA': 135, 'Base Stats: SpD': 135, 'Base Stats: Spe': 135,
                'Meta Usage (%)': 32.5
            }]
            threats_df = pd.DataFrame(sample_threats)
            processed_threats = preprocess_all_pokemon_data_for_threats(threats_df)
    
    # Ensure we have valid data before proceeding
    if processed_threats is None:
        st.error("Failed to process threats data. Please check your threats data file.")
        return
    
    # Use the processed data throughout the app
    threats_df = processed_threats
    
        # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Team Overview", "⚔️ Threat Analysis", "🎯 Survival Calculator", "🔧 Optimization"])
    
    with tab1:  # Team Overview tab
        st.header("Team Overview & Weaknesses")
        
        if not hasattr(processed_team, 'columns'):
            st.error("Team data not loaded correctly")
            return
        
        # 1. Display Team Data
        st.subheader("Your Team Composition")
        display_columns = {
            'Pokemon': 'Name',
            'Primary Type': 'Type1',
            'Secondary Type': 'Type2',
            'Item': 'Item',
            'Ability': 'Ability',
            'Moves': 'Moves'
        }
        
        display_df = processed_team[[col for col in display_columns.values() if col in processed_team.columns]]
        display_df.columns = [k for k, v in display_columns.items() if v in processed_team.columns]
        st.dataframe(display_df, use_container_width=True)
        
        # 2. Team Stats Summary
        st.subheader("Team Stats Summary")
        cols = st.columns(3)
        
        with cols[0]:
            avg_hp = processed_team['HP'].mean()
            st.metric("Average HP", f"{avg_hp:.0f}")
        
        with cols[1]:
            avg_speed = processed_team['Spe'].mean()
            st.metric("Average Speed", f"{avg_speed:.0f}")
        
        with cols[2]:
            physical_count = sum(processed_team['Atk'] >= processed_team['SpA'])
            st.metric("Physical Attackers", physical_count)
        
        # 3. Type Weakness Analysis
        st.subheader("Type Weakness Analysis")
        team_types = []
        for _, row in processed_team.iterrows():
            types = [row['Type1']]
            if pd.notna(row.get('Type2')):
                types.append(row['Type2'])
            team_types.append(types)
        
        type_chart = load_type_chart()
        weaknesses = analyze_team_weaknesses(team_types, type_chart)
        
        if weaknesses:
            weakness_df = pd.DataFrame(list(weaknesses.items()), columns=['Type', 'Weak Members'])
            fig = px.bar(weakness_df, x='Type', y='Weak Members', 
                        title="Team Type Weaknesses",
                        color='Weak Members',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        # 4. Speed Tier Analysis
        st.subheader("Speed Tier Distribution")
        speed_data = analyze_speed_tiers(processed_team)
        speed_df = pd.DataFrame(speed_data)
        fig = px.scatter(speed_df, x='Pokemon', y='Speed', color='Tier',
                        title="Team Speed Distribution",
                        hover_data=['Tier'])
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Trick Room Threshold")
        fig.add_hline(y=100, line_dash="dash", line_color="orange", 
                     annotation_text="Average Speed")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:  # Threat Analysis tab
        st.header("Meta Threat Analysis")
        
        if threats_df.empty:
            st.warning("Please upload threats data to use this feature")
            return
        
        # Ensure we have the required columns
        if 'Tier' not in threats_df.columns and 'Usage' in threats_df.columns:
            threats_df['Tier'] = threats_df['Usage'].apply(
                lambda x: 'S' if x >= 80 else 'A' if x >= 60 else 'B' if x >= 30 else 'C'
            )
        
        # Threat filtering UI
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Usage' in threats_df.columns:
                min_usage = st.slider(
                    "Minimum Usage % to Consider", 
                    0.0, 50.0, 5.0, 0.5,
                    help="Filter by minimum usage percentage"
                )
            else:
                st.warning("No usage data available")
                min_usage = 0.0
        
        with col2:
            available_tiers = threats_df['Tier'].unique()
            selected_tiers = st.multiselect(
                "Filter by Tier", 
                available_tiers, 
                default=list(available_tiers),
                help="Filter by threat tier (S=highest usage)"
            )
        
        # Filter threats
        filter_conditions = []
        if 'Usage' in threats_df.columns:
            filter_conditions.append(threats_df['Usage'] >= min_usage)
        if selected_tiers:
            filter_conditions.append(threats_df['Tier'].isin(selected_tiers))
        
        filtered_threats = threats_df[np.logical_and.reduce(filter_conditions)] if filter_conditions else threats_df
        
        if len(filtered_threats) == 0:
            st.warning("No threats match your filters. Try adjusting criteria.")
            return
        
        # Analyze matchups
        threat_analysis = analyze_comprehensive_threats(processed_team, filtered_threats, load_type_chart())
        
        # Critical threats display
        ohko_threats = threat_analysis[threat_analysis['OHKO Count'] >= 1]
        if len(ohko_threats) > 0:
            st.error(f"🚨 Critical Threat Warning: {len(ohko_threats)} threats can OHKO your team!")
            st.dataframe(
                ohko_threats[['Threat', 'Types', 'Usage %', 'OHKO Count', 'Most Dangerous Move', 'Worst Matchup']].round(1),
                use_container_width=True,
                hide_index=True
            )
        
        # 2HKO Analysis
        twohko_threats = threat_analysis[threat_analysis['2HKO Count'] >= 3]
        if len(twohko_threats) > 0:
            st.warning(f"⚠️ Warning: {len(twohko_threats)} threats can 2HKO most of your team")
        
        # Full analysis
        st.subheader("Complete Threat Assessment")
        
        def style_threat_analysis(val):
            if isinstance(val, (int, float)):
                if val >= 100:  # OHKO range
                    return 'background-color: #ffcdd2; color: black; font-weight: bold'
                elif val >= 75:  # Dangerous
                    return 'background-color: #fff3e0; color: black'
                elif val >= 50:  # Moderate
                    return 'background-color: #f3e5f5; color: black'
                else:  # Safe
                    return 'background-color: #e8f5e8; color: black'
            return ''
        
        def style_ohko_count(val):
            if val >= 2:
                return 'background-color: #ff1744; color: white; font-weight: bold'
            elif val >= 1:
                return 'background-color: #ff5722; color: white'
            elif val >= 0.5:
                return 'background-color: #ff9800; color: black'
            return ''
        
        styled_df = (threat_analysis.style
            .map(style_threat_analysis, subset=['Max Damage %'])  
            .map(style_ohko_count, subset=['OHKO Count'])        
            .format({
                'OHKO Count': '{:.1f}', 
                'Max Damage %': '{:.1f}%', 
                'Usage %': '{:.1f}%'
            }))
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Threat visualization
        fig = px.scatter(
            threat_analysis, 
            x='Usage %' if 'Usage %' in threat_analysis.columns else 'Tier',
            y='Max Damage %', 
            size='OHKO Count', 
            color='2HKO Count',
            hover_name='Threat',
            hover_data=['Most Dangerous Move', 'Worst Matchup'],
            title="Threat Danger Analysis",
            labels={
                'Max Damage %': 'Highest Damage % to Team',
                '2HKO Count': '2HKO Potential'
            },
            color_continuous_scale='Reds'
        )
        
        fig.add_hline(y=100, line_dash="dash", line_color="red", 
                     annotation_text="OHKO Threshold")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                     annotation_text="2HKO Threshold")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Advanced Survival Calculator")
        
        if threats_df.empty:
            st.warning("Please upload threats data to use this feature")
            return
        
        # Threat selection with validation
        try:
            available_threats = threats_df['Name'].unique()
            if len(available_threats) == 0:
                st.error("No threats available in the data")
                return
                
            selected_threat_name = st.selectbox("Select Threat to Analyze", available_threats)
            threat = threats_df[threats_df['Name'] == selected_threat_name].iloc[0]
            
            # Safe display of threat info
            threat_name = threat.get('Name', 'Unknown Threat')
            type1 = threat.get('Type1', 'Unknown')
            type2 = threat.get('Type2', '')
            types_display = f"{type1}{f'/{type2}' if type2 else ''}"
            
            st.markdown(f"### Comprehensive Analysis vs {threat_name} ({types_display})")
            
            # Get all estimated moves for this threat
            estimated_moves = estimate_move_powers(threat)
            
            # Create tabs for different move categories
            if estimated_moves.get('Physical') and estimated_moves.get('Special'):
                phys_tab, spec_tab, summary_tab = st.tabs(["Physical Moves", "Special Moves", "Summary"])
            elif estimated_moves.get('Physical'):
                phys_tab, summary_tab = st.tabs(["Physical Moves", "Summary"])
                spec_tab = None
            else:
                spec_tab, summary_tab = st.tabs(["Special Moves", "Summary"])
                phys_tab = None
            
            all_results = []
            
            # Physical moves analysis
            if phys_tab and estimated_moves.get('Physical'):
                with phys_tab:
                    st.subheader("Physical Move Analysis")
                    
                    for move_name, power_range, move_type in estimated_moves['Physical']:
                        st.markdown(f"**{move_name} ({move_type} type)**")
                        
                        results_for_move = []
                        for power in power_range:
                            move_results = []
                            for _, member in processed_team.iterrows():
                                min_dmg, max_dmg, defender_hp = simulate_damage_range(
                                    threat, member, power, move_type, 'Physical', type_chart
                                )
                                
                                min_percent = (min_dmg / defender_hp) * 100
                                max_percent = (max_dmg / defender_hp) * 100
                                
                                if min_dmg >= defender_hp:
                                    survival = "❌ OHKO"
                                    survival_num = 0
                                elif max_dmg >= defender_hp:
                                    survival_num = (1 - min_percent/100) * 100
                                    survival = f"⚠️ {survival_num:.1f}%-100% HP"
                                else:
                                    survival_num = (1 - max_percent/100) * 100
                                    survival = f"✅ {survival_num:.1f}%+ HP"
                                
                                type_eff = calculate_type_effectiveness(move_type, 
                                    [member['Type1'], member.get('Type2')], type_chart)
                                
                                move_results.append({
                                    'Pokémon': member['Name'],
                                    'Move': f"{move_name} ({power}BP)",
                                    'Damage': f"{min_dmg}-{max_dmg}",
                                    'Damage %': f"{min_percent:.1f}-{max_percent:.1f}%",
                                    'Type Eff': f"{type_eff}x",
                                    'Survival': survival,
                                    'Category': 'Physical'
                                })
                            
                            results_for_move.extend(move_results)
                            all_results.extend(move_results)
                        
                        # Display results for this move category
                        move_df = pd.DataFrame(results_for_move)
                        st.dataframe(move_df, use_container_width=True)
            
            # Special moves analysis
            if spec_tab and estimated_moves.get('Special'):
                with spec_tab:
                    st.subheader("Special Move Analysis")
                    
                    for move_name, power_range, move_type in estimated_moves['Special']:
                        st.markdown(f"**{move_name} ({move_type} type)**")
                        
                        results_for_move = []
                        for power in power_range:
                            move_results = []
                            for _, member in processed_team.iterrows():
                                min_dmg, max_dmg, defender_hp = simulate_damage_range(
                                    threat, member, power, move_type, 'Special', type_chart
                                )
                                
                                min_percent = (min_dmg / defender_hp) * 100
                                max_percent = (max_dmg / defender_hp) * 100
                                
                                if min_dmg >= defender_hp:
                                    survival = "❌ OHKO"
                                elif max_dmg >= defender_hp:
                                    survival = f"⚠️ {(1 - min_percent/100)*100:.1f}%-100% HP"
                                else:
                                    survival = f"✅ {(1 - max_percent/100)*100:.1f}%+ HP"
                                
                                type_eff = calculate_type_effectiveness(move_type, 
                                    [member['Type1'], member.get('Type2')], type_chart)
                                
                                move_results.append({
                                    'Pokémon': member['Name'],
                                    'Move': f"{move_name} ({power}BP)",
                                    'Damage': f"{min_dmg}-{max_dmg}",
                                    'Damage %': f"{min_percent:.1f}-{max_percent:.1f}%",
                                    'Type Eff': f"{type_eff}x",
                                    'Survival': survival,
                                    'Category': 'Special'
                                })
                            
                            results_for_move.extend(move_results)
                            all_results.extend(move_results)
                        
                        # Display results for this move category
                        move_df = pd.DataFrame(results_for_move)
                        st.dataframe(move_df, use_container_width=True)
            
            # Summary analysis
            with summary_tab:
                st.subheader("Worst-Case Scenario Summary")
                
                if all_results:
                    # Find the most dangerous move for each team member
                    summary_results = []
                    for _, member in processed_team.iterrows():
                        member_results = [r for r in all_results if r['Pokémon'] == member['Name']]
                        
                        # Find move that deals most damage
                        max_damage_result = max(member_results, 
                                              key=lambda x: float(x['Damage %'].split('-')[1].rstrip('%')))
                        
                        summary_results.append({
                            'Your Pokémon': member['Name'],
                            'Types': f"{member['Type1']}{f'/{member['Type2']}' if pd.notna(member.get('Type2')) else ''}",
                            'Most Dangerous Move': max_damage_result['Move'],
                            'Worst Damage %': max_damage_result['Damage %'],
                            'Survival Status': max_damage_result['Survival'],
                            'Type Effectiveness': max_damage_result['Type Eff']
                        })
                    
                    summary_df = pd.DataFrame(summary_results)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Visualization of worst-case damages
                    damage_viz_data = []
                    for result in summary_results:
                        max_damage_pct = float(result['Worst Damage %'].split('-')[1].rstrip('%'))
                        damage_viz_data.append({
                            'Pokémon': result['Your Pokémon'],
                            'Max Damage %': max_damage_pct,
                            'Move': result['Most Dangerous Move']
                        })
                    
                    fig = px.bar(damage_viz_data, x='Pokémon', y='Max Damage %', 
                               hover_data=['Move'],
                               title=f"Worst-Case Damage from {threat_name}",
                               color='Max Damage %',
                               color_continuous_scale='Reds')
                    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="OHKO Threshold")
                    fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="2HKO Threshold")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # OHKO/2HKO summary
                    ohko_count = sum(1 for r in summary_results if 'OHKO' in r['Survival Status'])
                    twohko_count = sum(1 for r in damage_viz_data if r['Max Damage %'] >= 50 and r['Max Damage %'] < 100)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("OHKO Potential", f"{ohko_count}/{len(processed_team)}", delta=f"{ohko_count} members")
                    with col2:
                        st.metric("2HKO Potential", f"{twohko_count}/{len(processed_team)}", delta=f"{twohko_count} members")  
                    with col3:
                        safe_count = len(processed_team) - ohko_count - twohko_count
                        st.metric("Safe Members", f"{safe_count}/{len(processed_team)}", delta=f"{safe_count} members")
        
        except Exception as e:
            st.error(f"Error in survival calculator: {str(e)}")
            st.write("Debug info - threats_df columns:", threats_df.columns.tolist())
    
    with tab4:
        st.header("Team Optimization Recommendations")
        
        if threats_df.empty:
            st.warning("Please upload threats data to use this feature")
            return
        
        # Advanced EV optimization
        st.subheader("🔧 Advanced EV Optimization")
        
        st.markdown("**Survival-Based EV Recommendations:**")
        top_threats = threats_df.nlargest(3, 'Usage') if 'Usage' in threats_df.columns else threats_df.head(3)
        
        optimization_recommendations = []
        
        for _, member in processed_team.iterrows():
            member_recs = {
                'Pokémon': member['Name'],
                'Current EVs': f"HP:{member.get('EV_HP', 0)} Atk:{member.get('EV_Atk', 0)} Def:{member.get('EV_Def', 0)} SpA:{member.get('EV_SpA', 0)} SpD:{member.get('EV_SpD', 0)} Spe:{member.get('EV_Spe', 0)}",
                'Survivability Issues': [],
                'Recommended Changes': []
            }
            
            for _, threat in top_threats.iterrows():
                estimated_moves = estimate_move_powers(threat)
                
                strongest_damage = 0
                strongest_move_info = None
                
                # Find the strongest move this threat can use
                for category, moves in estimated_moves.items():
                    for move_name, power_range, move_type in moves:
                        max_power = max(power_range)
                        min_dmg, max_dmg, defender_hp = simulate_damage_range(
                            threat, member, max_power, move_type, category, type_chart
                        )
                        
                        if max_dmg > strongest_damage:
                            strongest_damage = max_dmg
                            strongest_move_info = {
                                'move': move_name,
                                'power': max_power,
                                'type': move_type,
                                'category': category,
                                'damage_percent': (max_dmg / defender_hp) * 100
                            }
                
                # Analyze if member needs EV adjustments
                if strongest_move_info and strongest_move_info['damage_percent'] >= 100:
                    member_recs['Survivability Issues'].append(
                        f"OHKO'd by {threat['Name']}'s {strongest_move_info['move']} ({strongest_move_info['damage_percent']:.1f}%)"
                    )
                    
                    # Calculate required bulk
                    if strongest_move_info['category'] == 'Physical':
                        current_def_evs = member.get('EV_Def', 0)
                        current_hp_evs = member.get('EV_HP', 0)
                        
                        if current_def_evs < 100:
                            member_recs['Recommended Changes'].append(
                                f"Increase Defense EVs to survive {threat['Name']}"
                            )
                        elif current_hp_evs < 200:
                            member_recs['Recommended Changes'].append(
                                f"Increase HP EVs to survive {threat['Name']}"
                            )
                    else:  # Special
                        current_spd_evs = member.get('EV_SpD', 0)
                        current_hp_evs = member.get('EV_HP', 0)
                        
                        if current_spd_evs < 100:
                            member_recs['Recommended Changes'].append(
                                f"Increase Special Defense EVs to survive {threat['Name']}"
                            )
                        elif current_hp_evs < 200:
                            member_recs['Recommended Changes'].append(
                                f"Increase HP EVs to survive {threat['Name']}"
                            )
                
                elif strongest_move_info and strongest_move_info['damage_percent'] >= 75:
                    member_recs['Survivability Issues'].append(
                        f"Takes heavy damage from {threat['Name']}'s {strongest_move_info['move']} ({strongest_move_info['damage_percent']:.1f}%)"
                    )
            
            # Speed optimization
            member_speed = calculate_stats_with_evs(member['Spe'], member.get('EV_Spe', 0))
            if 95 <= member_speed <= 105:
                member_recs['Recommended Changes'].append("Awkward speed tier - consider full investment or minimal")
            
            # Mixed attacking optimization
            if member.get('EV_Atk', 0) > 0 and member.get('EV_SpA', 0) > 0:
                member_recs['Recommended Changes'].append("Consider focusing on one attacking stat")
            
            optimization_recommendations.append(member_recs)
        
        # Display optimization table
        for rec in optimization_recommendations:
            if rec['Survivability Issues'] or rec['Recommended Changes']:
                with st.expander(f"{rec['Pokémon']} - Optimization Needed"):
                    st.write(f"**Current EVs:** {rec['Current EVs']}")
                    
                    if rec['Survivability Issues']:
                        st.write("**Survivability Issues:**")
                        for issue in rec['Survivability Issues']:
                            st.write(f"• {issue}")
                    
                    if rec['Recommended Changes']:
                        st.write("**Recommended Changes:**")
                        for change in rec['Recommended Changes']:
                            st.write(f"• {change}")
        
        # Item optimization
        st.subheader("📦 Item Optimization")
        
        item_recommendations = []
        
        for _, member in processed_team.iterrows():
            current_item = member.get('Item', 'None')
            suggestions = []
            
            # Analyze if member is taking too much special damage
            special_threats = 0
            physical_threats = 0
            
            for _, threat in threats_df.head(5).iterrows():
                threat_spa = calculate_stats_with_evs(threat['SpA'], threat.get('EV_SpA', 0))
                threat_atk = calculate_stats_with_evs(threat['Atk'], threat.get('EV_Atk', 0))
                
                if threat_spa > threat_atk:
                    special_threats += 1
                else:
                    physical_threats += 1
            
            # Item suggestions based on role and threats
            if current_item != 'Assault Vest' and special_threats >= 3:
                suggestions.append("Assault Vest - for special bulk against meta threats")
            
            if current_item != 'Rocky Helmet' and physical_threats >= 3 and member.get('EV_HP', 0) >= 200:
                suggestions.append("Rocky Helmet - punish physical attackers")
            
            if current_item not in ['Life Orb', 'Choice Specs', 'Choice Band']:
                if member.get('EV_Atk', 0) >= 200 or member.get('EV_SpA', 0) >= 200:
                    suggestions.append("Life Orb/Choice item - maximize damage output")
            
            if member['Spe'] <= 70 and current_item != 'Room Service':
                suggestions.append("Room Service/Trick Room setup - for slow team members")
            
            if suggestions:
                item_recommendations.append({
                    'Pokémon': member['Name'],
                    'Current Item': current_item,
                    'Suggestions': suggestions
                })
        
        if item_recommendations:
            for rec in item_recommendations:
                st.info(f"**{rec['Pokémon']}** (Currently: {rec['Current Item']})")
                for suggestion in rec['Suggestions']:
                    st.write(f"• {suggestion}")
        
        # Team composition suggestions
        st.subheader("🎯 Team Composition Advice")
        
        weaknesses = analyze_team_weaknesses(
            [[row['Type1'], row['Type2']] for _, row in processed_team.iterrows() if pd.notna(row['Type1'])], 
            type_chart
        )
        speed_data = analyze_speed_tiers(processed_team)
        
        suggestions = []
        
        # Type coverage suggestions
        critical_weaknesses = {k: v for k, v in weaknesses.items() if v >= 3}
        for weakness_type, count in critical_weaknesses.items():
            # Suggest types that resist this weakness
            resisters = []
            for poke_type, matchups in type_chart.items():
                if weakness_type in matchups.get('resist', []):
                    resisters.append(poke_type)
            
            if resisters:
                suggestions.append(f"Add a {'/'.join(resisters[:3])} type to cover {weakness_type} weakness ({count} members affected)")
        
        # Speed control suggestions
        speed_tiers = {}
        for pokemon_data in speed_data:
            tier = pokemon_data['Tier']
            speed_tiers[tier] = speed_tiers.get(tier, 0) + 1
        
        if speed_tiers.get('Very Fast (121+)', 0) == 0:
            suggestions.append("Consider adding a very fast Pokémon (121+ Speed) or Tailwind support")
        
        if speed_tiers.get('Trick Room (≤70)', 0) >= 3:
            suggestions.append("Your team is slow - consider Trick Room support or speed control")
        
        # Display suggestions
        if suggestions:
            for suggestion in suggestions:
                st.info(f"💡 {suggestion}")
        else:
            st.success("✅ Your team composition looks well balanced!")
        
        # Specific threat counters
        st.subheader("🛡️ Recommended Threat Counters")
        
        top_threats = threats_df.nlargest(5, 'Usage') if 'Usage' in threats_df.columns else threats_df.head(5)
        
        counter_suggestions = []
        for _, threat in top_threats.iterrows():
            # Find what's super effective against this threat
            threat_types = [threat['Type1']]
            if pd.notna(threat.get('Type2')):
                threat_types.append(threat['Type2'])
            
            effective_types = []
            for attack_type in type_chart.keys():
                effectiveness = calculate_type_effectiveness(attack_type, threat_types, type_chart)
                if effectiveness > 1.0:
                    effective_types.append(attack_type)
            
            if effective_types:
                counter_suggestions.append({
                    'Threat': f"{threat['Name']} ({threat['Type1']}{f'/{threat['Type2']}' if pd.notna(threat.get('Type2')) else ''})",
                    'Usage %': f"{threat.get('Usage', 0):.1f}%" if 'Usage' in threat else "N/A",
                    'Super Effective Types': ', '.join(effective_types[:4]),
                    'Recommended Moves': f"{effective_types[0]} moves (e.g., {get_move_example(effective_types[0])})"
                })
        
        if counter_suggestions:
            counter_df = pd.DataFrame(counter_suggestions)
            st.dataframe(counter_df, use_container_width=True)
        
        # Advanced optimization section
        st.subheader("🔬 Advanced Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Item Recommendations:**")
            item_suggestions = [
                "Assault Vest for special bulk on mixed attackers",
                "Focus Sash for frail but crucial team members",
                "Life Orb for maximum damage output",
                "Rocky Helmet for passive damage on physical walls",
                "Choice items for speed/power boost with commitment"
            ]
            for suggestion in item_suggestions:
                st.markdown(f"• {suggestion}")
        
        with col2:
            st.markdown("**Ability Optimization:**")
            ability_tips = [
                "Intimidate for physical attack reduction",
                "Weather abilities for team synergy",
                "Speed boost abilities like Chlorophyll/Swift Swim",
                "Defensive abilities like Water Absorb/Flash Fire",
                "Priority abilities like Prankster for support moves"
            ]
            for tip in ability_tips:
                st.markdown(f"• {tip}")
        
        # Export functionality
        st.subheader("📊 Export Analysis")
        
        if st.button("Generate Detailed Report"):
            # Compile all analysis data
            report_data = {
                'Team Overview': processed_team.to_dict('records'),
                'Type Weaknesses': weaknesses,
                'Speed Analysis': speed_data,
                'Optimization Tips': optimization_recommendations,
                'Suggestions': suggestions
            }
            
            # Convert to text format for download
            report_text = "=== VGC TEAM ANALYSIS REPORT ===\n\n"
            report_text += f"Team: {', '.join(processed_team['Name'].tolist())}\n\n"
            
            report_text += "CRITICAL WEAKNESSES:\n"
            for weakness, count in list(weaknesses.items())[:5]:
                report_text += f"- {count} members weak to {weakness}\n"
            
            report_text += "\nOPTIMIZATION SUGGESTIONS:\n"
            for suggestion in suggestions:
                report_text += f"- {suggestion}\n"
            
            report_text += "\nEV OPTIMIZATION DETAILS:\n"
            for rec in optimization_recommendations:
                if rec['Survivability Issues'] or rec['Recommended Changes']:
                    report_text += f"\n{rec['Pokémon']}:\n"
                    report_text += f"  Current EVs: {rec['Current EVs']}\n"
                    
                    if rec['Survivability Issues']:
                        report_text += "  Issues:\n"
                        for issue in rec['Survivability Issues']:
                            report_text += f"    - {issue}\n"
                    
                    if rec['Recommended Changes']:
                        report_text += "  Recommendations:\n"
                        for change in rec['Recommended Changes']:
                            report_text += f"    - {change}\n"
            
            st.download_button(
                label="Download Analysis Report",
                data=report_text,
                file_name=f"vgc_analysis_{'-'.join(processed_team['Name'].str.lower().str.replace(' ', '_'))}.txt",
                mime="text/plain"
            )

# ========================
# 5. RUN THE APP
# ========================

if __name__ == "__main__":
    main()
