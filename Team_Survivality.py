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

def preprocess_all_pokemon_data_for_threats(df):
    """
    Convert Dataset 1 (All Pokémon Data) to threats format for the analyzer
    Updated to match your specific dataset columns
    """
    processed = pd.DataFrame()
    
    # Basic info
    processed['Name'] = df['Name']
    processed['Type1'] = df['Typing (Primary)']
    processed['Type2'] = df['Typing (Secondary)'].replace('NA', None)  # Handle NA values
    
    # Base stats (directly use numeric values)
    processed['HP'] = df['Base Stats: HP']
    processed['Atk'] = df['Base Stats: Atk']
    processed['Def'] = df['Base Stats: Def']
    processed['SpA'] = df['Base Stats: SpA']
    processed['SpD'] = df['Base Stats: SpD']
    processed['Spe'] = df['Base Stats: Spe']
    
    # Convert Meta Usage to Usage (simplified since your data is already numeric)
    def extract_usage(usage_text):
        if pd.isna(usage_text):
            return 0.0
        try:
            return float(usage_text)  # Direct conversion
        except:
            return 0.0
    
    processed['Usage'] = df['Meta Usage (%)'].apply(extract_usage)
    
    # Create tier based on usage
    def assign_tier(row):
        usage = row['Usage']
        
        if usage >= 80:
            return 'S'
        elif usage >= 60:
            return 'A'
        elif usage >= 30:
            return 'B'
        else:
            return 'C'
    
    processed['Tier'] = processed.apply(assign_tier, axis=1)
    
    return processed

def preprocess_team_data_for_analyzer(df):
    """
    Convert Dataset 2 (Team data) to team format for the analyzer
    Works with your team dataset format
    """
    processed = pd.DataFrame()
    
    # Basic info
    processed['Name'] = df['Pokemon']
    processed['Type1'] = df['Typing (Primary)']
    processed['Type2'] = df['Typing (Secondary)'].replace('NA', None)
    
    # Base stats
    processed['HP'] = df['Base Stats: HP']
    processed['Atk'] = df['Base Stats: Atk']
    processed['Def'] = df['Base Stats: Def']
    processed['SpA'] = df['Base Stats: SpA']
    processed['SpD'] = df['Base Stats: SpD']
    processed['Spe'] = df['Base Stats: Spe']
    
    # Parse EVs from text format "252 HP/252 Atk/4 SpD"
    def parse_evs(ev_text):
        ev_dict = {'HP': 0, 'Atk': 0, 'Def': 0, 'SpA': 0, 'SpD': 0, 'Spe': 0}
        
        if pd.isna(ev_text) or ev_text == 'NA':
            return ev_dict
        
        # Split by '/' and parse each part
        parts = str(ev_text).split('/')
        for part in parts:
            part = part.strip()
            # Look for patterns like "252 HP", "4 SpD", etc.
            match = re.match(r'(\d+)\s*(\w+)', part)
            if match:
                value, stat = match.groups()
                if stat in ev_dict:
                    ev_dict[stat] = int(value)
        
        return ev_dict
    
    # Apply EV parsing to each row
    ev_data = df['EVs'].apply(parse_evs)
    
    # Extract EV values into separate columns
    processed['EV_HP'] = [ev['HP'] for ev in ev_data]
    processed['EV_Atk'] = [ev['Atk'] for ev in ev_data]
    processed['EV_Def'] = [ev['Def'] for ev in ev_data]
    processed['EV_SpA'] = [ev['SpA'] for ev in ev_data]
    processed['EV_SpD'] = [ev['SpD'] for ev in ev_data]
    processed['EV_Spe'] = [ev['Spe'] for ev in ev_data]
    
    # Items and other info
    processed['Item'] = df['Item']
    processed['Ability'] = df['Ability']
    
    # Combine moves into comma-separated string
    moves = []
    for _, row in df.iterrows():
        move_list = []
        for i in range(1, 5):  # Move 1 through Move 4
            move_col = f'Move {i}'
            if move_col in df.columns:
                move = row.get(move_col, '')
                if pd.notna(move) and str(move).strip() != 'NA':
                    move_list.append(str(move).strip())
        moves.append(','.join(move_list))
    
    processed['Moves'] = moves
    
    return processed

def detect_data_format(df):
    """
    Detect if uploaded data is in raw format or analyzer format
    Updated to match your specific column names
    """
    # Check for analyzer format columns
    analyzer_team_cols = ['Name', 'Type1', 'Type2', 'HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe', 'EV_HP']
    analyzer_threats_cols = ['Name', 'Type1', 'Type2', 'HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe', 'Usage', 'Tier']
    
    # Check for raw format columns (your specific columns)
    raw_team_cols = ['Pokemon', 'Typing (Primary)', 'Base Stats: HP', 'EVs']
    raw_threats_cols = ['Name', 'Typing (Primary)', 'Base Stats: HP', 'Meta Usage (%)']
    
    if all(col in df.columns for col in analyzer_team_cols):
        return 'analyzer_team'
    elif all(col in df.columns for col in analyzer_threats_cols):
        return 'analyzer_threats'
    elif all(col in df.columns for col in raw_team_cols):
        return 'raw_team'
    elif all(col in df.columns for col in raw_threats_cols):
        return 'raw_threats'
    else:
        return 'unknown'

def validate_processed_data(team_df, threats_df):
    """
    Validate that processed data has all required columns
    """
    required_team_columns = [
        'Name', 'Type1', 'Type2', 'HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe',
        'EV_HP', 'EV_Atk', 'EV_Def', 'EV_SpA', 'EV_SpD', 'EV_Spe',
        'Item', 'Ability', 'Moves'
    ]
    
    required_threats_columns = [
        'Name', 'Type1', 'Type2', 'HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe',
        'Usage', 'Tier'
    ]
    
    team_valid = all(col in team_df.columns for col in required_team_columns)
    threats_valid = all(col in threats_df.columns for col in required_threats_columns)
    
    return team_valid, threats_valid


# ========================
# 2. TYPE CHART AND CONSTANTS
# ========================

@st.cache_data
def load_type_chart():
    """Complete type effectiveness chart for all 18 types"""
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

# Sample datasets
SAMPLE_TEAM = [
    {
        'Name': 'Iron Hands', 'Type1': 'Fighting', 'Type2': 'Electric',
        'HP': 154, 'Atk': 140, 'Def': 108, 'SpA': 50, 'SpD': 68, 'Spe': 50,
        'EV_HP': 252, 'EV_Atk': 252, 'EV_Def': 0, 'EV_SpA': 0, 'EV_SpD': 4, 'EV_Spe': 0,
        'Item': 'Assault Vest', 'Ability': 'Quark Drive',
        'Moves': 'Drain Punch,Thunder Punch,Fake Out,Wild Charge'
    },
    {
        'Name': 'Flutter Mane', 'Type1': 'Ghost', 'Type2': 'Fairy',
        'HP': 55, 'Atk': 55, 'Def': 55, 'SpA': 135, 'SpD': 135, 'Spe': 135,
        'EV_HP': 4, 'EV_Atk': 0, 'EV_Def': 0, 'EV_SpA': 252, 'EV_SpD': 0, 'EV_Spe': 252,
        'Item': 'Life Orb', 'Ability': 'Protosynthesis',
        'Moves': 'Moonblast,Shadow Ball,Protect,Dazzling Gleam'
    },
    {
        'Name': 'Garchomp', 'Type1': 'Dragon', 'Type2': 'Ground',
        'HP': 108, 'Atk': 130, 'Def': 95, 'SpA': 80, 'SpD': 85, 'Spe': 102,
        'EV_HP': 4, 'EV_Atk': 252, 'EV_Def': 0, 'EV_SpA': 0, 'EV_SpD': 0, 'EV_Spe': 252,
        'Item': 'Rocky Helmet', 'Ability': 'Rough Skin',
        'Moves': 'Dragon Claw,Earthquake,Rock Slide,Protect'
    }
]

SAMPLE_THREATS = [
    {'Name': 'Flutter Mane', 'Type1': 'Ghost', 'Type2': 'Fairy', 'HP': 55, 'Atk': 55, 'Def': 55, 'SpA': 135, 'SpD': 135, 'Spe': 135, 'Usage': 32.5, 'Tier': 'S'},
    {'Name': 'Garchomp', 'Type1': 'Dragon', 'Type2': 'Ground', 'HP': 108, 'Atk': 130, 'Def': 95, 'SpA': 80, 'SpD': 85, 'Spe': 102, 'Usage': 18.7, 'Tier': 'A'},
    {'Name': 'Amoonguss', 'Type1': 'Grass', 'Type2': 'Poison', 'HP': 114, 'Atk': 85, 'Def': 70, 'SpA': 85, 'SpD': 80, 'Spe': 30, 'Usage': 15.2, 'Tier': 'A'},
    {'Name': 'Tornadus', 'Type1': 'Flying', 'Type2': None, 'HP': 79, 'Atk': 115, 'Def': 70, 'SpA': 125, 'SpD': 80, 'Spe': 111, 'Usage': 12.8, 'Tier': 'B'},
    {'Name': 'Dragonite', 'Type1': 'Dragon', 'Type2': 'Flying', 'HP': 91, 'Atk': 134, 'Def': 95, 'SpA': 100, 'SpD': 100, 'Spe': 80, 'Usage': 11.3, 'Tier': 'B'}
]

# ========================
# 3. CORE ANALYSIS FUNCTIONS
# ========================

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

def estimate_move_powers(attacker: Dict) -> Dict[str, List[int]]:
    """Estimate realistic move power ranges based on Pokémon's attacking stats and common VGC moves"""
    
    # Get actual attacking stats with EVs
    physical_attack = calculate_stats_with_evs(attacker['Atk'], attacker.get('EV_Atk', 0))
    special_attack = calculate_stats_with_evs(attacker['SpA'], attacker.get('EV_SpA', 0))
    
    # Determine primary attacking style
    is_physical_focused = physical_attack > special_attack
    is_mixed = abs(physical_attack - special_attack) < 20
    
    # Common VGC move power distributions
    move_powers = {
        'weak': [60, 70, 75],      # Utility moves, priority moves
        'medium': [80, 85, 90],    # Standard STAB moves
        'strong': [95, 100, 110],  # Strong STAB moves
        'powerful': [120, 130, 140] # Z-moves, choice locked moves, risky moves
    }
    
    # Estimate move arsenal based on stats and meta trends
    estimated_moves = {
        'Physical': [],
        'Special': []
    }
    
    if is_physical_focused or is_mixed:
        # Physical movesets
        estimated_moves['Physical'] = [
            ('STAB Physical (Medium)', move_powers['medium'], attacker['Type1']),
            ('STAB Physical (Strong)', move_powers['strong'], attacker['Type1']),
            ('Coverage Physical', move_powers['medium'], 'Normal'),  # Neutral coverage
        ]
        
        # Add second STAB if dual-type
        if pd.notna(attacker.get('Type2')):
            estimated_moves['Physical'].append(
                ('STAB Physical Type2', move_powers['medium'], attacker['Type2'])
            )
    
    if not is_physical_focused or is_mixed:
        # Special movesets
        estimated_moves['Special'] = [
            ('STAB Special (Medium)', move_powers['medium'], attacker['Type1']),
            ('STAB Special (Strong)', move_powers['strong'], attacker['Type1']),
            ('Coverage Special', move_powers['medium'], 'Normal'),
        ]
        
        # Add second STAB if dual-type
        if pd.notna(attacker.get('Type2')):
            estimated_moves['Special'].append(
                ('STAB Special Type2', move_powers['medium'], attacker['Type2'])
            )
    
    # Add signature/powerful moves for high-stat Pokémon
    if physical_attack >= 130 or special_attack >= 130:
        if is_physical_focused:
            estimated_moves['Physical'].append(
                ('Signature Physical', move_powers['powerful'], attacker['Type1'])
            )
        else:
            estimated_moves['Special'].append(
                ('Signature Special', move_powers['powerful'], attacker['Type1'])
            )
    
    return estimated_moves

def simulate_damage_range(attacker: Dict, defender: Dict, move_power: int, move_type: str, 
                         move_category: str, type_chart: Dict) -> Tuple[int, int, int]:
    """Calculate damage range (min, max) using Gen 9 damage formula"""
    level = 50
    
    # Calculate actual stats with EVs
    if move_category == 'Physical':
        attack_stat = calculate_stats_with_evs(attacker['Atk'], attacker.get('EV_Atk', 0))
        defense_stat = calculate_stats_with_evs(defender['Def'], defender.get('EV_Def', 0))
    else:  # Special
        attack_stat = calculate_stats_with_evs(attacker['SpA'], attacker.get('EV_SpA', 0))
        defense_stat = calculate_stats_with_evs(defender['SpD'], defender.get('EV_SpD', 0))
    
    defender_hp = calculate_hp_with_evs(defender['HP'], defender.get('EV_HP', 0))
    
    # Apply item modifiers to stats before damage calculation
    if defender.get('Item') == 'Assault Vest' and move_category == 'Special':
        defense_stat = math.floor(defense_stat * 1.5)
    
    if attacker.get('Item') == 'Choice Band' and move_category == 'Physical':
        attack_stat = math.floor(attack_stat * 1.5)
    elif attacker.get('Item') == 'Choice Specs' and move_category == 'Special':
        attack_stat = math.floor(attack_stat * 1.5)
    
    # Base damage calculation (Gen 9 formula)
    base_damage = ((2 * level / 5 + 2) * move_power * attack_stat / defense_stat) / 50 + 2
    
    # Apply modifiers
    stab = 1.5 if move_type in [attacker['Type1'], attacker.get('Type2')] else 1.0
    type_eff = calculate_type_effectiveness(move_type, [defender['Type1'], defender.get('Type2')], type_chart)
    
    # Item damage modifiers
    item_modifier = 1.0
    if attacker.get('Item') == 'Life Orb':
        item_modifier = 1.3
    elif attacker.get('Item') == 'Expert Belt' and type_eff > 1.0:
        item_modifier = 1.2
    
    # Weather and field modifiers (simplified)
    weather_modifier = 1.0
    # Could add weather detection based on team composition
    
    final_damage = base_damage * stab * type_eff * item_modifier * weather_modifier
    
    # Damage rolls (85% to 100%)
    min_damage = math.floor(final_damage * 0.85)
    max_damage = math.floor(final_damage)
    
    return min_damage, max_damage, defender_hp

def analyze_comprehensive_threats(team_df: pd.DataFrame, threats_df: pd.DataFrame, type_chart: Dict) -> pd.DataFrame:
    """Analyze threats using estimated move power ranges for realistic damage calculations"""
    results = []
    
    for _, threat in threats_df.iterrows():
        # Get estimated moveset for this threat
        estimated_moves = estimate_move_powers(threat)
        
        threat_data = {
            'Threat': threat['Name'],
            'Types': f"{threat['Type1']}{f'/{threat["Type2"]}' if pd.notna(threat.get('Type2')) else ''}",
            'Usage %': threat.get('Usage', 0),
            'Tier': threat.get('Tier', 'Unknown'),
            'OHKO Count': 0,
            '2HKO Count': 0,
            'Worst Matchup': '',
            'Max Damage %': 0,
            'Most Dangerous Move': '',
            'Speed Advantage': 0
        }
        
        most_damage = 0
        most_dangerous_move = ''
        
        # Test all estimated moves against all team members
        for category, moves in estimated_moves.items():
            if not moves:  # Skip empty categories
                continue
                
            for move_name, power_range, move_type in moves:
                # Test with maximum power from the range
                max_power = max(power_range)
                
                for _, team_member in team_df.iterrows():
                    min_dmg, max_dmg, defender_hp = simulate_damage_range(
                        threat, team_member, max_power, move_type, category, type_chart
                    )
                    
                    damage_percent = (max_dmg / defender_hp) * 100
                    
                    # Track most dangerous combination
                    if damage_percent > most_damage:
                        most_damage = damage_percent
                        threat_data['Worst Matchup'] = team_member['Name']
                        most_dangerous_move = f"{move_name} ({max_power}BP)"
                        threat_data['Max Damage %'] = damage_percent
                        threat_data['Most Dangerous Move'] = most_dangerous_move
                    
                    # Count OHKO and 2HKO potential
                    if min_dmg >= defender_hp:  # Guaranteed OHKO
                        threat_data['OHKO Count'] += 1
                    elif max_dmg >= defender_hp:  # Possible OHKO
                        threat_data['OHKO Count'] += 0.5  # Partial count for roll-dependent
                    elif max_dmg >= defender_hp * 0.5:  # 2HKO potential
                        threat_data['2HKO Count'] += 1
        
        # Speed comparison
        threat_speed = calculate_stats_with_evs(threat['Spe'], threat.get('EV_Spe', 0))
        for _, member in team_df.iterrows():
            member_speed = calculate_stats_with_evs(member['Spe'], member.get('EV_Spe', 0))
            if member_speed > threat_speed:
                threat_data['Speed Advantage'] += 1
        
        results.append(threat_data)
    
    return pd.DataFrame(results).sort_values('Max Damage %', ascending=False)

def analyze_team_weaknesses(team_df: pd.DataFrame, type_chart: Dict) -> Dict[str, int]:
    """Analyze team-wide type weaknesses"""
    weaknesses = {}
    all_types = list(type_chart.keys())
    
    for attack_type in all_types:
        weak_count = 0
        for _, pokemon in team_df.iterrows():
            types = [pokemon['Type1']]
            if pd.notna(pokemon.get('Type2')):
                types.append(pokemon['Type2'])
            
            effectiveness = calculate_type_effectiveness(attack_type, types, type_chart)
            if effectiveness > 1.0:
                weak_count += 1
        
        if weak_count > 0:
            weaknesses[attack_type] = weak_count
    
    return dict(sorted(weaknesses.items(), key=lambda x: x[1], reverse=True))

def analyze_speed_tiers(team_df: pd.DataFrame) -> Dict[str, List]:
    """Analyze team speed distribution"""
    speed_data = []
    
    for _, pokemon in team_df.iterrows():
        actual_speed = calculate_stats_with_evs(pokemon['Spe'], pokemon.get('EV_Spe', 0))
        speed_data.append({
            'Name': pokemon['Name'],
            'Speed': actual_speed,
            'Tier': get_speed_tier(actual_speed)
        })
    
    return speed_data

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

def get_move_example(move_type: str) -> str:
    """Get example move for a given type"""
    move_examples = {
        'Fire': 'Flamethrower/Heat Wave',
        'Water': 'Surf/Hydro Pump',
        'Electric': 'Thunderbolt/Thunder',
        'Grass': 'Energy Ball/Leaf Storm',
        'Ice': 'Ice Beam/Blizzard',
        'Fighting': 'Close Combat/Focus Blast',
        'Poison': 'Sludge Bomb/Poison Jab',
        'Ground': 'Earthquake/Earth Power',
        'Flying': 'Air Slash/Hurricane',
        'Psychic': 'Psychic/Psyshock',
        'Bug': 'Bug Buzz/U-turn',
        'Rock': 'Rock Slide/Stone Edge',
        'Ghost': 'Shadow Ball/Shadow Claw',
        'Dragon': 'Dragon Pulse/Dragon Claw',
        'Dark': 'Dark Pulse/Knock Off',
        'Steel': 'Flash Cannon/Iron Head',
        'Fairy': 'Moonblast/Dazzling Gleam'
    }
    return move_examples.get(move_type, f'{move_type} move')

# ========================
# 4. STREAMLIT APP
# ========================

def main():
    st.set_page_config(page_title="VGC Team Analyzer", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🎮 VGC Team Analyzer with Data Preprocessing")
    st.markdown("**Comprehensive Pokémon VGC team analysis with automatic data preprocessing and threat detection**")
    
    # Sidebar for data upload
    with st.sidebar:
        st.header("📁 Data Upload & Processing")
        
        # Team upload
        team_file = st.file_uploader(
            "Upload Your Team Data (CSV)", 
            type=["csv"],
            help="Upload either raw team data or preprocessed analyzer format"
        )
        
        # Threats upload
        threats_file = st.file_uploader(
            "Upload Pokémon Dataset for Threats (CSV)", 
            type=["csv"],
            help="Upload either raw Pokémon data or preprocessed analyzer format"
        )
        
        st.markdown("---")
        st.markdown("### 🔄 Data Processing Status")
        
        # Processing status will be shown here
        processing_status = st.empty()
        
        st.markdown("---")
        st.markdown("### 📋 Supported Data Formats")
        
        with st.expander("Raw Team Data Format"):
            st.code("""Pokemon,Typing (Primary),Typing (Secondary),Base Stats: HP,Base Stats: Atk,Base Stats: Def,Base Stats: SpA,Base Stats: SpD,Base Stats: Spe,EVs,Item,Ability,Move 1,Move 2,Move 3,Move 4
Iron Hands,Fighting,Electric,154,140,108,50,68,50,"252 HP/252 Atk/4 SpD",Assault Vest,Quark Drive,Drain Punch,Thunder Punch,Fake Out,Wild Charge""")
        
        with st.expander("Raw Threats Data Format"):
            st.code("""Name,Typing (Primary),Typing (Secondary),Base Stats: HP,Base Stats: Atk,Base Stats: Def,Base Stats: SpA,Base Stats: SpD,Base Stats: Spe,Meta Usage (%),Role(s)
Flutter Mane,Ghost,Fairy,55,55,55,135,135,135,32.5,Special Sweeper
Garchomp,Dragon,Ground,108,130,95,80,85,102,18.7,Physical Sweeper""")
        
        with st.expander("Preprocessed Team Format"):
            st.code("""Name,Type1,Type2,HP,Atk,Def,SpA,SpD,Spe,EV_HP,EV_Atk,EV_Def,EV_SpA,EV_SpD,EV_Spe,Item,Ability,Moves
Iron Hands,Fighting,Electric,154,140,108,50,68,50,252,252,0,0,4,0,Assault Vest,Quark Drive,"Drain Punch,Thunder Punch,Fake Out,Wild Charge\"""")
        
        with st.expander("Preprocessed Threats Format"):
            st.code("""Name,Type1,Type2,HP,Atk,Def,SpA,SpD,Spe,Usage,Tier
Flutter Mane,Ghost,Fairy,55,55,55,135,135,135,32.5,S
Garchomp,Dragon,Ground,108,130,95,80,85,102,18.7,A""")
    
    # Load data with preprocessing
    type_chart = load_type_chart()
    team_df = None
    threats_df = None
    
    # Process team data
    if team_file is not None:
        try:
            raw_team_df = pd.read_csv(team_file)
            team_format = detect_data_format(raw_team_df)
            
            if team_format == 'raw_team':
                processing_status.info("🔄 Processing raw team data...")
                team_df = preprocess_team_data_for_analyzer(raw_team_df)
                processing_status.success("✅ Team data processed successfully!")
            elif team_format == 'analyzer_team':
                processing_status.success("✅ Team data in correct format!")
                team_df = raw_team_df
            else:
                processing_status.error("❌ Unrecognized team data format. Please check column names.")
                st.error("Team data format not recognized. Please ensure your CSV has the required columns.")
        except Exception as e:
            processing_status.error(f"❌ Error processing team data: {str(e)}")
            st.error(f"Error processing team data: {str(e)}")
    else:
        processing_status.info("🔄 Using sample team data. Upload your own team above for personalized analysis.")
        team_df = pd.DataFrame(SAMPLE_TEAM)
    
    # Process threats data
    if threats_file is not None:
        try:
            raw_threats_df = pd.read_csv(threats_file)
            threats_format = detect_data_format(raw_threats_df)
            
            if threats_format == 'raw_threats':
                processing_status.info("🔄 Processing raw threats data...")
                threats_df = preprocess_all_pokemon_data_for_threats(raw_threats_df)
                processing_status.success("✅ Threats data processed successfully!")
            elif threats_format == 'analyzer_threats':
                processing_status.success("✅ Threats data in correct format!")
                threats_df = raw_threats_df
            else:
                processing_status.error("❌ Unrecognized threats data format. Please check column names.")
                st.error("Threats data format not recognized. Please ensure your CSV has the required columns.")
        except Exception as e:
            processing_status.error(f"❌ Error processing threats data: {str(e)}")
            st.error(f"Error processing threats data: {str(e)}")
    else:
        processing_status.info("🔄 Using sample threat data. Upload complete Pokémon dataset for comprehensive analysis.")
        threats_df = pd.DataFrame(SAMPLE_THREATS)
    
    # Validate data
    if team_df is None or len(team_df) == 0:
        st.error("❌ Team data is empty or could not be processed. Please check your CSV file.")
        st.stop()
    
    if threats_df is None or len(threats_df) == 0:
        st.error("❌ Threats data is empty or could not be processed. Please check your CSV file.")
        st.stop()
    
    # Validate processed data
    team_valid, threats_valid = validate_processed_data(team_df, threats_df)
    
    if not team_valid:
        st.error("❌ Processed team data is missing required columns.")
        st.stop()
    
    if not threats_valid:
        st.error("❌ Processed threats data is missing required columns.")
        st.stop()
    
    # Display processing summary
    with st.sidebar:
        st.markdown("### 📊 Data Summary")
        st.metric("Team Size", len(team_df))
        st.metric("Threats Dataset", len(threats_df))
        
        # Show download options for processed data
        if team_file is not None or threats_file is not None:
            st.markdown("### 💾 Download Processed Data")
            
            if team_file is not None:
                csv_team = team_df.to_csv(index=False)
                st.download_button(
                    label="Download Processed Team Data",
                    data=csv_team,
                    file_name="processed_team.csv",
                    mime="text/csv"
                )
            
            if threats_file is not None:
                csv_threats = threats_df.to_csv(index=False)
                st.download_button(
                    label="Download Processed Threats Data",
                    data=csv_threats,
                    file_name="processed_threats.csv",
                    mime="text/csv"
                )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🛡️ Team Overview", "⚔️ Threat Analysis", "🎯 Survival Calculator", "🔧 Optimization"])
    
    with tab1:  # Team Overview tab
        st.header("Team Overview & Weaknesses")
        
        # 1. Display Raw Team Data Exactly As Uploaded
        st.subheader("Your Team Composition")
        st.dataframe(team_df[[
            'Pokemon', 'Typing (Primary)', 'Typing (Secondary)',
            'Item', 'Ability', 'EVs', 'Nature',
            'Move 1', 'Move 2', 'Move 3', 'Move 4'
        ]], use_container_width=True, height=400)
    
        # 2. Calculate and Display Team Stats (using raw data)
        st.subheader("Team Stats Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_hp = team_df['Base Stats: HP'].mean()
            st.metric("Average HP", f"{avg_hp:.0f}")
        
        with col2:
            avg_speed = team_df['Base Stats: Spe'].mean()
            st.metric("Average Speed", f"{avg_speed:.0f}")
        
        with col3:
            physical_attackers = len([x for x in team_df['Role'] if 'Physical' in str(x)])
            st.metric("Physical Attackers", physical_attackers)
    
        # 3. Type Weakness Analysis (using raw data)
        st.subheader("Type Weakness Analysis")
        
        # Convert raw team data to types list for analysis
        team_types = []
        for _, row in team_df.iterrows():
            types = [row['Typing (Primary)']]
            if pd.notna(row['Typing (Secondary)']) and row['Typing (Secondary)'] != 'NA':
                types.append(row['Typing (Secondary)'])
            team_types.append(types)
        
        weaknesses = analyze_team_weaknesses(team_types, load_type_chart())
        
        if weaknesses:
            weakness_df = pd.DataFrame(list(weaknesses.items()), columns=['Type', 'Weak Members'])
            fig = px.bar(weakness_df, x='Type', y='Weak Members', 
                        title="Team Type Weaknesses",
                        color='Weak Members',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
        # 4. Speed Tier Analysis (using raw data)
        st.subheader("Speed Tier Distribution")
        
        speed_data = []
        for _, row in team_df.iterrows():
            speed_data.append({
                'Pokemon': row['Pokemon'],
                'Speed': row['Base Stats: Spe'],
                'Tier': get_speed_tier(row['Base Stats: Spe'])
            })
        
        speed_df = pd.DataFrame(speed_data)
        fig = px.scatter(speed_df, x='Pokemon', y='Speed', color='Tier',
                       title="Team Speed Distribution",
                       hover_data=['Tier'])
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Trick Room Threshold")
        fig.add_hline(y=100, line_dash="dash", line_color="orange", 
                     annotation_text="Average Speed")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Meta Threat Analysis")
        
        # Threat filtering
        col1, col2 = st.columns(2)
        with col1:
            min_usage = st.slider("Minimum Usage % to Consider", 0.0, 50.0, 5.0, 0.5)
        with col2:
            available_tiers = threats_df['Tier'].unique()
            selected_tiers = st.multiselect("Filter by Tier", available_tiers, default=list(available_tiers))
        
        # Filter threats
        filtered_threats = threats_df[
            (threats_df['Usage'] >= min_usage) & 
            (threats_df['Tier'].isin(selected_tiers) if selected_tiers else True)
        ]
        
        if len(filtered_threats) > 0:
            # Analyze matchups with comprehensive move estimation
            threat_analysis = analyze_comprehensive_threats(team_df, filtered_threats, type_chart)
            
            # Critical threats display
            ohko_threats = threat_analysis[threat_analysis['OHKO Count'] >= 1]
            if len(ohko_threats) > 0:
                st.error(f"🚨 {len(ohko_threats)} threats have OHKO potential!")
                
                # Show OHKO threats in detail
                st.dataframe(
                    ohko_threats[['Threat', 'Types', 'Usage %', 'OHKO Count', 'Most Dangerous Move', 'Worst Matchup']].round(1),
                    use_container_width=True
                )
            
            # 2HKO Analysis
            twohko_threats = threat_analysis[threat_analysis['2HKO Count'] >= 3]
            if len(twohko_threats) > 0:
                st.warning(f"⚠️ {len(twohko_threats)} threats can 2HKO most of your team")
            
            # Full analysis with enhanced columns
            st.subheader("Complete Threat Assessment")
            
            # Enhanced styling for the new columns
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
             .format({'OHKO Count': '{:.1f}', 'Max Damage %': '{:.1f}%', 'Usage %': '{:.1f}%'}))
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Enhanced threat visualization
            fig = px.scatter(threat_analysis, 
                           x='Usage %', y='Max Damage %', 
                           size='OHKO Count', 
                           color='2HKO Count',
                           hover_name='Threat',
                           hover_data=['Most Dangerous Move', 'Worst Matchup'],
                           title="Threat Danger Analysis: OHKO Potential vs Meta Relevance",
                           labels={'Max Damage %': 'Highest Damage % to Team',
                                  '2HKO Count': '2HKO Potential'},
                           color_continuous_scale='Reds')
            
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="OHKO Threshold")
            fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="2HKO Threshold")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No threats found matching your criteria. Try adjusting the filters.")
    
    with tab3:
        st.header("Advanced Survival Calculator")
        
        if len(threats_df) > 0:
            # Threat selection
            selected_threat_name = st.selectbox("Select Threat to Analyze", threats_df['Name'].unique())
            threat = threats_df[threats_df['Name'] == selected_threat_name].iloc[0]
            
            st.markdown(f"### Comprehensive Analysis vs {threat['Name']} ({threat['Type1']}{f'/{threat["Type2"]}' if pd.notna(threat.get('Type2')) else ''})")
            
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
                            for _, member in team_df.iterrows():
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
                                
                                type_eff = calculate_type_effectiveness(move_type, [member['Type1'], member.get('Type2')], type_chart)
                                
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
                            for _, member in team_df.iterrows():
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
                                
                                type_eff = calculate_type_effectiveness(move_type, [member['Type1'], member.get('Type2')], type_chart)
                                
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
                    for _, member in team_df.iterrows():
                        member_results = [r for r in all_results if r['Pokémon'] == member['Name']]
                        
                        # Find move that deals most damage
                        max_damage_result = max(member_results, 
                                              key=lambda x: float(x['Damage %'].split('-')[1].rstrip('%')))
                        
                        summary_results.append({
                            'Your Pokémon': member['Name'],
                            'Types': f"{member['Type1']}{f'/{member["Type2"]}' if pd.notna(member.get('Type2')) else ''}",
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
                               title=f"Worst-Case Damage from {threat['Name']}",
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
                        st.metric("OHKO Potential", f"{ohko_count}/6", delta=f"{ohko_count} members")
                    with col2:
                        st.metric("2HKO Potential", f"{twohko_count}/6", delta=f"{twohko_count} members")  
                    with col3:
                        safe_count = 6 - ohko_count - twohko_count
                        st.metric("Safe Members", f"{safe_count}/6", delta=f"{safe_count} members")
        else:
            st.error("No threat data available for survival calculation.")
    
    with tab4:
        st.header("Team Optimization Recommendations")
        
        # Advanced EV optimization with move power estimation
        st.subheader("🔧 Advanced EV Optimization")
        
        if len(threats_df) > 0:
            st.markdown("**Survival-Based EV Recommendations:**")
            
            # Select top threats for optimization
            top_threats = threats_df.nlargest(3, 'Usage')
            
            optimization_recommendations = []
            
            for _, member in team_df.iterrows():
                member_recs = {
                    'Pokémon': member['Name'],
                    'Current EVs': f"HP:{member.get('EV_HP', 0)} Atk:{member.get('EV_Atk', 0)} Def:{member.get('EV_Def', 0)} SpA:{member.get('EV_SpA', 0)} SpD:{member.get('EV_SpD', 0)} Spe:{member.get('EV_Spe', 0)}",
                    'Survivability Issues': [],
                    'Recommended Changes': []
                }
                
                for _, threat in top_threats.iterrows():
                    # Get threat's estimated strongest move
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
        
        # Item optimization based on estimated damage
        st.subheader("📦 Item Optimization")
        
        item_recommendations = []
        
        for _, member in team_df.iterrows():
            current_item = member.get('Item', 'None')
            suggestions = []
            
            # Analyze if member is taking too much special damage
            special_threats = 0
            physical_threats = 0
            
            if len(threats_df) > 0:
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
        
        weaknesses = analyze_team_weaknesses(team_df, type_chart)
        speed_data = analyze_speed_tiers(team_df)
        
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
        
        if len(threats_df) > 0:
            top_threats = threats_df.nlargest(5, 'Usage')
            
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
                        'Threat': f"{threat['Name']} ({threat['Type1']}{f'/{threat["Type2"]}' if pd.notna(threat.get("Type2")) else ''})",
                        'Usage %': f"{threat.get('Usage', 0):.1f}%",
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
                'Team Overview': team_df.to_dict('records'),
                'Type Weaknesses': weaknesses,
                'Speed Analysis': speed_data,
                'Optimization Tips': optimization_recommendations,
                'Suggestions': suggestions
            }
            
            # Convert to text format for download
            report_text = "=== VGC TEAM ANALYSIS REPORT ===\n\n"
            report_text += f"Team: {', '.join(team_df['Name'].tolist())}\n\n"
            
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
                file_name=f"vgc_analysis_{'-'.join(team_df['Name'].str.lower().str.replace(' ', '_'))}.txt",
                mime="text/plain"
            )

# ========================
# 5. RUN THE APP
# ========================

if __name__ == "__main__":
    main()
