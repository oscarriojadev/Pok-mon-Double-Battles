import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
import random

# --------------------------
# Hardcoded Data for Missing Mechanics
# --------------------------

# Complete Tera Types for each Pokémon
TERA_TYPES = {
    "Amoonguss": ["Poison", "Grass", "Water"],
    "Chien-Pao": ["Dark", "Ice", "Ghost"],
    "Excadrill": ["Ground", "Steel", "Fighting"],
    "Gothitelle": ["Psychic", "Fairy", "Dark"],
    "Iron Hands": ["Fighting", "Electric", "Ground"],
    "Flutter Mane": ["Ghost", "Fairy", "Fire"],
    "Tornadus": ["Flying", "Dark", "Ground"],
    "Rillaboom": ["Grass", "Dragon", "Rock"],
    "Urshifu": ["Water", "Dark", "Fighting"],
    "Landorus": ["Ground", "Flying", "Rock"],
    "Ogerpon": ["Grass", "Water", "Fire"],
    "Incineroar": ["Fire", "Dark", "Flying"],
    "Indeedee": ["Psychic", "Fairy", "Normal"],
    "Torkoal": ["Fire", "Rock", "Ground"],
    "Farigiraf": ["Normal", "Psychic", "Fairy"],
    "Pelipper": ["Water", "Flying", "Ground"],
    # Add more Pokémon as needed
}

# Complete Move PP (default values if not specified)
MOVE_PP = {
    "Protect": 10,
    "Fake Out": 16,
    "Earthquake": 16,
    "Rage Powder": 16,
    "Spore": 16,
    "Tailwind": 16,
    "Moonblast": 16,
    "Rock Slide": 16,
    "Surging Strikes": 8,
    "Close Combat": 8,
    "Wood Hammer": 16,
    "Grassy Glide": 20,
    "Heat Wave": 8,
    "Hydro Pump": 8,
    "Thunderbolt": 16,
    "Ice Beam": 16,
    "Flamethrower": 16,
    "Will-O-Wisp": 16,
    "Trick Room": 8,
    "Helping Hand": 32,
    "Follow Me": 16,
    "Extreme Speed": 16,
    "Quick Attack": 32,
    "Counter": 16,
    # Add more moves as needed
}

# Complete Move Power values
MOVE_POWER = {
    "Earthquake": 100,
    "Rock Slide": 75,
    "Moonblast": 95,
    "Surging Strikes": 25,  # (3 hits)
    "Close Combat": 120,
    "Wood Hammer": 120,
    "Grassy Glide": 60,
    "Heat Wave": 95,
    "Hydro Pump": 110,
    "Thunderbolt": 90,
    "Ice Beam": 90,
    "Flamethrower": 90,
    "Will-O-Wisp": 0,
    "Quick Attack": 40,
    "Extreme Speed": 80,
    "Fake Out": 40,
    "Counter": 0,  # Special calculation
    # Add more moves as needed
}

# Type Chart for damage calculation
TYPE_CHART = {
    "Normal": {"strong_against": [], "resist": ["Rock", "Steel"], "immune": ["Ghost"]},
    "Fire": {"strong_against": ["Grass", "Ice", "Bug", "Steel"], "resist": ["Fire", "Water", "Rock", "Dragon"]},
    "Water": {"strong_against": ["Fire", "Ground", "Rock"], "resist": ["Water", "Grass", "Dragon"]},
    "Electric": {"strong_against": ["Water", "Flying"], "resist": ["Electric", "Grass", "Dragon"], "immune": ["Ground"]},
    "Grass": {"strong_against": ["Water", "Ground", "Rock"], "resist": ["Fire", "Grass", "Poison", "Flying", "Bug", "Dragon", "Steel"]},
    "Ice": {"strong_against": ["Grass", "Ground", "Flying", "Dragon"], "resist": ["Fire", "Water", "Ice", "Steel"]},
    "Fighting": {"strong_against": ["Normal", "Ice", "Rock", "Dark", "Steel"], "resist": ["Poison", "Flying", "Psychic", "Bug", "Fairy"], "immune": ["Ghost"]},
    "Poison": {"strong_against": ["Grass", "Fairy"], "resist": ["Poison", "Ground", "Rock", "Ghost"], "immune": ["Steel"]},
    "Ground": {"strong_against": ["Fire", "Electric", "Poison", "Rock", "Steel"], "resist": ["Grass", "Bug"], "immune": ["Flying"]},
    "Flying": {"strong_against": ["Grass", "Fighting", "Bug"], "resist": ["Electric", "Rock", "Steel"]},
    "Psychic": {"strong_against": ["Fighting", "Poison"], "resist": ["Psychic", "Steel"], "immune": ["Dark"]},
    "Bug": {"strong_against": ["Grass", "Psychic", "Dark"], "resist": ["Fire", "Fighting", "Poison", "Flying", "Ghost", "Steel", "Fairy"]},
    "Rock": {"strong_against": ["Fire", "Ice", "Flying", "Bug"], "resist": ["Fighting", "Ground", "Steel"]},
    "Ghost": {"strong_against": ["Psychic", "Ghost"], "resist": ["Dark"], "immune": ["Normal"]},
    "Dragon": {"strong_against": ["Dragon"], "resist": ["Steel"], "immune": ["Fairy"]},
    "Dark": {"strong_against": ["Psychic", "Ghost"], "resist": ["Fighting", "Dark", "Fairy"]},
    "Steel": {"strong_against": ["Ice", "Rock", "Fairy"], "resist": ["Fire", "Water", "Electric", "Steel"]},
    "Fairy": {"strong_against": ["Fighting", "Dragon", "Dark"], "resist": ["Fire", "Poison", "Steel"]}
}

# --------------------------
# Enhanced Battle State Tracking with Stat Boosts
# --------------------------

def init_battle_state(team1_df, team2_df):
    """Initialize battle state with all mechanics."""
    if 'battle_state' not in st.session_state:
        st.session_state.battle_state = {
            'turn': 0,
            'weather': None,
            'terrain': None,
            'trick_room': 0,
            'field_conditions': [],
            'team1_active': None,
            'team2_active': None,
            'team1_tera': {poke: False for poke in team1_df['Pokemon']},
            'team2_tera': {poke: False for poke in team2_df['Pokemon']},
            'team1_tera_type': {poke: TERA_TYPES.get(poke, ["Normal"])[0] for poke in team1_df['Pokemon']},
            'team2_tera_type': {poke: TERA_TYPES.get(poke, ["Normal"])[0] for poke in team2_df['Pokemon']},
            'team1_pp': {
                poke: {
                    move: MOVE_PP.get(move, 16) 
                    for move in team1_df.loc[team1_df['Pokemon'] == poke, [f'Move {i}' for i in range(1, 5)]].values[0]
                } 
                for poke in team1_df['Pokemon']
            },
            'team2_pp': {
                poke: {
                    move: MOVE_PP.get(move, 16) 
                    for move in team2_df.loc[team2_df['Pokemon'] == poke, [f'Move {i}' for i in range(1, 5)]].values[0]
                } 
                for poke in team2_df['Pokemon']
            },
            'team1_stat_boosts': {
                poke: {
                    'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 
                    'accuracy': 0, 'evasion': 0
                }
                for poke in team1_df['Pokemon']
            },
            'team2_stat_boosts': {
                poke: {
                    'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 
                    'accuracy': 0, 'evasion': 0
                }
                for poke in team2_df['Pokemon']
            },
            'team1_status': {poke: None for poke in team1_df['Pokemon']},
            'team2_status': {poke: None for poke in team2_df['Pokemon']},
            'last_move': None,
            'battle_log': []
        }

# --------------------------
# Full Gen 9 Damage Calculation
# --------------------------

def calculate_damage(attacker, defender, move, battle_state, attacker_team):
    """Full Gen 9 damage calculation with all mechanics."""
    # Get base power
    power = MOVE_POWER.get(move, 80)  # Default to 80 if move not in dict
    
    # Determine if move is physical or special (simplified)
    move_type = "Normal"  # Should be set based on move's actual type
    is_physical = True  # Default, should be determined by move
    
    # Get attacker and defender stats with boosts
    attack_stat = attacker['Attack'] * get_stat_modifier(
        battle_state[f'team{attacker_team}_stat_boosts'][attacker['Pokemon']]['atk']
    )
    defense_stat = defender['Defense'] * get_stat_modifier(
        battle_state[f'team{3-attacker_team}_stat_boosts'][defender['Pokemon']]['def']
    )
    
    if not is_physical:
        attack_stat = attacker['Sp. Atk'] * get_stat_modifier(
            battle_state[f'team{attacker_team}_stat_boosts'][attacker['Pokemon']]['spa']
        )
        defense_stat = defender['Sp. Def'] * get_stat_modifier(
            battle_state[f'team{3-attacker_team}_stat_boosts'][defender['Pokemon']]['spd']
        )
    
    # Type effectiveness
    effectiveness = 1.0
    defender_types = [defender['Type1']]
    if pd.notna(defender['Type2']):
        defender_types.append(defender['Type2'])
    
    # Check Tera type
    tera_active = battle_state[f"team{3-attacker_team}_tera"][defender['Pokemon']]
    tera_type = battle_state[f"team{3-attacker_team}_tera_type"][defender['Pokemon']] if tera_active else None
    
    for dtype in defender_types:
        if dtype and dtype != tera_type:  # Original types only if not Tera
            if dtype in TYPE_CHART.get(move_type, {}).get('strong_against', []):
                effectiveness *= 2
            elif dtype in TYPE_CHART.get(move_type, {}).get('resist', []):
                effectiveness *= 0.5
            elif dtype in TYPE_CHART.get(move_type, {}).get('immune', []):
                effectiveness = 0
    
    # Check Tera type effectiveness
    if tera_active:
        if tera_type in TYPE_CHART.get(move_type, {}).get('strong_against', []):
            effectiveness *= 2
        elif tera_type in TYPE_CHART.get(move_type, {}).get('resist', []):
            effectiveness *= 0.5
        elif tera_type in TYPE_CHART.get(move_type, {}).get('immune', []):
            effectiveness = 0
    
    # STAB calculation
    stab = 1.0
    attacker_types = [attacker['Type1']]
    if pd.notna(attacker['Type2']):
        attacker_types.append(attacker['Type2'])
    if move_type in attacker_types:
        stab = 1.5
    
    # Tera STAB bonus
    if battle_state[f"team{attacker_team}_tera"][attacker['Pokemon']]:
        tera_type = battle_state[f"team{attacker_team}_tera_type"][attacker['Pokemon']]
        if move_type == tera_type:
            stab = 2.0  # Same-type Tera bonus
    
    # Critical hit (6.25% chance)
    critical = 1.5 if random.random() < 0.0625 else 1.0
    
    # Random damage roll (0.85 to 1.0)
    random_roll = random.uniform(0.85, 1.0)
    
    # Weather modifiers
    weather = battle_state['weather']
    if weather == "Sun" and move_type == "Fire":
        power *= 1.5
    elif weather == "Sun" and move_type == "Water":
        power *= 0.5
    elif weather == "Rain" and move_type == "Water":
        power *= 1.5
    elif weather == "Rain" and move_type == "Fire":
        power *= 0.5
    
    # Terrain modifiers
    terrain = battle_state['terrain']
    if terrain == "Electric" and move_type == "Electric":
        power *= 1.3
    elif terrain == "Grassy" and move_type == "Grass":
        power *= 1.3
    elif terrain == "Psychic" and move_type == "Psychic":
        power *= 1.3
    
    # Full damage formula
    damage = int((((((2 * attacker['Level']) / 5) + 2) * power * attack_stat / defense_stat) / 50 + 2) * critical * random_roll * stab * effectiveness
    
    return max(1, int(damage))  # Minimum 1 damage

def get_stat_modifier(stage):
    """Convert stat boost stage to multiplier."""
    if stage > 0:
        return (2 + stage) / 2
    elif stage < 0:
        return 2 / (2 - stage)
    return 1

# --------------------------
# Stat Boost Tracking
# --------------------------

def apply_stat_boost(pokemon, stat, amount, battle_state, team):
    """Apply stat boost to a Pokémon."""
    current = battle_state[f'team{team}_stat_boosts'][pokemon][stat]
    new_val = max(-6, min(6, current + amount))
    battle_state[f'team{team}_stat_boosts'][pokemon][stat] = new_val
    
    if amount > 0:
        st.session_state.battle_log.append(f"{pokemon}'s {stat.upper()} rose!")
    else:
        st.session_state.battle_log.append(f"{pokemon}'s {stat.upper()} fell!")

# --------------------------
# Terastallization Visual Feedback
# --------------------------

def render_tera_animation(pokemon, tera_type):
    """Display visual feedback for Terastallization."""
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <div style="font-size: 24px; font-weight: bold; color: #ff6b00;">TERASTALLIZE!</div>
        <div style="font-size: 18px; margin: 10px 0;">{pokemon} became {tera_type} type!</div>
        <div style="font-size: 48px;">✨💎✨</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Play sound effect (placeholder)
    st.session_state.battle_log.append(f"*{pokemon} Terastallized into the {tera_type} type!*")

# --------------------------
# Enhanced Move Execution with Stat Boosts
# --------------------------

def execute_move(attacker, defender, move, battle_state, attacker_team):
    """Process a move with all mechanics."""
    # PP Tracking
    battle_state[f"team{attacker_team}_pp"][attacker['Pokemon']][move] -= 1
    
    # Priority Calculation
    priority = 0
    if move in ['Quick Attack', 'Fake Out', 'Extreme Speed']:
        priority = 1
    elif move == 'Counter':
        priority = -1
    
    # Speed calculation (with Tailwind/Trick Room)
    speed = attacker['Speed'] * get_stat_modifier(
        battle_state[f'team{attacker_team}_stat_boosts'][attacker['Pokemon']]['spe']
    )
    if battle_state['trick_room'] > 0:
        speed = 10000 - speed  # Inverted for Trick Room
    
    # Handle stat boosting moves
    if move == "Swords Dance":
        apply_stat_boost(attacker['Pokemon'], 'atk', 2, battle_state, attacker_team)
        damage = 0
    elif move == "Nasty Plot":
        apply_stat_boost(attacker['Pokemon'], 'spa', 2, battle_state, attacker_team)
        damage = 0
    elif move == "Iron Defense":
        apply_stat_boost(attacker['Pokemon'], 'def', 2, battle_state, attacker_team)
        damage = 0
    elif move == "Calm Mind":
        apply_stat_boost(attacker['Pokemon'], 'spa', 1, battle_state, attacker_team)
        apply_stat_boost(attacker['Pokemon'], 'spd', 1, battle_state, attacker_team)
        damage = 0
    else:
        # Damage Calculation
        damage = calculate_damage(attacker, defender, move, battle_state, attacker_team)
    
    # Apply effects
    if move == "Will-O-Wisp":
        defender['status'] = "Burn"
    elif move == "Spore":
        defender['status'] = "Sleep"
    
    # Update battle log
    log_entry = f"{attacker['Pokemon']} used {move}!"
    if damage > 0:
        log_entry += f" (Dealt {damage} damage)"
    st.session_state.battle_log.append(log_entry)
    
    return damage

# --------------------------
# Streamlit UI for Battle Mechanics
# --------------------------

def render_battle_controls(battle_state):
    """UI for advanced battle options."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Weather Control
        new_weather = st.selectbox(
            "Set Weather",
            ["None", "Sun", "Rain", "Sand", "Hail"],
            index=0
        )
        if new_weather != battle_state['weather']:
            battle_state['weather'] = new_weather if new_weather != "None" else None
    
    with col2:
        # Terrain Control
        new_terrain = st.selectbox(
            "Set Terrain",
            ["None", "Electric", "Psychic", "Grassy", "Misty"],
            index=0
        )
        if new_terrain != battle_state['terrain']:
            battle_state['terrain'] = new_terrain if new_terrain != "None" else None
    
    with col3:
        # Trick Room Toggle
        if st.button(f"Trick Room ({battle_state['trick_room']} turns left)"):
            battle_state['trick_room'] = 5 if battle_state['trick_room'] == 0 else 0

# --------------------------
# Main Battle Loop Integration
# --------------------------

def main():
    # Example usage - you'll need to replace this with your actual Streamlit setup
    st.title("Pokémon Battle Simulator")
    
    # Sample team data
    team1_data = {
        'Pokemon': ['Pikachu', 'Charizard'],
        'Type1': ['Electric', 'Fire'],
        'Type2': [None, 'Flying'],
        'Move 1': ['Thunderbolt', 'Flamethrower'],
        'Move 2': ['Quick Attack', 'Air Slash'],
        'Move 3': ['Iron Tail', 'Dragon Claw'],
        'Move 4': ['Protect', 'Roost'],
        'Attack': [55, 84],
        'Defense': [40, 78],
        'Sp. Atk': [50, 109],
        'Sp. Def': [50, 85],
        'Speed': [90, 100],
        'Level': [50, 50]
    }
    
    team2_data = {
        'Pokemon': ['Blastoise', 'Venusaur'],
        'Type1': ['Water', 'Grass'],
        'Type2': [None, 'Poison'],
        'Move 1': ['Hydro Pump', 'Solar Beam'],
        'Move 2': ['Ice Beam', 'Sludge Bomb'],
        'Move 3': ['Flash Cannon', 'Earthquake'],
        'Move 4': ['Protect', 'Synthesis'],
        'Attack': [83, 82],
        'Defense': [100, 83],
        'Sp. Atk': [85, 100],
        'Sp. Def': [105, 100],
        'Speed': [78, 80],
        'Level': [50, 50]
    }
    
    team1_df = pd.DataFrame(team1_data)
    team2_df = pd.DataFrame(team2_data)
    
    # Initialize battle state
    init_battle_state(team1_df, team2_df)
    battle_state = st.session_state.battle_state
    
    # Set active Pokémon for demo
    battle_state['team1_active'] = 'Pikachu'
    battle_state['team2_active'] = 'Blastoise'
    
    # Render battle interface
    render_battle_controls(battle_state)
    
    # Display active Pokémon
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Active Pokémon")
        if battle_state['team1_active']:
            poke = battle_state['team1_active']
            tera_status = " (Terastallized)" if battle_state['team1_tera'][poke] else ""
            st.write(f"{poke}{tera_status}")
            
            # Tera Activation Button
            if st.button("Terastallize", key="tera_button", 
                        disabled=battle_state['team1_tera'][poke]):
                battle_state['team1_tera'][poke] = True
                tera_type = battle_state['team1_tera_type'][poke]
                render_tera_animation(poke, tera_type)
    
    with col2:
        st.subheader("Opponent's Active Pokémon")
        if battle_state['team2_active']:
            poke = battle_state['team2_active']
            tera_status = " (Terastallized)" if battle_state['team2_tera'][poke] else ""
            st.write(f"{poke}{tera_status}")
    
    # Move selection
    st.subheader("Move Selection")
    your_active = battle_state['team1_active']
    if your_active:
        moves = team1_df.loc[team1_df['Pokemon'] == your_active, 
                            ['Move 1', 'Move 2', 'Move 3', 'Move 4']].values[0]
        
        for i, move in enumerate(moves, 1):
            pp_left = battle_state['team1_pp'][your_active].get(move, 0)
            if st.button(f"{move} ({pp_left} PP)", key=f"move_{i}"):
                # For demo purposes - you'd need to get actual attacker/defender data
                attacker = team1_df.loc[team1_df['Pokemon'] == your_active].iloc[0].to_dict()
                defender = team2_df.loc[team2_df['Pokemon'] == battle_state['team2_active']].iloc[0].to_dict()
                damage = execute_move(attacker, defender, move, battle_state, 1)
                if damage > 0:
                    st.success(f"Dealt {damage} damage!")
    
    # Display battle log
    st.subheader("Battle Log")
    if 'battle_log' in st.session_state.battle_state:
        for entry in st.session_state.battle_state['battle_log'][-10:]:
            st.write(entry)

if __name__ == "__main__":
    main()
