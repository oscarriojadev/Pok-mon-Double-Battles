import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from collections import defaultdict

# Complete type effectiveness chart (same as before)
TYPE_CHART = {
    # ... (keep the existing TYPE_CHART dictionary)
}

ALL_TYPES = sorted(TYPE_CHART.keys())

# Damage calculation constants
BASE_DAMAGE_CONSTANT = 0.85
CRITICAL_HIT_MULTIPLIER = 1.5
STAB_MULTIPLIER = 1.5
WEAKNESS_MULTIPLIER = 2
RESISTANCE_MULTIPLIER = 0.5
IMMUNE_MULTIPLIER = 0

# Load data with caching (same as before)
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# ... (keep all existing helper functions like calculate_pokemon_similarity, create_radar_chart, etc.)

# New damage calculation functions
def calculate_damage(attacker, defender, move_power, move_type, is_physical, is_critical=False):
    """Calculate damage range between two Pokémon"""
    # Get relevant stats
    attack_stat = attacker['Attack'] if is_physical else attacker['Sp. Atk']
    defense_stat = defender['Defense'] if is_physical else defender['Sp. Def']
    
    # Apply nature modifiers if available
    if 'Nature' in attacker and isinstance(attacker['Nature'], str):
        nature = attacker['Nature']
        if '+' in nature and '-' in nature:
            plus_part = nature.split('+')[1].split(')')[0].split(' ')[0]
            minus_part = nature.split('-')[1].split(')')[0].split(' ')[0]
            
            if is_physical:
                if plus_part == 'Atk':
                    attack_stat *= 1.1
                elif minus_part == 'Atk':
                    attack_stat *= 0.9
            else:
                if plus_part == 'SpA':
                    attack_stat *= 1.1
                elif minus_part == 'SpA':
                    attack_stat *= 0.9
    
    # Calculate base damage
    level = 50  # Standard VGC level
    base_damage = (((2 * level / 5 + 2) * move_power * attack_stat / defense_stat) / 50 + 2)
    
    # Apply modifiers
    modifier = BASE_DAMAGE_CONSTANT
    
    # STAB check
    if move_type in [attacker['Type1'], attacker.get('Type2', '')]:
        modifier *= STAB_MULTIPLIER
    
    # Type effectiveness
    effectiveness = 1
    defender_types = [t for t in [defender['Type1'], defender.get('Type2', '')] if pd.notna(t) and t != '']
    for def_type in defender_types:
        if move_type in TYPE_CHART[def_type]['weak']:
            effectiveness *= WEAKNESS_MULTIPLIER
        elif move_type in TYPE_CHART[def_type]['resist']:
            effectiveness *= RESISTANCE_MULTIPLIER
        elif move_type in TYPE_CHART[def_type]['immune']:
            effectiveness *= IMMUNE_MULTIPLIER
    
    modifier *= effectiveness
    
    # Critical hit
    if is_critical:
        modifier *= CRITICAL_HIT_MULTIPLIER
    
    # Apply item modifiers if available
    if 'Item' in attacker and isinstance(attacker['Item'], str):
        item = attacker['Item']
        if 'Life Orb' in item:
            modifier *= 1.3
        elif 'Choice Band' in item and is_physical:
            modifier *= 1.5
        elif 'Choice Specs' in item and not is_physical:
            modifier *= 1.5
    
    # Calculate damage range
    min_damage = int(base_damage * modifier * 0.85)
    max_damage = int(base_damage * modifier)
    
    return min_damage, max_damage, effectiveness

def calculate_speed_tiers(df):
    """Calculate speed tiers for all Pokémon in the dataset"""
    speed_tiers = []
    
    for _, row in df.iterrows():
        speed = row['Speed']
        
        # Apply nature modifiers if available
        if 'Nature' in row and isinstance(row['Nature'], str):
            nature = row['Nature']
            if '+' in nature and '-' in nature:
                plus_part = nature.split('+')[1].split(')')[0].split(' ')[0]
                minus_part = nature.split('-')[1].split(')')[0].split(' ')[0]
                
                if plus_part == 'Spe':
                    speed *= 1.1
                elif minus_part == 'Spe':
                    speed *= 0.9
        
        # Apply item modifiers if available
        if 'Item' in row and isinstance(row['Item'], str):
            item = row['Item']
            if 'Choice Scarf' in item:
                speed *= 1.5
        
        speed_tiers.append({
            'Pokemon': row['Pokemon'],
            'Base Speed': row['Speed'],
            'Adjusted Speed': int(speed),
            'Team': row.get('Team', ''),
            'Item': row.get('Item', ''),
            'Nature': row.get('Nature', '')
        })
    
    return pd.DataFrame(speed_tiers).sort_values('Adjusted Speed', ascending=False)

def calculate_priority_layers(team_df):
    """Analyze priority move usage in a team"""
    priority_moves = [
        'Aqua Jet', 'Bullet Punch', 'Ice Shard', 'Mach Punch', 'Quick Attack',
        'Shadow Sneak', 'Sucker Punch', 'Extreme Speed', 'Vacuum Wave',
        'Water Shuriken', 'First Impression', 'Fake Out', 'Grassy Glide'
    ]
    
    priority_users = []
    
    for _, pokemon in team_df.iterrows():
        moves = [pokemon[f'Move {i}'] for i in range(1,5)]
        for move in moves:
            if pd.notna(move) and any(pm in str(move) for pm in priority_moves):
                priority_users.append({
                    'Pokemon': pokemon['Pokemon'],
                    'Move': move,
                    'Priority': 1  # Base priority, could be enhanced for different tiers
                })
    
    return pd.DataFrame(priority_users)

def calculate_speed_control_effectiveness(team_df):
    """Analyze speed control moves in a team"""
    speed_control_moves = {
        'Tailwind': 2,
        'Electroweb': -1,
        'Icy Wind': -1,
        'Rock Tomb': -1,
        'Bulldoze': -1,
        'Trick Room': -7,  # Special case
        'Quash': -1,
        'After You': 0,
        'Sticky Web': -1  # Hazard but affects speed
    }
    
    control_users = []
    
    for _, pokemon in team_df.iterrows():
        moves = [pokemon[f'Move {i}'] for i in range(1,5)]
        for move in moves:
            if pd.notna(move):
                for sc_move, priority in speed_control_moves.items():
                    if sc_move in str(move):
                        control_users.append({
                            'Pokemon': pokemon['Pokemon'],
                            'Move': move,
                            'Type': sc_move,
                            'Effect': f"{'+' if priority > 0 else ''}{priority} Speed",
                            'Priority': priority
                        })
    
    return pd.DataFrame(control_users)

# Main app with new tabs
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer")
    
    st.title("⚔️ Pokémon Competitive Team Analyzer")
    st.write("""
    Analyze competitive Pokémon teams with advanced damage calculations, speed tier analysis,
    and comprehensive team synergy evaluation.
    """)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Pokémon Data CSV", type=["csv"])
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()
    
    df = load_data(uploaded_file)
    
    # Ensure required columns exist (same as before)
    required_cols = ['Pokemon', 'Team', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        st.stop()
    
    # Add default columns if not present (same as before)
    # ... (keep the existing column checks)
    
    # New tabs for core mechanics
    tab_names = [
        "🧮 Damage Calculator",
        "🏃 Speed Analysis",
        "📈 Survival Calculator"
    ]
    
    tabs = st.tabs(tab_names)
    
        # New Core Mechanics tabs
    with tabs[1]:  # Damage Calculator
        st.header("🧮 Damage Calculator")
        st.write("Calculate damage ranges between Pokémon with EV/IV/Nature/Item modifiers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            attacker = st.selectbox("Attacking Pokémon", sorted(df['Pokemon'].unique()), key='damage_attacker')
            attacker_data = df[df['Pokemon'] == attacker].iloc[0]
            
            st.subheader("Attacker Details")
            st.write(f"**Type:** {attacker_data['Type1']}" + 
                     (f"/{attacker_data['Type2']}" if pd.notna(attacker_data['Type2']) else ""))
            st.write(f"**Attack:** {attacker_data['Attack']}")
            st.write(f"**Sp. Atk:** {attacker_data['Sp. Atk']}")
            st.write(f"**Nature:** {attacker_data.get('Nature', 'Unknown')}")
            st.write(f"**Item:** {attacker_data.get('Item', 'None')}")
            
            move_power = st.slider("Move Power", 10, 250, 90)
            move_type = st.selectbox("Move Type", ALL_TYPES)
            is_physical = st.checkbox("Physical Move", True)
            is_critical = st.checkbox("Critical Hit", False)
        
        with col2:
            defender = st.selectbox("Defending Pokémon", sorted(df['Pokemon'].unique()), key='damage_defender')
            defender_data = df[df['Pokemon'] == defender].iloc[0]
            
            st.subheader("Defender Details")
            st.write(f"**Type:** {defender_data['Type1']}" + 
                     (f"/{defender_data['Type2']}" if pd.notna(defender_data['Type2']) else ""))
            st.write(f"**Defense:** {defender_data['Defense']}")
            st.write(f"**Sp. Def:** {defender_data['Sp. Def']}")
            st.write(f"**HP:** {defender_data['HP']}")
            st.write(f"**Item:** {defender_data.get('Item', 'None')}")
        
        if st.button("Calculate Damage"):
            min_dmg, max_dmg, effectiveness = calculate_damage(
                attacker_data, defender_data, move_power, move_type, is_physical, is_critical
            )
            
            st.subheader("Damage Calculation Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Minimum Damage", min_dmg)
                st.metric("Maximum Damage", max_dmg)
                st.metric("Damage % of HP", f"{(max_dmg / defender_data['HP']) * 100:.1f}%")
            
            with col2:
                if effectiveness == 0:
                    st.error("No Effect (Immune)")
                elif effectiveness < 1:
                    st.warning(f"Not Very Effective (×{effectiveness})")
                elif effectiveness == 1:
                    st.info("Normal Effectiveness")
                elif effectiveness > 1:
                    st.success(f"Super Effective (×{effectiveness})")
                
                if is_critical:
                    st.info("Critical Hit! (×1.5)")
                if move_type in [attacker_data['Type1'], attacker_data.get('Type2', '')]:
                    st.info("STAB Bonus (×1.5)")
            
            with col3:
                # OHKO/2HKO analysis
                hp = defender_data['HP']
                if max_dmg >= hp:
                    st.error("Guaranteed OHKO")
                elif min_dmg >= hp:
                    st.warning("Possible OHKO (high roll)")
                elif max_dmg * 2 >= hp:
                    st.success("Guaranteed 2HKO")
                elif min_dmg * 2 >= hp:
                    st.info("Possible 2HKO (high rolls)")
                else:
                    st.info("3HKO or more")
                
                # Survival benchmark
                survival_hits = hp // max_dmg + (1 if hp % max_dmg else 0)
                st.metric("Survival Hits", survival_hits)
    
    with tabs[2]:  # Speed Analysis
        st.header("🏃 Speed Tier Analysis")
        st.write("Analyze speed tiers and priority move usage across teams")
        
        tab1, tab2, tab3 = st.tabs(["Speed Tiers", "Priority Layers", "Speed Control"])
        
        with tab1:
            st.subheader("Pokémon Speed Tiers")
            speed_df = calculate_speed_tiers(df)
            
            # Filter by team if desired
            selected_team = st.selectbox("Filter by Team", ['All Teams'] + sorted(df['Team'].unique()))
            
            if selected_team != 'All Teams':
                speed_df = speed_df[speed_df['Team'] == selected_team]
            
            st.dataframe(
                speed_df.head(50),
                column_config={
                    "Base Speed": st.column_config.NumberColumn(format="%d"),
                    "Adjusted Speed": st.column_config.NumberColumn(format="%d")
                },
                hide_index=True
            )
            
            # Visualization
            st.subheader("Speed Tier Visualization")
            fig = px.bar(
                speed_df.head(20),
                x='Pokemon',
                y='Adjusted Speed',
                color='Team',
                title="Top 20 Speed Tiers",
                hover_data=['Item', 'Nature']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Priority Move Analysis")
            selected_team = st.selectbox("Select Team for Priority Analysis", sorted(df['Team'].unique()))
            
            team_df = df[df['Team'] == selected_team]
            priority_df = calculate_priority_layers(team_df)
            
            if not priority_df.empty:
                st.write(f"**Priority Move Users in {selected_team}**")
                st.dataframe(priority_df, hide_index=True)
                
                # Priority visualization
                fig = px.bar(
                    priority_df,
                    x='Pokemon',
                    y='Priority',
                    color='Move',
                    title=f"Priority Move Layers in {selected_team}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No priority moves detected in {selected_team}")
        
        with tab3:
            st.subheader("Speed Control Effectiveness")
            selected_team = st.selectbox("Select Team for Speed Control", sorted(df['Team'].unique()))
            
            team_df = df[df['Team'] == selected_team]
            control_df = calculate_speed_control_effectiveness(team_df)
            
            if not control_df.empty:
                st.write(f"**Speed Control in {selected_team}**")
                st.dataframe(control_df, hide_index=True)
                
                # Speed control visualization
                fig = px.bar(
                    control_df,
                    x='Pokemon',
                    y='Priority',
                    color='Move',
                    title=f"Speed Control in {selected_team}",
                    labels={'Priority': 'Speed Modifier'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No speed control moves detected in {selected_team}")
    
    with tabs[3]:  # Survival Calculator
        st.header("📈 Survival Benchmark Calculator")
        st.write("Determine if your Pokémon can survive specific attacks")
        
        col1, col2 = st.columns(2)
        
        with col1:
            defender = st.selectbox("Your Pokémon", sorted(df['Pokemon'].unique()), key='survival_pokemon')
            defender_data = df[df['Pokemon'] == defender].iloc[0]
            
            st.subheader("Defender Details")
            st.write(f"**HP:** {defender_data['HP']}")
            st.write(f"**Defense:** {defender_data['Defense']}")
            st.write(f"**Sp. Def:** {defender_data['Sp. Def']}")
            st.write(f"**Type:** {defender_data['Type1']}" + 
                    (f"/{defender_data['Type2']}" if pd.notna(defender_data['Type2']) else ""))
            
            # Defender item effects
            defender_item = st.selectbox("Defender Item", [
                'None', 'Leftovers', 'Sitrus Berry', 'Assault Vest', 'Rocky Helmet', 'Other'
            ])
        
        with col2:
            move_power = st.slider("Incoming Move Power", 10, 250, 100)
            move_type = st.selectbox("Move Type", ALL_TYPES, key='survival_move_type')
            is_physical = st.checkbox("Physical Move", True, key='survival_physical')
            is_critical = st.checkbox("Critical Hit", False, key='survival_critical')
            
            # Attacker stats (simplified)
            attacker_atk = st.slider(
                "Attacker's Attack Stat", 
                50, 300, 
                value=150 if is_physical else 100,
                key='survival_atk'
            )
            
            # Simulate attacker data
            attacker_data = {
                'Attack': attacker_atk if is_physical else 0,
                'Sp. Atk': 0 if is_physical else attacker_atk,
                'Type1': move_type,  # For STAB calculation
                'Item': 'None',
                'Nature': 'Neutral'
            }
        
        if st.button("Calculate Survival"):
            min_dmg, max_dmg, _ = calculate_damage(
                attacker_data, defender_data, move_power, move_type, is_physical, is_critical
            )
            
            # Apply defender item effects
            hp = defender_data['HP']
            if defender_item == 'Leftovers':
                hp += hp // 16  # Approximate Leftovers recovery
            elif defender_item == 'Sitrus Berry':
                hp += hp // 4  # Sitrus Berry recovery
            
            st.subheader("Survival Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Minimum Damage", min_dmg)
                st.metric("Maximum Damage", max_dmg)
                st.metric("Damage % of HP", f"{(max_dmg / hp) * 100:.1f}%")
            
            with col2:
                # Survival benchmarks
                if max_dmg >= hp:
                    st.error("Guaranteed OHKO")
                elif min_dmg >= hp:
                    st.warning("Possible OHKO (high roll)")
                elif max_dmg * 2 >= hp:
                    st.success("Guaranteed 2HKO")
                elif min_dmg * 2 >= hp:
                    st.info("Possible 2HKO (high rolls)")
                else:
                    st.info("3HKO or more")
                
                # EV investment suggestion
                if max_dmg >= hp:
                    needed_hp = max_dmg - hp + 1
                    st.warning(f"Invest {needed_hp} HP EVs to survive")
                elif max_dmg * 2 >= hp:
                    needed_hp = (max_dmg * 2 - hp) // 2 + 1
                    st.info(f"Invest {needed_hp} HP EVs to survive 2 hits")

if __name__ == "__main__":
    main()
