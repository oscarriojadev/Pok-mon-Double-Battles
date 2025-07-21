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

def analyze_threat_matrix(team1_df, team2_df):
    """Create a threat assessment matrix between two teams"""
    threat_matrix = pd.DataFrame(index=team1_df['Pokemon'], columns=team2_df['Pokemon'])
    
    for _, pokemon1 in team1_df.iterrows():
        for _, pokemon2 in team2_df.iterrows():
            # Calculate type advantage
            type1 = pokemon1['Type1']
            type2_1 = pokemon2['Type1']
            type2_2 = pokemon2['Type2'] if pd.notna(pokemon2['Type2']) else None
            
            # Get strengths against opponent's types
            strengths = TYPE_CHART[type1]['strong_against']
            advantage = 0
            if type2_1 in strengths:
                advantage += 1
            if type2_2 and type2_2 in strengths:
                advantage += 1
                
            # Get weaknesses from opponent's types
            weaknesses = []
            if type2_1 in TYPE_CHART:
                weaknesses.extend(TYPE_CHART[type2_1]['strong_against'])
            if type2_2 and type2_2 in TYPE_CHART:
                weaknesses.extend(TYPE_CHART[type2_2]['strong_against'])
            
            disadvantage = 0
            if type1 in weaknesses:
                disadvantage += weaknesses.count(type1)
                
            # Calculate threat score (positive is advantage, negative is disadvantage)
            threat_score = advantage - disadvantage
            threat_matrix.at[pokemon1['Pokemon'], pokemon2['Pokemon']] = threat_score
    
    return threat_matrix

def find_counterplay_windows(team1_df, team2_df):
    """Identify potential counterplay opportunities between teams"""
    windows = []
    
    for _, pokemon1 in team1_df.iterrows():
        for _, pokemon2 in team2_df.iterrows():
            # Check speed tiers
            speed_diff = pokemon1['Speed'] - pokemon2['Speed']
            
            # Check priority moves
            priority_moves1 = [move for move in [pokemon1[f'Move {i}'] for i in range(1,5) 
                             if move in ['Quick Attack', 'Extreme Speed', 'Aqua Jet', 'Bullet Punch', 'Ice Shard', 'Mach Punch', 'Vacuum Wave', 'Sucker Punch']]
            priority_moves2 = [move for move in [pokemon2[f'Move {i}'] for i in range(1,5) 
                             if move in ['Quick Attack', 'Extreme Speed', 'Aqua Jet', 'Bullet Punch', 'Ice Shard', 'Mach Punch', 'Vacuum Wave', 'Sucker Punch']]
            
            # Check setup opportunities
            setup_moves1 = [move for move in [pokemon1[f'Move {i}'] for i in range(1,5)] 
                          if move in ['Swords Dance', 'Nasty Plot', 'Bulk Up', 'Calm Mind', 'Dragon Dance']]
            setup_moves2 = [move for move in [pokemon2[f'Move {i}'] for i in range(1,5)] 
                          if move in ['Swords Dance', 'Nasty Plot', 'Bulk Up', 'Calm Mind', 'Dragon Dance']]
            
            # Check defensive options
            defensive_moves1 = [move for move in [pokemon1[f'Move {i}'] for i in range(1,5)] 
                             if move in ['Protect', 'Detect', 'King\'s Shield', 'Spiky Shield']]
            defensive_moves2 = [move for move in [pokemon2[f'Move {i}'] for i in range(1,5)] 
                             if move in ['Protect', 'Detect', 'King\'s Shield', 'Spiky Shield']]
            
            # Generate recommendations
            recommendations = []
            
            if speed_diff > 0 and not priority_moves2:
                recommendations.append(f"{pokemon1['Pokemon']} can outspeed {pokemon2['Pokemon']}")
            elif speed_diff < 0 and priority_moves1:
                recommendations.append(f"{pokemon1['Pokemon']} has priority moves to compensate for lower speed")
                
            if setup_moves1 and not setup_moves2:
                recommendations.append(f"{pokemon1['Pokemon']} can setup against {pokemon2['Pokemon']}")
                
            if defensive_moves1:
                recommendations.append(f"{pokemon1['Pokemon']} has defensive options against {pokemon2['Pokemon']}")
                
            if recommendations:
                windows.append({
                    'attacker': pokemon1['Pokemon'],
                    'defender': pokemon2['Pokemon'],
                    'recommendations': recommendations
                })
    
    return windows

def track_battle_state():
    """Initialize or update the battle state tracker"""
    if 'battle_state' not in st.session_state:
        st.session_state.battle_state = {
            'turn': 0,
            'team1_active': None,
            'team2_active': None,
            'team1_status': {},
            'team2_status': {},
            'field_conditions': [],
            'team1_remaining': [],
            'team2_remaining': []
        }
    return st.session_state.battle_state

def update_battle_state(battle_state, updates):
    """Update the battle state with new information"""
    for key, value in updates.items():
        battle_state[key] = value
    st.session_state.battle_state = battle_state
    return battle_state

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Battle Analyzer")
    
    st.title("⚔️ Pokémon Competitive Battle Analyzer")
    st.write("""
    Advanced tools for analyzing Pokémon battles with real-time tracking and threat assessment.
    """)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Pokémon Data CSV", type=["csv"])
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()
    
    df = load_data(uploaded_file)
    
    # Ensure required columns exist
    required_cols = ['Pokemon', 'Team', 'Type1', 'Type2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed',
                    'Move 1', 'Move 2', 'Move 3', 'Move 4', 'Item', 'Ability']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        st.stop()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Advanced Matchup Analysis", 
        "⚠️ Threat Assessment Matrix",
        "📊 Live Battle Tracker"
    ])
    
    with tab1:
        st.header("Advanced Matchup Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Your Team", sorted(df['Team'].unique()), key='team1_analysis')
            team1_df = df[df['Team'] == team1]
            
        with col2:
            team2 = st.selectbox("Select Opponent Team", sorted(df['Team'].unique()), key='team2_analysis')
            team2_df = df[df['Team'] == team2]
        
        if team1 and team2:
            # Automated threat identification
            st.subheader("Automated Threat Identification")
            
            # Get team types
            team1_types = []
            for _, row in team1_df.iterrows():
                team1_types.append(row['Type1'])
                if pd.notna(row['Type2']) and row['Type2'] != '':
                    team1_types.append(row['Type2'])
            
            team2_types = []
            for _, row in team2_df.iterrows():
                team2_types.append(row['Type1'])
                if pd.notna(row['Type2']) and row['Type2'] != '':
                    team2_types.append(row['Type2'])
            
            # Calculate coverage
            team1_coverage = calculate_team_coverage(team1_types)
            team2_coverage = calculate_team_coverage(team2_types)
            
            # Find threats (team1's strengths vs team2's weaknesses)
            threats = []
            for t in team1_coverage['offensive_coverage']:
                if t in team2_coverage['uncovered_weaknesses']:
                    threats.append(f"Your {t} attacks exploit opponent's {t} weakness")
            
            if threats:
                st.success("Identified Threats:")
                for threat in threats:
                    st.write(f"✅ {threat}")
            else:
                st.warning("No clear type advantages identified")
            
            # Find vulnerabilities (team2's strengths vs team1's weaknesses)
            vulnerabilities = []
            for t in team2_coverage['offensive_coverage']:
                if t in team1_coverage['uncovered_weaknesses']:
                    vulnerabilities.append(f"Opponent's {t} attacks exploit your {t} weakness")
            
            if vulnerabilities:
                st.error("Identified Vulnerabilities:")
                for vuln in vulnerabilities:
                    st.write(f"⚠️ {vuln}")
            
            st.divider()
            
            # Counterplay window finder
            st.subheader("Counterplay Opportunities")
            windows = find_counterplay_windows(team1_df, team2_df)
            
            if windows:
                st.write("Potential counterplay scenarios:")
                for window in windows[:5]:  # Show top 5
                    st.write(f"**{window['attacker']} vs {window['defender']}**")
                    for rec in window['recommendations']:
                        st.write(f"- {rec}")
            else:
                st.info("No clear counterplay opportunities identified")
            
            st.divider()
            
            # Playstyle recommendations
            st.subheader("Playstyle Recommendations")
            
            # Analyze team speed tiers
            team1_avg_speed = team1_df['Speed'].mean()
            team2_avg_speed = team2_df['Speed'].mean()
            
            if team1_avg_speed > team2_avg_speed + 10:
                st.success("**Offensive Recommendation:** Your team is faster on average - consider aggressive play")
            elif team1_avg_speed < team2_avg_speed - 10:
                st.warning("**Defensive Recommendation:** Your team is slower on average - consider defensive plays and priority moves")
            else:
                st.info("Teams are similarly paced - focus on strategic matchups")
            
            # Check for setup opportunities
            setup_users = []
            for _, pokemon in team1_df.iterrows():
                moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                if any(move in ['Swords Dance', 'Nasty Plot', 'Bulk Up', 'Calm Mind', 'Dragon Dance'] for move in moves):
                    setup_users.append(pokemon['Pokemon'])
            
            if setup_users:
                st.success(f"**Setup Potential:** These Pokémon can boost stats: {', '.join(setup_users)}")
            
            # Check for stall potential
            defensive_pokemon = team1_df[(team1_df['HP'] > 80) & (team1_df['Defense'] > 80) & (team1_df['Sp. Def'] > 80)]
            if not defensive_pokemon.empty:
                st.info(f"**Stall Potential:** These Pokémon can take hits: {', '.join(defensive_pokemon['Pokemon'])}")
    
    with tab2:
        st.header("Threat Assessment Matrix")
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select Your Team", sorted(df['Team'].unique()), key='team1_matrix')
            team1_df = df[df['Team'] == team1]
            
        with col2:
            team2 = st.selectbox("Select Opponent Team", sorted(df['Team'].unique()), key='team2_matrix')
            team2_df = df[df['Team'] == team2]
        
        if team1 and team2:
            # Generate threat matrix
            threat_matrix = analyze_threat_matrix(team1_df, team2_df)
            
            # Display as heatmap
            fig = px.imshow(threat_matrix,
                           labels=dict(x="Opponent Pokémon", y="Your Pokémon", color="Threat Score"),
                           x=threat_matrix.columns,
                           y=threat_matrix.index,
                           aspect="auto")
            fig.update_xaxes(side="top")
            st.plotly_chart(fig, use_container_width=True)
            
            # Interpretation guide
            st.subheader("Matrix Interpretation")
            st.write("""
            - **Positive values (Green):** Your Pokémon has type advantage
            - **Zero values (White):** Neutral matchup
            - **Negative values (Red):** Opponent has type advantage
            """)
            
            # Find best and worst matchups
            max_threat = threat_matrix.max().max()
            min_threat = threat_matrix.min().min()
            
            if max_threat > 0:
                best_matchup = threat_matrix[threat_matrix == max_threat].stack().index[0]
                st.success(f"**Best Matchup:** {best_matchup[0]} vs {best_matchup[1]} (Advantage: +{max_threat})")
            
            if min_threat < 0:
                worst_matchup = threat_matrix[threat_matrix == min_threat].stack().index[0]
                st.error(f"**Worst Matchup:** {worst_matchup[0]} vs {worst_matchup[1]} (Disadvantage: {min_threat})")
            
            # Pattern recognition trainer
            st.divider()
            st.subheader("Pattern Recognition Trainer")
            
            selected_pokemon = st.selectbox("Select your Pokémon to analyze", team1_df['Pokemon'])
            opponent_pokemon = st.selectbox("Select opponent Pokémon", team2_df['Pokemon'])
            
            if selected_pokemon and opponent_pokemon:
                # Get matchup data
                matchup_score = threat_matrix.at[selected_pokemon, opponent_pokemon]
                
                # Get detailed type info
                your_poke = team1_df[team1_df['Pokemon'] == selected_pokemon].iloc[0]
                opp_poke = team2_df[team2_df['Pokemon'] == opponent_pokemon].iloc[0]
                
                st.write(f"### {selected_pokemon} ({your_poke['Type1']}{'/' + your_poke['Type2'] if pd.notna(your_poke['Type2']) else ''})")
                st.write(f"### vs {opponent_pokemon} ({opp_poke['Type1']}{'/' + opp_poke['Type2'] if pd.notna(opp_poke['Type2']) else ''})")
                
                if matchup_score > 0:
                    st.success(f"Type Advantage: +{matchup_score}")
                elif matchup_score < 0:
                    st.error(f"Type Disadvantage: {matchup_score}")
                else:
                    st.info("Neutral Matchup")
                
                # Show move effectiveness
                st.write("**Your Moves Against Opponent:**")
                your_moves = [your_poke[f'Move {i}'] for i in range(1,5) if pd.notna(your_poke[f'Move {i}'])]
                
                for move in your_moves:
                    # Simplified move type analysis (would need full move data for accuracy)
                    st.write(f"- {move}: Effectiveness depends on move type")
                
                st.write("**Opponent's Moves Against You:**")
                opp_moves = [opp_poke[f'Move {i}'] for i in range(1,5) if pd.notna(opp_poke[f'Move {i}'])]
                
                for move in opp_moves:
                    # Simplified move type analysis
                    st.write(f"- {move}: Effectiveness depends on move type")
    
    with tab3:
        st.header("Live Battle Tracker")
        
        # Initialize or get battle state
        battle_state = track_battle_state()
        
        col1, col2 = st.columns(2)
        
        with col1:
            your_team = st.selectbox("Select Your Team", sorted(df['Team'].unique()), key='your_team')
            your_team_df = df[df['Team'] == your_team]
            your_active = st.selectbox("Your Active Pokémon", your_team_df['Pokemon'], 
                                     key='your_active', index=0)
            
            # Status conditions
            your_status = st.multiselect("Status Conditions", 
                                       ['Burn', 'Freeze', 'Paralysis', 'Poison', 'Sleep', 'Toxic', 'Confusion'],
                                       key='your_status')
            
        with col2:
            opp_team = st.selectbox("Select Opponent Team", sorted(df['Team'].unique()), key='opp_team')
            opp_team_df = df[df['Team'] == opp_team]
            opp_active = st.selectbox("Opponent Active Pokémon", opp_team_df['Pokemon'],
                                    key='opp_active', index=0)
            
            # Status conditions
            opp_status = st.multiselect("Opponent Status Conditions", 
                                      ['Burn', 'Freeze', 'Paralysis', 'Poison', 'Sleep', 'Toxic', 'Confusion'],
                                      key='opp_status')
        
        # Field conditions
        field_conditions = st.multiselect("Field Conditions",
                                         ['Sun', 'Rain', 'Sand', 'Hail', 
                                          'Electric Terrain', 'Psychic Terrain', 'Grassy Terrain', 'Misty Terrain',
                                          'Trick Room', 'Gravity', 'Magic Room', 'Wonder Room',
                                          'Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web'],
                                         key='field_conditions')
        
        # Turn counter
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Next Turn"):
                battle_state = update_battle_state(battle_state, {'turn': battle_state['turn'] + 1})
        with col2:
            if st.button("Reset Battle"):
                battle_state = update_battle_state(battle_state, {'turn': 0})
        with col3:
            st.write(f"**Current Turn:** {battle_state['turn']}")
        
        # Update battle state
        battle_state = update_battle_state(battle_state, {
            'team1_active': your_active,
            'team2_active': opp_active,
            'team1_status': your_status,
            'team2_status': opp_status,
            'field_conditions': field_conditions,
            'team1_remaining': your_team_df['Pokemon'].tolist(),
            'team2_remaining': opp_team_df['Pokemon'].tolist()
        })
        
        # Display battle state
        st.divider()
        st.subheader("Current Battle State")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Your Active:** {battle_state['team1_active']}")
            if battle_state['team1_status']:
                st.write("**Status Conditions:**")
                for status in battle_state['team1_status']:
                    st.write(f"- {status}")
            
            st.write("**Remaining Pokémon:**")
            for poke in battle_state['team1_remaining']:
                if poke != battle_state['team1_active']:
                    st.write(f"- {poke}")
        
        with col2:
            st.write(f"**Opponent Active:** {battle_state['team2_active']}")
            if battle_state['team2_status']:
                st.write("**Status Conditions:**")
                for status in battle_state['team2_status']:
                    st.write(f"- {status}")
            
            st.write("**Remaining Pokémon:**")
            for poke in battle_state['team2_remaining']:
                if poke != battle_state['team2_active']:
                    st.write(f"- {poke}")
        
        if battle_state['field_conditions']:
            st.write("**Field Conditions:**")
            for condition in battle_state['field_conditions']:
                st.write(f"- {condition}")
        
        # Resource depletion calculator
        st.divider()
        st.subheader("Resource Tracker")
        
        # Calculate remaining resources
        your_remaining = len([p for p in battle_state['team1_remaining'] if p != battle_state['team1_active']])
        opp_remaining = len([p for p in battle_state['team2_remaining'] if p != battle_state['team2_active']])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Your Remaining Pokémon", your_remaining)
        with col2:
            st.metric("Opponent Remaining Pokémon", opp_remaining)
        
        # HP tracker (simplified - would need more complex implementation for actual HP tracking)
        if st.checkbox("Track HP (Simplified)"):
            your_hp = st.slider(f"{battle_state['team1_active']} HP %", 0, 100, 100)
            opp_hp = st.slider(f"{battle_state['team2_active']} HP %", 0, 100, 100)
            
            if your_hp <= 0:
                st.error(f"{battle_state['team1_active']} has fainted!")
            if opp_hp <= 0:
                st.error(f"{battle_state['team2_active']} has fainted!")

if __name__ == "__main__":
    main()
