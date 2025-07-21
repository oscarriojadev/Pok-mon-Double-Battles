import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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

def create_type_heatmap(team_types):
    """Create a heatmap of type effectiveness for the team"""
    # Create a matrix of type effectiveness
    type_matrix = []
    for attack_type in ALL_TYPES:
        row = []
        for defense_type in ALL_TYPES:
            effectiveness = 1.0  # Neutral
            if attack_type in TYPE_CHART[defense_type]['weak']:
                effectiveness = 2.0  # Super effective
            elif attack_type in TYPE_CHART[defense_type]['resist']:
                effectiveness = 0.5  # Not very effective
            elif attack_type in TYPE_CHART[defense_type]['immune']:
                effectiveness = 0.0  # Immune
            row.append(effectiveness)
        type_matrix.append(row)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=type_matrix,
        x=ALL_TYPES,
        y=ALL_TYPES,
        colorscale='RdYlGn',
        zmin=0,
        zmax=2,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Type Effectiveness Heatmap",
        xaxis_title="Defending Type",
        yaxis_title="Attacking Type",
        width=800,
        height=800
    )
    
    return fig

def create_team_coverage_chart(team_coverage):
    """Create a radar chart showing team coverage"""
    categories = ['Offensive Coverage', 'Defensive Coverage', 'Type Synergy', 'Weakness Protection']
    
    # Calculate scores (simplified for demo)
    offensive_score = len(team_coverage['good_coverage']) / len(ALL_TYPES) * 100
    defensive_score = (len(ALL_TYPES) - len(team_coverage['uncovered_weaknesses'])) / len(ALL_TYPES) * 100
    synergy_score = (len(team_coverage['resisted_types']) + len(team_coverage['immune_types'])) / (len(ALL_TYPES) * 2) * 100
    protection_score = (len(ALL_TYPES) - len(team_coverage['uncovered_weaknesses'])) / len(ALL_TYPES) * 100
    
    values = [offensive_score, defensive_score, synergy_score, protection_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Coverage Score'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title='Team Coverage Radar Chart'
    )
    
    return fig

def create_battle_flow_diagram(team_df):
    """Create a flowchart of the team's battle strategy"""
    # Extract key phases from the team data
    phases = {
        'Early Game': team_df['Early Game'].iloc[0] if 'Early Game' in team_df.columns else None,
        'Mid Game': team_df['Mid Game'].iloc[0] if 'Mid Game' in team_df.columns else None,
        'Late Game': team_df['Late Game'].iloc[0] if 'Late Game' in team_df.columns else None,
        'Win Condition': team_df['Win Condition'].iloc[0] if 'Win Condition' in team_df.columns else None
    }
    
    # Create nodes and edges for the flowchart
    nodes = []
    edges = []
    
    if phases['Early Game']:
        nodes.append(dict(
            label="Early Game",
            description=phases['Early Game'],
            shape="square",  # Changed from "box" to "square"
            style="filled",
            fillcolor="#FFD700"  # Gold
        ))
    
    if phases['Mid Game']:
        nodes.append(dict(
            label="Mid Game",
            description=phases['Mid Game'],
            shape="square",  # Changed from "box" to "square"
            style="filled",
            fillcolor="#FFA500"  # Orange
        ))
        if phases['Early Game']:
            edges.append(("Early Game", "Mid Game"))
    
    if phases['Late Game']:
        nodes.append(dict(
            label="Late Game",
            description=phases['Late Game'],
            shape="square",  # Changed from "box" to "square"
            style="filled",
            fillcolor="#FF6347"  # Tomato
        ))
        if phases['Mid Game']:
            edges.append(("Mid Game", "Late Game"))
    
    if phases['Win Condition']:
        nodes.append(dict(
            label="Win Condition",
            description=phases['Win Condition'],
            shape="diamond",
            style="filled",
            fillcolor="#90EE90"  # Light green
        ))
        if phases['Late Game']:
            edges.append(("Late Game", "Win Condition"))
    
    # Create the figure
    fig = go.Figure()
    
    # Add nodes
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['label']],
            y=[1],
            mode="markers+text",
            marker=dict(
                size=40,
                color=node['fillcolor'],
                symbol=node['shape'],  # Now using valid symbol "square" or "diamond"
                line=dict(width=2, color='black')
            ),
            text=node['label'],
            textposition="middle center",
            hovertext=node['description'],
            hoverinfo="text",
            name=node['label']
        ))
    
    # Add edges
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[edge[0], edge[1]],
            y=[1, 1],
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none",
            showlegend=False
        ))
    
    fig.update_layout(
        title="Battle Flow Diagram",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        height=400
    )
    
    return fig

def create_turn_sequence_planner(team_df):
    """Create a turn sequence planner for the team"""
    # Get the first 3 turns strategy
    turn_sequence = []
    for i in range(1, 4):
        turn_key = f"Turn {i}"
        if turn_key in team_df.columns:
            turn_desc = team_df[turn_key].iloc[0]
            if pd.notna(turn_desc):
                turn_sequence.append((f"Turn {i}", turn_desc))
    
    # Create the figure
    fig = go.Figure()
    
    for i, (turn, desc) in enumerate(turn_sequence):
        fig.add_trace(go.Scatter(
            x=[i+1],
            y=[1],
            mode="markers+text",
            marker=dict(size=30, color="#4B9CD3"),
            text=turn,
            textposition="middle center",
            hovertext=desc,
            hoverinfo="text",
            name=turn
        ))
    
    if turn_sequence:
        fig.update_layout(
            title="Turn Sequence Planner",
            xaxis=dict(
                range=[0, len(turn_sequence)+1],
                tickvals=list(range(1, len(turn_sequence)+1)),
                ticktext=[t[0] for t in turn_sequence]
            ),
            yaxis=dict(visible=False),
            showlegend=False,
            height=300
        )
    else:
        fig.update_layout(
            title="No Turn Sequence Data Available",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            showlegend=False,
            height=300
        )
    
    return fig

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer")
    
    st.title("⚔️ Pokémon Competitive Team Analyzer")
    st.write("""
    Analyze competitive Pokémon teams with enhanced visualizations and battle flow diagrams.
    """)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Pokémon Data CSV", type=["csv"])
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()
    
    df = load_data(uploaded_file)
    
    # Ensure required columns exist
    required_cols = ['Pokemon', 'Team', 'Type1', 'Type2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        st.stop()
    
    # Add default columns if not present
    if 'PrimaryRole' not in df.columns:
        df['PrimaryRole'] = 'Unknown'
    if 'SecondaryRole' not in df.columns:
        df['SecondaryRole'] = ''
    if 'Win Condition' not in df.columns:
        df['Win Condition'] = ''
    if 'Early Game' not in df.columns:
        df['Early Game'] = ''
    if 'Mid Game' not in df.columns:
        df['Mid Game'] = ''
    if 'Late Game' not in df.columns:
        df['Late Game'] = ''
    
    # Main tabs - only keeping the enhanced visualization and battle flow tabs
    tab1, tab2 = st.tabs([
        "📊 Enhanced Data Visualization", 
        "⚡ Battle Flow Diagrams"
    ])
    
    with tab1:
        st.header("Enhanced Data Visualization")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()))
        
        team_df = df[df['Team'] == selected_team]
        if not team_df.empty:
            # Get all types for the team
            team_types = []
            for _, row in team_df.iterrows():
                team_types.append(row['Type1'])
                if pd.notna(row['Type2']) and row['Type2'] != '':
                    team_types.append(row['Type2'])
            
            coverage = calculate_team_coverage(team_types)
            
            # Interactive Type Chart
            st.subheader("Interactive Type Effectiveness")
            col1, col2 = st.columns(2)
            
            with col1:
                attack_type = st.selectbox("Select Attack Type", ALL_TYPES, key='attack_type')
                defense_type = st.selectbox("Select Defense Type", ALL_TYPES, key='defense_type')
                
                effectiveness = 1.0
                if attack_type in TYPE_CHART[defense_type]['weak']:
                    effectiveness = 2.0
                    st.success(f"Super effective! (2x damage)")
                elif attack_type in TYPE_CHART[defense_type]['resist']:
                    effectiveness = 0.5
                    st.warning(f"Not very effective (0.5x damage)")
                elif attack_type in TYPE_CHART[defense_type]['immune']:
                    effectiveness = 0.0
                    st.error(f"No effect! (0x damage)")
                else:
                    st.info(f"Normal effectiveness (1x damage)")
                
                st.metric("Damage Multiplier", effectiveness)
            
            with col2:
                st.plotly_chart(create_type_heatmap(team_types), use_container_width=True)
            
            # Team Coverage Heat Map
            st.subheader("Team Coverage Heat Map")
            st.plotly_chart(create_team_coverage_chart(coverage), use_container_width=True)
            
            # Weakness Exploitation Calculator
            st.subheader("Weakness Exploitation Calculator")
            st.write("Find which types your team can exploit in opponents")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Your Team's Excellent Coverage (3+):**")
                if coverage['excellent_coverage']:
                    for t in coverage['excellent_coverage']:
                        st.success(f"{t} - Great against opponents weak to {t}")
                else:
                    st.warning("No excellent coverage")
            
            with col2:
                st.write("**Opponent Types to Target:**")
                for t in coverage['good_coverage']:
                    st.info(f"{t} - Good coverage against this type")
            
            # Defensive Synergy Mapper
            st.subheader("Defensive Synergy Mapper")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("❌ **Uncovered Weaknesses**")
                if coverage['uncovered_weaknesses']:
                    for t in coverage['uncovered_weaknesses']:
                        st.error(t)
                else:
                    st.success("All weaknesses covered!")
            
            with col2:
                st.write("🛡️ **Resisted Types**")
                if coverage['resisted_types']:
                    for t in coverage['resisted_types']:
                        st.info(t)
                else:
                    st.warning("No resisted types")
            
            with col3:
                st.write("✅ **Immune Types**")
                if coverage['immune_types']:
                    for t in coverage['immune_types']:
                        st.success(t)
                else:
                    st.warning("No immunities")
        else:
            st.warning("No data available for selected team")
    
    with tab2:
        st.header("Battle Flow Diagrams")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='battle_flow_team')
        
        team_df = df[df['Team'] == selected_team]
        if not team_df.empty:
            # Win Condition Pathway Visualization
            st.subheader("Win Condition Pathway")
            st.plotly_chart(create_battle_flow_diagram(team_df), use_container_width=True)
            
            # Turn Sequence Planner
            st.subheader("Turn Sequence Planner")
            st.plotly_chart(create_turn_sequence_planner(team_df), use_container_width=True)
            
            # Setup Opportunity Mapper
            st.subheader("Setup Opportunities")
            
            setup_pokemon = []
            for _, row in team_df.iterrows():
                moves = [row['Move 1'], row['Move 2'], row['Move 3'], row['Move 4']]
                setup_moves = ['Swords Dance', 'Nasty Plot', 'Bulk Up', 'Calm Mind', 'Dragon Dance']
                for move in moves:
                    if any(sm in str(move) for sm in setup_moves):
                        setup_pokemon.append(row['Pokemon'])
                        break
            
            if setup_pokemon:
                st.write("**Pokémon with Setup Moves:**")
                for pokemon in setup_pokemon:
                    st.success(f"- {pokemon}")
                
                st.write("**Best Setup Opportunities:**")
                st.info("""
                - Use redirection (Follow Me/Rage Powder) to safely setup
                - Setup when opponent has a Pokémon locked into an ineffective move
                - Setup after eliminating threats to your Pokémon
                """)
            else:
                st.warning("No setup moves detected on this team")
            
            # Endgame Scenario Flowchart
            st.subheader("Endgame Scenarios")
            
            win_condition = team_df['Win Condition'].iloc[0] if 'Win Condition' in team_df.columns else None
            if pd.notna(win_condition) and win_condition != '':
                st.write(f"**Win Condition:** {win_condition}")
                
                if "Perish Song" in win_condition:
                    st.info("""
                    **Perish Song Endgame Strategy:**
                    1. Trap opponents with Shadow Tag/Arena Trap
                    2. Use Protect/Redirection to survive turns
                    3. Clean up with priority moves
                    """)
                elif "Sand" in win_condition:
                    st.info("""
                    **Sandstorm Endgame Strategy:**
                    1. Keep sand up with Sand Stream
                    2. Use Sand Rush/Sand Force Pokémon
                    3. Chip down opponents with residual damage
                    """)
                elif "Tailwind" in win_condition:
                    st.info("""
                    **Tailwind Endgame Strategy:**
                    1. Maintain speed control with Tailwind
                    2. Use fast sweepers to clean up
                    3. Protect Tailwind setter for reapplication
                    """)
                else:
                    st.info("""
                    **General Endgame Strategy:**
                    1. Preserve your win condition Pokémon
                    2. Eliminate counters to your win condition
                    3. Position your win condition for cleanup
                    """)
            else:
                st.warning("No explicit win condition defined for this team")
        else:
            st.warning("No data available for selected team")

if __name__ == "__main__":
    main()
