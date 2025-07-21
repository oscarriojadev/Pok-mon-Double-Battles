import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from collections import defaultdict
import random

# Complete type effectiveness chart (unchanged)
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

def calculate_team_coverage(team_types):
    # Calculate team's defensive coverage
    weaknesses = defaultdict(int)
    resists = defaultdict(int)
    immunities = set()
    
    # Count weaknesses and resistances across all team types
    for t in team_types:
        type_data = TYPE_CHART.get(t, {})
        
        # Count weaknesses
        for weak_to in type_data.get('weak', []):
            weaknesses[weak_to] += 1
            
        # Count resistances
        for resists_to in type_data.get('resist', []):
            resists[resists_to] += 1
            
        # Track immunities
        for immune_to in type_data.get('immune', []):
            immunities.add(immune_to)
    
    # Determine coverage
    uncovered_weaknesses = []
    resisted_types = []
    
    for t in ALL_TYPES:
        # If more weaknesses than resistances for this type, it's a problem
        if weaknesses.get(t, 0) > resists.get(t, 0):
            uncovered_weaknesses.append(t)
        
        # If we have at least one resistance to this type
        if resists.get(t, 0) > 0:
            resisted_types.append(t)
    
    return {
        'uncovered_weaknesses': uncovered_weaknesses,
        'resisted_types': resisted_types,
        'immune_types': list(immunities)
    }

# Load data with caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Fixed machine learning similarity function (unchanged)
def calculate_pokemon_similarity(df, selected_pokemon):
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    temp_df = df.reset_index(drop=True)
    
    if selected_pokemon not in temp_df['Pokemon'].values:
        return pd.DataFrame(columns=df.columns.tolist() + ['Similarity'])
    
    numeric_df = temp_df[stats].fillna(0)
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(numeric_df)
    
    try:
        pokemon_index = temp_df.index[temp_df['Pokemon'] == selected_pokemon][0]
        similarities = cosine_similarity([scaled_stats[pokemon_index]], scaled_stats)[0]
    except IndexError:
        return pd.DataFrame(columns=df.columns.tolist() + ['Similarity'])
    
    result_df = df.copy()
    result_df['Similarity'] = similarities
    return result_df.sort_values('Similarity', ascending=False)

# New feature: Opponent Tendency Tracker
class OpponentTracker:
    def __init__(self):
        self.lead_patterns = defaultdict(int)
        self.switch_patterns = defaultdict(int)
        self.protect_usage = defaultdict(int)
        self.move_preferences = defaultdict(lambda: defaultdict(int))
        self.team_revealed = []
    
    def record_lead(self, pokemon):
        self.lead_patterns[pokemon] += 1
    
    def record_switch(self, from_pokemon, to_pokemon):
        self.switch_patterns[(from_pokemon, to_pokemon)] += 1
    
    def record_protect(self, pokemon):
        self.protect_usage[pokemon] += 1
    
    def record_move(self, pokemon, move):
        self.move_preferences[pokemon][move] += 1
    
    def record_team_member(self, pokemon):
        if pokemon not in self.team_revealed:
            self.team_revealed.append(pokemon)
    
    def get_most_common_leads(self, n=3):
        return sorted(self.lead_patterns.items(), key=lambda x: -x[1])[:n]
    
    def get_most_common_switches(self, n=3):
        return sorted(self.switch_patterns.items(), key=lambda x: -x[1])[:n]
    
    def get_protect_users(self, n=3):
        return sorted(self.protect_usage.items(), key=lambda x: -x[1])[:n]
    
    def get_move_preferences(self, pokemon):
        return sorted(self.move_preferences[pokemon].items(), key=lambda x: -x[1])
    
    def get_predicted_team(self, all_pokemon, n=6):
        revealed = self.team_revealed.copy()
        remaining = [p for p in all_pokemon if p not in revealed]
        if len(revealed) >= n:
            return revealed[:n]
        return revealed + random.sample(remaining, n - len(revealed))

# New feature: Game State Evaluator
class GameStateEvaluator:
    def __init__(self, team_df):
        self.team_df = team_df
        self.remaining_pokemon = team_df['Pokemon'].tolist()
        self.opponent_remaining = 6
        self.field_conditions = {
            'weather': None,
            'terrain': None,
            'trick_room': False,
            'tailwind': {'player': False, 'opponent': False}
        }
    
    def update_remaining(self, pokemon, is_player=True):
        if is_player:
            if pokemon in self.remaining_pokemon:
                self.remaining_pokemon.remove(pokemon)
        else:
            self.opponent_remaining -= 1
    
    def update_field(self, condition, value, is_player=True):
        if condition == 'weather':
            self.field_conditions['weather'] = value
        elif condition == 'terrain':
            self.field_conditions['terrain'] = value
        elif condition == 'trick_room':
            self.field_conditions['trick_room'] = value
        elif condition == 'tailwind':
            key = 'player' if is_player else 'opponent'
            self.field_conditions['tailwind'][key] = value
    
    def calculate_win_probability(self):
        # Simple heuristic based on remaining pokemon and field conditions
        player_advantage = len(self.remaining_pokemon) / 6
        opponent_advantage = self.opponent_remaining / 6
        
        # Adjust for field conditions
        if self.field_conditions['trick_room']:
            # Check if team has slow pokemon
            avg_speed = self.team_df['Speed'].mean()
            if avg_speed < 70:  # Team benefits from trick room
                player_advantage *= 1.2
        
        if self.field_conditions['tailwind']['player']:
            player_advantage *= 1.1
        
        if self.field_conditions['tailwind']['opponent']:
            opponent_advantage *= 1.1
        
        total = player_advantage + opponent_advantage
        if total == 0:
            return 0.5
        return player_advantage / total
    
    def evaluate_endgame(self):
        if len(self.remaining_pokemon) == 1 and self.opponent_remaining == 1:
            # 1v1 scenario
            player_poke = self.remaining_pokemon[0]
            player_stats = self.team_df[self.team_df['Pokemon'] == player_poke].iloc[0]
            
            # Very simple evaluation - could be enhanced with actual matchup data
            speed_advantage = player_stats['Speed'] > 100  # Assuming opponent has avg speed
            bulk_advantage = (player_stats['HP'] + player_stats['Defense'] + player_stats['Sp. Def']) > 300
            offense_advantage = (player_stats['Attack'] + player_stats['Sp. Atk']) > 200
            
            if speed_advantage and (bulk_advantage or offense_advantage):
                return "Favorable", 0.7
            elif speed_advantage or bulk_advantage or offense_advantage:
                return "Even", 0.5
            else:
                return "Unfavorable", 0.3
        elif len(self.remaining_pokemon) > self.opponent_remaining:
            return "Advantage", 0.7
        elif len(self.remaining_pokemon) < self.opponent_remaining:
            return "Disadvantage", 0.3
        else:
            return "Even", 0.5
    
    def get_resource_advantage(self):
        # Count remaining resources (pokemon, field conditions)
        resources = {
            'pokemon_remaining': len(self.remaining_pokemon),
            'opponent_remaining': self.opponent_remaining,
            'weather': self.field_conditions['weather'],
            'terrain': self.field_conditions['terrain'],
            'trick_room': self.field_conditions['trick_room'],
            'player_tailwind': self.field_conditions['tailwind']['player'],
            'opponent_tailwind': self.field_conditions['tailwind']['opponent']
        }
        return resources

# New feature: Mental Game & Pattern Recognition
class MentalGameAnalyzer:
    def __init__(self, team_df):
        self.team_df = team_df
        self.bluff_opportunities = []
        self.pressure_situations = []
        self.adaptation_strategies = []
    
    def identify_bluff_opportunities(self):
        # Identify pokemon that can bluff different sets
        for _, row in self.team_df.iterrows():
            item = row['Item']
            moves = [row['Move 1'], row['Move 2'], row['Move 3'], row['Move 4']]
            
            # Check for choice item bluff
            if 'Choice' in str(item):
                self.bluff_opportunities.append(
                    f"{row['Pokemon']} appears to be locked into a move but might not be"
                )
            
            # Check for coverage bluff
            coverage_moves = [m for m in moves if m in ['Ice Beam', 'Thunderbolt', 'Flamethrower', 'Energy Ball']]
            if len(coverage_moves) >= 2:
                self.bluff_opportunities.append(
                    f"{row['Pokemon']} has multiple coverage moves and can bluff different sets"
                )
        
        return self.bluff_opportunities
    
    def identify_pressure_situations(self):
        # Identify when the team can apply pressure
        for _, row in self.team_df.iterrows():
            if row['PrimaryRole'] == 'Sweeper' and 'Swords Dance' in [row['Move 1'], row['Move 2'], row['Move 3'], row['Move 4']]:
                self.pressure_situations.append(
                    f"{row['Pokemon']} can apply setup pressure with Swords Dance"
                )
            
            if row['Speed'] > 100:  # Fast pokemon
                self.pressure_situations.append(
                    f"{row['Pokemon']} can apply speed pressure with {row['Speed']} base speed"
                )
        
        return self.pressure_situations
    
    def generate_adaptation_strategies(self, opponent_tracker):
        # Generate strategies based on opponent patterns
        common_leads = opponent_tracker.get_most_common_leads()
        if common_leads:
            lead = common_leads[0][0]
            self.adaptation_strategies.append(
                f"Opponent frequently leads with {lead} - consider adjusting your lead"
            )
        
        protect_users = opponent_tracker.get_protect_users()
        if protect_users:
            poke = protect_users[0][0]
            self.adaptation_strategies.append(
                f"{poke} frequently uses Protect - consider using moves that bypass Protect or double-targeting"
            )
        
        return self.adaptation_strategies
    
    def get_pattern_recognition(self, opponent_tracker):
        patterns = []
        
        # Lead patterns
        common_leads = opponent_tracker.get_most_common_leads()
        if common_leads:
            patterns.append("**Lead Patterns:**")
            for lead, count in common_leads:
                patterns.append(f"- {lead}: {count} times")
        
        # Switch patterns
        common_switches = opponent_tracker.get_most_common_switches()
        if common_switches:
            patterns.append("\n**Switch Patterns:**")
            for (from_poke, to_poke), count in common_switches:
                patterns.append(f"- From {from_poke} to {to_poke}: {count} times")
        
        # Protect usage
        protect_users = opponent_tracker.get_protect_users()
        if protect_users:
            patterns.append("\n**Protect Usage:**")
            for poke, count in protect_users:
                patterns.append(f"- {poke}: {count} times")
        
        return "\n".join(patterns)

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer")
    
    st.title("🧠 Advanced Pokémon Team Analyzer")
    st.write("""
    Analyze competitive Pokémon teams with enhanced mental game tracking, opponent pattern recognition,
    and game state evaluation features.
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
    
    # Initialize session state for opponent tracking
    if 'opponent_tracker' not in st.session_state:
        st.session_state.opponent_tracker = OpponentTracker()
    
    # Initialize game state evaluator
    selected_team = st.sidebar.selectbox("Select Your Team", sorted(df['Team'].unique()))
    team_df = df[df['Team'] == selected_team]
    game_state = GameStateEvaluator(team_df)
    
    # Initialize mental game analyzer
    mental_game = MentalGameAnalyzer(team_df)
    
    # Main tabs - replaced with new features
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Mental Game & Patterns", 
        "👀 Opponent Tracker", 
        "📊 Game State Evaluator",
        "⚔️ Team Analysis"
    ])
    
    with tab1:
        st.header("Mental Game & Pattern Recognition")
        
        st.subheader("Bluffing Opportunities")
        bluff_ops = mental_game.identify_bluff_opportunities()
        if bluff_ops:
            for op in bluff_ops:
                st.info(op)
        else:
            st.warning("No obvious bluffing opportunities identified")
        
        st.subheader("Pressure Situations")
        pressure_sits = mental_game.identify_pressure_situations()
        if pressure_sits:
            for sit in pressure_sits:
                st.success(sit)
        else:
            st.warning("No clear pressure situations identified")
        
        st.subheader("Adaptation Strategies")
        adapt_strats = mental_game.generate_adaptation_strategies(st.session_state.opponent_tracker)
        if adapt_strats:
            for strat in adapt_strats:
                st.info(strat)
        else:
            st.warning("No specific adaptation strategies yet - track more opponent data")
    
    with tab2:
        st.header("Opponent Tendency Tracker")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Record Opponent Actions")
            action_type = st.selectbox("Action Type", ["Lead", "Switch", "Protect", "Move", "Revealed Team Member"])
            
            if action_type == "Lead":
                pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
                if st.button("Record Lead"):
                    st.session_state.opponent_tracker.record_lead(pokemon)
                    st.success(f"Recorded {pokemon} as lead")
            
            elif action_type == "Switch":
                col1, col2 = st.columns(2)
                with col1:
                    from_poke = st.selectbox("From", sorted(df['Pokemon'].unique()))
                with col2:
                    to_poke = st.selectbox("To", sorted(df['Pokemon'].unique()))
                if st.button("Record Switch"):
                    st.session_state.opponent_tracker.record_switch(from_poke, to_poke)
                    st.success(f"Recorded switch from {from_poke} to {to_poke}")
            
            elif action_type == "Protect":
                pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
                if st.button("Record Protect"):
                    st.session_state.opponent_tracker.record_protect(pokemon)
                    st.success(f"Recorded Protect by {pokemon}")
            
            elif action_type == "Move":
                pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
                move = st.text_input("Move Used")
                if st.button("Record Move") and move:
                    st.session_state.opponent_tracker.record_move(pokemon, move)
                    st.success(f"Recorded {move} by {pokemon}")
            
            elif action_type == "Revealed Team Member":
                pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
                if st.button("Record Team Member"):
                    st.session_state.opponent_tracker.record_team_member(pokemon)
                    st.success(f"Recorded {pokemon} as revealed team member")
        
        with col2:
            st.subheader("Opponent Patterns")
            patterns = mental_game.get_pattern_recognition(st.session_state.opponent_tracker)
            st.markdown(patterns if patterns else "No patterns recorded yet")
            
            st.subheader("Predicted Team")
            predicted_team = st.session_state.opponent_tracker.get_predicted_team(df['Pokemon'].unique())
            st.write("Based on revealed members and common patterns:")
            for poke in predicted_team:
                st.write(f"- {poke}")
    
    with tab3:
        st.header("Game State Evaluator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Update Game State")
            
            st.write("**Your Team Status**")
            pokemon_remaining = st.multiselect(
                "Your remaining Pokémon",
                options=team_df['Pokemon'].unique(),
                default=team_df['Pokemon'].unique()
            )
            game_state.remaining_pokemon = list(pokemon_remaining)
            
            st.write("**Opponent Status**")
            opponent_remaining = st.slider("Opponent remaining Pokémon", 0, 6, 6)
            game_state.opponent_remaining = opponent_remaining
            
            st.write("**Field Conditions**")
            weather = st.selectbox("Weather", ["None", "Sun", "Rain", "Sand", "Hail"])
            game_state.update_field('weather', weather if weather != "None" else None)
            
            terrain = st.selectbox("Terrain", ["None", "Electric", "Grassy", "Misty", "Psychic"])
            game_state.update_field('terrain', terrain if terrain != "None" else None)
            
            trick_room = st.checkbox("Trick Room Active")
            game_state.update_field('trick_room', trick_room)
            
            player_tailwind = st.checkbox("Your Tailwind Active")
            game_state.update_field('tailwind', player_tailwind, is_player=True)
            
            opponent_tailwind = st.checkbox("Opponent Tailwind Active")
            game_state.update_field('tailwind', opponent_tailwind, is_player=False)
        
        with col2:
            st.subheader("Game State Analysis")
            
            win_prob = game_state.calculate_win_probability()
            st.metric("Win Probability", f"{win_prob*100:.1f}%")
            
            endgame_eval, endgame_prob = game_state.evaluate_endgame()
            st.metric("Endgame Evaluation", endgame_eval, delta=f"{endgame_prob*100:.1f}%")
            
            st.write("**Resource Advantage**")
            resources = game_state.get_resource_advantage()
            st.write(f"- Pokémon remaining: You {resources['pokemon_remaining']} vs Opponent {resources['opponent_remaining']}")
            st.write(f"- Weather: {resources['weather'] or 'None'}")
            st.write(f"- Terrain: {resources['terrain'] or 'None'}")
            st.write(f"- Trick Room: {'Active' if resources['trick_room'] else 'Inactive'}")
            st.write(f"- Tailwind: You {'Active' if resources['player_tailwind'] else 'Inactive'} vs Opponent {'Active' if resources['opponent_tailwind'] else 'Inactive'}")
            
            st.write("**Recommended Actions**")
            if resources['pokemon_remaining'] < resources['opponent_remaining']:
                st.warning("You're down on Pokémon - consider playing more cautiously")
            elif resources['pokemon_remaining'] > resources['opponent_remaining']:
                st.success("You have Pokémon advantage - consider applying pressure")
            
            if resources['trick_room'] and team_df['Speed'].mean() < 70:
                st.success("Trick Room benefits your slow team - maintain it")
            elif resources['trick_room'] and team_df['Speed'].mean() > 80:
                st.warning("Trick Room helps opponent - consider disrupting it")
    
    with tab4:
        st.header("Team Analysis")
        
        st.subheader("Team Composition")
        role_dist = team_df['PrimaryRole'].value_counts().reset_index()
        fig = px.pie(role_dist, names='PrimaryRole', values='count', title='Role Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Team Members")
        st.dataframe(team_df[['Pokemon', 'PrimaryRole', 'Item', 'Ability', 'Move 1', 'Move 2', 'Move 3', 'Move 4']], hide_index=True)
        
        st.subheader("Type Coverage")
        team_types = []
        for _, row in team_df.iterrows():
            team_types.append(row['Type1'])
            if pd.notna(row['Type2']) and row['Type2'] != '':
                team_types.append(row['Type2'])
        
        coverage = calculate_team_coverage(team_types)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Uncovered Weaknesses**")
            if coverage['uncovered_weaknesses']:
                for t in coverage['uncovered_weaknesses']:
                    st.error(t)
            else:
                st.success("All weaknesses covered!")
        
        with col2:
            st.write("**Resisted Types**")
            if coverage['resisted_types']:
                for t in coverage['resisted_types']:
                    st.info(t)
            else:
                st.warning("No resisted types")
        
        with col3:
            st.write("**Immune Types**")
            if coverage['immune_types']:
                for t in coverage['immune_types']:
                    st.success(t)
            else:
                st.warning("No immunities")

if __name__ == "__main__":
    main()
