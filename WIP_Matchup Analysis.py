import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import random
import joblib

# --------------------------
# Initialize Battle State for Double Battles
# --------------------------

def init_battle_state(team1_df, team2_df, team1_active=None, team2_active=None):
    """Initialize the battle state dictionary for double battles"""
    if 'battle_state' not in st.session_state:
        st.session_state.battle_state = {
            'team1_active': team1_active if team1_active else [],
            'team2_active': team2_active if team2_active else [],
            'weather': None,
            'terrain': None,
            'team1_pp': defaultdict(dict),
            'team2_pp': defaultdict(dict),
            'team1_tera': defaultdict(bool),
            'team2_tera': defaultdict(bool),
            'team1_tera_type': defaultdict(str),
            'team2_tera_type': defaultdict(str),
            'team1_bench': [p for p in team1_df['Pokemon'].unique() if p not in team1_active] if team1_active else [],
            'team2_bench': [p for p in team2_df['Pokemon'].unique() if p not in team2_active] if team2_active else []
        }
    
    # Initialize PP for each Pokémon's moves
    for _, row in team1_df.iterrows():
        for i in range(1, 5):
            move = row[f'Move {i}']
            if pd.notna(move):
                st.session_state.battle_state['team1_pp'][row['Pokemon']][move] = 5  # Default PP
    
    for _, row in team2_df.iterrows():
        for i in range(1, 5):
            move = row[f'Move {i}']
            if pd.notna(move):
                st.session_state.battle_state['team2_pp'][row['Pokemon']][move] = 5  # Default PP
    
    # Initialize Tera types
    for _, row in team1_df.iterrows():
        st.session_state.battle_state['team1_tera_type'][row['Pokemon']] = row['Type1']
    
    for _, row in team2_df.iterrows():
        st.session_state.battle_state['team2_tera_type'][row['Pokemon']] = row['Type1']

# --------------------------
# Battle Controls for Double Battles
# --------------------------

def render_battle_controls(battle_state, team1_df, team2_df):
    """Render the battle control buttons for double battles"""
    st.subheader("Battle Controls")
    
    # Switch Pokémon controls
    if st.button("Switch Pokémon", key="switch_pokemon_button"):
        st.session_state.show_switch_menu = True
    
    # Use Item button
    if st.button("Use Item", key="use_item_button"):
        st.session_state.show_item_menu = True
    
    # Flee Battle button
    if st.button("Flee Battle", key="flee_battle_button"):
        st.session_state.battle_log.append("You fled from the battle!")
        st.session_state.battle_active = False
    
    # Switch Pokémon menu
    if st.session_state.get('show_switch_menu', False):
        st.subheader("Switch Pokémon")
        active_pokemon = battle_state['team1_active']
        bench_pokemon = [p for p in team1_df['Pokemon'].unique() if p not in active_pokemon]
        
        if len(battle_state['team1_active']) < 2 and bench_pokemon:
            selected = st.selectbox("Choose a Pokémon to send out", bench_pokemon, key="switch_selectbox")
            if st.button("Confirm Switch", key="confirm_switch_button"):
                battle_state['team1_active'].append(selected)
                battle_state['team1_bench'].remove(selected)
                st.session_state.battle_log.append(f"You sent out {selected}!")
                st.session_state.show_switch_menu = False
                st.rerun()
        else:
            st.warning("You already have 2 Pokémon in battle or no Pokémon left on bench!")
            if st.button("Cancel", key="cancel_switch_button"):
                st.session_state.show_switch_menu = False

# --------------------------
# Execute Move Function for Double Battles
# --------------------------

def execute_move(attacker, defender, move, battle_state, team, target_position=None):
    """Execute a move in double battles and return damage dealt"""
    # Simplified damage calculation
    damage = random.randint(10, 50)
    
    # Track PP usage
    if team == 1:
        battle_state['team1_pp'][attacker['Pokemon']][move] -= 1
    else:
        battle_state['team2_pp'][attacker['Pokemon']][move] -= 1
    
    # Log the move with target information - using full Pokémon names
    target_info = f" on opponent's {defender['Pokemon']}" if target_position is not None else ""
    st.session_state.battle_log.append(f"{attacker['Pokemon']} used {move}{target_info}!")
    
    return damage

# --------------------------
# Tera Animation (placeholder)
# --------------------------

def render_tera_animation(pokemon, tera_type):
    """Placeholder for tera animation"""
    st.session_state.battle_log.append(f"{pokemon} Terastallized to {tera_type} type!")

# --------------------------
# Machine Learning Model for Opponent Move Selection (Updated for Doubles)
# --------------------------

class MovePredictor:
    def __init__(self):
        self.model = None
        self.le_pokemon = LabelEncoder()
        self.le_moves = LabelEncoder()
        self.le_types = LabelEncoder()
        
    def train(self, battle_data):
        """Train the move prediction model for double battles"""
        # Prepare features and target
        X = battle_data[['attacker', 'defender', 'attacker_type1', 'attacker_type2', 
                        'defender_type1', 'defender_type2', 'weather', 'terrain', 'ally', 'opponent']]
        y = battle_data['move_used']
        
        # Encode categorical features
        X['attacker'] = self.le_pokemon.fit_transform(X['attacker'])
        X['defender'] = self.le_pokemon.transform(X['defender'])
        X['attacker_type1'] = self.le_types.fit_transform(X['attacker_type1'])
        X['attacker_type2'] = self.le_types.transform(X['attacker_type2'].fillna('None'))
        X['defender_type1'] = self.le_types.transform(X['defender_type1'])
        X['defender_type2'] = self.le_types.transform(X['defender_type2'].fillna('None'))
        X['weather'] = self.le_types.fit_transform(X['weather'].fillna('None'))
        X['terrain'] = self.le_types.transform(X['terrain'].fillna('None'))
        X['ally'] = self.le_pokemon.transform(X['ally'].fillna('None'))
        X['opponent'] = self.le_pokemon.transform(X['opponent'].fillna('None'))
        
        # Encode target
        y = self.le_moves.fit_transform(y)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
    def predict_move(self, attacker, defender, attacker_types, defender_types, weather, terrain, ally=None, opponent=None):
        """Predict the best move for the opponent in double battles"""
        if not self.model:
            return random.choice(defender['moves'])  # Fallback if no model
            
        # Prepare input features
        input_data = pd.DataFrame({
            'attacker': [attacker['Pokemon']],
            'defender': [defender['Pokemon']],
            'attacker_type1': [attacker_types[0]],
            'attacker_type2': [attacker_types[1] if len(attacker_types) > 1 else 'None'],
            'defender_type1': [defender_types[0]],
            'defender_type2': [defender_types[1] if len(defender_types) > 1 else 'None'],
            'weather': [weather if weather else 'None'],
            'terrain': [terrain if terrain else 'None'],
            'ally': [ally['Pokemon'] if ally else 'None'],
            'opponent': [opponent['Pokemon'] if opponent else 'None']
        })
        
        # Transform features
        input_data['attacker'] = self.le_pokemon.transform(input_data['attacker'])
        input_data['defender'] = self.le_pokemon.transform(input_data['defender'])
        input_data['attacker_type1'] = self.le_types.transform(input_data['attacker_type1'])
        input_data['attacker_type2'] = self.le_types.transform(input_data['attacker_type2'])
        input_data['defender_type1'] = self.le_types.transform(input_data['defender_type1'])
        input_data['defender_type2'] = self.le_types.transform(input_data['defender_type2'])
        input_data['weather'] = self.le_types.transform(input_data['weather'])
        input_data['terrain'] = self.le_types.transform(input_data['terrain'])
        input_data['ally'] = self.le_pokemon.transform(input_data['ally'])
        input_data['opponent'] = self.le_pokemon.transform(input_data['opponent'])
        
        # Predict and decode move
        pred = self.model.predict(input_data)
        return self.le_moves.inverse_transform(pred)[0]

# --------------------------
# Initialize or Load ML Model
# --------------------------

@st.cache_resource
def load_move_predictor():
    """Load or initialize the move predictor"""
    try:
        predictor = joblib.load('move_predictor.joblib')
    except:
        predictor = MovePredictor()
        # Load some default battle data if available
        try:
            default_data = pd.read_csv('battle_history.csv')
            predictor.train(default_data)
        except:
            pass
    return predictor

move_predictor = load_move_predictor()

# --------------------------
# Enhanced Battle Simulation with ML Opponent (Double Battles)
# --------------------------

def opponent_turn(battle_state, team1_df, team2_df):
    """Handle the opponent's turn with ML move selection in double battles"""
    if len(battle_state['team2_active']) > 0:
        try:
            # Get all active Pokémon
            team1_active = [team1_df.loc[team1_df['Pokemon'] == p].iloc[0].to_dict() for p in battle_state['team1_active']]
            team2_active = [team2_df.loc[team2_df['Pokemon'] == p].iloc[0].to_dict() for p in battle_state['team2_active']]
            
            # Process each opponent Pokémon's turn
            for attacker in team2_active:
                # Get available moves with PP
                available_moves = [
                    move for move in [attacker[f'Move {i}'] for i in range(1, 5)] 
                    if battle_state['team2_pp'][attacker['Pokemon']].get(move, 0) > 0
                ]
                
                if not available_moves:
                    st.session_state.battle_log.append(f"{attacker['Pokemon']} has no moves left!")
                    continue
                    
                # Get types
                attacker_types = [attacker['Type1']]
                if pd.notna(attacker['Type2']):
                    attacker_types.append(attacker['Type2'])
                
                # Choose a random target from opponent's active Pokémon
                defender = random.choice(team1_active)
                defender_types = [defender['Type1']]
                if pd.notna(defender['Type2']):
                    defender_types.append(defender['Type2'])
                
                # Get ally and other opponent for double battle context
                ally = random.choice([p for p in team2_active if p['Pokemon'] != attacker['Pokemon']]) if len(team2_active) > 1 else None
                opponent = random.choice([p for p in team1_active if p['Pokemon'] != defender['Pokemon']]) if len(team1_active) > 1 else None
                
                # Predict best move
                predicted_move = move_predictor.predict_move(
                    attacker=attacker,
                    defender=defender,
                    attacker_types=attacker_types,
                    defender_types=defender_types,
                    weather=battle_state['weather'],
                    terrain=battle_state['terrain'],
                    ally=ally,
                    opponent=opponent
                )
                
                # If predicted move not available (shouldn't happen), choose random
                move = predicted_move if predicted_move in available_moves else random.choice(available_moves)
                
                # Execute move - now showing full Pokémon names
                damage = execute_move(attacker, defender, move, battle_state, 2, target_position=team1_active.index(defender))
                if damage > 0:
                    st.session_state.battle_log.append(f"Opponent's {attacker['Pokemon']} dealt {damage} damage to {defender['Pokemon']} with {move}!")
        except IndexError:
            st.session_state.battle_log.append("Error: Selected Pokémon not found in team!")

# --------------------------
# Main Function with Team Selection for Double Battles
# --------------------------

def main():
    st.title("Pokémon Double Battle Simulator (4v4) with AI Opponent")
    
    # Initialize session state variables
    if 'battle_log' not in st.session_state:
        st.session_state.battle_log = []
    if 'show_switch_menu' not in st.session_state:
        st.session_state.show_switch_menu = False
    if 'show_item_menu' not in st.session_state:
        st.session_state.show_item_menu = False
    if 'battle_active' not in st.session_state:
        st.session_state.battle_active = False
    if 'full_dataset' not in st.session_state:
        st.session_state.full_dataset = None
    
    # File upload for the complete dataset
    st.sidebar.header("Upload Pokémon Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file with all Pokémon data", type="csv", key="dataset_uploader")
    
    if uploaded_file is not None:
        try:
            # Load the complete dataset
            full_dataset = pd.read_csv(uploaded_file)
            st.session_state.full_dataset = full_dataset
            
            # Team selection
            st.subheader("Team Selection (4 Pokémon each)")
            
            # Your team selection
            st.write("### Your Team")
            your_team = st.multiselect(
                "Select your 4 Pokémon team",
                options=full_dataset['Pokemon'].unique(),
                max_selections=4,
                key="your_team_select"
            )
            
            # Opponent team selection
            st.write("### Opponent Team")
            available_opponents = [p for p in full_dataset['Pokemon'].unique() if p not in your_team]
            opponent_team = st.multiselect(
                "Select opponent's 4 Pokémon team",
                options=available_opponents,
                max_selections=4,
                key="opponent_team_select"
            )
            
            # Filter the teams from the full dataset
            if len(your_team) == 4 and len(opponent_team) == 4:
                team1_df = full_dataset[full_dataset['Pokemon'].isin(your_team)].copy()
                team2_df = full_dataset[full_dataset['Pokemon'].isin(opponent_team)].copy()
                
                # Active Pokémon selection (2 for each side)
                st.write("### Select Active Pokémon (2 each)")
                col1, col2 = st.columns(2)
                with col1:
                    team1_active = st.multiselect(
                        "Your active Pokémon (select 2)", 
                        your_team,
                        default=your_team[:2],
                        max_selections=2,
                        key="team1_active_select"
                    )
                with col2:
                    team2_active = st.multiselect(
                        "Opponent's active Pokémon (select 2)", 
                        opponent_team,
                        default=opponent_team[:2],
                        max_selections=2,
                        key="team2_active_select"
                    )
                
                # Start battle button
                if st.button("Start Battle", key="start_battle_button") and len(team1_active) == 2 and len(team2_active) == 2:
                    st.session_state.battle_active = True
                    init_battle_state(team1_df, team2_df, team1_active, team2_active)
                    st.session_state.battle_log.append("Double battle started!")
                    st.session_state.battle_log.append(f"Your active Pokémon: {', '.join(team1_active)}")
                    st.session_state.battle_log.append(f"Opponent's active Pokémon: {', '.join(team2_active)}")
                    st.rerun()
            
            # Battle display
            if st.session_state.battle_active and 'battle_state' in st.session_state:
                battle_state = st.session_state.battle_state
                
                # Display teams
                st.subheader("Your Team")
                st.dataframe(team1_df[['Pokemon', 'Type1', 'Type2'] + [f'Move {i}' for i in range(1, 5)]], key="team1_df_display")
                
                st.subheader("Opponent Team")
                st.dataframe(team2_df[['Pokemon', 'Type1', 'Type2']], key="team2_df_display")
                
                # Battle controls
                render_battle_controls(battle_state, team1_df, team2_df)
                
                # Display active Pokémon
                st.subheader("Active Pokémon")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Your Active Pokémon")
                    for idx, poke in enumerate(battle_state['team1_active']):
                        tera_status = " (Terastallized)" if battle_state['team1_tera'][poke] else ""
                        st.write(f"- {poke}{tera_status}")
                        
                        # Tera Activation Button for each Pokémon
                        if st.button(f"Terastallize {poke}", key=f"tera_1_{idx}_{poke}", 
                                    disabled=battle_state['team1_tera'][poke]):
                            battle_state['team1_tera'][poke] = True
                            tera_type = battle_state['team1_tera_type'][poke]
                            render_tera_animation(poke, tera_type)
                
                with col2:
                    st.write("### Opponent's Active Pokémon")
                    for idx, poke in enumerate(battle_state['team2_active']):
                        tera_status = " (Terastallized)" if battle_state['team2_tera'][poke] else ""
                        st.write(f"- {poke}{tera_status}")
                
                # Move selection for each of your active Pokémon
                st.subheader("Your Move Selection")
                for poke_idx, poke_name in enumerate(battle_state['team1_active']):
                    active_pokemon = team1_df.loc[team1_df['Pokemon'] == poke_name]
                    if not active_pokemon.empty:
                        poke = active_pokemon.iloc[0]
                        st.write(f"#### {poke_name}'s Moves")
                        moves = poke[['Move 1', 'Move 2', 'Move 3', 'Move 4']].values
                        
                        # Choose target for each move
                        target_options = battle_state['team2_active']
                        for move_idx, move in enumerate(moves, 1):
                            if pd.notna(move):
                                pp_left = battle_state['team1_pp'][poke_name].get(move, 0)
                                target = st.selectbox(
                                    f"Target for {move} ({pp_left} PP)", 
                                    target_options,
                                    key=f"target_1_{poke_idx}_{move_idx}"
                                )
                                if st.button(f"Use {move}", key=f"move_1_{poke_idx}_{move_idx}"):
                                    defender = team2_df.loc[team2_df['Pokemon'] == target].iloc[0].to_dict()
                                    damage = execute_move(
                                        poke.to_dict(), 
                                        defender, 
                                        move, 
                                        battle_state, 
                                        1,
                                        target_position=battle_state['team2_active'].index(target)
                                    )
                                    if damage > 0:
                                        st.success(f"{poke_name} dealt {damage} damage to {target}!")
                                    
                                    # Opponent's turn
                                    opponent_turn(battle_state, team1_df, team2_df)
                
                # Display battle log
                st.subheader("Battle Log")
                if 'battle_log' in st.session_state:
                    for entry in st.session_state.battle_log[-10:]:
                        st.write(entry)
                
                # Save battle data for ML training
                if st.button("Save Battle Data", key="save_battle_data_button"):
                    save_battle_data(team1_df, team2_df, battle_state)
                    st.success("Battle data saved for future AI training!")
        
        except Exception as e:
            st.error(f"Error loading team data: {str(e)}")
    else:
        st.info("Please upload a CSV file with Pokémon data to begin.")
        st.markdown("""
        ### Expected CSV Format:
        - **Pokemon**: Name of the Pokémon
        - **Type1**: Primary type
        - **Type2**: Secondary type (leave empty if none)
        - **Move 1** to **Move 4**: The Pokémon's moves
        - **Attack**, **Defense**, **Sp. Atk**, **Sp. Def**, **Speed**: Base stats
        - **Level**: Pokémon level (default 50)
        """)

def save_battle_data(team1_df, team2_df, battle_state):
    """Save battle data for ML training"""
    try:
        # Load existing data or create new
        try:
            battle_data = pd.read_csv('battle_history.csv')
        except:
            battle_data = pd.DataFrame(columns=[
                'attacker', 'defender', 'attacker_type1', 'attacker_type2',
                'defender_type1', 'defender_type2', 'weather', 'terrain', 
                'ally', 'opponent', 'move_used'
            ])
        
        # Add new battle data
        new_data = []
        for log in st.session_state.battle_log:
            if 'used' in log:
                parts = log.split()
                pokemon = parts[0]
                move = parts[2].replace('!', '')
                target = parts[-1] if 'opponent' in log else None
                
                # Determine which team the attacker belongs to
                if pokemon in team1_df['Pokemon'].values:
                    attacker = team1_df.loc[team1_df['Pokemon'] == pokemon].iloc[0]
                    defender = team2_df.loc[team2_df['Pokemon'] == target].iloc[0] if target else None
                    ally = [p for p in battle_state['team1_active'] if p != pokemon][0] if len(battle_state['team1_active']) > 1 else None
                    opponent = [p for p in battle_state['team2_active'] if p != target][0] if target and len(battle_state['team2_active']) > 1 else None
                else:
                    attacker = team2_df.loc[team2_df['Pokemon'] == pokemon].iloc[0]
                    defender = team1_df.loc[team1_df['Pokemon'] == target].iloc[0] if target else None
                    ally = [p for p in battle_state['team2_active'] if p != pokemon][0] if len(battle_state['team2_active']) > 1 else None
                    opponent = [p for p in battle_state['team1_active'] if p != target][0] if target and len(battle_state['team1_active']) > 1 else None
                
                new_data.append({
                    'attacker': attacker['Pokemon'],
                    'defender': defender['Pokemon'] if defender else 'None',
                    'attacker_type1': attacker['Type1'],
                    'attacker_type2': attacker['Type2'] if pd.notna(attacker['Type2']) else 'None',
                    'defender_type1': defender['Type1'] if defender else 'None',
                    'defender_type2': defender['Type2'] if defender and pd.notna(defender['Type2']) else 'None',
                    'weather': battle_state['weather'] if battle_state['weather'] else 'None',
                    'terrain': battle_state['terrain'] if battle_state['terrain'] else 'None',
                    'ally': ally if ally else 'None',
                    'opponent': opponent if opponent else 'None',
                    'move_used': move
                })
        
        if new_data:
            battle_data = pd.concat([battle_data, pd.DataFrame(new_data)], ignore_index=True)
            battle_data.to_csv('battle_history.csv', index=False)
            
            # Retrain model with new data
            move_predictor.train(battle_data)
            joblib.dump(move_predictor, 'move_predictor.joblib')
            
    except Exception as e:
        st.error(f"Error saving battle data: {str(e)}")

if __name__ == "__main__":
    main()
