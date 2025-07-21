import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import random
import joblib

# --------------------------
# Machine Learning Model for Opponent Move Selection
# --------------------------

class MovePredictor:
    def __init__(self):
        self.model = None
        self.le_pokemon = LabelEncoder()
        self.le_moves = LabelEncoder()
        self.le_types = LabelEncoder()
        
    def train(self, battle_data):
        """Train the move prediction model"""
        # Prepare features and target
        X = battle_data[['attacker', 'defender', 'attacker_type1', 'attacker_type2', 
                        'defender_type1', 'defender_type2', 'weather', 'terrain']]
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
        
        # Encode target
        y = self.le_moves.fit_transform(y)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
    def predict_move(self, attacker, defender, attacker_types, defender_types, weather, terrain):
        """Predict the best move for the opponent"""
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
            'terrain': [terrain if terrain else 'None']
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
# Enhanced Battle Simulation with ML Opponent
# --------------------------

def opponent_turn(battle_state, team1_df, team2_df):
    """Handle the opponent's turn with ML move selection"""
    if battle_state['team2_active']:
        defender = team1_df.loc[team1_df['Pokemon'] == battle_state['team1_active']].iloc[0].to_dict()
        attacker = team2_df.loc[team2_df['Pokemon'] == battle_state['team2_active']].iloc[0].to_dict()
        
        # Get available moves with PP
        available_moves = [
            move for move in [attacker[f'Move {i}'] for i in range(1, 5)] 
            if battle_state['team2_pp'][attacker['Pokemon']].get(move, 0) > 0
        ]
        
        if not available_moves:
            st.session_state.battle_log.append(f"{attacker['Pokemon'] has no moves left!")
            return
            
        # Get types
        attacker_types = [attacker['Type1']]
        if pd.notna(attacker['Type2']):
            attacker_types.append(attacker['Type2'])
            
        defender_types = [defender['Type1']]
        if pd.notna(defender['Type2']):
            defender_types.append(defender['Type2'])
        
        # Predict best move
        predicted_move = move_predictor.predict_move(
            attacker=attacker,
            defender=defender,
            attacker_types=attacker_types,
            defender_types=defender_types,
            weather=battle_state['weather'],
            terrain=battle_state['terrain']
        )
        
        # If predicted move not available (shouldn't happen), choose random
        move = predicted_move if predicted_move in available_moves else random.choice(available_moves)
        
        # Execute move
        damage = execute_move(attacker, defender, move, battle_state, 2)
        if damage > 0:
            st.session_state.battle_log.append(f"Opponent's {attacker['Pokemon']} dealt {damage} damage with {move}!")

# --------------------------
# Modified Main Function with ML Opponent
# --------------------------

def main():
    st.title("Pokémon Battle Simulator with AI Opponent")
    
    # File upload for teams
    st.sidebar.header("Upload Your Team Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file with Pokémon data", type="csv")
    
    if uploaded_file is not None:
        try:
            team1_df = pd.read_csv(uploaded_file)
            # Generate a random opponent team from the same dataset
            team2_df = team1_df.sample(n=min(6, len(team1_df))).copy()
            
            # Initialize battle state
            init_battle_state(team1_df, team2_df)
            battle_state = st.session_state.battle_state
            
            # Set initial active Pokémon
            if not battle_state['team1_active']:
                battle_state['team1_active'] = team1_df.iloc[0]['Pokemon']
            if not battle_state['team2_active']:
                battle_state['team2_active'] = team2_df.iloc[0]['Pokemon']
            
            # Display teams
            st.subheader("Your Team")
            st.dataframe(team1_df[['Pokemon', 'Type1', 'Type2'] + [f'Move {i}' for i in range(1, 5)]])
            
            st.subheader("Opponent Team")
            st.dataframe(team2_df[['Pokemon', 'Type1', 'Type2']])
            
            # Battle controls
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
            st.subheader("Your Move Selection")
            your_active = battle_state['team1_active']
            if your_active:
                moves = team1_df.loc[team1_df['Pokemon'] == your_active, 
                                    ['Move 1', 'Move 2', 'Move 3', 'Move 4']].values[0]
                
                for i, move in enumerate(moves, 1):
                    pp_left = battle_state['team1_pp'][your_active].get(move, 0)
                    if st.button(f"{move} ({pp_left} PP)", key=f"move_{i}"):
                        attacker = team1_df.loc[team1_df['Pokemon'] == your_active].iloc[0].to_dict()
                        defender = team2_df.loc[team2_df['Pokemon'] == battle_state['team2_active']].iloc[0].to_dict()
                        damage = execute_move(attacker, defender, move, battle_state, 1)
                        if damage > 0:
                            st.success(f"Dealt {damage} damage!")
                        
                        # Opponent's turn
                        opponent_turn(battle_state, team1_df, team2_df)
            
            # Display battle log
            st.subheader("Battle Log")
            if 'battle_log' in st.session_state:
                for entry in st.session_state.battle_log[-10:]:
                    st.write(entry)
            
            # Save battle data for ML training
            if st.button("Save Battle Data"):
                save_battle_data(team1_df, team2_df, battle_state)
                st.success("Battle data saved for future AI training!")
        
        except Exception as e:
            st.error(f"Error loading team data: {str(e)}")
    else:
        st.info("Please upload a CSV file with your Pokémon team data to begin.")
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
                'defender_type1', 'defender_type2', 'weather', 'terrain', 'move_used'
            ])
        
        # Add new battle data
        new_data = []
        for log in st.session_state.battle_log:
            if 'used' in log:
                parts = log.split()
                pokemon = parts[0]
                move = parts[2].replace('!', '')
                
                # Determine which team the attacker belongs to
                if pokemon in team1_df['Pokemon'].values:
                    attacker = team1_df.loc[team1_df['Pokemon'] == pokemon].iloc[0]
                    defender = team2_df.loc[team2_df['Pokemon'] == battle_state['team2_active']].iloc[0]
                else:
                    attacker = team2_df.loc[team2_df['Pokemon'] == pokemon].iloc[0]
                    defender = team1_df.loc[team1_df['Pokemon'] == battle_state['team1_active']].iloc[0]
                
                new_data.append({
                    'attacker': attacker['Pokemon'],
                    'defender': defender['Pokemon'],
                    'attacker_type1': attacker['Type1'],
                    'attacker_type2': attacker['Type2'] if pd.notna(attacker['Type2']) else 'None',
                    'defender_type1': defender['Type1'],
                    'defender_type2': defender['Type2'] if pd.notna(defender['Type2']) else 'None',
                    'weather': battle_state['weather'] if battle_state['weather'] else 'None',
                    'terrain': battle_state['terrain'] if battle_state['terrain'] else 'None',
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
