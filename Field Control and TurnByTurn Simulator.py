import streamlit as st
import pandas as pd
import numpy as np

# Load data with caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer")
    
    st.title("⚔️ Pokémon Competitive Team Analyzer")
    st.write("""
    Analyze competitive Pokémon teams with strategic depth analysis.
    """)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Pokémon Data CSV", type=["csv"])
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis")
        st.stop()
    
    df = load_data(uploaded_file)
    
    # Ensure required columns exist
    required_cols = ['Pokemon', 'Team', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 
                    'Ability', 'Move 1', 'Move 2', 'Move 3', 'Move 4', 'PrimaryRole', 'SecondaryRole']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        st.stop()
    
    # Add default columns if not present
    if 'Win Condition' not in df.columns:
        df['Win Condition'] = ''
    
    # Strategic Depth Analysis tab
    st.header("Strategic Depth Analysis")
    selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='strategic_team')
    
    team_df = df[df['Team'] == selected_team]
    if not team_df.empty:
        # Field Control Manager section
        st.subheader("🌦️ Field Control Manager")
        
        # Weather condition analyzer
        weather_col, terrain_col = st.columns(2)
        
        with weather_col:
            st.write("### Weather Condition Analysis")
            weather_abilities = []
            weather_moves = []
            
            for _, pokemon in team_df.iterrows():
                ability = pokemon['Ability']
                moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                
                # Check for weather-setting abilities
                if ability in ['Drizzle', 'Drought', 'Sand Stream', 'Snow Warning']:
                    weather_abilities.append(f"{pokemon['Pokemon']} ({ability})")
                
                # Check for weather-setting moves
                for move in moves:
                    if move in ['Rain Dance', 'Sunny Day', 'Sandstorm', 'Hail']:
                        weather_moves.append(f"{pokemon['Pokemon']} ({move})")
            
            if weather_abilities:
                st.success("**Weather-Setting Abilities:**")
                for item in weather_abilities:
                    st.write(f"- {item}")
            else:
                st.warning("No weather-setting abilities")
            
            if weather_moves:
                st.success("**Weather-Setting Moves:**")
                for item in weather_moves:
                    st.write(f"- {item}")
            else:
                st.warning("No weather-setting moves")
            
            # Weather interactions
            st.write("**Weather Interactions:**")
            weather_interactions = []
            for _, pokemon in team_df.iterrows():
                ability = pokemon['Ability']
                if ability in ['Swift Swim', 'Chlorophyll', 'Sand Rush', 'Slush Rush']:
                    weather_interactions.append(f"{pokemon['Pokemon']} benefits from weather ({ability})")
                elif ability in ['Dry Skin', 'Hydration', 'Rain Dish']:
                    weather_interactions.append(f"{pokemon['Pokemon']} benefits from rain ({ability})")
                elif ability in ['Solar Power', 'Leaf Guard']:
                    weather_interactions.append(f"{pokemon['Pokemon']} benefits from sun ({ability})")
            
            if weather_interactions:
                for item in weather_interactions:
                    st.info(f"- {item}")
            else:
                st.info("No notable weather interactions")
        
        with terrain_col:
            st.write("### Terrain Effect Analysis")
            terrain_abilities = []
            terrain_moves = []
            
            for _, pokemon in team_df.iterrows():
                ability = pokemon['Ability']
                moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                
                # Check for terrain-setting abilities
                if ability in ['Electric Surge', 'Psychic Surge', 'Grassy Surge', 'Misty Surge']:
                    terrain_abilities.append(f"{pokemon['Pokemon']} ({ability})")
                
                # Check for terrain-setting moves
                for move in moves:
                    if move in ['Electric Terrain', 'Psychic Terrain', 'Grassy Terrain', 'Misty Terrain']:
                        terrain_moves.append(f"{pokemon['Pokemon']} ({move})")
            
            if terrain_abilities:
                st.success("**Terrain-Setting Abilities:**")
                for item in terrain_abilities:
                    st.write(f"- {item}")
            else:
                st.warning("No terrain-setting abilities")
            
            if terrain_moves:
                st.success("**Terrain-Setting Moves:**")
                for item in terrain_moves:
                    st.write(f"- {item}")
            else:
                st.warning("No terrain-setting moves")
            
            # Terrain interactions
            st.write("**Terrain Interactions:**")
            terrain_interactions = []
            for _, pokemon in team_df.iterrows():
                ability = pokemon['Ability']
                if ability in ['Surge Surfer', 'Grassy Pelt']:
                    terrain_interactions.append(f"{pokemon['Pokemon']} benefits from terrain ({ability})")
                elif ability in ['Mimicry']:
                    terrain_interactions.append(f"{pokemon['Pokemon']} adapts to terrain ({ability})")
            
            if terrain_interactions:
                for item in terrain_interactions:
                    st.info(f"- {item}")
            else:
                st.info("No notable terrain interactions")
        
        # Hazard management planner
        st.subheader("⚠️ Hazard Management")
        hazard_col, room_col = st.columns(2)
        
        with hazard_col:
            st.write("### Hazard Control")
            hazard_setters = []
            hazard_removers = []
            
            for _, pokemon in team_df.iterrows():
                moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                
                # Check for hazard-setting moves
                for move in moves:
                    if move in ['Stealth Rock', 'Spikes', 'Toxic Spikes', 'Sticky Web']:
                        hazard_setters.append(f"{pokemon['Pokemon']} ({move})")
                
                # Check for hazard-removing moves
                for move in moves:
                    if move in ['Rapid Spin', 'Defog', 'Mortal Spin']:
                        hazard_removers.append(f"{pokemon['Pokemon']} ({move})")
            
            if hazard_setters:
                st.success("**Hazard Setters:**")
                for item in hazard_setters:
                    st.write(f"- {item}")
            else:
                st.warning("No hazard-setting moves")
            
            if hazard_removers:
                st.success("**Hazard Removers:**")
                for item in hazard_removers:
                    st.write(f"- {item}")
            else:
                st.warning("No hazard-removing moves")
        
        with room_col:
            st.write("### Room Effect Strategy")
            room_users = []
            
            for _, pokemon in team_df.iterrows():
                moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                
                # Check for room-effect moves
                for move in moves:
                    if move in ['Trick Room', 'Magic Room', 'Wonder Room']:
                        room_users.append(f"{pokemon['Pokemon']} ({move})")
            
            if room_users:
                st.success("**Room Effect Users:**")
                for item in room_users:
                    st.write(f"- {item}")
                
                # Speed analysis for Trick Room
                if 'Trick Room' in ' '.join(str(move) for move in room_users):
                    st.write("**Trick Room Speed Analysis:**")
                    trick_room_pokemon = []
                    for _, pokemon in team_df.iterrows():
                        if pokemon['Speed'] < 70:  # Good for Trick Room
                            trick_room_pokemon.append(f"{pokemon['Pokemon']} (Speed: {pokemon['Speed']} - Good)")
                        elif pokemon['Speed'] > 100:  # Bad for Trick Room
                            trick_room_pokemon.append(f"{pokemon['Pokemon']} (Speed: {pokemon['Speed']} - Bad)")
                    
                    if trick_room_pokemon:
                        for item in trick_room_pokemon:
                            if "Good)" in item:
                                st.success(f"- {item}")
                            else:
                                st.warning(f"- {item}")
            else:
                st.warning("No room-effect moves")
        
        # Turn-by-Turn Simulator section
        st.subheader("⏱️ Turn-by-Turn Simulator")
        
        # Decision tree visualization
        st.write("### Decision Tree Visualization")
        st.write("Key decision points based on team strategy:")
        
        # Get the team's win condition
        win_condition = team_df['Win Condition'].iloc[0] if 'Win Condition' in team_df.columns and not team_df['Win Condition'].isna().all() else "Not specified"
        
        if "Perish Song" in win_condition:
            st.write("""
            **Perish Song Strategy:**
            1. Lead with trapper (Gothitelle/Politoed) + redirection (Amoonguss)
            2. Set up Perish Song with protection
            3. Cycle through Pokémon to maintain trap
            4. Clean up with priority moves
            """)
        elif "Sand" in win_condition:
            st.write("""
            **Sandstorm Strategy:**
            1. Lead with sand setter (Tyranitar) + sweeper (Excadrill)
            2. Set up sand and start sweeping
            3. Use redirection/support to enable sweeper
            4. Clean up with remaining Pokémon
            """)
        elif "Tailwind" in win_condition:
            st.write("""
            **Tailwind Strategy:**
            1. Lead with speed control (Tornadus/Whimsicott) + offensive threat
            2. Set up Tailwind immediately
            3. Maintain offensive pressure
            4. Clean up with speed advantage
            """)
        else:
            st.info("No specific win condition strategy identified")
        
        # Resource management tracker
        st.write("### Resource Management")
        resource_col1, resource_col2 = st.columns(2)
        
        with resource_col1:
            st.write("**HP Recovery Options:**")
            recovery_users = []
            for _, pokemon in team_df.iterrows():
                moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                for move in moves:
                    if move in ['Recover', 'Roost', 'Synthesis', 'Moonlight', 'Morning Sun', 'Slack Off', 'Soft-Boiled']:
                        recovery_users.append(f"{pokemon['Pokemon']} ({move})")
            
            if recovery_users:
                for item in recovery_users:
                    st.success(f"- {item}")
            else:
                st.warning("No HP recovery moves")
        
        with resource_col2:
            st.write("**PP Management Considerations:**")
            low_pp_moves = []
            for _, pokemon in team_df.iterrows():
                moves = [pokemon[f'Move {i}'] for i in range(1,5)]
                for move in moves:
                    if move in ['Fire Blast', 'Hydro Pump', 'Blizzard', 'Thunder', 'Focus Blast']:
                        low_pp_moves.append(f"{pokemon['Pokemon']} ({move} - 5 PP)")
            
            if low_pp_moves:
                st.warning("Low PP moves (consider PP Max):")
                for item in low_pp_moves:
                    st.write(f"- {item}")
            else:
                st.success("No extremely low PP moves")
        
        # Momentum shift analyzer
        st.write("### Momentum Analysis")
        momentum_moves = []
        for _, pokemon in team_df.iterrows():
            moves = [pokemon[f'Move {i}'] for i in range(1,5)]
            for move in moves:
                if move in ['U-turn', 'Volt Switch', 'Parting Shot', 'Flip Turn']:
                    momentum_moves.append(f"{pokemon['Pokemon']} ({move})")
        
        if momentum_moves:
            st.success("**Momentum-Gaining Moves:**")
            for item in momentum_moves:
                st.write(f"- {item}")
        else:
            st.warning("No momentum-gaining moves")
        
        # Pivot timing optimizer
        st.write("### Pivot Optimization")
        pivot_users = []
        for _, pokemon in team_df.iterrows():
            if 'Pivot' in pokemon['PrimaryRole'] or 'Pivot' in str(pokemon['SecondaryRole']):
                pivot_users.append(pokemon['Pokemon'])
        
        if pivot_users:
            st.success("**Designated Pivots:**")
            for item in pivot_users:
                st.write(f"- {item}")
            
            st.write("**Recommended Pivot Timing:**")
            st.write("""
            - Early game: Use pivots to safely bring in sweepers
            - Mid game: Use pivots to maintain favorable matchups
            - Late game: Use pivots to preserve win conditions
            """)
        else:
            st.info("No designated pivots on this team")
    
    else:
        st.warning("No data available for selected team")

if __name__ == "__main__":
    main()
