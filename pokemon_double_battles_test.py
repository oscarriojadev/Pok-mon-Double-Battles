import streamlit as st
import pandas as pd
import numpy as np

# Load the dataset from user upload
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Main function
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer")
    
    st.title("Pokémon Competitive Team Analyzer")
    
    # File uploader widget
    uploaded_file = st.sidebar.file_uploader(
        "Upload Pokémon Team CSV", 
        type=["csv"],
        help="Upload the Pokémon Double Teams database CSV file"
    )
    
    if uploaded_file is None:
        st.info("Please upload a CSV file to continue")
        st.stop()
    
    df = load_data(uploaded_file)
    
    if df is None:
        st.error("Failed to load the CSV file")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_team = st.sidebar.selectbox("Select Team", sorted(df['Team'].unique()))
    selected_role = st.sidebar.selectbox("Select Role", sorted(df['PrimaryRole'].unique()))
    
    # Filter data
    filtered_df = df[(df['Team'] == selected_team) & (df['PrimaryRole'] == selected_role)]
    
    # Main display
    tab1, tab2, tab3 = st.tabs(["Team Overview", "Pokémon Details", "Team Synergy"])
    
    with tab1:
        st.header(f"Team: {selected_team}")
        st.subheader(f"Role: {selected_role}")
        
        # Team stats
        team_members = df[df['Team'] == selected_team]['Pokemon'].unique()
        st.write(f"Team Members: {', '.join(team_members)}")
        
        # Display team composition
        team_composition = df[df['Team'] == selected_team].groupby('PrimaryRole').size().reset_index(name='Count')
        st.bar_chart(team_composition.set_index('PrimaryRole'))
        
        # Show top Pokémon in this team
        st.dataframe(filtered_df[['Pokemon', 'Item', 'Ability', 'Move 1', 'Move 2', 'Move 3', 'Move 4']].head(10))
    
    with tab2:
        if not filtered_df.empty:
            selected_pokemon = st.selectbox("Select Pokémon", filtered_df['Pokemon'].unique())
            pokemon_data = filtered_df[filtered_df['Pokemon'] == selected_pokemon].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Info")
                st.write(f"**Pokémon:** {pokemon_data['Pokemon']}")
                st.write(f"**Item:** {pokemon_data['Item']}")
                st.write(f"**Ability:** {pokemon_data['Ability']}")
                st.write(f"**Nature:** {pokemon_data['Nature']}")
                st.write(f"**EVs:** {pokemon_data['EVs']}")
                
                st.subheader("Moveset")
                st.write(f"1. {pokemon_data['Move 1']}")
                st.write(f"2. {pokemon_data['Move 2']}")
                st.write(f"3. {pokemon_data['Move 3']}")
                st.write(f"4. {pokemon_data['Move 4']}")
            
            with col2:
                st.subheader("Strategy")
                st.write(pokemon_data['Key Strategy'])
                
                st.subheader("Good Teammates")
                st.write(pokemon_data['Good Teammates'])
                
                st.subheader("Counters")
                st.write(pokemon_data['Counters'])
                
                st.subheader("Performance Scores")
                st.metric("Role Score", pokemon_data['RoleScore'])
                st.metric("Stat Score", pokemon_data['StatScore'])
                st.metric("Move Score", pokemon_data['MoveScore'])
        else:
            st.warning("No Pokémon match the selected filters")
    
    with tab3:
        st.header("Team Synergy Analysis")
        
        if selected_team:
            team_df = df[df['Team'] == selected_team]
            
            # Show team strategy
            st.subheader("Overall Team Strategy")
            sample_strategy = team_df['Key Strategy'].iloc[0]
            st.write(sample_strategy)
            
            # Show team phases
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Early Game")
                st.write(team_df['Early Game'].iloc[0])
            with col2:
                st.subheader("Mid Game")
                st.write(team_df['Mid Game'].iloc[0])
            with col3:
                st.subheader("Late Game")
                st.write(team_df['EndgameWinCon'].iloc[0])
            
            # Show key synergies
            st.subheader("Key Synergies")
            st.write(team_df['Key Synergies'].iloc[0])
            
            # Show win conditions
            st.subheader("Win Conditions")
            st.write(team_df['Win Condition'].iloc[0])

if __name__ == "__main__":
    main()
