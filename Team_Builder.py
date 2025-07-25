import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

# Load data function with caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded file
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = pd.read_csv(stringio)
        
        # Clean and preprocess data
        data = data.dropna(how='all', axis=1)  # Remove empty columns
        
        # Handle 'Meta Usage (%)' column - convert only numeric percentages
        if 'Meta Usage (%)' in data.columns:
            # Remove % sign if present and try to convert to float
            data['Meta Usage (%)'] = data['Meta Usage (%)'].apply(
                lambda x: float(str(x).rstrip('%')) if '%' in str(data['Meta Usage (%)'].iloc[0]) 
                else pd.to_numeric(data['Meta Usage (%)'], errors='coerce'))
        
        # Create a list of Pokémon for each team
        team_data = data.groupby('Team Number').agg({
            'Team Name': 'first',
            'Pokemon': list,
            'Role': list,
            'Typing (Primary)': list,
            'Typing (Secondary)': list,
            'Format Viability': 'first',
            'Pivot Synergy Rating (1-20)': 'mean',
            'Bulk Score': 'mean',
            'Damage Output Score': 'mean',
            'Meta Usage (%)': 'mean',
            'Archetype Suitability': 'first'
        }).reset_index()
        
        return data, team_data
    return None, None

# Main app function
def main():
    st.set_page_config(page_title="Pokémon Team Builder", page_icon="⚔️", layout="wide")
    
    st.title("⚔️ Pokémon Team Building Recommender")
    st.markdown("""
    Build competitive Pokémon teams based on meta trends, synergies, and roles. 
    Upload your team data CSV to get started.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Pokémon Teams CSV", type="csv")
    data, team_data = load_data(uploaded_file)
    
    if data is None or team_data is None:
        st.info("Please upload a CSV file to begin.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Archetype filter
    archetypes = team_data['Archetype Suitability'].unique()
    selected_archetype = st.sidebar.multiselect(
        "Team Archetype", 
        archetypes, 
        default=archetypes
    )
    
    # Format viability filter
    min_viability = st.sidebar.slider(
        "Minimum Format Viability", 
        min_value=0, 
        max_value=100, 
        value=50,
        step=5,
        format="%d%%"
    )
    
    # Bulk and damage filters
    min_bulk = st.sidebar.slider(
        "Minimum Bulk Score", 
        min_value=0, 
        max_value=100, 
        value=50
    )
    
    min_damage = st.sidebar.slider(
        "Minimum Damage Output", 
        min_value=0, 
        max_value=100, 
        value=50
    )
    
    # Apply filters
    filtered_teams = team_data[
        (team_data['Archetype Suitability'].isin(selected_archetype)) &
        (team_data['Format Viability'].astype(str).str.rstrip('%').astype(float) >= min_viability) &
        (team_data['Bulk Score'] >= min_bulk) &
        (team_data['Damage Output Score'] >= min_damage)
    ]
    
    # Main content
    st.header("Recommended Teams")
    
    if len(filtered_teams) == 0:
        st.warning("No teams match your filters. Try adjusting your criteria.")
    else:
        # Display team cards
        cols = st.columns(3)
        for idx, (_, row) in enumerate(filtered_teams.iterrows()):
            with cols[idx % 3]:
                with st.expander(f"**{row['Team Name']}** (Rating: {row['Format Viability']})"):
                    st.markdown(f"""
                    **Archetype**: {row['Archetype Suitability']}  
                    **Avg Bulk**: {row['Bulk Score']:.0f}/100  
                    **Avg Damage**: {row['Damage Output Score']:.0f}/100  
                    **Synergy**: {row['Pivot Synergy Rating (1-20)']:.1f}/20  
                    **Meta Usage**: {row['Meta Usage (%)']:.1f}% if available
                    """)
                    
                    st.markdown("**Team Members:**")
                    for pokemon, role, type1, type2 in zip(
                        row['Pokemon'], 
                        row['Role'], 
                        row['Typing (Primary)'], 
                        row['Typing (Secondary)']
                    ):
                        type2_display = f"/{type2}" if pd.notna(type2) and str(type2) != 'nan' else ""
                        st.markdown(f"- {pokemon} ({type1}{type2_display}) - *{role}*")
                    
                    # Button to see detailed team view
                    if st.button("View Details", key=f"details_{row['Team Number']}"):
                        st.session_state['selected_team'] = row['Team Number']
    
        # Team comparison visualization
        st.header("Team Comparison")
        compare_col1, compare_col2 = st.columns(2)
        
        with compare_col1:
            team1 = st.selectbox(
                "Select first team to compare",
                filtered_teams['Team Name'],
                key="team1"
            )
        
        with compare_col2:
            team2 = st.selectbox(
                "Select second team to compare",
                filtered_teams['Team Name'],
                index=1 if len(filtered_teams) > 1 else 0,
                key="team2"
            )
        
        if team1 and team2:
            compare_data = filtered_teams[
                filtered_teams['Team Name'].isin([team1, team2])
            ].set_index('Team Name')[
                ['Bulk Score', 'Damage Output Score', 'Pivot Synergy Rating (1-20)', 'Meta Usage (%)']
            ].T.reset_index()
            
            fig = px.bar(
                compare_data, 
                x='index', 
                y=[team1, team2],
                barmode='group',
                labels={'index': 'Metric', 'value': 'Score'},
                title="Team Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed team view
        if 'selected_team' in st.session_state:
            st.header("Team Details")
            selected_team = st.session_state['selected_team']
            team_members = data[data['Team Number'] == selected_team]
            
            for _, member in team_members.iterrows():
                with st.expander(f"**{member['Pokemon']}** ({member['Typing (Primary)']}/{member.get('Typing (Secondary)', '')})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        **Role**: {member['Role']}  
                        **Item**: {member['Item']}  
                        **Ability**: {member['Ability']}  
                        **Nature**: {member['Nature']}  
                        **EVs**: {member['EVs']}  
                        **Speed Tier**: {member['Speed Tier']}
                        """)
                    
                    with col2:
                        st.markdown(f"""
                        **Move 1**: {member['Move 1']}  
                        **Move 2**: {member['Move 2']}  
                        **Move 3**: {member['Move 3']}  
                        **Move 4**: {member['Move 4']}  
                        **Synergy Notes**: {member.get('Synergy Notes', 'N/A')}
                        """)
            
            # Type coverage analysis
            st.subheader("Type Coverage Analysis")
            all_types = []
            for _, member in team_members.iterrows():
                types = [member['Typing (Primary)']]
                if pd.notna(member.get('Typing (Secondary)')):
                    types.append(member['Typing (Secondary)'])
                all_types.extend(types)
            
            type_counts = pd.Series(all_types).value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            
            fig = px.bar(
                type_counts, 
                x='Type', 
                y='Count',
                title="Team Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Weakness analysis
            st.subheader("Team Weakness Analysis")
            weaknesses = []
            for _, member in team_members.iterrows():
                if pd.notna(member.get('Weakness Overlaps')):
                    weaknesses.extend(member['Weakness Overlaps'].split(','))
            
            if weaknesses:
                weakness_counts = pd.Series(weaknesses).value_counts().reset_index()
                weakness_counts.columns = ['Weakness', 'Count']
                
                fig = px.bar(
                    weakness_counts, 
                    x='Weakness', 
                    y='Count',
                    title="Common Team Weaknesses"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No weakness data available for this team.")
    
    # "Fill a Gap" tool
    st.sidebar.header("Fill a Gap Tool")
    st.sidebar.markdown("Find Pokémon to cover your team's weaknesses")
    
    needed_type = st.sidebar.selectbox(
        "Needed Type Coverage",
        sorted(data['Typing (Primary)'].unique()),
        index=0
    )
    
    needed_role = st.sidebar.selectbox(
        "Needed Role",
        sorted(data['Role'].unique()),
        index=0
    )
    
    if st.sidebar.button("Find Recommendations"):
        recommendations = data[
            (data['Typing (Primary)'] == needed_type) |
            (data['Typing (Secondary)'] == needed_type)
        ].sort_values('Damage Output Score', ascending=False)
        
        if len(recommendations) > 0:
            st.sidebar.success(f"Top {needed_type} {needed_role} recommendations:")
            for _, rec in recommendations.head(3).iterrows():
                st.sidebar.markdown(f"""
                **{rec['Pokemon']}**  
                Role: {rec['Role']}  
                Damage: {rec['Damage Output Score']}/100  
                Bulk: {rec['Bulk Score']}/100
                """)
        else:
            st.sidebar.warning("No matching Pokémon found. Try different criteria.")

if __name__ == "__main__":
    main()
