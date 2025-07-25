import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Read the file directly with pandas
            data = pd.read_csv(uploaded_file, sep='\t')
            
            # Rest of your function remains the same...
            data.columns = data.columns.str.strip().str.replace('"', '')
            
            st.write("Columns in uploaded data:", list(data.columns))
            
            required_columns = {
                'Team Number': 'Team Number',
                'Team Name': 'Team Name', 
                'Pokemon': 'Pokemon',
                'Role': 'Role'
            }
            
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return None, None
            
            data = data.dropna(how='all', axis=1)
            
            if 'Meta Usage (%)' in data.columns:
                data['Meta Usage (%)'] = pd.to_numeric(
                    data['Meta Usage (%)'].astype(str).str.replace('%', ''),
                    errors='coerce'
                )
            
            agg_dict = {
                'Team Name': 'first',
                'Pokemon': list,
                'Role': list
            }
            
            optional_columns = {
                'Typing (Primary)': list,
                'Typing (Secondary)': list,
                'Format Viability': 'first',
                'Pivot Synergy Rating (1-20)': 'mean',
                'Bulk Score': 'mean',
                'Damage Output Score': 'mean',
                'Meta Usage (%)': 'mean',
                'Archetype Suitability': 'first'
            }
            
            for col, agg_func in optional_columns.items():
                if col in data.columns:
                    agg_dict[col] = agg_func
            
            team_data = data.groupby('Team Number').agg(agg_dict).reset_index()
            
            return data, team_data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None
    return None, None


def main():
    st.set_page_config(page_title="Pokémon Team Builder", page_icon="⚔️", layout="wide")
    
    st.title("⚔️ Pokémon Team Building Recommender")
    st.markdown("""
    Build competitive Pokémon teams based on meta trends, synergies, and roles. 
    Upload your team data CSV to get started.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Pokémon Teams CSV", type=["csv", "tsv"])
    data, team_data = load_data(uploaded_file)
    
    if data is None or team_data is None:
        st.info("Please upload a CSV/TSV file to begin.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Archetype filter
    archetypes = team_data['Archetype Suitability'].unique() if 'Archetype Suitability' in team_data.columns else []
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
    ) if 'Format Viability' in team_data.columns else 0
    
    # Bulk and damage filters
    min_bulk = st.sidebar.slider(
        "Minimum Bulk Score", 
        min_value=0, 
        max_value=100, 
        value=50
    ) if 'Bulk Score' in team_data.columns else 0
    
    min_damage = st.sidebar.slider(
        "Minimum Damage Output", 
        min_value=0, 
        max_value=100, 
        value=50
    ) if 'Damage Output Score' in team_data.columns else 0
    
    # Apply filters
    filter_conditions = []
    
    if selected_archetype and 'Archetype Suitability' in team_data.columns:
        filter_conditions.append(team_data['Archetype Suitability'].isin(selected_archetype))
    
    if 'Format Viability' in team_data.columns:
        filter_conditions.append(team_data['Format Viability'].astype(str).str.rstrip('%').astype(float) >= min_viability)
    
    if 'Bulk Score' in team_data.columns:
        filter_conditions.append(team_data['Bulk Score'] >= min_bulk)
    
    if 'Damage Output Score' in team_data.columns:
        filter_conditions.append(team_data['Damage Output Score'] >= min_damage)
    
    if filter_conditions:
        filtered_teams = team_data[pd.concat(filter_conditions, axis=1).all(axis=1)]
    else:
        filtered_teams = team_data
    
    # Main content
    st.header("Recommended Teams")
    
    if len(filtered_teams) == 0:
        st.warning("No teams match your filters. Try adjusting your criteria.")
    else:
        # Display team cards
        cols = st.columns(3)
        for idx, (_, row) in enumerate(filtered_teams.iterrows()):
            with cols[idx % 3]:
                with st.expander(f"**{row['Team Name']}**"):
                    # Dynamic metric display
                    metrics = []
                    
                    if 'Format Viability' in row:
                        metrics.append(f"**Rating**: {row['Format Viability']}")
                    if 'Archetype Suitability' in row:
                        metrics.append(f"**Archetype**: {row['Archetype Suitability']}")
                    if 'Bulk Score' in row:
                        metrics.append(f"**Avg Bulk**: {row['Bulk Score']:.0f}/100")
                    if 'Damage Output Score' in row:
                        metrics.append(f"**Avg Damage**: {row['Damage Output Score']:.0f}/100")
                    if 'Pivot Synergy Rating (1-20)' in row:
                        metrics.append(f"**Synergy**: {row['Pivot Synergy Rating (1-20)']:.1f}/20")
                    if 'Meta Usage (%)' in row:
                        usage = f"{row['Meta Usage (%)']:.1f}%" if not pd.isna(row['Meta Usage (%)']) else "N/A"
                        metrics.append(f"**Meta Usage**: {usage}")
                    
                    st.markdown("\n".join(metrics))
                    
                    st.markdown("**Team Members:**")
                    for pokemon, role in zip(row['Pokemon'], row['Role']):
                        type_info = ""
                        if 'Typing (Primary)' in row and idx < len(row['Typing (Primary)']):
                            type1 = row['Typing (Primary)'][idx]
                            type2 = row['Typing (Secondary)'][idx] if 'Typing (Secondary)' in row and idx < len(row['Typing (Secondary)']) else None
                            type_info = f" ({type1}{f'/{type2}' if type2 else ''})"
                        st.markdown(f"- {pokemon}{type_info} - *{role}*")
    
        # Team comparison visualization
        if len(filtered_teams) > 1:
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
                    index=1,
                    key="team2"
                )
            
            if team1 and team2:
                compare_metrics = []
                if 'Bulk Score' in filtered_teams.columns:
                    compare_metrics.append('Bulk Score')
                if 'Damage Output Score' in filtered_teams.columns:
                    compare_metrics.append('Damage Output Score')
                if 'Pivot Synergy Rating (1-20)' in filtered_teams.columns:
                    compare_metrics.append('Pivot Synergy Rating (1-20)')
                if 'Meta Usage (%)' in filtered_teams.columns:
                    compare_metrics.append('Meta Usage (%)')
                
                if compare_metrics:
                    compare_data = filtered_teams[
                        filtered_teams['Team Name'].isin([team1, team2])
                    ].set_index('Team Name')[compare_metrics].T.reset_index()
                    
                    fig = px.bar(
                        compare_data, 
                        x='index', 
                        y=[team1, team2],
                        barmode='group',
                        labels={'index': 'Metric', 'value': 'Score'},
                        title="Team Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # "Fill a Gap" tool
    if data is not None:
        st.sidebar.header("Fill a Gap Tool")
        st.sidebar.markdown("Find Pokémon to cover your team's weaknesses")
        
        type_col = 'Typing (Primary)' if 'Typing (Primary)' in data.columns else None
        role_col = 'Role' if 'Role' in data.columns else None
        
        if type_col and role_col:
            needed_type = st.sidebar.selectbox(
                "Needed Type Coverage",
                sorted(data[type_col].unique()),
                index=0
            )
            
            needed_role = st.sidebar.selectbox(
                "Needed Role",
                sorted(data[role_col].unique()),
                index=0
            )
            
            if st.sidebar.button("Find Recommendations"):
                type_condition = (data[type_col] == needed_type)
                if 'Typing (Secondary)' in data.columns:
                    type_condition |= (data['Typing (Secondary)'] == needed_type)
                
                recommendations = data[
                    type_condition &
                    (data[role_col] == needed_role)
                ].sort_values('Damage Output Score' if 'Damage Output Score' in data.columns else 'Pokemon', 
                            ascending=False)
                
                if len(recommendations) > 0:
                    st.sidebar.success(f"Top {needed_type} {needed_role} recommendations:")
                    for _, rec in recommendations.head(3).iterrows():
                        rec_info = [f"**{rec['Pokemon']}**", f"Role: {rec['Role']}"]
                        if 'Damage Output Score' in rec:
                            rec_info.append(f"Damage: {rec['Damage Output Score']}/100")
                        if 'Bulk Score' in rec:
                            rec_info.append(f"Bulk: {rec['Bulk Score']}/100")
                        st.sidebar.markdown("\n".join(rec_info))
                else:
                    st.sidebar.warning("No matching Pokémon found. Try different criteria.")

if __name__ == "__main__":
    main()
