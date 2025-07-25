import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Optional

# Constants
TYPE_COLORS = {
    'Normal': '#A8A878', 'Fire': '#F08030', 'Water': '#6890F0',
    'Electric': '#F8D030', 'Grass': '#78C850', 'Ice': '#98D8D8',
    'Fighting': '#C03028', 'Poison': '#A040A0', 'Ground': '#E0C068',
    'Flying': '#A890F0', 'Psychic': '#F85888', 'Bug': '#A8B820',
    'Rock': '#B8A038', 'Ghost': '#705898', 'Dragon': '#7038F8',
    'Dark': '#705848', 'Steel': '#B8B8D0', 'Fairy': '#EE99AC'
}

# Data Loading with enhanced caching
@st.cache_data
def load_data(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if uploaded_file is None:
        return None, None
    
    try:
        # Read with automatic separator detection
        data = pd.read_csv(uploaded_file, sep=None, engine='python')
        
        # Clean column names
        data.columns = (
            data.columns
            .str.strip()
            .str.replace('"', '')
            .str.replace('  ', ' ')
            .str.replace('(', ' (')
        )
        
        # Convert numeric columns safely
        def safe_convert(col):
            if col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = (
                        data[col]
                        .astype(str)
                        .str.replace('%', '')
                        .str.replace('−', '-')  # Handle different minus signs
                    )
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(0)
        
        numeric_cols = [
            'Format Viability', 'Pivot Synergy Rating (1-20)',
            'Bulk Score', 'Damage Output Score', 'Meta Usage (%)'
        ]
        
        for col in numeric_cols:
            safe_convert(col)
        
        # Verify required columns
        required_cols = ['Team Number', 'Team Name', 'Pokemon', 'Role']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None, None
        
        # Create team data aggregation
        agg_dict = {
            'Team Name': 'first',
            'Pokemon': list,
            'Role': list,
            'Typing (Primary)': list,
            'Typing (Secondary)': list,
            'Format Viability': 'mean',
            'Pivot Synergy Rating (1-20)': 'mean',
            'Bulk Score': 'mean',
            'Damage Output Score': 'mean',
            'Meta Usage (%)': 'mean',
            'Archetype Suitability': 'first'
        }
        
        # Only include columns that exist in the data
        agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}
        
        team_data = data.groupby('Team Number', as_index=False).agg(agg_dict)
        
        return data, team_data
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Visualization Functions
def plot_team_comparison(team1, team2, metrics):
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=[metric],
            y=[team1[metric]],
            name=team1['Team Name'],
            marker_color='#FF6B6B',
            text=[f"{team1[metric]:.1f}"],
            textposition='auto'
        ))
        fig.add_trace(go.Bar(
            x=[metric],
            y=[team2[metric]],
            name=team2['Team Name'],
            marker_color='#4ECDC4',
            text=[f"{team2[metric]:.1f}"],
            textposition='auto'
        ))
    
    fig.update_layout(
        barmode='group',
        title='Team Comparison',
        yaxis_title='Score',
        showlegend=True
    )
    return fig

def plot_type_coverage(types_list):
    type_counts = pd.Series(types_list).value_counts()
    fig = px.bar(
        x=type_counts.index,
        y=type_counts.values,
        color=type_counts.index,
        color_discrete_map=TYPE_COLORS,
        labels={'x': 'Type', 'y': 'Count'},
        title='Team Type Coverage'
    )
    fig.update_layout(showlegend=False)
    return fig

# Main App
def main():
    st.set_page_config(
        page_title="Pokémon Team Builder",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚔️ Pokémon Team Building Recommender")
    st.markdown("""
    Build competitive Pokémon teams based on meta trends, synergies, and roles. 
    Upload your team data CSV to get started.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your Pokémon Teams CSV",
        type=["csv", "tsv"],
        help="Upload a CSV/TSV file with your Pokémon team data"
    )
    
    data, team_data = load_data(uploaded_file)
    
    if data is None:
        st.info("Please upload a file to begin analysis.")
        return
    
    # Sidebar Filters
    st.sidebar.header("Team Filters")
    
    # Archetype filter
    archetypes = team_data['Archetype Suitability'].unique()
    selected_archetypes = st.sidebar.multiselect(
        "Team Archetypes",
        archetypes,
        default=archetypes,
        help="Filter teams by their strategic archetype"
    )
    
    # Metric filters
    min_viability = st.sidebar.slider(
        "Minimum Format Viability",
        0, 100, 50,
        help="Filter by team viability score"
    )
    
    min_bulk = st.sidebar.slider(
        "Minimum Bulk Score",
        0, 100, 50,
        help="Filter by team defensive bulk"
    )
    
    min_damage = st.sidebar.slider(
        "Minimum Damage Output",
        0, 100, 50,
        help="Filter by team offensive power"
    )
    
    # Apply filters
    filtered_teams = team_data[
        (team_data['Archetype Suitability'].isin(selected_archetypes)) &
        (team_data['Format Viability'] >= min_viability) &
        (team_data['Bulk Score'] >= min_bulk) &
        (team_data['Damage Output Score'] >= min_damage)
    ]
    
    # Main Content
    st.header("Recommended Teams")
    
    if len(filtered_teams) == 0:
        st.warning("No teams match your filters. Try adjusting your criteria.")
    else:
        # Team Cards
        cols = st.columns(3)
        for idx, (_, row) in enumerate(filtered_teams.iterrows()):
            with cols[idx % 3]:
                with st.expander(f"**{row['Team Name']}**", expanded=False):
                    # Team Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Viability", f"{row['Format Viability']:.1f}")
                        st.metric("Bulk", f"{row['Bulk Score']:.1f}")
                    with col2:
                        st.metric("Damage", f"{row['Damage Output Score']:.1f}")
                        st.metric("Synergy", f"{row['Pivot Synergy Rating (1-20)']:.1f}/20")
                    
                    # Team Members
                    st.subheader("Team Members")
                    for i, (pokemon, role) in enumerate(zip(row['Pokemon'], row['Role'])):
                        type1 = row['Typing (Primary)'][i] if i < len(row['Typing (Primary)']) else "?"
                        type2 = row['Typing (Secondary)'][i] if i < len(row['Typing (Secondary)']) else None
                        
                        type_display = f"{type1}{f'/{type2}' if type2 else ''}"
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <div style="width: 100px; font-weight: bold;">{pokemon}</div>
                            <div style="background-color: {TYPE_COLORS.get(type1, '#777')}; 
                                        color: white; padding: 2px 8px; border-radius: 4px; 
                                        margin-right: 4px;">
                                {type1}
                            </div>
                            {f'<div style="background-color: {TYPE_COLORS.get(type2, "#777")}; color: white; padding: 2px 8px; border-radius: 4px;">{type2}</div>' if type2 else ''}
                            <div style="margin-left: auto; font-style: italic;">{role}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Team Comparison Tool
        st.header("Team Comparison")
        if len(filtered_teams) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                team1 = st.selectbox(
                    "Select first team",
                    filtered_teams['Team Name'],
                    key="team1"
                )
            with col2:
                team2 = st.selectbox(
                    "Select second team",
                    filtered_teams['Team Name'],
                    index=min(1, len(filtered_teams)-1),
                    key="team2"
                )
            
            team1_data = filtered_teams[filtered_teams['Team Name'] == team1].iloc[0]
            team2_data = filtered_teams[filtered_teams['Team Name'] == team2].iloc[0]
            
            metrics = [
                'Format Viability', 'Bulk Score',
                'Damage Output Score', 'Pivot Synergy Rating (1-20)'
            ]
            metrics = [m for m in metrics if m in filtered_teams.columns]
            
            st.plotly_chart(
                plot_team_comparison(team1_data, team2_data, metrics),
                use_container_width=True
            )
        
        # Type Coverage Analysis
        st.header("Type Coverage Analysis")
        if 'Typing (Primary)' in data.columns:
            all_types = data['Typing (Primary)'].tolist()
            if 'Typing (Secondary)' in data.columns:
                all_types += data['Typing (Secondary)'].dropna().tolist()
            
            st.plotly_chart(
                plot_type_coverage(all_types),
                use_container_width=True
            )
    
    # Fill a Gap Tool
    st.sidebar.header("Fill a Gap Tool")
    if 'Typing (Primary)' in data.columns and 'Role' in data.columns:
        needed_type = st.sidebar.selectbox(
            "Needed Type Coverage",
            sorted(data['Typing (Primary)'].unique()),
            index=0,
            help="Select a type you need coverage for"
        )
        
        needed_role = st.sidebar.selectbox(
            "Needed Role",
            sorted(data['Role'].unique()),
            index=0,
            help="Select the role you need filled"
        )
        
        if st.sidebar.button("Find Recommendations"):
            type_condition = (data['Typing (Primary)'] == needed_type)
            if 'Typing (Secondary)' in data.columns:
                type_condition |= (data['Typing (Secondary)'] == needed_type)
            
            recommendations = data[
                type_condition &
                (data['Role'] == needed_role)
            ].sort_values(
                'Damage Output Score' if 'Damage Output Score' in data.columns else 'Pokemon',
                ascending=False
            )
            
            if len(recommendations) > 0:
                st.sidebar.success(f"Top {needed_type} {needed_role} recommendations:")
                for _, rec in recommendations.head(3).iterrows():
                    rec_info = [
                        f"**{rec['Pokemon']}**",
                        f"**Role:** {rec['Role']}",
                        f"**Primary Type:** {rec['Typing (Primary)']}",
                        f"**Secondary Type:** {rec.get('Typing (Secondary)', 'None')}",
                    ]
                    
                    if 'Damage Output Score' in rec:
                        rec_info.append(f"**Damage:** {rec['Damage Output Score']:.1f}")
                    if 'Bulk Score' in rec:
                        rec_info.append(f"**Bulk:** {rec['Bulk Score']:.1f}")
                    
                    st.sidebar.markdown("<br>".join(rec_info))
                    st.sidebar.markdown("---")
            else:
                st.sidebar.warning("No matching Pokémon found. Try different criteria.")

if __name__ == "__main__":
    main()
