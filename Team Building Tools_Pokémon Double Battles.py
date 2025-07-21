import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from collections import defaultdict
from itertools import combinations

def load_data(uploaded_file):
    """Load and preprocess the uploaded CSV file"""
    df = pd.read_csv(uploaded_file)
    
    # Add default columns if not present
    if 'PrimaryRole' not in df.columns:
        df['PrimaryRole'] = ''
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
    if 'Counters' not in df.columns:
        df['Counters'] = ''
    
    return df

def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer Pro")
    
    st.title("⚔️ Pokémon Competitive Team Analyzer Pro")
    st.write("""
    Advanced toolkit for competitive Pokémon team analysis with enhanced team building tools and meta analysis.
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
    
    # Add default columns if not present (same as before)
    
    # Main tabs - expanded with new features
    tab_names = [
        "🏆 Team Overview", 
        "🔍 Pokémon Analysis", 
        "📊 Team Comparison", 
        "🤖 ML Recommendations",
        "🛡️ Type Coverage",
        "🔄 Team Synergy",
        "⚔️ Team Matchup",
        "🧩 Enhanced Team Building",
        "📈 Meta Analysis"
    ]
    
    tabs = st.tabs(tab_names)
    
    # [Previous tab implementations remain unchanged until the new tabs]
    
    with tabs[7]:  # Enhanced Team Building tab
        st.header("🧩 Enhanced Team Building Tools")
        
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='enhanced_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            st.subheader("🎭 Role Synergy Compatibility Matrix")
            
            # Create role compatibility matrix
            roles = team_df['PrimaryRole'].unique()
            compatibility = pd.DataFrame(index=roles, columns=roles)
            
            # Simple compatibility scoring (would be more sophisticated in real implementation)
            ROLE_SYNERGY = {
                'Sweeper': {'Support': 2, 'Wallbreaker': 1, 'Disruptor': 1},
                'Wallbreaker': {'Support': 1, 'Tank': 1, 'Sweeper': 1},
                'Support': {'Sweeper': 2, 'Tank': 1, 'Disruptor': 1},
                'Tank': {'Support': 1, 'Wallbreaker': 1},
                'Disruptor': {'Sweeper': 1, 'Wallbreaker': 1}
            }
            
            for r1 in roles:
                for r2 in roles:
                    if r1 == r2:
                        compatibility.loc[r1, r2] = 0
                    else:
                        score = ROLE_SYNERGY.get(r1, {}).get(r2, 0) + ROLE_SYNERGY.get(r2, {}).get(r1, 0)
                        compatibility.loc[r1, r2] = score
            
            # Visualize compatibility matrix
            fig = px.imshow(compatibility, 
                           labels=dict(x="Role", y="Role", color="Synergy"),
                           x=compatibility.columns,
                           y=compatibility.index,
                           title="Role Synergy Compatibility Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🔄 Secondary Role Optimization")
            
            # Analyze secondary role coverage
            primary_roles = team_df['PrimaryRole'].value_counts()
            secondary_roles = team_df['SecondaryRole'].value_counts()
            
            st.write("**Current Role Distribution:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Primary Roles:")
                st.write(primary_roles)
            with col2:
                st.write("Secondary Roles:")
                st.write(secondary_roles)
            
            # Suggest secondary role improvements
            st.write("**Suggested Secondary Role Improvements:**")
            role_coverage = defaultdict(int)
            for _, row in team_df.iterrows():
                role_coverage[row['PrimaryRole']] += 1
                if pd.notna(row['SecondaryRole']) and row['SecondaryRole'] != '':
                    role_coverage[row['SecondaryRole']] += 0.5
            
            suggested_roles = []
            for role in ['Support', 'Disruptor', 'Tank']:
                if role_coverage.get(role, 0) < 1:
                    suggested_roles.append(role)
            
            if suggested_roles:
                st.warning(f"Consider adding these secondary roles: {', '.join(suggested_roles)}")
                st.write("Potential candidates:")
                for pokemon in team_df['Pokemon']:
                    pokemon_data = df[df['Pokemon'] == pokemon].iloc[0]
                    if pokemon_data['SecondaryRole'] == '':
                        st.write(f"- {pokemon} could take on: {', '.join(suggested_roles)}")
            else:
                st.success("Team has good secondary role coverage!")
            
            st.subheader("🗺️ Win Condition Pathway Mapper")
            
            # Extract win conditions
            win_conditions = team_df['Win Condition'].dropna().unique()
            
            if len(win_conditions) > 0:
                st.write("**Primary Win Conditions:**")
                for wc in win_conditions:
                    st.write(f"- {wc}")
                
                # Create pathway visualization
                st.write("**Execution Pathway:**")
                phases = ['Early Game', 'Mid Game', 'Late Game']
                pathway = {}
                
                for phase in phases:
                    phase_data = team_df[phase].dropna().unique()
                    if len(phase_data) > 0:
                        pathway[phase] = phase_data[0]
                
                if pathway:
                    for phase, desc in pathway.items():
                        st.write(f"**{phase}:** {desc}")
                    
                    # Visualize as flowchart
                    st.graphviz_chart("""
                    digraph {
                        node [shape=box]
                        Early -> Mid -> Late
                        Early [label="Early Game"]
                        Mid [label="Mid Game"]
                        Late [label="Late Game"]
                    }
                    """)
                else:
                    st.warning("No phase information available")
                
                # Identify backup win conditions
                st.subheader("🛡️ Backup Win Condition Identifier")
                
                # Look for alternative strategies in the team
                backup_conditions = []
                if 'Perish Song' in str(win_conditions):
                    backup_conditions.append("Stall with Protect/Recover")
                if 'Sand' in str(win_conditions):
                    backup_conditions.append("Weather-based damage")
                if 'Tailwind' in str(win_conditions):
                    backup_conditions.append("Speed control")
                
                if backup_conditions:
                    st.success(f"Potential backup win conditions: {', '.join(backup_conditions)}")
                else:
                    st.warning("No clear backup win conditions identified")
            else:
                st.warning("No explicit win conditions defined for this team")
    
    with tabs[8]:  # Meta Analysis tab
        st.header("📈 Meta Analysis Tools")
        
        st.subheader("📊 Usage Statistics")
        
        # Calculate usage statistics
        usage_stats = df['Pokemon'].value_counts().reset_index()
        usage_stats.columns = ['Pokemon', 'Usage Count']
        
        fig = px.bar(usage_stats.head(20), 
                     x='Pokemon', y='Usage Count',
                     title="Top 20 Most Used Pokémon")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🏛️ Popular Team Archetypes")
        
        # Detect common team archetypes
        team_archetypes = df['Team'].value_counts().reset_index()
        team_archetypes.columns = ['Team Name', 'Count']
        
        # Simple archetype detection based on team name
        archetype_map = {
            'Sand': ['Sand', 'Drill', 'Tyranitar'],
            'Rain': ['Rain', 'Pelipper', 'Swift Swim'],
            'Trick Room': ['TR', 'Trick Room', 'Slow'],
            'Hyper Offense': ['Hyper', 'Offense', 'Sweep'],
            'Balance': ['Balance', 'Core', 'Pivot']
        }
        
        detected_archetypes = []
        for team in team_archetypes['Team Name']:
            for archetype, keywords in archetype_map.items():
                if any(kw in team for kw in keywords):
                    detected_archetypes.append(archetype)
                    break
        
        if detected_archetypes:
            archetype_counts = pd.Series(detected_archetypes).value_counts()
            fig = px.pie(archetype_counts, 
                          names=archetype_counts.index,
                          values=archetype_counts.values,
                          title="Team Archetype Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not detect common archetypes from team names")
        
        st.subheader("⚠️ Meta Threat Assessment")
        
        # Identify common threats
        top_pokemon = usage_stats.head(10)['Pokemon'].tolist()
        st.write("**Top Meta Threats:**")
        for i, pokemon in enumerate(top_pokemon, 1):
            st.write(f"{i}. {pokemon}")
        
        # Suggest counters
        st.write("**Suggested Counters:**")
        for pokemon in top_pokemon:
            pokemon_data = df[df['Pokemon'] == pokemon].iloc[0]
            counters = pokemon_data['Counters']
            if pd.notna(counters):
                st.write(f"- **{pokemon}** is countered by: {counters}")
        
        st.subheader("🔄 Counter-Meta Team Suggestions")
        
        # Suggest anti-meta team compositions
        anti_meta_suggestions = {
            'Stall': "Use bulky teams with recovery to outlast hyper offense",
            'Hyper Offense': "Use priority moves and focus sash to disrupt sweepers",
            'Weather Teams': "Use opposing weather or weather cancellers",
            'Trick Room': "Use Taunt or fast Pokémon that can function outside TR"
        }
        
        # Determine most common archetype to counter
        if detected_archetypes:
            main_archetype = archetype_counts.index[0]
            st.write(f"Most common archetype: **{main_archetype}**")
            st.write(f"**Counter strategy:** {anti_meta_suggestions.get(main_archetype, 'Use balanced team with diverse answers')}")
            
            # Suggest specific Pokémon
            st.write("**Suggested Anti-Meta Pokémon:**")
            if main_archetype == 'Hyper Offense':
                st.write("- Toxapex (bulky wall)")
                st.write("- Ferrothorn (hazard setter)")
                st.write("- Porygon2 (TR counter)")
            elif main_archetype == 'Rain':
                st.write("- Gastrodon (Storm Drain)")
                st.write("- Ferrothorn (resists water)")
                st.write("- Torkoal (sun setter)")
        else:
            st.info("No dominant archetype detected - balanced teams recommended")

if __name__ == "__main__":
    main()
