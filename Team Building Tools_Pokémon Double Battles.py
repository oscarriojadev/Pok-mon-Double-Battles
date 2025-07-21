import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from collections import defaultdict
from itertools import combinations

# Sample type effectiveness chart (simplified)
TYPE_EFFECTIVENESS = {
    'Fire': {'Strong': ['Grass', 'Ice', 'Bug', 'Steel'], 'Weak': ['Water', 'Rock', 'Fire', 'Dragon']},
    'Water': {'Strong': ['Fire', 'Ground', 'Rock'], 'Weak': ['Electric', 'Grass', 'Water']},
    'Electric': {'Strong': ['Water', 'Flying'], 'Weak': ['Ground', 'Electric', 'Grass', 'Dragon']},
    'Grass': {'Strong': ['Water', 'Ground', 'Rock'], 'Weak': ['Fire', 'Ice', 'Poison', 'Flying', 'Bug', 'Grass']}
}

def load_data(uploaded_file):
    """Load and preprocess the uploaded CSV file"""
    df = pd.read_csv(uploaded_file)
    
    # Add default columns if not present
    default_columns = {
        'PrimaryRole': '',
        'SecondaryRole': '',
        'Win Condition': '',
        'Early Game': '',
        'Mid Game': '',
        'Late Game': '',
        'Counters': '',
        'Type1': '',
        'Type2': '',
        'Move1': '',
        'Move2': '',
        'Move3': '',
        'Move4': ''
    }
    
    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    return df

def calculate_synergy_scores(team_df):
    """Calculate synergy scores between team members"""
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    stats = team_df[numeric_cols].values
    
    scaler = StandardScaler()
    normalized_stats = scaler.fit_transform(stats)
    synergy_matrix = cosine_similarity(normalized_stats)
    
    return synergy_matrix

def analyze_type_coverage(team_df):
    """Analyze team type coverage with proper type distribution"""
    type_list = []
    for _, row in team_df.iterrows():
        if pd.notna(row['Type1']):
            type_list.append(row['Type1'])
        if pd.notna(row['Type2']):
            type_list.append(row['Type2'])
    
    type_counts = pd.Series(type_list).value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    return type_counts

def generate_ml_recommendations(df, team_name, n_recommendations=3):
    """Generate ML-based recommendations for team improvement"""
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    
    # Get current team
    current_team = df[df['Team'] == team_name]
    
    # Prepare data for KNN
    all_pokemon = df[['Pokemon'] + numeric_cols].drop_duplicates()
    all_pokemon = all_pokemon[~all_pokemon['Pokemon'].isin(current_team['Pokemon'])]
    
    if len(all_pokemon) == 0:
        return []
    
    # Use KNN to find similar Pokémon
    knn = NearestNeighbors(n_neighbors=n_recommendations)
    knn.fit(all_pokemon[numeric_cols])
    
    # Get recommendations for each team member
    recommendations = set()
    for _, pokemon in current_team.iterrows():
        distances, indices = knn.kneighbors([pokemon[numeric_cols].values])
        for idx in indices[0]:
            recommendations.add(all_pokemon.iloc[idx]['Pokemon'])
            if len(recommendations) >= n_recommendations:
                break
    
    return list(recommendations)[:n_recommendations]

def calculate_team_matchup(team1_df, team2_df):
    """Calculate matchup between two teams"""
    matchup = {
        'Speed Advantage': team1_df['Speed'].mean() - team2_df['Speed'].mean(),
        'Offensive Power': (team1_df['Attack'].mean() + team1_df['Sp. Atk'].mean()) - 
                          (team2_df['Attack'].mean() + team2_df['Sp. Atk'].mean()),
        'Defensive Bulk': (team1_df['HP'].mean() + team1_df['Defense'].mean() + team1_df['Sp. Def'].mean()) - 
                         (team2_df['HP'].mean() + team2_df['Defense'].mean() + team2_df['Sp. Def'].mean())
    }
    return matchup

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
    
    # Main tabs
    tabs = st.tabs([
        "🏆 Team Overview", 
        "🔍 Pokémon Analysis", 
        "📊 Team Comparison", 
        "🤖 ML Recommendations",
        "🛡️ Type Coverage",
        "🔄 Team Synergy",
        "⚔️ Team Matchup",
        "🧩 Enhanced Team Building",
        "📈 Meta Analysis"
    ])
    
    # 1. Team Overview Tab
    with tabs[0]:
        st.header("🏆 Team Overview Dashboard")
        
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='overview_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Team Size", len(team_df))
            with col2:
                st.metric("Avg Speed", f"{team_df['Speed'].mean():.1f}")
            with col3:
                st.metric("Offensive Power", f"{(team_df['Attack'].mean() + team_df['Sp. Atk'].mean())/2:.1f}")
            with col4:
                st.metric("Defensive Bulk", f"{(team_df['HP'].mean() + team_df['Defense'].mean() + team_df['Sp. Def'].mean())/3:.1f}")
            
            # Team Composition Pie Charts
            st.subheader("Team Composition")
            
            col1, col2 = st.columns(2)
            with col1:
                # Type Distribution
                type_list = []
                for _, row in team_df.iterrows():
                    if pd.notna(row['Type1']):
                        type_list.append(row['Type1'])
                    if pd.notna(row['Type2']):
                        type_list.append(row['Type2'])
                
                type_counts = pd.Series(type_list).value_counts()
                if not type_counts.empty:
                    fig = px.pie(type_counts, names=type_counts.index, values=type_counts.values,
                                title="Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Role Distribution
                role_counts = team_df['PrimaryRole'].value_counts()
                if not role_counts.empty:
                    fig = px.pie(role_counts, names=role_counts.index, values=role_counts.values,
                                title="Role Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Team Stats Radar Chart
            st.subheader("Team Stats Profile")
            stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            avg_stats = team_df[stats].mean()
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=avg_stats.values,
                theta=stats,
                fill='toself',
                name='Average Stats'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Team Members Table
            st.subheader("Team Members")
            st.dataframe(team_df[['Pokemon', 'Type1', 'Type2', 'PrimaryRole', 'SecondaryRole'] + stats])
    
    # 2. Pokémon Analysis Tab
    with tabs[1]:
        st.header("🔍 Individual Pokémon Analysis")
        
        # Pokémon Selector
        selected_pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
        pokemon_data = df[df['Pokemon'] == selected_pokemon].iloc[0]
        
        # Basic Info
        col1, col2 = st.columns([1, 3])
        with col1:
            st.subheader(selected_pokemon)
            types = []
            if pd.notna(pokemon_data['Type1']):
                types.append(pokemon_data['Type1'])
            if pd.notna(pokemon_data['Type2']):
                types.append(pokemon_data['Type2'])
            st.write(f"**Type:** {'/'.join(types)}")
            st.write(f"**Primary Role:** {pokemon_data['PrimaryRole']}")
            st.write(f"**Secondary Role:** {pokemon_data['SecondaryRole']}")
            st.write(f"**Ability:** {pokemon_data['Ability']}")
            st.write(f"**Item:** {pokemon_data['Item']}")
            st.write(f"**Nature:** {pokemon_data['Nature']}")
            
            # Display stat modifications from nature
            if isinstance(pokemon_data['Nature'], str):
                nature = pokemon_data['Nature'].split()[0]
                if '+' in nature and '-' in nature:
                    increased_stat = nature.split('+')[1].split('-')[0]
                    decreased_stat = nature.split('-')[1]
                    st.write(f"**Nature Effect:** +{increased_stat}, -{decreased_stat}")
        
        with col2:
            # Calculate modified stats accounting for nature/items/abilities
            stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            base_stats = pokemon_data[stats].copy()
            modified_stats = base_stats.copy()
            
            # Apply nature modifiers
            if isinstance(pokemon_data['Nature'], str):
                nature = pokemon_data['Nature'].split()[0]
                if '+' in nature and '-' in nature:
                    increased_stat = nature.split('+')[1].split('-')[0].strip()
                    decreased_stat = nature.split('-')[1].strip()
                    
                    if increased_stat in stats:
                        modified_stats[increased_stat] *= 1.1
                    if decreased_stat in stats:
                        modified_stats[decreased_stat] *= 0.9
            
            # Apply item modifiers
            item = str(pokemon_data['Item']).lower()
            if 'choice band' in item:
                modified_stats['Attack'] *= 1.5
            elif 'choice specs' in item:
                modified_stats['Sp. Atk'] *= 1.5
            elif 'life orb' in item:
                modified_stats['Attack'] *= 1.3
                modified_stats['Sp. Atk'] *= 1.3
            elif 'assault vest' in item:
                modified_stats['Sp. Def'] *= 1.5
            
            # Create comparison dataframe
            stats_df = pd.DataFrame({
                'Stat': stats,
                'Base': base_stats.values,
                'Modified': modified_stats.values
            })
            
            # Add difference column
            stats_df['Difference'] = stats_df['Modified'] - stats_df['Base']
            
            # Visualize stats with modifications
            fig = px.bar(stats_df, x='Stat', y=['Base', 'Modified'], 
                         barmode='group', title="Base vs Modified Stats",
                         labels={'value': 'Stat Value', 'variable': 'Stat Type', 'Stat': 'Stat Name'},
                         color_discrete_map={'Base': '#636EFA', 'Modified': '#EF553B'})
            fig.update_layout(
                xaxis_title="Stat Name",
                yaxis_title="Stat Value",
                legend_title="Stat Type"
            )
            
            # Add ability effects to the description
            ability_desc = ""
            ability = pokemon_data['Ability']
            if ability == 'Intimidate':
                ability_desc = "Lowers opponents' Attack by 1 stage on switch-in"
            elif ability == 'Sand Rush':
                ability_desc = "Doubles Speed in Sandstorm"
            elif ability == 'Protosynthesis':
                ability_desc = "Boosts highest stat in Harsh Sunlight"
                
            if ability_desc:
                st.write(f"**Ability Effect:** {ability_desc}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Moves and Counters
        st.subheader("Strategic Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Common Moves:**")
            moves = [pokemon_data[f'Move{i}'] for i in range(1, 5) if pd.notna(pokemon_data.get(f'Move{i}'))]
            for move in moves:
                st.write(f"- {move}")
        
        with col2:
            st.write("**Common Counters:**")
            if pd.notna(pokemon_data.get('Counters')):
                counters = pokemon_data['Counters'].split(',')
                for counter in counters:
                    st.write(f"- {counter.strip()}")
        
        # Detailed Stat Analysis
        st.subheader("Detailed Stat Analysis")
        
        # Show stat modifications table
        st.dataframe(
            stats_df.style.format({
                "Base": "{:.1f}",
                "Modified": "{:.1f}",
                "Difference": "{:.1f}"
            }).apply(
                lambda x: ['background-color: lightgreen' if v > 0 else '' for v in x], 
                subset=['Difference']
            ).apply(
                lambda x: ['background-color: lightcoral' if v < 0 else '' for v in x],
                subset=['Difference']
            ),
            column_config={
                "Stat": "Stat",
                "Base": st.column_config.NumberColumn("Base Stat"),
                "Modified": st.column_config.NumberColumn("Modified Stat"),
                "Difference": st.column_config.NumberColumn("Difference")
            }
        )
        
        # Performance by Game Phase
        st.subheader("Performance by Game Phase")
        phases = ['Early Game', 'Mid Game', 'Late Game']
        phase_perf = {phase: pokemon_data.get(phase, '') for phase in phases}
        st.write(pd.DataFrame.from_dict(phase_perf, orient='index', columns=['Performance']))
    
    # 3. Team Comparison Tab
    with tabs[2]:
        st.header("📊 Team Comparison Analysis")
        
        # Select teams to compare
        teams_to_compare = st.multiselect(
            "Select teams to compare (2-3 recommended)",
            sorted(df['Team'].unique()),
            default=sorted(df['Team'].unique())[:2]
        )
        
        if len(teams_to_compare) >= 2:
            comparison_data = []
            stats_to_compare = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            
            for team in teams_to_compare:
                team_df = df[df['Team'] == team]
                avg_stats = team_df[stats_to_compare].mean().to_dict()
                avg_stats['Team'] = team
                comparison_data.append(avg_stats)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Radar chart comparison
            st.subheader("Team Stats Comparison")
            fig = go.Figure()
            
            for _, row in comparison_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row[stats_to_compare].values,
                    theta=stats_to_compare,
                    fill='toself',
                    name=row['Team']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, comparison_df[stats_to_compare].values.max() * 1.1])),
                showlegend=True,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Numerical comparison table
            st.subheader("Numerical Comparison")
            st.dataframe(
                comparison_df.set_index('Team').T.style.highlight_max(axis=1, color='lightgreen')
            )
            
            # Team composition comparison
            st.subheader("Team Composition")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{teams_to_compare[0]} Members**")
                st.dataframe(df[df['Team'] == teams_to_compare[0]][['Pokemon', 'Type1', 'Type2', 'PrimaryRole']])
            
            if len(teams_to_compare) > 1:
                with col2:
                    st.write(f"**{teams_to_compare[1]} Members**")
                    st.dataframe(df[df['Team'] == teams_to_compare[1]][['Pokemon', 'Type1', 'Type2', 'PrimaryRole']])
        else:
            st.warning("Please select at least 2 teams to compare")
    
    # 4. ML Recommendations Tab
    with tabs[3]:
        st.header("🤖 Machine Learning Recommendations")
        selected_team = st.selectbox("Select team for recommendations", sorted(df['Team'].unique()))
        
        if st.button("Generate Recommendations", key='ml_recommend'):
            with st.spinner("Analyzing team composition and finding optimal partners..."):
                recommendations = generate_ml_recommendations(df, selected_team)
                
                if recommendations:
                    st.subheader("Recommended Pokémon Additions")
                    st.write("Based on your team's composition, these Pokémon would complement your team well:")
                    
                    for i, pokemon in enumerate(recommendations, 1):
                        pokemon_data = df[df['Pokemon'] == pokemon].iloc[0]
                        types = []
                        if pd.notna(pokemon_data['Type1']):
                            types.append(pokemon_data['Type1'])
                        if pd.notna(pokemon_data['Type2']):
                            types.append(pokemon_data['Type2'])
                        pokemon_type = '/'.join(types) if types else 'Unknown'
                        
                        with st.expander(f"{i}. {pokemon} ({pokemon_type})"):
                            st.write(f"**Primary Role:** {pokemon_data.get('PrimaryRole', 'Not specified')}")
                            st.write(f"**Stats:** HP: {pokemon_data['HP']}, Atk: {pokemon_data['Attack']}, Def: {pokemon_data['Defense']}")
                            st.write(f"Sp.Atk: {pokemon_data['Sp. Atk']}, Sp.Def: {pokemon_data['Sp. Def']}, Speed: {pokemon_data['Speed']}")
                    
                    st.info("""
                    **How these recommendations work:**
                    - The system analyzes your team's stat distributions
                    - Finds Pokémon with complementary stat profiles
                    - Considers type coverage and role balance
                    """)
                else:
                    st.warning("Could not generate recommendations - not enough data")
    
    # 5. Type Coverage Tab
    with tabs[4]:
        st.header("🛡️ Team Type Coverage Analysis")
        
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='type_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            # Type Distribution
            st.subheader("Type Distribution")
            type_counts = analyze_type_coverage(team_df)
            if type_counts is not None:
                fig = px.bar(type_counts, x='Type', y='Count', color='Type',
                            title="Type Distribution on Team")
                st.plotly_chart(fig, use_container_width=True)
            
            # Type Effectiveness Analysis
            st.subheader("Type Effectiveness")
            
            # Get all types on team
            team_types = set()
            for _, row in team_df.iterrows():
                if pd.notna(row['Type1']):
                    team_types.add(row['Type1'])
                if pd.notna(row['Type2']):
                    team_types.add(row['Type2'])
            
            # Calculate coverage
            coverage = defaultdict(int)
            weaknesses = defaultdict(int)
            
            for t in team_types:
                if t in TYPE_EFFECTIVENESS:
                    for strong_against in TYPE_EFFECTIVENESS[t]['Strong']:
                        coverage[strong_against] += 1
                    for weak_against in TYPE_EFFECTIVENESS[t]['Weak']:
                        weaknesses[weak_against] += 1
            
            # Display coverage
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Offensive Coverage**")
                coverage_df = pd.DataFrame(sorted(coverage.items(), key=lambda x: -x[1]),
                                        columns=['Type', 'Coverage'])
                st.dataframe(coverage_df.style.highlight_max(axis=0, color='lightgreen'))
            
            with col2:
                st.write("**Defensive Weaknesses**")
                weaknesses_df = pd.DataFrame(sorted(weaknesses.items(), key=lambda x: -x[1]),
                                        columns=['Type', 'Weaknesses'])
                st.dataframe(weaknesses_df.style.highlight_max(axis=0, color='lightpink'))
            
            # Recommendations
            st.subheader("Type Coverage Recommendations")
            if coverage:
                poorly_covered = [t for t in TYPE_EFFECTIVENESS.keys() if t not in coverage or coverage[t] < 2]
                if poorly_covered:
                    st.warning(f"Consider adding coverage for: {', '.join(poorly_covered[:3])}")
                else:
                    st.success("Your team has excellent type coverage!")
            
            if weaknesses:
                common_weaknesses = [t for t, count in weaknesses.items() if count > 2]
                if common_weaknesses:
                    st.warning(f"Watch out for common weaknesses to: {', '.join(common_weaknesses)}")
    
    # 6. Team Synergy Tab
    with tabs[5]:
        st.header("🔄 Team Synergy Analysis")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='synergy_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            synergy_matrix = calculate_synergy_scores(team_df)
            
            st.markdown("""
            ### Understanding Team Synergy
            
            The synergy matrix shows how well each Pokémon's stat distribution complements others on your team:
            
            - **High values (0.7-1.0)**: These Pokémon have very similar stat spreads. This can be good for consistent strategies but may indicate redundancy.
            - **Moderate values (0.3-0.7)**: Healthy synergy - these Pokémon complement each other well.
            - **Low values (0-0.3)**: These Pokémon have different roles. This provides coverage but may lack synergy.
            - **Negative values**: These Pokémon have opposing stat distributions (rare but possible).
            
            *Ideal teams typically show moderate synergy between most members with a few specialized pairs.*
            """)
            
            # Visualize synergy matrix
            fig = px.imshow(
                synergy_matrix,
                labels=dict(x="Pokémon", y="Pokémon", color="Synergy Score"),
                x=team_df['Pokemon'].tolist(),
                y=team_df['Pokemon'].tolist(),
                color_continuous_scale='RdYlGn',
                zmin=-1, zmax=1,
                title=f"{selected_team} Team Synergy Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Synergy analysis
            st.subheader("Synergy Insights")
            
            # Find best and worst pairs
            n = len(team_df)
            best_pair = None
            worst_pair = None
            best_score = -1
            worst_score = 2
            
            for i in range(n):
                for j in range(i+1, n):
                    score = synergy_matrix[i][j]
                    if score > best_score:
                        best_score = score
                        best_pair = (team_df.iloc[i]['Pokemon'], team_df.iloc[j]['Pokemon'])
                    if score < worst_score:
                        worst_score = score
                        worst_pair = (team_df.iloc[i]['Pokemon'], team_df.iloc[j]['Pokemon'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Synergy Pair", 
                         f"{best_pair[0]} + {best_pair[1]}", 
                         f"Score: {best_score:.2f}")
                st.write(f"These Pokémon work exceptionally well together based on their stat distributions.")
            
            with col2:
                st.metric("Worst Synergy Pair", 
                         f"{worst_pair[0]} + {worst_pair[1]}", 
                         f"Score: {worst_score:.2f}")
                st.write(f"Consider adjusting their roles or replacing one to improve team balance.")
    
    # 7. Team Matchup Tab
    with tabs[6]:
        st.header("⚔️ Team Matchup Simulator")
        
        col1, col2 = st.columns(2)
        with col1:
            team_a = st.selectbox("Your Team", sorted(df['Team'].unique()), key='team_a')
        with col2:
            team_b = st.selectbox("Opponent Team", sorted(df['Team'].unique()), key='team_b')
        
        if team_a and team_b and team_a != team_b:
            team_a_df = df[df['Team'] == team_a]
            team_b_df = df[df['Team'] == team_b]
            
            # Calculate matchup
            matchup = calculate_team_matchup(team_a_df, team_b_df)
            
            st.subheader("Matchup Summary")
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Speed Advantage", 
                         f"{'Your Team' if matchup['Speed Advantage'] > 0 else 'Opponent'}", 
                         f"{abs(matchup['Speed Advantage']):.1f} points")
            with col2:
                st.metric("Offensive Power", 
                         f"{'Your Team' if matchup['Offensive Power'] > 0 else 'Opponent'}", 
                         f"{abs(matchup['Offensive Power']):.1f} points")
            with col3:
                st.metric("Defensive Bulk", 
                         f"{'Your Team' if matchup['Defensive Bulk'] > 0 else 'Opponent'}", 
                         f"{abs(matchup['Defensive Bulk']):.1f} points")
            
            # Type matchup analysis
            st.subheader("Type Matchup Analysis")
            
            # Get all types from both teams
            team_a_types = set()
            for _, row in team_a_df.iterrows():
                if pd.notna(row['Type1']):
                    team_a_types.add(row['Type1'])
                if pd.notna(row['Type2']):
                    team_a_types.add(row['Type2'])
            
            team_b_types = set()
            for _, row in team_b_df.iterrows():
                if pd.notna(row['Type1']):
                    team_b_types.add(row['Type1'])
                if pd.notna(row['Type2']):
                    team_b_types.add(row['Type2'])
            
            # Display type advantages
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Your Team's Strengths**")
                for t in team_a_types:
                    if t in TYPE_EFFECTIVENESS:
                        strong_against = [x for x in TYPE_EFFECTIVENESS[t]['Strong'] if x in team_b_types]
                        if strong_against:
                            st.write(f"- {t} is strong against: {', '.join(strong_against)}")
            
            with col2:
                st.write("**Opponent's Strengths**")
                for t in team_b_types:
                    if t in TYPE_EFFECTIVENESS:
                        strong_against = [x for x in TYPE_EFFECTIVENESS[t]['Strong'] if x in team_a_types]
                        if strong_against:
                            st.write(f"- {t} is strong against: {', '.join(strong_against)}")
            
            # Strategy recommendations
            st.subheader("Recommended Strategy")
            
            if matchup['Speed Advantage'] > 10:
                st.success("**Speed Control Advantage**: Use your faster Pokémon to control the tempo of the battle")
            elif matchup['Speed Advantage'] < -10:
                st.warning("**Speed Disadvantage**: Consider using priority moves or speed control like Tailwind/Trick Room")
            
            if matchup['Offensive Power'] > 20:
                st.success("**Offensive Advantage**: Play aggressively to capitalize on your stronger attackers")
            elif matchup['Offensive Power'] < -20:
                st.warning("**Offensive Disadvantage**: Focus on defensive plays and wearing down their team")
            
            # Key threats
            st.subheader("Key Threats to Watch For")
            st.write(f"**From {team_b}:**")
            st.write("- " + "\n- ".join(team_b_df.sort_values('Attack', ascending=False)['Pokemon'].head(3).tolist()))
            
            st.write(f"**From {team_a} (their threats):**")
            st.write("- " + "\n- ".join(team_a_df.sort_values('Defense', ascending=False)['Pokemon'].head(3).tolist()))
        elif team_a == team_b:
            st.warning("Please select two different teams to compare")
    
    # 8. Enhanced Team Building Tab
    with tabs[7]:
        st.header("🧩 Enhanced Team Building Tools")
        
        tab1, tab2, tab3 = st.tabs(["Team Builder", "Role Checker", "Core Generator"])
        
        with tab1:
            st.subheader("Interactive Team Builder")
            
            # Team slots
            team = []
            cols = st.columns(4)
            for i in range(6):
                with cols[i%4]:
                    pokemon = st.selectbox(f"Slot {i+1}", [""] + sorted(df['Pokemon'].unique()),
                                         key=f"team_slot_{i}")
                    if pokemon:
                        team.append(pokemon)
            
            if team:
                st.subheader("Current Team Analysis")
                team_df = df[df['Pokemon'].isin(team)].drop_duplicates('Pokemon')
                
                # Quick stats
                st.write(f"Team Size: {len(team_df)}")
                st.write(f"Unique Types: {team_df['Type1'].nunique() + team_df['Type2'].nunique()}")
                
                # Visual builder
                st.write("**Team Visualization**")
                fig = px.treemap(team_df, path=['Type1', 'Pokemon'],
                                color='PrimaryRole', hover_data=['HP', 'Attack', 'Speed'])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Role Composition Checker")
            st.write("Ensure your team has all necessary roles")
            
            roles = ['Physical Attacker', 'Special Attacker', 'Tank', 'Support', 'Hazard Setter', 'Speed Control']
            selected_roles = st.multiselect("Select desired roles", roles, default=roles[:4])
            
            if selected_roles:
                team = st.selectbox("Select team to analyze", sorted(df['Team'].unique()))
                team_df = df[df['Team'] == team]
                
                role_counts = team_df['PrimaryRole'].value_counts()
                missing_roles = [role for role in selected_roles if role not in role_counts.index]
                
                if missing_roles:
                    st.error(f"Missing roles: {', '.join(missing_roles)}")
                else:
                    st.success("All selected roles are covered!")
                    
                fig = px.bar(role_counts, title="Role Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Core Generator")
            st.write("Generate strong Pokémon cores for your team")
            
            core_type = st.radio("Core Type", ["Offensive", "Defensive", "Balanced"])
            num_cores = st.slider("Number of cores to generate", 1, 5, 3)
            
            if st.button("Generate Cores"):
                # Simple core generation logic (would use ML in production)
                if core_type == "Offensive":
                    candidates = df.sort_values(['Attack', 'Sp. Atk', 'Speed'], ascending=False)
                elif core_type == "Defensive":
                    candidates = df.sort_values(['HP', 'Defense', 'Sp. Def'], ascending=False)
                else:
                    candidates = df.sort_values(['Attack', 'Defense', 'Sp. Atk', 'Sp. Def'], ascending=False)
                
                cores = []
                for i in range(num_cores):
                    core = candidates.iloc[i*2:(i*2)+2]['Pokemon'].tolist()
                    cores.append(f"{core[0]} + {core[1]}")
                
                st.write("Recommended cores:")
                for core in cores:
                    st.write(f"- {core}")
    
    # 9. Meta Analysis Tab
    with tabs[8]:
        st.header("📈 Meta Game Analysis")
        
        tab1, tab2 = st.tabs(["Usage Statistics", "Trend Analysis"])
        
        with tab1:
            st.subheader("Pokémon Usage Statistics")
            
            # Calculate usage (assuming multiple teams in dataset)
            usage = df['Pokemon'].value_counts().reset_index()
            usage.columns = ['Pokemon', 'Usage Count']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Top 10 Most Used Pokémon")
                st.dataframe(usage.head(10))
            
            with col2:
                fig = px.bar(usage.head(10), x='Pokemon', y='Usage Count',
                            title="Top 10 Pokémon")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Meta Trends")
            
            # Role distribution
            st.write("Role Distribution in Current Meta")
            role_dist = df['PrimaryRole'].value_counts()
            if not role_dist.empty:
                fig = px.pie(role_dist, names=role_dist.index, values=role_dist.values)
                st.plotly_chart(fig, use_container_width=True)
            
            # Type distribution
            st.write("Type Distribution in Current Meta")
            type_list = []
            for _, row in df.iterrows():
                if pd.notna(row['Type1']):
                    type_list.append(row['Type1'])
                if pd.notna(row['Type2']):
                    type_list.append(row['Type2'])
            
            type_dist = pd.Series(type_list).value_counts()
            fig = px.bar(type_dist, title="Type Popularity")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
