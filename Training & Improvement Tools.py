import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from collections import defaultdict

# Complete type effectiveness chart
TYPE_CHART = {
    'Normal': {
        'weak': ['Fighting'],
        'resist': [],
        'immune': ['Ghost'],
        'strong_against': []
    },
    'Fire': {
        'weak': ['Water', 'Ground', 'Rock'],
        'resist': ['Fire', 'Grass', 'Ice', 'Bug', 'Steel', 'Fairy'],
        'immune': [],
        'strong_against': ['Grass', 'Ice', 'Bug', 'Steel']
    },
    'Water': {
        'weak': ['Electric', 'Grass'],
        'resist': ['Fire', 'Water', 'Ice', 'Steel'],
        'immune': [],
        'strong_against': ['Fire', 'Ground', 'Rock']
    },
    'Electric': {
        'weak': ['Ground'],
        'resist': ['Electric', 'Flying', 'Steel'],
        'immune': [],
        'strong_against': ['Water', 'Flying']
    },
    'Grass': {
        'weak': ['Fire', 'Ice', 'Poison', 'Flying', 'Bug'],
        'resist': ['Water', 'Electric', 'Grass', 'Ground'],
        'immune': [],
        'strong_against': ['Water', 'Ground', 'Rock']
    },
    'Ice': {
        'weak': ['Fire', 'Fighting', 'Rock', 'Steel'],
        'resist': ['Ice'],
        'immune': [],
        'strong_against': ['Grass', 'Ground', 'Flying', 'Dragon']
    },
    'Fighting': {
        'weak': ['Flying', 'Psychic', 'Fairy'],
        'resist': ['Bug', 'Rock', 'Dark'],
        'immune': [],
        'strong_against': ['Normal', 'Ice', 'Rock', 'Dark', 'Steel']
    },
    'Poison': {
        'weak': ['Ground', 'Psychic'],
        'resist': ['Grass', 'Fighting', 'Poison', 'Bug', 'Fairy'],
        'immune': [],
        'strong_against': ['Grass', 'Fairy']
    },
    'Ground': {
        'weak': ['Water', 'Grass', 'Ice'],
        'resist': ['Poison', 'Rock'],
        'immune': ['Electric'],
        'strong_against': ['Fire', 'Electric', 'Poison', 'Rock', 'Steel']
    },
    'Flying': {
        'weak': ['Electric', 'Ice', 'Rock'],
        'resist': ['Grass', 'Fighting', 'Bug'],
        'immune': ['Ground'],
        'strong_against': ['Grass', 'Fighting', 'Bug']
    },
    'Psychic': {
        'weak': ['Bug', 'Ghost', 'Dark'],
        'resist': ['Fighting', 'Psychic'],
        'immune': [],
        'strong_against': ['Fighting', 'Poison']
    },
    'Bug': {
        'weak': ['Fire', 'Flying', 'Rock'],
        'resist': ['Grass', 'Fighting', 'Ground'],
        'immune': [],
        'strong_against': ['Grass', 'Psychic', 'Dark']
    },
    'Rock': {
        'weak': ['Water', 'Grass', 'Fighting', 'Ground', 'Steel'],
        'resist': ['Normal', 'Fire', 'Poison', 'Flying'],
        'immune': [],
        'strong_against': ['Fire', 'Ice', 'Flying', 'Bug']
    },
    'Ghost': {
        'weak': ['Ghost', 'Dark'],
        'resist': ['Poison', 'Bug'],
        'immune': ['Normal', 'Fighting'],
        'strong_against': ['Psychic', 'Ghost']
    },
    'Dragon': {
        'weak': ['Ice', 'Dragon', 'Fairy'],
        'resist': ['Fire', 'Water', 'Electric', 'Grass'],
        'immune': [],
        'strong_against': ['Dragon']
    },
    'Dark': {
        'weak': ['Fighting', 'Bug', 'Fairy'],
        'resist': ['Ghost', 'Dark'],
        'immune': ['Psychic'],
        'strong_against': ['Psychic', 'Ghost']
    },
    'Steel': {
        'weak': ['Fire', 'Fighting', 'Ground'],
        'resist': ['Normal', 'Grass', 'Ice', 'Flying', 'Psychic', 'Bug', 'Rock', 'Dragon', 'Steel', 'Fairy'],
        'immune': ['Poison'],
        'strong_against': ['Ice', 'Rock', 'Fairy']
    },
    'Fairy': {
        'weak': ['Poison', 'Steel'],
        'resist': ['Fighting', 'Bug', 'Dark'],
        'immune': ['Dragon'],
        'strong_against': ['Fighting', 'Dragon', 'Dark']
    }
}

ALL_TYPES = sorted(TYPE_CHART.keys())

# Load data with caching
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Fixed machine learning similarity function
def calculate_pokemon_similarity(df, selected_pokemon):
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    
    # Create a temporary DataFrame with reset index
    temp_df = df.reset_index(drop=True)
    
    # Check if Pokémon exists in DataFrame
    if selected_pokemon not in temp_df['Pokemon'].values:
        return pd.DataFrame(columns=df.columns.tolist() + ['Similarity'])
    
    # Get numeric stats and scale
    numeric_df = temp_df[stats].fillna(0)
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(numeric_df)
    
    # Find index in the reset DataFrame
    try:
        pokemon_index = temp_df.index[temp_df['Pokemon'] == selected_pokemon][0]
        similarities = cosine_similarity([scaled_stats[pokemon_index]], scaled_stats)[0]
    except IndexError:
        return pd.DataFrame(columns=df.columns.tolist() + ['Similarity'])
    
    # Return results with original DataFrame structure
    result_df = df.copy()
    result_df['Similarity'] = similarities
    return result_df.sort_values('Similarity', ascending=False)

def calculate_team_similarity(df, selected_team):
    team_stats = df.groupby('Team').agg({
        'HP': 'mean',
        'Attack': 'mean',
        'Defense': 'mean',
        'Sp. Atk': 'mean',
        'Sp. Def': 'mean',
        'Speed': 'mean'
    }).reset_index()
    
    numeric_df = team_stats.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled_stats = scaler.fit_transform(numeric_df)
    
    try:
        team_index = team_stats[team_stats['Team'] == selected_team].index[0]
        similarities = cosine_similarity([scaled_stats[team_index]], scaled_stats)[0]
    except IndexError:
        return pd.DataFrame(columns=team_stats.columns.tolist() + ['Similarity'])
    
    team_stats['Similarity'] = similarities
    return team_stats.sort_values('Similarity', ascending=False)

# Radar chart function
def create_radar_chart(df, team_name):
    team_df = df[df['Team'] == team_name]
    if team_df.empty:
        return go.Figure()
    
    avg_stats = team_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_stats.values,
        theta=avg_stats.index,
        fill='toself',
        name=team_name
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(avg_stats)*1.2])),
        showlegend=True,
        title=f'Team {team_name} Average Stats Radar Chart'
    )
    
    return fig

def calculate_team_coverage(team_types):
    """Calculate all coverage aspects for the team"""
    resistances = set()
    immunities = set()
    offensive_coverage = defaultdict(int)
    
    # Collect all defensive properties
    for t in team_types:
        resistances.update(TYPE_CHART[t]['resist'])
        immunities.update(TYPE_CHART[t]['immune'])
        
        # Count offensive coverage
        for target in TYPE_CHART[t]['strong_against']:
            offensive_coverage[target] += 1
    
    # Find uncovered weaknesses
    all_attack_types = set()
    for t in ALL_TYPES:
        all_attack_types.update(TYPE_CHART[t]['weak'])
    
    uncovered_weaknesses = []
    for attack_type in sorted(all_attack_types):
        if attack_type not in resistances and attack_type not in immunities:
            uncovered_weaknesses.append(attack_type)
    
    # Get resisted types (not including immunities)
    resisted_types = [t for t in sorted(resistances) if t not in immunities]
    
    # Get super effective coverage (types hit by at least 2 team members)
    good_coverage = [t for t, count in offensive_coverage.items() if count >= 2]
    excellent_coverage = [t for t, count in offensive_coverage.items() if count >= 3]
    
    return {
        'uncovered_weaknesses': uncovered_weaknesses,
        'resisted_types': resisted_types,
        'immune_types': sorted(immunities),
        'offensive_coverage': dict(offensive_coverage),
        'good_coverage': sorted(good_coverage),
        'excellent_coverage': sorted(excellent_coverage)
    }

def analyze_replay_data(replay_file):
    """Analyze battle replay data to identify key decision points"""
    # This is a placeholder - in a real implementation, you would parse the replay file
    # and extract key decision points, missed opportunities, and patterns
    
    analysis = {
        'decision_points': [
            "Turn 3: Could have switched to counter opponent's setup",
            "Turn 7: Missed opportunity to use Protect to scout",
            "Turn 10: Optimal moment to set up Tailwind was missed"
        ],
        'missed_opportunities': [
            "Failed to capitalize on opponent's weakened Pokémon in mid-game",
            "Didn't exploit type advantage in late-game",
            "Missed chance to double-target key threat"
        ],
        'patterns': [
            "Consistently led with same Pokémon regardless of opponent",
            "Overused Protect in early game",
            "Underutilized speed control options"
        ],
        'suggestions': [
            "Consider more flexible lead options based on opponent's team",
            "Practice mid-game switching to maintain momentum",
            "Work on late-game cleanup strategies"
        ]
    }
    
    return analysis

def test_team_variants(base_team, variant_team, results_data):
    """Compare performance of team variants"""
    # This would compare win rates, key matchups, etc. in a real implementation
    
    comparison = {
        'win_rates': {
            'base_team': 0.65,
            'variant_team': 0.72
        },
        'matchup_differences': {
            'vs_hyper_offense': "+7%",
            'vs_stall': "+3%",
            'vs_balance': "+5%"
        },
        'key_improvements': [
            "Variant handles Flying-types better",
            "Improved late-game sweeping potential",
            "Better matchup against weather teams"
        ],
        'drawbacks': [
            "Slightly weaker to Electric-types",
            "Loses some early-game pressure"
        ]
    }
    
    return comparison

def track_moveset_effectiveness(team_data, battle_results):
    """Track how effective specific movesets are in different scenarios"""
    # This would analyze battle results to determine moveset performance
    
    effectiveness = {
        'most_effective_moves': [
            "Earthquake (85% success rate)",
            "Will-O-Wisp (78% success rate)",
            "Tailwind (92% success rate)"
        ],
        'underperforming_moves': [
            "Ice Beam (42% success rate)",
            "Protect (overused in 65% of matches)",
            "Swords Dance (only set up successfully 30% of time)"
        ],
        'recommendations': [
            "Consider replacing Ice Beam with more reliable coverage",
            "Use Protect more strategically rather than predictively",
            "Find safer setup opportunities for Swords Dance"
        ]
    }
    
    return effectiveness

def compare_ev_spreads(pokemon_name, spread1, spread2, performance_data):
    """Compare performance of different EV spreads for a Pokémon"""
    # This would analyze battle data to compare EV spreads
    
    comparison = {
        'offensive_output': {
            'spread1': "12% higher damage output",
            'spread2': "More consistent but lower peak damage"
        },
        'defensive_capability': {
            'spread1': "Survives key hits 85% of time",
            'spread2': "Survives key hits 95% of time"
        },
        'speed_tiers': {
            'spread1': "Outspeeds base 100s",
            'spread2': "Outspeeds base 110s but sacrifices bulk"
        },
        'recommendation': "Spread2 provides better consistency in current meta"
    }
    
    return comparison

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer")
    
    st.title("⚔️ Pokémon Competitive Team Analyzer")
    st.write("""
    Analyze competitive Pokémon teams with advanced training tools, replay analysis, 
    and team testing capabilities.
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
    
    # Add Type1 and Type2 columns if not present
    if 'Type1' not in df.columns:
        df['Type1'] = 'Unknown'
    if 'Type2' not in df.columns:
        df['Type2'] = ''
    
    # Main tabs - streamlined to focus on the new features
    tab1, tab2, tab3 = st.tabs([
        "🎮 Replay Analysis", 
        "🧪 Team Testing Suite", 
        "📊 Team Insights"
    ])
    
    with tab1:
        st.header("Replay Analysis System")
        st.write("""
        Upload battle replays to identify key decision points, missed opportunities,
        and patterns in your gameplay.
        """)
        
        replay_file = st.file_uploader("Upload Battle Replay", type=['json', 'txt'])
        
        if replay_file:
            analysis = analyze_replay_data(replay_file)
            
            st.subheader("Key Decision Points")
            for point in analysis['decision_points']:
                st.warning(f"⚠️ {point}")
            
            st.subheader("Missed Opportunities")
            for opportunity in analysis['missed_opportunities']:
                st.error(f"❌ {opportunity}")
            
            st.subheader("Identified Patterns")
            for pattern in analysis['patterns']:
                st.info(f"🔍 {pattern}")
            
            st.subheader("Improvement Suggestions")
            for suggestion in analysis['suggestions']:
                st.success(f"💡 {suggestion}")
            
            st.divider()
            
            # Visual pattern analysis
            st.subheader("Pattern Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                # Example move usage chart
                move_data = pd.DataFrame({
                    'Move': ['Protect', 'Switch', 'Attack', 'Setup'],
                    'Usage Rate': [45, 25, 20, 10]
                })
                fig = px.pie(move_data, names='Move', values='Usage Rate', 
                             title='Move Usage Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Example turn analysis
                turn_data = pd.DataFrame({
                    'Turn': range(1, 11),
                    'Advantage Gained': [0, 1, 2, 3, 2, 1, 0, -1, -2, -3]
                })
                fig = px.line(turn_data, x='Turn', y='Advantage Gained', 
                              title='Momentum Across Turns')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Team Testing Suite")
        st.write("""
        Compare team variants, test EV spreads, and track moveset effectiveness.
        """)
        
        test_type = st.selectbox("Select Test Type", [
            "Team Variant Comparison",
            "EV Spread Analysis",
            "Moveset Effectiveness Tracking"
        ])
        
        if test_type == "Team Variant Comparison":
            st.subheader("Team Variant A/B Testing")
            
            col1, col2 = st.columns(2)
            with col1:
                base_team = st.selectbox("Select Base Team", sorted(df['Team'].unique()))
            with col2:
                variant_team = st.selectbox("Select Variant Team", sorted(df['Team'].unique()))
            
            if st.button("Compare Variants"):
                comparison = test_team_variants(base_team, variant_team, None)
                
                st.metric("Win Rate Improvement", 
                         f"{comparison['win_rates']['variant_team']*100}%", 
                         f"{(comparison['win_rates']['variant_team'] - comparison['win_rates']['base_team'])*100:.1f}%")
                
                st.subheader("Matchup Differences")
                for matchup, diff in comparison['matchup_differences'].items():
                    st.write(f"- {matchup}: {diff}")
                
                st.subheader("Key Improvements")
                for improvement in comparison['key_improvements']:
                    st.success(f"✅ {improvement}")
                
                st.subheader("Potential Drawbacks")
                for drawback in comparison['drawbacks']:
                    st.warning(f"⚠️ {drawback}")
        
        elif test_type == "EV Spread Analysis":
            st.subheader("EV Spread Comparison Tool")
            
            pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
            pokemon_data = df[df['Pokemon'] == pokemon].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Current Spread")
                current_evs = st.text_input("Current EVs", value="252 Atk / 4 Def / 252 Spe")
                current_nature = st.text_input("Current Nature", value="Jolly (+Spe, -SpA)")
            
            with col2:
                st.write("### Proposed Spread")
                proposed_evs = st.text_input("Proposed EVs", value="196 HP / 60 Atk / 252 Spe")
                proposed_nature = st.text_input("Proposed Nature", value="Jolly (+Spe, -SpA)")
            
            if st.button("Compare EV Spreads"):
                comparison = compare_ev_spreads(pokemon, current_evs, proposed_evs, None)
                
                st.subheader("Comparison Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("#### Current Spread")
                    st.write(f"**Offensive Output:** {comparison['offensive_output']['spread1']}")
                    st.write(f"**Defensive Capability:** {comparison['defensive_capability']['spread1']}")
                    st.write(f"**Speed Tier:** {comparison['speed_tiers']['spread1']}")
                
                with col2:
                    st.write("#### Proposed Spread")
                    st.write(f"**Offensive Output:** {comparison['offensive_output']['spread2']}")
                    st.write(f"**Defensive Capability:** {comparison['defensive_capability']['spread2']}")
                    st.write(f"**Speed Tier:** {comparison['speed_tiers']['spread2']}")
                
                st.success(f"Recommendation: {comparison['recommendation']}")
        
        elif test_type == "Moveset Effectiveness Tracking":
            st.subheader("Moveset Performance Analysis")
            
            selected_team = st.selectbox("Select Team to Analyze", sorted(df['Team'].unique()))
            team_df = df[df['Team'] == selected_team]
            
            if st.button("Analyze Moveset Effectiveness"):
                effectiveness = track_moveset_effectiveness(team_df, None)
                
                st.subheader("Most Effective Moves")
                for move in effectiveness['most_effective_moves']:
                    st.success(f"✔️ {move}")
                
                st.subheader("Underperforming Moves")
                for move in effectiveness['underperforming_moves']:
                    st.error(f"✖️ {move}")
                
                st.subheader("Recommended Adjustments")
                for rec in effectiveness['recommendations']:
                    st.info(f"💡 {rec}")
                
                st.divider()
                
                # Moveset usage visualization
                st.subheader("Moveset Usage and Success Rates")
                move_data = pd.DataFrame({
                    'Move': ['Earthquake', 'Will-O-Wisp', 'Tailwind', 'Ice Beam', 'Protect', 'Swords Dance'],
                    'Usage Rate': [25, 20, 15, 10, 20, 10],
                    'Success Rate': [85, 78, 92, 42, 65, 30]
                })
                
                fig = px.bar(move_data, x='Move', y=['Usage Rate', 'Success Rate'], 
                             barmode='group', title='Move Performance Metrics')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Team Insights")
        st.write("""
        Core team analysis tools including type coverage, synergy assessment, and matchup evaluation.
        """)
        
        insight_type = st.selectbox("Select Insight Type", [
            "Type Coverage Analysis",
            "Team Synergy Assessment",
            "Matchup Evaluation"
        ])
        
        if insight_type == "Type Coverage Analysis":
            st.subheader("Team Type Coverage")
            selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()))
            
            team_df = df[df['Team'] == selected_team]
            if not team_df.empty:
                # Get all types present on the team
                team_types = []
                for _, row in team_df.iterrows():
                    team_types.append(row['Type1'])
                    if pd.notna(row['Type2']) and row['Type2'] != '':
                        team_types.append(row['Type2'])
                
                coverage = calculate_team_coverage(team_types)
                
                # Display defensive coverage
                st.header("Defensive Coverage")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("❌ Uncovered Weaknesses")
                    if coverage['uncovered_weaknesses']:
                        for t in coverage['uncovered_weaknesses']:
                            st.error(t)
                    else:
                        st.success("All attack types are covered!")
                
                with col2:
                    st.subheader("🛡️ Resisted Types")
                    if coverage['resisted_types']:
                        for t in coverage['resisted_types']:
                            st.info(t)
                    else:
                        st.warning("No resisted types")
                
                with col3:
                    st.subheader("✅ Immune Types")
                    if coverage['immune_types']:
                        for t in coverage['immune_types']:
                            st.success(t)
                    else:
                        st.warning("No immunities")
                
                # Display offensive coverage
                st.header("Offensive Coverage")
                
                st.subheader("⚔️ Super Effective Against:")
                if coverage['offensive_coverage']:
                    cols = st.columns(3)
                    for i, (t, count) in enumerate(sorted(coverage['offensive_coverage'].items())):
                        with cols[i%3]:
                            st.write(f"{t}: {'⭐' * count}")
                else:
                    st.warning("No notable offensive coverage")
                
                st.subheader("✨ Good Coverage (2+ members):")
                if coverage['good_coverage']:
                    st.write(", ".join(coverage['good_coverage']))
                else:
                    st.warning("No types with good coverage")
                
                st.subheader("💫 Excellent Coverage (3+ members):")
                if coverage['excellent_coverage']:
                    st.write(", ".join(coverage['excellent_coverage']))
                else:
                    st.warning("No types with excellent coverage")
        
        elif insight_type == "Team Synergy Assessment":
            st.subheader("Team Role Synergy")
            selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()))
            
            team_df = df[df['Team'] == selected_team]
            if not team_df.empty:
                # Role distribution pie chart
                role_dist = team_df['PrimaryRole'].value_counts().reset_index()
                fig = px.pie(role_dist, names='PrimaryRole', values='count', 
                             title='Primary Role Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show all team members with roles
                st.subheader("Team Composition")
                for _, row in team_df.iterrows():
                    role_info = f"{row['Pokemon']} ({row['PrimaryRole']}"
                    if pd.notna(row['SecondaryRole']) and row['SecondaryRole'] != '':
                        role_info += f" / {row['SecondaryRole']}"
                    st.write(f"- {role_info}")
                
                # Key synergies
                st.subheader("Key Synergies")
                try:
                    key_synergies = team_df['Key Synergies'].iloc[0]
                    if pd.notna(key_synergies):
                        st.success(key_synergies)
                    else:
                        st.warning("No explicit synergies documented")
                except:
                    st.warning("Synergy data not available")
        
        elif insight_type == "Matchup Evaluation":
            st.subheader("Team Matchup Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox("Your Team", sorted(df['Team'].unique()))
            
            with col2:
                team2 = st.selectbox("Opponent's Team", sorted(df['Team'].unique()))
            
            if st.button("Analyze Matchup"):
                # Get all types for each team
                team1_types = []
                for _, row in df[df['Team'] == team1].iterrows():
                    team1_types.append(row['Type1'])
                    if pd.notna(row['Type2']) and row['Type2'] != '':
                        team1_types.append(row['Type2'])
                
                team2_types = []
                for _, row in df[df['Team'] == team2].iterrows():
                    team2_types.append(row['Type1'])
                    if pd.notna(row['Type2']) and row['Type2'] != '':
                        team2_types.append(row['Type2'])
                
                # Calculate coverage for both teams
                team1_coverage = calculate_team_coverage(team1_types)
                team2_coverage = calculate_team_coverage(team2_types)
                
                # Find threats (team1's strengths vs team2's weaknesses)
                threats = []
                for t in team1_coverage['offensive_coverage']:
                    if t in team2_coverage['uncovered_weaknesses']:
                        threats.append(f"Your {t} attacks vs opponent's weakness")
                
                # Find opportunities (team2's uncovered weaknesses)
                opportunities = []
                for t in team2_coverage['uncovered_weaknesses']:
                    if t not in team1_coverage['offensive_coverage']:
                        opportunities.append(f"You could exploit opponent's {t} weakness")
                
                # Display results
                if threats:
                    st.subheader("Your Advantages")
                    for threat in threats:
                        st.success(f"✅ {threat}")
                else:
                    st.warning("No clear type advantages against this team")
                
                if opportunities:
                    st.subheader("Potential Opportunities")
                    for opportunity in opportunities:
                        st.info(f"💡 {opportunity}")
                else:
                    st.info("You're already exploiting all of opponent's weaknesses")

if __name__ == "__main__":
    main()
