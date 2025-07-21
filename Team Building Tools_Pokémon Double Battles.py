import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from collections import defaultdict
from itertools import combinations

# Sample type data for demonstration
POKEMON_TYPES = {
    'Pikachu': 'Electric',
    'Charizard': 'Fire/Flying',
    'Blastoise': 'Water',
    'Venusaur': 'Grass/Poison',
    'Gyarados': 'Water/Flying',
    'Dragonite': 'Dragon/Flying',
    'Snorlax': 'Normal',
    'Alakazam': 'Psychic',
    'Gengar': 'Ghost/Poison',
    'Machamp': 'Fighting'
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
        'Type': '',  # Adding default type column
        'Moves': ''  # Adding default moves column
    }
    
    for col, default_val in default_columns.items():
        if col not in df.columns:
            df[col] = default_val
    
    # If Type column is empty, populate with sample data
    if df['Type'].isnull().all() or (df['Type'] == '').all():
        df['Type'] = df['Pokemon'].map(POKEMON_TYPES).fillna('Unknown')
    
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
    if 'Type' not in team_df.columns:
        return None
    
    # Split dual types and count each type separately
    type_list = []
    for types in team_df['Type']:
        for t in str(types).split('/'):
            if t and t != 'Unknown':
                type_list.append(t.strip())
    
    type_counts = pd.Series(type_list).value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    return type_counts

def generate_ml_recommendations(df, team_name):
    """Generate ML-based recommendations for team improvement"""
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[numeric_cols])
    
    team_cluster = df[df['Team'] == team_name]['cluster'].mode()[0]
    recommendations = df[df['cluster'] == team_cluster].sample(3)['Pokemon'].tolist()
    
    return recommendations

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
    
    with tabs[0]:  # Team Overview
        st.header("🏆 Team Overview Dashboard")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()))
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            # Team summary stats
            st.subheader(f"Team Composition: {selected_team}")
            st.dataframe(team_df[['Pokemon', 'Type', 'PrimaryRole', 'SecondaryRole']])
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Team Size", len(team_df))
            with col2:
                st.metric("Average Speed", f"{team_df['Speed'].mean():.1f}")
            with col3:
                st.metric("Total Power", f"{team_df['Attack'].sum() + team_df['Sp. Atk'].sum():.1f}")
            
            # Team stats radar chart
            st.subheader("Team Stats Distribution")
            avg_stats = team_df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
            fig = go.Figure(go.Scatterpolar(
                r=avg_stats.values,
                theta=avg_stats.index,
                fill='toself'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # Pokémon Analysis
        st.header("🔍 Individual Pokémon Analysis")
        selected_pokemon = st.selectbox("Select Pokémon", sorted(df['Pokemon'].unique()))
        pokemon_data = df[df['Pokemon'] == selected_pokemon].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Base Stats")
            stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
            values = [pokemon_data[stat] for stat in stats]
            
            fig = go.Figure(go.Scatterpolar(
                r=values,
                theta=stats,
                fill='toself'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Pokémon Details")
            st.write(f"**Type:** {pokemon_data.get('Type', 'Unknown')}")
            st.write(f"**Primary Role:** {pokemon_data.get('PrimaryRole', 'Not specified')}")
            st.write(f"**Secondary Role:** {pokemon_data.get('SecondaryRole', 'Not specified')}")
            st.write(f"**Win Condition:** {pokemon_data.get('Win Condition', 'Not specified')}")
            
            st.subheader("Stat Comparison")
            team_avg = df[df['Team'] == pokemon_data['Team']][stats].mean()
            comparison = pd.DataFrame({
                'Stat': stats,
                'Pokemon': values,
                'Team Average': team_avg.values
            })
            st.bar_chart(comparison.set_index('Stat'))
    
    with tabs[4]:  # Type Coverage
        st.header("🛡️ Team Type Coverage Analysis")
        selected_team = st.selectbox("Select team for type analysis", sorted(df['Team'].unique()), key='type_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            type_counts = analyze_type_coverage(team_df)
            
            if type_counts is not None and not type_counts.empty:
                st.subheader("Type Distribution")
                fig = px.pie(type_counts, values='Count', names='Type', 
                            title="Team Type Composition",
                            hole=0.3)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Type Coverage Assessment")
                
                # Sample type effectiveness (would be more sophisticated in real implementation)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info("**Strong Against**")
                    st.write("- Water\n- Ground\n- Rock")
                with col2:
                    st.warning("**Weak Against**")
                    st.write("- Electric\n- Psychic\n- Dragon")
                with col3:
                    st.error("**Missing Coverage**")
                    st.write("- Steel\n- Fairy\n- Dark")
                
                st.subheader("Suggested Improvements")
                st.write("Consider adding Pokémon with these types to improve coverage:")
                st.write("- Steel type for defensive utility")
                st.write("- Fairy type for Dragon resistance")
                st.write("- Dark type for Psychic immunity")
            else:
                st.warning("Could not analyze type coverage - type data may be missing")
    
    with tabs[7]:  # Enhanced Team Building
        st.header("🧩 Enhanced Team Building Tools")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='enhanced_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            # Role analysis
            st.subheader("Role Composition")
            roles = team_df['PrimaryRole'].value_counts()
            if not roles.empty:
                fig = px.pie(roles, values=roles.values, names=roles.index,
                            title="Primary Role Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No role data available for this team")
            
            # Team balance assessment
            st.subheader("Team Balance Assessment")
            offense = team_df['Attack'].mean() + team_df['Sp. Atk'].mean()
            defense = team_df['Defense'].mean() + team_df['Sp. Def'].mean()
            speed = team_df['Speed'].mean()
            
            balance_df = pd.DataFrame({
                'Category': ['Offense', 'Defense', 'Speed'],
                'Score': [offense, defense, speed]
            })
            
            fig = px.bar(balance_df, x='Category', y='Score', 
                        title="Team Balance Metrics",
                        color='Category')
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Team Building Recommendations")
            if offense > defense + 50:
                st.warning("Your team is offense-heavy. Consider adding more defensive Pokémon.")
            elif defense > offense + 50:
                st.warning("Your team is defense-heavy. Consider adding more offensive threats.")
            else:
                st.success("Your team has good offensive/defensive balance!")
            
            if speed < 80:
                st.warning("Your team is slow. Consider adding speed control options like Tailwind or Trick Room.")
    
    with tabs[8]:  # Meta Analysis
        st.header("📈 Meta Analysis Dashboard")
        
        # Usage statistics
        st.subheader("Pokémon Usage Statistics")
        usage = df['Pokemon'].value_counts().head(20).reset_index()
        usage.columns = ['Pokemon', 'Usage Count']
        fig = px.bar(usage, x='Pokemon', y='Usage Count', 
                     title="Top 20 Most Used Pokémon")
        st.plotly_chart(fig, use_container_width=True)
        
        # Team archetypes
        st.subheader("Popular Team Archetypes")
        team_counts = df['Team'].value_counts().head(10).reset_index()
        team_counts.columns = ['Team', 'Count']
        fig = px.pie(team_counts, values='Count', names='Team',
                    title="Most Common Team Archetypes")
        st.plotly_chart(fig, use_container_width=True)
        
        # Meta threats
        st.subheader("Top Meta Threats")
        top_threats = df['Pokemon'].value_counts().head(5).index.tolist()
        for i, threat in enumerate(top_threats, 1):
            st.write(f"{i}. {threat}")
        
        # Counter recommendations
        st.subheader("Anti-Meta Recommendations")
        st.write("Consider these Pokémon to counter the current meta:")
        st.write("- Toxapex (bulky wall)")
        st.write("- Ferrothorn (hazard setter)")
        st.write("- Clefable (Fairy type with great utility)")

if __name__ == "__main__":
    main()
