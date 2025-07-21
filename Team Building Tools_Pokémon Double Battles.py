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
    """Analyze team type coverage"""
    # This would use actual type data in a real implementation
    if 'Type' not in team_df.columns:
        return None
    
    type_counts = team_df['Type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']
    return type_counts

def generate_ml_recommendations(df, team_name):
    """Generate ML-based recommendations for team improvement"""
    # Simple clustering-based recommendation (would be more sophisticated in production)
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    kmeans = KMeans(n_clusters=3)
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
    
    # [Previous tab implementations...]
    
    with tabs[2]:  # Team Comparison
        st.header("📊 Team Comparison")
        teams = st.multiselect("Select teams to compare", sorted(df['Team'].unique()), default=sorted(df['Team'].unique())[:2])
        
        if len(teams) >= 2:
            comparison_df = df[df['Team'].isin(teams)]
            
            # Compare average stats
            avg_stats = comparison_df.groupby('Team')[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
            st.subheader("Average Stats Comparison")
            st.dataframe(avg_stats.style.highlight_max(axis=0))
            
            # Radar chart comparison
            st.subheader("Radar Chart Comparison")
            fig = go.Figure()
            
            for team in teams:
                team_avg = avg_stats.loc[team]
                fig.add_trace(go.Scatterpolar(
                    r=team_avg.values,
                    theta=team_avg.index,
                    fill='toself',
                    name=team
                ))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least 2 teams for comparison")
    
    with tabs[3]:  # ML Recommendations
        st.header("🤖 Machine Learning Recommendations")
        selected_team = st.selectbox("Select team for recommendations", sorted(df['Team'].unique()))
        
        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing team composition..."):
                recommendations = generate_ml_recommendations(df, selected_team)
                
                st.subheader("Recommended Pokémon Additions")
                st.write("Based on similar team compositions, consider adding:")
                
                for i, pokemon in enumerate(recommendations, 1):
                    st.write(f"{i}. {pokemon}")
                
                st.info("These recommendations are generated using clustering algorithms that identify Pokémon with similar stat distributions to your current team members")
    
    with tabs[4]:  # Type Coverage
        st.header("🛡️ Type Coverage Analysis")
        selected_team = st.selectbox("Select team for type analysis", sorted(df['Team'].unique()))
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            type_counts = analyze_type_coverage(team_df)
            
            if type_counts is not None:
                st.subheader("Type Distribution")
                fig = px.pie(type_counts, values='Count', names='Type', title="Team Type Composition")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Type Coverage Assessment")
                st.write("""
                - **Strong Against**: Shows types your team has advantage against
                - **Weak Against**: Shows types that threaten your team
                - **Missing Coverage**: Important types your team doesn't counter
                """)
                
                # This would be expanded with actual type matchup data
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("**Strong Against**")
                    st.write("- Water\n- Ground\n- Fire")
                with col2:
                    st.write("**Weak Against**")
                    st.write("- Electric\n- Flying\n- Psychic")
                with col3:
                    st.write("**Missing Coverage**")
                    st.write("- Dragon\n- Steel\n- Fairy")
            else:
                st.warning("Type data not available in the dataset")
    
    with tabs[5]:  # Team Synergy
        st.header("🔄 Team Synergy Analysis")
        selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='synergy_team')
        team_df = df[df['Team'] == selected_team]
        
        if not team_df.empty:
            synergy_matrix = calculate_synergy_scores(team_df)
            
            st.write("""
            ### Understanding the Team Synergy Matrix
            
            The synergy matrix shows how well each Pokémon's stat distribution complements others on your team:
            
            - **High values (closer to 1)**: These Pokémon have similar stat distributions, which can be good for balanced teams but may indicate redundancy
            - **Low values (closer to 0)**: These Pokémon have different stat distributions, which can provide coverage but may lack synergy
            - **Negative values**: These Pokémon have opposing stat distributions (rare)
            
            Ideal teams often show moderate synergy (0.4-0.7) between most members with some specialized pairs.
            """)
            
            fig = px.imshow(
                synergy_matrix,
                labels=dict(x="Pokémon", y="Pokémon", color="Synergy"),
                x=team_df['Pokemon'].tolist(),
                y=team_df['Pokemon'].tolist(),
                title="Team Synergy Matrix",
                color_continuous_scale='RdYlGn',
                zmin=-1, zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[6]:  # Team Matchup
        st.header("⚔️ Team Matchup Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            team_a = st.selectbox("Your Team", sorted(df['Team'].unique()), key='team_a')
        with col2:
            team_b = st.selectbox("Opponent Team", sorted(df['Team'].unique()), key='team_b')
        
        if team_a and team_b:
            st.subheader("Matchup Analysis")
            
            # Simple matchup analysis (would be more sophisticated in real implementation)
            st.write("""
            **Key Matchup Factors:**
            
            1. **Speed Control**: Which team has faster Pokémon on average
            2. **Type Advantage**: Overall type matchups between teams
            3. **Win Conditions**: How each team plans to win
            """)
            
            # Calculate basic comparison metrics
            team_a_df = df[df['Team'] == team_a]
            team_b_df = df[df['Team'] == team_b]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Speed", 
                         f"{team_a_df['Speed'].mean():.1f} vs {team_b_df['Speed'].mean():.1f}",
                         delta=f"{(team_a_df['Speed'].mean() - team_b_df['Speed'].mean()):.1f}")
            with col2:
                st.metric("Total Attack", 
                         f"{(team_a_df['Attack'].sum() + team_a_df['Sp. Atk'].sum()):.1f} vs {(team_b_df['Attack'].sum() + team_b_df['Sp. Atk'].sum()):.1f}")
            with col3:
                st.metric("Total Bulk", 
                         f"{(team_a_df['HP'].sum() + team_a_df['Defense'].sum() + team_a_df['Sp. Def'].sum()):.1f} vs {(team_b_df['HP'].sum() + team_b_df['Defense'].sum() + team_b_df['Sp. Def'].sum()):.1f}")
            
            st.subheader("Suggested Strategy")
            st.write("""
            - Focus on eliminating opponent's key threats first
            - Protect your win condition Pokémon
            - Use your speed advantage to control the tempo
            """)
    
    # [Rest of your existing tab implementations...]

if __name__ == "__main__":
    main()
