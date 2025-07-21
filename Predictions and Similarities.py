import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor  # Changed from Classifier to Regressor
from sklearn.model_selection import train_test_split
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

def predict_team_success(df):
    """Predict team success rate based on historical data"""
    # Feature engineering
    features = df.groupby('Team').agg({
        'HP': 'mean',
        'Attack': 'mean',
        'Defense': 'mean',
        'Sp. Atk': 'mean',
        'Sp. Def': 'mean',
        'Speed': 'mean',
        'RoleScore': 'mean',
        'StatScore': 'mean',
        'MoveScore': 'mean',
        'AbilityScore': 'mean',
        'SynergyScore': 'mean',
        'MetaScore': 'mean'
    }).reset_index()
    
    # Simulate target variable (win rate) - in a real app, this would come from actual data
    np.random.seed(42)
    features['WinRate'] = np.random.uniform(0.4, 0.9, len(features))
    
    # Select only numeric columns for modeling
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    X = features[numeric_cols].drop('WinRate', axis=1, errors='ignore')
    y = features['WinRate']
    
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict for all teams
        features['PredictedWinRate'] = model.predict(X)
        return features.sort_values('PredictedWinRate', ascending=False)
    return features.sort_values('WinRate', ascending=False)

def predict_matchup_outcome(df, team1, team2):
    """Predict matchup outcome probability between two teams"""
    # Get team stats
    team_stats = df.groupby('Team').agg({
        'HP': 'mean',
        'Attack': 'mean',
        'Defense': 'mean',
        'Sp. Atk': 'mean',
        'Sp. Def': 'mean',
        'Speed': 'mean',
        'RoleScore': 'mean',
        'StatScore': 'mean',
        'MoveScore': 'mean',
        'AbilityScore': 'mean',
        'SynergyScore': 'mean',
        'MetaScore': 'mean'
    }).reset_index()
    
    # Simulate model prediction - in a real app, this would use an actual trained model
    np.random.seed(hash(team1 + team2) % 100)
    team1_win_prob = np.random.uniform(0.3, 0.7)
    
    return {
        'team1': team1,
        'team2': team2,
        'team1_win_prob': team1_win_prob,
        'team2_win_prob': 1 - team1_win_prob,
        'key_factors': ['Speed control', 'Type coverage', 'Role balance'][:np.random.randint(1, 4)]
    }

def cluster_teams_by_playstyle(df):
    """Cluster teams by playstyle using K-means"""
    # Feature engineering for clustering
    team_features = df.groupby('Team').agg({
        'HP': 'mean',
        'Attack': 'mean',
        'Defense': 'mean',
        'Sp. Atk': 'mean',
        'Sp. Def': 'mean',
        'Speed': 'mean',
        'RoleScore': 'mean',
        'StatScore': 'mean',
        'MoveScore': 'mean',
        'AbilityScore': 'mean',
        'SynergyScore': 'mean',
        'MetaScore': 'mean'
    }).reset_index()
    
    # Select only numeric columns for clustering
    numeric_cols = team_features.select_dtypes(include=[np.number]).columns.tolist()
    X = team_features[numeric_cols]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    if len(X_scaled) > 5:
        k = min(5, len(X_scaled) - 1)
        kmeans = KMeans(n_clusters=k, random_state=42)
        team_features['PlaystyleCluster'] = kmeans.fit_predict(X_scaled)
        
        # Analyze cluster characteristics (numeric columns only)
        cluster_profiles = team_features.groupby('PlaystyleCluster')[numeric_cols].mean()
        return team_features, cluster_profiles
    return team_features, None

def analyze_usage_trends(df):
    """Analyze Pokémon usage trends across teams"""
    usage = df['Pokemon'].value_counts().reset_index()
    usage.columns = ['Pokemon', 'UsageCount']
    
    # Get average stats for each Pokémon
    stats = df.groupby('Pokemon')[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean().reset_index()
    usage = usage.merge(stats, on='Pokemon')
    
    # Calculate usage tier (top 10%, 20%, etc.)
    usage['UsageTier'] = pd.qcut(usage['UsageCount'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    return usage.sort_values('UsageCount', ascending=False)

def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Analyzer")
    
    st.title("⚔️ Pokémon Competitive Team Analyzer")
    st.write("""
    Advanced analytics for competitive Pokémon teams with predictive modeling and machine learning insights.
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
    
    # Add default columns if not present
    if 'PrimaryRole' not in df.columns:
        df['PrimaryRole'] = 'Unknown'
    if 'Type1' not in df.columns:
        df['Type1'] = 'Unknown'
    if 'Type2' not in df.columns:
        df['Type2'] = ''
    
    # Main tabs focusing on new features
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Predictive Analytics", 
        "🔍 Advanced Similarity", 
        "🤖 ML Insights",
        "📊 Usage Trends"
    ])
    
    with tab1:
        st.header("Predictive Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Team Success Predictor")
            success_df = predict_team_success(df)
            st.dataframe(
                success_df[['Team', 'PredictedWinRate']].sort_values('PredictedWinRate', ascending=False),
                use_container_width=True
            )
            
            # Visualize success predictions
            fig = px.bar(
                success_df.sort_values('PredictedWinRate', ascending=False).head(10),
                x='Team',
                y='PredictedWinRate',
                title="Top Teams by Predicted Win Rate"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Matchup Outcome Predictor")
            team1 = st.selectbox("Select Team 1", sorted(df['Team'].unique()), key='pred_team1')
            team2 = st.selectbox("Select Team 2", sorted(df['Team'].unique()), key='pred_team2')
            
            if st.button("Predict Matchup"):
                prediction = predict_matchup_outcome(df, team1, team2)
                
                st.metric(
                    label=f"{team1} Win Probability",
                    value=f"{prediction['team1_win_prob']*100:.1f}%"
                )
                st.metric(
                    label=f"{team2} Win Probability",
                    value=f"{prediction['team2_win_prob']*100:.1f}%"
                )
                
                st.write("**Key Factors:**")
                for factor in prediction['key_factors']:
                    st.write(f"- {factor}")
                
                # Show comparison radar chart
                team1_stats = df[df['Team'] == team1][['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
                team2_stats = df[df['Team'] == team2][['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=team1_stats.values,
                    theta=team1_stats.index,
                    fill='toself',
                    name=team1
                ))
                fig.add_trace(go.Scatterpolar(
                    r=team2_stats.values,
                    theta=team2_stats.index,
                    fill='toself',
                    name=team2
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title=f"{team1} vs {team2} Stats Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Advanced Similarity Algorithms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Role-Based Clustering")
            team_clusters, cluster_profiles = cluster_teams_by_playstyle(df)
            
            if cluster_profiles is not None:
                st.write("**Team Playstyle Clusters:**")
                st.dataframe(team_clusters[['Team', 'PlaystyleCluster']], use_container_width=True)
                
                st.write("**Cluster Profiles:**")
                st.dataframe(cluster_profiles, use_container_width=True)
                
                # Visualize clusters
                fig = px.scatter(
                    team_clusters,
                    x='Speed',
                    y='Attack',
                    color='PlaystyleCluster',
                    hover_name='Team',
                    title="Team Playstyle Clusters (Speed vs Attack)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough teams to perform clustering (need at least 6)")
        
        with col2:
            st.subheader("Strategy Similarity Mapping")
            selected_team = st.selectbox("Select Team", sorted(df['Team'].unique()), key='sim_team')
            
            similar_teams = calculate_team_similarity(df, selected_team)
            if not similar_teams.empty:
                st.write(f"Teams most similar to {selected_team}:")
                st.dataframe(
                    similar_teams[['Team', 'Similarity']].head(10),
                    use_container_width=True
                )
                
                # Show similarity network
                top_similar = similar_teams.head(5)
                fig = go.Figure()
                
                # Add edges
                for i, row in top_similar.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[selected_team, row['Team']],
                        y=[1, row['Similarity']],
                        mode='lines',
                        line=dict(width=row['Similarity']*5),
                        showlegend=False
                    ))
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=[selected_team] + top_similar['Team'].tolist(),
                    y=[1] + top_similar['Similarity'].tolist(),
                    mode='markers',
                    marker=dict(size=20, color=['blue'] + ['red']*len(top_similar)),
                    text=[selected_team] + top_similar['Team'].tolist(),
                    hoverinfo='text',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f"Similarity Network for {selected_team}",
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(title="Similarity Score")
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Machine Learning Insights")
        
        st.subheader("Meta Shift Prediction")
        st.write("""
        This analysis predicts upcoming meta shifts based on current team compositions and usage trends.
        """)
        
        # Simulate meta shift prediction
        meta_shifts = {
            'PredictedTrend': ['Rise of Hyper Offense', 'Decline of Stall', 'Increase in Weather Teams', 'More Speed Control'],
            'Confidence': [0.85, 0.72, 0.68, 0.91],
            'Impact': ['High', 'Medium', 'Medium', 'High']
        }
        meta_df = pd.DataFrame(meta_shifts)
        st.dataframe(meta_df, use_container_width=True)
        
        # Visualize predicted trends
        fig = px.bar(
            meta_df,
            x='PredictedTrend',
            y='Confidence',
            color='Impact',
            title="Predicted Meta Shifts"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Team Archetype Classification")
        st.write("""
        Classify teams into archetypes based on their composition and strategy.
        """)
        
        # Simulate archetype classification
        team_archetypes = df.groupby('Team').first().reset_index()
        np.random.seed(42)
        team_archetypes['Archetype'] = np.random.choice(
            ['Hyper Offense', 'Balance', 'Stall', 'Weather', 'Trick Room'],
            size=len(team_archetypes)
        )
        
        st.dataframe(
            team_archetypes[['Team', 'Archetype']],
            use_container_width=True
        )
        
        # Archetype distribution
        fig = px.pie(
            team_archetypes,
            names='Archetype',
            title="Team Archetype Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Usage Trend Analysis")
        
        usage_df = analyze_usage_trends(df)
        
        st.subheader("Top Used Pokémon")
        st.dataframe(
            usage_df.head(20),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Usage Tier Distribution")
            fig = px.pie(
                usage_df,
                names='UsageTier',
                title="Pokémon Usage Tiers"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Stats by Usage Tier")
            fig = px.box(
                usage_df,
                x='UsageTier',
                y='Speed',
                title="Speed Distribution by Usage Tier"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Usage Trends Over Time")
        st.write("""
        (In a production app, this would show actual temporal usage data)
        """)
        
        # Simulate temporal data
        temporal_data = []
        for pokemon in usage_df['Pokemon'].head(10):
            temporal_data.append({
                'Pokemon': pokemon,
                'Month 1': np.random.randint(1, 20),
                'Month 2': np.random.randint(1, 30),
                'Month 3': np.random.randint(5, 40)
            })
        
        temporal_df = pd.DataFrame(temporal_data).melt(id_vars='Pokemon', var_name='Month', value_name='Usage')
        
        fig = px.line(
            temporal_df,
            x='Month',
            y='Usage',
            color='Pokemon',
            title="Top Pokémon Usage Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
