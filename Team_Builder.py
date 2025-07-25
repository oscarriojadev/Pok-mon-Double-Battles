import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Read file directly with pandas
            data = pd.read_csv(uploaded_file, sep='\t')  # First try tab-separated
            
            # If that fails (only got 1 column), try comma-separated
            if len(data.columns) == 1:
                uploaded_file.seek(0)  # Reset file pointer
                data = pd.read_csv(uploaded_file)  # Try comma-separated
            
            # Clean column names
            data.columns = data.columns.str.strip().str.replace('"', '').str.replace('  ', ' ')
            
            # Convert numeric columns safely
            def safe_convert(col):
                if col in data.columns:
                    # Handle percentage columns
                    if '%' in col:
                        data[col] = data[col].astype(str).str.replace('%', '')
                    # Convert to numeric, coercing errors to NaN
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    # Fill NaN with 0 if needed
                    data[col] = data[col].fillna(0)
            
            numeric_cols = [
                'Format Viability',
                'Pivot Synergy Rating (1-20)',
                'Bulk Score',
                'Damage Output Score',
                'Meta Usage (%)'
            ]
            
            for col in numeric_cols:
                safe_convert(col)
            
            # Verify required columns
            required_cols = ['Team Number', 'Team Name', 'Pokemon', 'Role']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return None, None
            
            # Define aggregation
            agg_dict = {
                'Team Name': 'first',
                'Pokemon': list,
                'Role': list
            }
            
            # Add numeric aggregations
            numeric_aggs = {
                'Format Viability': 'mean',
                'Pivot Synergy Rating (1-20)': 'mean',
                'Bulk Score': 'mean',
                'Damage Output Score': 'mean',
                'Meta Usage (%)': 'mean'
            }
            
            for col, agg_func in numeric_aggs.items():
                if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                    agg_dict[col] = agg_func
            
            # Add non-numeric aggregations
            text_cols = {
                'Typing (Primary)': list,
                'Typing (Secondary)': list,
                'Archetype Suitability': 'first'
            }
            
            for col, agg_func in text_cols.items():
                if col in data.columns:
                    agg_dict[col] = agg_func
            
            # Group and aggregate
            team_data = data.groupby('Team Number', as_index=False).agg(agg_dict)
            
            return data, team_data
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None
    return None, None

def main():
    st.set_page_config(page_title="Pokémon Team Builder", page_icon="⚔️", layout="wide")
    st.title("⚔️ Pokémon Team Building Recommender")
    
    uploaded_file = st.file_uploader("Upload your Pokémon Teams CSV", type=["csv", "tsv"])
    data, team_data = load_data(uploaded_file)
    
    if data is None:
        return
    
    # Filter section - with fixed numeric conversion
    st.sidebar.header("Filters")
    
    if 'Format Viability' in team_data.columns:
        min_viability = st.sidebar.slider(
            "Minimum Format Viability",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        )
        # Safe conversion for filtering
        team_data['Format Viability'] = pd.to_numeric(team_data['Format Viability'], errors='coerce').fillna(0)
        filtered_teams = team_data[team_data['Format Viability'] >= min_viability]
    else:
        filtered_teams = team_data
    
    # Display results
    st.header("Recommended Teams")
    for _, row in filtered_teams.iterrows():
        with st.expander(f"Team {row['Team Number']}: {row['Team Name']}"):
            st.write(f"**Format Viability:** {row.get('Format Viability', 'N/A')}")
            st.write("**Pokémon:**")
            for pokemon, role in zip(row['Pokemon'], row['Role']):
                st.write(f"- {pokemon} ({role})")

if __name__ == "__main__":
    main()
