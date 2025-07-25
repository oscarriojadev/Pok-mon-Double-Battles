@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded file with quoted headers
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        data = pd.read_csv(stringio, sep='\t', quotechar='"')  # Using tab separator and quotechar
        
        # Clean and preprocess data
        data = data.dropna(how='all', axis=1)  # Remove empty columns
        
        # Clean column names by removing quotes and extra spaces
        data.columns = data.columns.str.replace('"', '').str.strip()
        
        # Handle 'Meta Usage (%)' column - convert only numeric percentages
        if 'Meta Usage (%)' in data.columns:
            # First convert to string, then process percentages
            data['Meta Usage (%)'] = data['Meta Usage (%)'].astype(str)
            # Remove % sign and convert to float where possible
            data['Meta Usage (%)'] = pd.to_numeric(
                data['Meta Usage (%)'].str.replace('%', ''),
                errors='coerce'
            )
        
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
