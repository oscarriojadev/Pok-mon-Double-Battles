# Pokémon Competitive Team Analyzer

![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

A Streamlit web application for analyzing competitive Pokémon teams, their compositions, and synergies.

## Features

- **Team Overview**: View team composition, members, and role distribution
- **Pokémon Details**: Examine individual Pokémon builds with movesets, items, and strategies
- **Team Synergy**: Analyze team strategies across different game phases
- **Interactive Filters**: Filter by team and role to focus your analysis
- **CSV Upload**: Use your own Pokémon team data in CSV format

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pokemon-team-analyzer.git
   cd pokemon-team-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your Pokémon team data in CSV format (see `sample_data` for format reference)
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Upload your CSV file when prompted
4. Use the sidebar filters to explore teams and roles

## Data Format Requirements

Your CSV file should include these columns (at minimum):
- `Team` - Team identifier/name
- `Pokemon` - Pokémon name
- `PrimaryRole` - Primary role (e.g., "Physical Sweeper")
- `Item` - Held item
- `Ability` - Pokémon ability
- `Move 1` to `Move 4` - Moveset
- `Key Strategy` - Strategy description
- `Good Teammates` - Synergistic teammates
- `Counters` - Common counters
- `Early Game`, `Mid Game`, `EndgameWinCon` - Phase strategies

## Example Data

A sample CSV structure:

| Team       | Pokemon   | PrimaryRole    | Item        | Ability    | Move 1      | ... |
|------------|-----------|----------------|-------------|------------|-------------|-----|
| Rain Team  | Pelipper  | Weather Setter | Damp Rock   | Drizzle    | Hurricane   | ... |
| Rain Team  | Swampert  | Physical Wall  | Leftovers   | Torrent    | Flip Turn   | ... |

## Screenshots

![Team Overview Tab](screenshots/team-overview.png)
*Team composition and role distribution*

![Pokémon Details Tab](screenshots/pokemon-details.png)
*Detailed Pokémon build information*

## Contributing

Contributions are welcome! Please open an issue or pull request for:
- Bug fixes
- New features
- Documentation improvements

## License

MIT License - See [LICENSE](LICENSE) for details
