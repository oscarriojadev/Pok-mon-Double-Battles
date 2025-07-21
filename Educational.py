import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from collections import defaultdict
import random

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

def display_type_info(attack_type):
    """Display detailed information about a specific type"""
    st.write(f"### {attack_type} Attacks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Super Effective Against:**")
        super_effective = [t for t in ALL_TYPES 
                         if attack_type in TYPE_CHART[t]['weak']]
        for t in super_effective:
            st.write(f"- {t}")
    
    with col2:
        st.write("**Resisted By:**")
        resisted_by = [t for t in ALL_TYPES 
                     if attack_type in TYPE_CHART[t]['resist']]
        if not resisted_by:
            st.write("(No types resist this)")
        else:
            for t in resisted_by:
                st.write(f"- {t}")
    
    st.write("**Ineffective Against:**")
    immune = [t for t in ALL_TYPES 
             if attack_type in TYPE_CHART[t]['immune']]
    if not immune:
        st.write("(No types are immune to this)")
    else:
        for t in immune:
            st.write(f"- {t}")

def generate_type_quiz():
    """Generate a random type effectiveness quiz question"""
    attacking_type = random.choice(ALL_TYPES)
    defending_type = random.choice(ALL_TYPES)
    
    # Get effectiveness
    if attacking_type in TYPE_CHART[defending_type]['immune']:
        correct_answer = "No effect (immune)"
    elif attacking_type in TYPE_CHART[defending_type]['resist']:
        correct_answer = "Not very effective (resisted)"
    elif attacking_type in TYPE_CHART[defending_type]['weak']:
        correct_answer = "Super effective"
    else:
        correct_answer = "Normal effectiveness"
    
    return {
        'question': f"How effective is {attacking_type} against {defending_type}?",
        'options': [
            "Super effective",
            "Not very effective (resisted)",
            "Normal effectiveness",
            "No effect (immune)"
        ],
        'correct': correct_answer,
        'explanation': f"{attacking_type} attacks are {correct_answer.lower()} against {defending_type} Pokémon."
    }

def generate_speed_tier_question(df):
    """Generate a speed tier comparison question"""
    pokemon_list = df['Pokemon'].unique()
    if len(pokemon_list) < 2:
        return None
    
    pokemon1, pokemon2 = random.sample(list(pokemon_list), 2)
    speed1 = df[df['Pokemon'] == pokemon1]['Speed'].values[0]
    speed2 = df[df['Pokemon'] == pokemon2]['Speed'].values[0]
    
    if speed1 == speed2:
        return generate_speed_tier_question(df)  # Try again if speeds are equal
    
    faster = pokemon1 if speed1 > speed2 else pokemon2
    speed_diff = abs(speed1 - speed2)
    
    return {
        'question': f"Which Pokémon is faster: {pokemon1} or {pokemon2}?",
        'options': [pokemon1, pokemon2, "They have the same speed"],
        'correct': faster,
        'explanation': f"{faster} is faster ({max(speed1, speed2)} vs {min(speed1, speed2)} speed). The difference is {speed_diff} points."
    }

def generate_damage_calc_question():
    """Generate a damage calculation concept question"""
    questions = [
        {
            'question': "Which of these factors does NOT affect damage calculation?",
            'options': [
                "Pokémon level",
                "Base friendship",
                "Weather conditions",
                "STAB (Same Type Attack Bonus)"
            ],
            'correct': "Base friendship",
            'explanation': "Base friendship doesn't affect damage calculation. Level, weather, and STAB all influence damage."
        },
        {
            'question': "What does STAB stand for in Pokémon battles?",
            'options': [
                "Same Type Attack Bonus",
                "Special Technique Amplification Boost",
                "Standard Tactical Advantage Bonus",
                "Super Effective Attack Boost"
            ],
            'correct': "Same Type Attack Bonus",
            'explanation': "STAB means Same Type Attack Bonus - a 1.5x damage boost when a Pokémon uses a move of its own type."
        },
        {
            'question': "What is the damage multiplier when a move is super effective?",
            'options': ["1.5x", "2x", "0.5x", "1x"],
            'correct': "2x",
            'explanation': "Super effective moves deal 2x damage. Some types have double weaknesses (4x) when both types are weak to the move."
        }
    ]
    return random.choice(questions)

def generate_team_building_question():
    """Generate a team building strategy question"""
    questions = [
        {
            'question': "Which of these is NOT a common team role in competitive Pokémon?",
            'options': [
                "Wallbreaker",
                "Happiness Manager",
                "Pivot",
                "Hazard Setter"
            ],
            'correct': "Happiness Manager",
            'explanation': "Happiness Manager isn't a competitive role. Wallbreakers break defensive cores, pivots switch safely, and hazard setters lay entry hazards."
        },
        {
            'question': "What's the purpose of a 'pivot' Pokémon on a team?",
            'options': [
                "To deal massive damage",
                "To switch in safely and maintain momentum",
                "To set up entry hazards",
                "To heal other team members"
            ],
            'correct': "To switch in safely and maintain momentum",
            'explanation': "Pivots use moves like U-turn or Volt Switch to switch out while dealing damage, or have abilities/typing that let them switch into attacks safely."
        },
        {
            'question': "Which item is typically used by setup sweepers?",
            'options': [
                "Leftovers",
                "Life Orb",
                "Choice Scarf",
                "Bright Powder"
            ],
            'correct': "Life Orb",
            'explanation': "Life Orb provides a 1.3x damage boost at the cost of some HP, ideal for sweepers that boost their stats."
        }
    ]
    return random.choice(questions)

def generate_battle_scenario(df):
    """Generate a battle scenario for practice"""
    scenarios = [
        {
            'scenario': "Your opponent has set up Stealth Rock and has a Dragon-type sweeper in the back. What should you prioritize?",
            'options': [
                "Set up your own hazards",
                "Remove hazards with Rapid Spin/Defog",
                "Switch in your Fairy-type to counter the Dragon",
                "Sacrifice a Pokémon to gain momentum"
            ],
            'correct': "Remove hazards with Rapid Spin/Defog",
            'explanation': "Hazard control is crucial when facing setup sweepers to preserve your defensive answers."
        },
        {
            'scenario': "You're facing a rain team with Swift Swim users. What's the best counterplay?",
            'options': [
                "Set up your own weather",
                "Use priority moves",
                "Bring Pokémon with Water resistance",
                "All of the above"
            ],
            'correct': "All of the above",
            'explanation': "Against weather teams, changing the weather, using priority, and resistant Pokémon are all valid strategies."
        },
        {
            'scenario': "Your opponent has a bulky Steel-type that keeps using Protect. How do you break through?",
            'options': [
                "Use Taunt to prevent Protect",
                "Predict with a Fighting-type move",
                "Set up hazards to wear it down",
                "All of the above"
            ],
            'correct': "All of the above",
            'explanation': "Stallbreakers use multiple methods to break defensive cores - Taunt, prediction, and residual damage."
        }
    ]
    return random.choice(scenarios)

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Pokémon Team Educator")
    
    st.title("📚 Pokémon Competitive Team Educator")
    st.write("""
    Interactive learning tools to master competitive Pokémon team building and battle strategies.
    """)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Pokémon Data CSV (Optional)", type=["csv"])
    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Type Effectiveness Trainer", 
        "⚡ Speed Tier Quizzes", 
        "💥 Damage Calculation", 
        "🏆 Team Building Workshops"
    ])
    
    with tab1:
        st.header("Type Effectiveness Trainer")
        st.write("Test your knowledge of Pokémon type matchups")
        
        if st.button("Generate New Question"):
            if 'type_question' not in st.session_state:
                st.session_state.type_question = generate_type_quiz()
        
        if 'type_question' in st.session_state:
            question = st.session_state.type_question
            st.subheader(question['question'])
            
            selected = st.radio("Select your answer:", question['options'], key="type_quiz")
            
            if st.button("Submit Answer"):
                if selected == question['correct']:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {question['correct']}")
                
                st.write(question['explanation'])
                
                # Show detailed type info
                attack_type = question['question'].split()[3]
                display_type_info(attack_type)
        
        st.divider()
        st.subheader("Type Effectiveness Reference")
        selected_type = st.selectbox("Select a type to learn about:", ALL_TYPES)
        display_type_info(selected_type)
    
    with tab2:
        st.header("Speed Tier Quizzes")
        st.write("Test your knowledge of Pokémon speed tiers and matchups")
        
        if df is None:
            st.warning("Upload a Pokémon dataset to enable speed tier questions")
        else:
            if st.button("Generate Speed Question"):
                st.session_state.speed_question = generate_speed_tier_question(df)
            
            if 'speed_question' in st.session_state:
                question = st.session_state.speed_question
                st.subheader(question['question'])
                
                selected = st.radio("Select your answer:", question['options'], key="speed_quiz")
                
                if st.button("Check Answer"):
                    if selected == question['correct']:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Incorrect. The correct answer is: {question['correct']}")
                    
                    st.write(question['explanation'])
            
            st.divider()
            st.subheader("Speed Tier Importance")
            st.write("""
            Understanding speed tiers is crucial for competitive Pokémon:
            
            - Faster Pokémon attack first (unless priority moves are used)
            - Speed determines who sets up weather/terrain last (overwriting opponents)
            - Many sweepers aim to outspeed common threats
            - Speed control moves (Tailwind, Thunder Wave) can change matchups
            """)
    
    with tab3:
        st.header("Damage Calculation Lessons")
        st.write("Learn how damage is calculated in Pokémon battles")
        
        if st.button("Generate Damage Calculation Question"):
            st.session_state.damage_question = generate_damage_calc_question()
        
        if 'damage_question' in st.session_state:
            question = st.session_state.damage_question
            st.subheader(question['question'])
            
            selected = st.radio("Select your answer:", question['options'], key="damage_quiz")
            
            if st.button("Submit Damage Answer"):
                if selected == question['correct']:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {question['correct']}")
                
                st.write(question['explanation'])
        
        st.divider()
        st.subheader("Damage Calculation Formula")
        st.write("""
        The basic damage formula is:
        
        ```
        Damage = ((((2 × Level ÷ 5 + 2) × Power × [Attack/Defense]) ÷ 50 + 2) × Modifiers
        ```
        
        **Modifiers include:**
        - STAB (Same Type Attack Bonus): 1.5x
        - Type effectiveness: 2x, 4x, 0.5x, or 0.25x
        - Critical hit: 1.5x
        - Weather: 1.5x for matching types
        - Other: Items, abilities, etc.
        """)
    
    with tab4:
        st.header("Team Building Workshops")
        st.write("Practice building balanced teams and battle scenarios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Team Building Concepts")
            if st.button("Generate Team Building Question"):
                st.session_state.team_question = generate_team_building_question()
            
            if 'team_question' in st.session_state:
                question = st.session_state.team_question
                st.write(question['question'])
                
                selected = st.radio("Select answer:", question['options'], key="team_quiz")
                
                if st.button("Check Team Answer"):
                    if selected == question['correct']:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Incorrect. The correct answer is: {question['correct']}")
                    
                    st.write(question['explanation'])
        
        with col2:
            st.subheader("Battle Scenario Practice")
            if st.button("Generate Battle Scenario"):
                st.session_state.battle_scenario = generate_battle_scenario(df) if df else generate_battle_scenario(None)
            
            if 'battle_scenario' in st.session_state:
                scenario = st.session_state.battle_scenario
                st.write(scenario['scenario'])
                
                selected = st.radio("Choose your action:", scenario['options'], key="battle_quiz")
                
                if st.button("Submit Battle Choice"):
                    if selected == scenario['correct']:
                        st.success("✅ Good decision!")
                    else:
                        st.error(f"❌ Suboptimal choice. Better option: {scenario['correct']}")
                    
                    st.write(scenario['explanation'])
        
        st.divider()
        st.subheader("Team Building Fundamentals")
        st.write("""
        **Core Team Roles:**
        
        1. **Sweepers**: Deal heavy damage, often after setting up
        2. **Walls/Tanks**: Take hits and stall opponents
        3. **Pivots**: Switch in safely and maintain momentum
        4. **Hazard Setters**: Set Stealth Rock/Spikes
        5. **Hazard Control**: Remove hazards with Rapid Spin/Defog
        6. **Support**: Provide healing, screens, or speed control
        
        **Team Synergy Checklist:**
        - Cover all major weaknesses
        - Have multiple win conditions
        - Include speed control options
        - Balance physical/special attackers
        - Consider weather/terrain strategies
        """)

if __name__ == "__main__":
    main()
