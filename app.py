import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import plotly.express as px

# Set up page config
st.set_page_config(
    page_title="FIFA 2026 World Cup Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for modern visual layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFD700, #FFA500, #FF4500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #88888b;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        margin-bottom: 1.5rem;
    }
    
    .predictor-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1rem;
        border-left: 4px solid #FFD700;
        padding-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 4. Cache the simulation results for 24 hours using st.cache_data
@st.cache_data(ttl=1)
def run_simulation_pipeline():
    import os
    os.environ["FOOTBALL_API_KEY"] = st.secrets["FOOTBALL_API_KEY"]
    # Import pipeline scripts
    import fetch_data
    import merge_data
    import simulate_tournament
    
    # Run fetch_data logic (safely, fallback to local file if internet/API fails)
    try:
        #fetch_data.main()
        pass
    except Exception as e:
        print(f"Warning: fetch_data failed ({e}). Using existing wc_matches.csv.")
        
    # Run merge_data logic
    #merge_data.main()
    
    # Simulate tournament logic to get win probabilities
    model = joblib.load(simulate_tournament.MODEL_FILE)
    feature_columns = joblib.load(simulate_tournament.FEATURE_COLUMNS_FILE)
    
    matches = simulate_tournament.load_match_history()
    latest_rankings = simulate_tournament.load_current_rankings()
    profiles = simulate_tournament.build_team_profiles(matches, latest_rankings)
    h2h_lookup = simulate_tournament.build_h2h_rates(matches)
    
    probabilities = simulate_tournament.build_matchup_probabilities(
        model, feature_columns, profiles, h2h_lookup
    )
    st.write("Spain vs Australia:", probabilities.get(("Spain", "Australia")))
    st.write("Australia vs Spain:", probabilities.get(("Australia", "Spain")))
    st.write("Argentina vs Jordan:", probabilities.get(("Argentina", "Jordan")))
    st.write("France vs Curaçao:", probabilities.get(("France", "Curaçao")))
    
    all_teams = [team for group in simulate_tournament.GROUPS.values() for team in group]
    win_counts = {team: 0 for team in all_teams}
    
    for _ in range(simulate_tournament.SIMULATIONS):
        champion = simulate_tournament.simulate_tournament(probabilities)
        win_counts[champion] += 1
        
    results = [
        {"team": team, "wins": wins, "probability": wins / simulate_tournament.SIMULATIONS * 100}
        for team, wins in win_counts.items()
    ]
    results = sorted(results, key=lambda x: x["probability"], reverse=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return results, probabilities, timestamp, all_teams

# 1. On load, run pipeline using st.spinner
with st.spinner("Running simulation pipeline (fetching new matches, merging datasets, and running tournament simulation)..."):
    results, probabilities, timestamp, all_teams = run_simulation_pipeline()

# Layout
st.markdown('<div class="main-title">FIFA 2026 World Cup Predictor</div>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 0.9rem; color: #a0a0a5; font-style: italic; margin-bottom: 0.5rem;">Predictions based on historical match data and FIFA rankings. Does not account for injuries or squad changes.</div>', unsafe_allow_html=True)
# 5. Show last updated timestamp
st.markdown(f'<div class="subtitle">Live match data fetched & 10,000 tournament simulations executed • Last Updated: <b>{timestamp}</b></div>', unsafe_allow_html=True)

col1, col2 = st.columns([5, 4], gap="large")

with col1:
    st.markdown('<div class="predictor-header">Tournament Win Probabilities</div>', unsafe_allow_html=True)
    
    # Prepare top 20 data for plotting
    df_results = pd.DataFrame(results).head(20)
    
    # 2. Shows a bar chart of top 20 teams using plotly
    fig = px.bar(
        df_results,
        x="probability",
        y="team",
        orientation="h",
        labels={"probability": "Win Probability (%)", "team": "Team"},
        color="probability",
        color_continuous_scale=["#FF4500", "#FFA500", "#FFD700"],
    )
    
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        margin=dict(l=0, r=0, t=10, b=0),
        height=550,
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col2:
    # 3. Match Predictor section
    st.markdown('<div class="predictor-header">Head-to-Head Match Predictor</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Select any two teams qualified for the 2026 World Cup to calculate their direct matchup probabilities 
    based on historical matchups, form profiles, and current rankings.
    """)
    
    sorted_teams = sorted(all_teams)
    
    # Ensure default picks are different
    default_a = "Argentina" if "Argentina" in sorted_teams else sorted_teams[0]
    default_b = "Spain" if "Spain" in sorted_teams else sorted_teams[1]
    
    idx_a = sorted_teams.index(default_a)
    idx_b = sorted_teams.index(default_b)
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        team_a = st.selectbox("Team A (Home)", sorted_teams, index=idx_a)
    with col_sel2:
        team_b = st.selectbox("Team B (Away)", sorted_teams, index=idx_b)
        
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    if team_a == team_b:
        st.info("Please select two different teams to compute head-to-head probabilities.")
    else:
        # Retrieve matchup probability
        p_a = probabilities.get((team_a, team_b), 0.5)
        p_b = 1.0 - p_a
        
        pct_a = p_a * 100
        pct_b = p_b * 100
        
        # Display probabilities side-by-side
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>{team_a}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: #FFD700; margin-top: 0;'>{pct_a:.1f}%</h1>", unsafe_allow_html=True)
        with col_res2:
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>{team_b}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: #FF4500; margin-top: 0;'>{pct_b:.1f}%</h1>", unsafe_allow_html=True)
            
        # A simple visual probability bar (premium look horizontal split bar)
        st.markdown(f"""
        <div style="display: flex; width: 100%; height: 28px; border-radius: 14px; overflow: hidden; background-color: #2c2c2e; font-family: 'Outfit', sans-serif; font-weight: 600; color: white; margin-top: 1rem; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="width: {pct_a}%; background: linear-gradient(90deg, #FFA500, #FFD700); display: flex; align-items: center; justify-content: center; transition: width 0.6s ease; font-size: 0.95rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                {pct_a:.1f}%
            </div>
            <div style="width: {pct_b}%; background: linear-gradient(90deg, #FF4500, #FF6347); display: flex; align-items: center; justify-content: center; transition: width 0.6s ease; font-size: 0.95rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                {pct_b:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display team quick stats/ranks from the simulator context
        import simulate_tournament
        rankings = simulate_tournament.load_current_rankings()
        
        rank_a = int(rankings.get(simulate_tournament.to_ranking_name(team_a), 100))
        rank_b = int(rankings.get(simulate_tournament.to_ranking_name(team_b), 100))
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin-top: 0; color: #ffffff; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem;">Quick Matchup Facts</h4>
            <table style="width: 100%; color: #d0d0d2; font-size: 0.95rem; line-height: 2;">
                <tr>
                    <td>FIFA World Ranking</td>
                    <td style="text-align: right;"><b>{team_a}</b> (Rank {rank_a}) vs <b>{team_b}</b> (Rank {rank_b})</td>
                </tr>
                <tr>
                    <td>Ranking Difference</td>
                    <td style="text-align: right; color: #FFD700;"><b>{abs(rank_a - rank_b)} positions</b></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
