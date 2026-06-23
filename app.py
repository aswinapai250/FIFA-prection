import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import plotly.express as px
import simulate_tournament

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
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0A0A0A !important;
        color: #F5F5F0 !important;
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #C9A227, #E8C766);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #8A8A85;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #161616;
        border: 1px solid #2A2A2A;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .predictor-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #F5F5F0;
        margin-bottom: 1rem;
        border-left: 4px solid #C9A227;
        padding-left: 0.5rem;
    }

    /* Clean flat buttons and selectboxes */
    div.stButton > button, button[data-testid^="baseButton"], div[data-baseweb="select"] > div {
        background-color: #161616 !important;
        color: #F5F5F0 !important;
        border: 1px solid #2A2A2A !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover, button[data-testid^="baseButton"]:hover {
        background-color: #2A2A2A !important;
        border-color: #C9A227 !important;
    }
    div.stButton > button:active, button[data-testid^="baseButton"]:active {
        background-color: #121212 !important;
    }

    /* Override Streamlit default tab styling to gold */
    button[data-baseweb="tab"] {
        color: #8A8A85 !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        transition: color 0.2s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #C9A227 !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    div[data-baseweb="tab-highlight-indicator"] {
        display: block !important;
        background-color: #C9A227 !important;
    }

    /* Hide fullscreen button on images */
    button[aria-label="Fullscreen"], [data-testid="stImageFullscreenBtn"] {
        display: none !important;
    }

    /* Hide heading anchor links */
    .anchor-link, [data-testid="stHeaderActionElements"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. Cache the simulation results for 24 hours using st.cache_data
@st.cache_data(ttl=24 * 3600)

def run_simulation_pipeline():
    import os
    os.environ["FOOTBALL_API_KEY"] = st.secrets["FOOTBALL_API_KEY"]
    # Import pipeline scripts
    import fetch_data
    import merge_data
    import simulate_tournament
    
    # Run fetch_data logic (safely, fallback to local file if internet/API fails)
    try:
        fetch_data.main()
        merge_data.main()
    except Exception as e:
        print(f"Warning: data refresh failed ({e}). Using existing all_matches.csv.")
        
    # Run merge_data logic
    #merge_data.main()
    
    # Simulate tournament logic to get win probabilities
    import train_model
    model, feature_columns = train_model.train_and_get_model()
    
    matches = simulate_tournament.load_match_history()
    latest_rankings = simulate_tournament.load_current_rankings()
    profiles = simulate_tournament.build_team_profiles(matches, latest_rankings)
    h2h_lookup = simulate_tournament.build_h2h_rates(matches)
    
    probabilities = simulate_tournament.build_matchup_probabilities(
        model, feature_columns, profiles, h2h_lookup
    )
    
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
    
    group_tables, played_matches, h2h_results = simulate_tournament.build_real_group_state()
    remaining_matches = simulate_tournament.get_groups_with_remaining_matches()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return results, probabilities, timestamp, all_teams, group_tables, remaining_matches, h2h_results

# 1. On load, run pipeline using st.spinner
with st.spinner("Running simulation pipeline (fetching new matches, merging datasets, and running tournament simulation)..."):
    results, probabilities, timestamp, all_teams, group_tables, remaining_matches, h2h_results = run_simulation_pipeline()

# Layout
# Header layout with logo and title
logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
with logo_col:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/1/17/2026_FIFA_World_Cup_emblem.svg/960px-2026_FIFA_World_Cup_emblem.svg.png", width=120)
with title_col:
    st.markdown('<div class="main-title" style="margin-top: 1rem;">FIFA 2026 World Cup Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.9rem; color: #8A8A85; font-style: italic; margin-bottom: 0.5rem;">Predictions based on historical match data and FIFA rankings. Does not account for injuries or squad changes.</div>', unsafe_allow_html=True)
    # 5. Show last updated timestamp
    st.markdown(f'<div class="subtitle" style="margin-bottom: 1.5rem;">Live match data fetched & 10,000 tournament simulations executed • Last Updated: <b>{timestamp}</b></div>', unsafe_allow_html=True)

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
        text=df_results["probability"].round(1).astype(str) + "%",
    )
    
    fig.update_traces(
        marker_color="#C9A227",
        textposition="outside",
        cliponaxis=False,
        hoverinfo="skip",
        hovertemplate=None,
    )
    
    fig.update_layout(
        yaxis={"categoryorder": "total ascending", "fixedrange": True},
        xaxis={"fixedrange": True},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#ffffff",
        margin=dict(l=0, r=40, t=10, b=0),
        height=600,
        dragmode=False,
        bargap=0.3,
        hovermode=False,
    )
    
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        title_text="",
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        title_text="",
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})
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
        team_a = st.selectbox("Team A", sorted_teams, index=idx_a)
    with col_sel2:
        team_b = st.selectbox("Team B", sorted_teams, index=idx_b)
        
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
        gold_gradient = "linear-gradient(90deg, #C9A227, #E8C766)"
        silver_gradient = "linear-gradient(90deg, #555555, #888888)"

        if pct_a >= pct_b:
            color_a, color_b = "#E8C766", "#AAAAAA"
            bar_a, bar_b = gold_gradient, silver_gradient
        else:
            color_a, color_b = "#AAAAAA", "#E8C766"
            bar_a, bar_b = silver_gradient, gold_gradient

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>{team_a}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: {color_a}; margin-top: 0;'>{pct_a:.1f}%</h1>", unsafe_allow_html=True)
        with col_res2:
            st.markdown(f"<h3 style='text-align: center; margin-bottom: 0;'>{team_b}</h3>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; color: {color_b}; margin-top: 0;'>{pct_b:.1f}%</h1>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display: flex; width: 100%; height: 28px; border-radius: 14px; overflow: hidden; background-color: #2c2c2e; font-family: 'Outfit', sans-serif; font-weight: 600; color: white; margin-top: 1rem; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="width: {pct_a}%; background: {bar_a}; display: flex; align-items: center; justify-content: center; transition: width 0.6s ease; font-size: 0.95rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                {pct_a:.1f}%
            </div>
            <div style="width: {pct_b}%; background: {bar_b}; display: flex; align-items: center; justify-content: center; transition: width 0.6s ease; font-size: 0.95rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
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
            <h4 style="margin-top: 0; color: #F5F5F0; border-bottom: 1px solid #2A2A2A; padding-bottom: 0.5rem;">Quick Matchup Facts</h4>
            <table style="width: 100%; color: #8A8A85; font-size: 0.95rem; line-height: 2;">
                <tr>
                    <td>FIFA World Ranking</td>
                    <td style="text-align: right; color: #F5F5F0;"><b>{team_a}</b> (Rank {rank_a}) vs <b>{team_b}</b> (Rank {rank_b})</td>
                </tr>
                <tr>
                    <td>Ranking Difference</td>
                    <td style="text-align: right; color: #C9A227;"><b>{abs(rank_a - rank_b)} positions</b></td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# Live Group Stage Standings section
st.markdown('<div class="predictor-header" style="margin-top: 2rem;">Live Group Stage Standings</div>', unsafe_allow_html=True)

# Custom CSS for the custom HTML standings table and badges
st.markdown("""
<style>
    .custom-standings-table {
        width: 100%;
        border-collapse: fixed;
        border-spacing: 0 6px;
        font-family: 'Outfit', sans-serif;
        margin-bottom: 1rem;
        background: #161616;
        border: 1px solid #2A2A2A;
        border-radius: 12px;
        padding: 12px;
    }
    .custom-standings-table th {
        background-color: rgba(255, 255, 255, 0.02);
        color: #8A8A85;
        font-weight: 600;
        text-align: left;
        padding: 10px 15px;
        font-size: 0.9rem;
        border: none;
    }
    .custom-standings-table th:first-child {
        border-top-left-radius: 8px;
        border-bottom-left-radius: 8px;
    }
    .custom-standings-table th:last-child {
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
    }
    .custom-standings-row {
        background: rgba(255, 255, 255, 0.02);
    }
    .custom-standings-row td {
        padding: 12px 15px;
        font-size: 0.95rem;
        border: none;
        vertical-align: middle;
    }
    .custom-standings-row.qualifying td {
        color: #F5F5F0;
    }
    .custom-standings-row.qualifying td:first-child {
        border-left: 4px solid #C9A227;
        border-top-left-radius: 8px;
        border-bottom-left-radius: 8px;
    }
    .custom-standings-row.non-qualifying td {
        color: #8A8A85;
    }
    .custom-standings-row.non-qualifying td:first-child {
        border-left: 4px solid #333333;
        border-top-left-radius: 8px;
        border-bottom-left-radius: 8px;
    }
    .custom-standings-row td:last-child {
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
    }
    .custom-pts-cell {
        font-weight: bold;
        color: #C9A227 !important;
    }
    .fixture-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .fixture-badge {
        background-color: #1a1a1a;
        border: 1px solid #2A2A2A;
        border-radius: 16px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #8A8A85;
        font-family: 'Outfit', sans-serif;
    }
    .complete-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(46, 204, 113, 0.1);
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 16px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: #2ECC71;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        margin-bottom: 1.5rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

groups_keys = sorted(list(simulate_tournament.GROUPS.keys()))
tabs = st.tabs([f"Group {g}" for g in groups_keys])

for tab, group_name in zip(tabs, groups_keys):
    with tab:
        table = group_tables[group_name]
        sorted_teams = simulate_tournament.sort_group_table(table, h2h_results)
        
        # Build custom styled HTML table
        table_html = """
        <table class="custom-standings-table">
            <thead>
                <tr>
                    <th style="width: 60px;">Pos</th>
                    <th style="width: 220px;">Team</th>
                    <th style="width: 60px;">MP</th>
                    <th style="width: 60px;">W</th>
                    <th style="width: 60px;">D</th>
                    <th style="width: 60px;">L</th>
                    <th style="width: 80px;">Pts</th>
                </tr>
            </thead>
            <tbody>
"""
        for pos_idx, team in enumerate(sorted_teams, start=1):
            stats = table[team]
            row_class = "qualifying" if pos_idx <= 2 else "non-qualifying"
            table_html += f"""
                <tr class="custom-standings-row {row_class}">
                    <td>{pos_idx}</td>
                    <td><b>{team}</b></td>
                    <td>{stats["mp"]}</td>
                    <td>{stats["w"]}</td>
                    <td>{stats["d"]}</td>
                    <td>{stats["l"]}</td>
                    <td class="custom-pts-cell">{stats["points"]}</td>
                </tr>
            """
        table_html += """
            </tbody>
        </table>
        """
        import re
        table_html = re.sub(r'\n\s+', '', table_html)
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Build remaining fixtures or complete badge
        rem_fixtures = remaining_matches.get(group_name, [])
        if len(rem_fixtures) == 0:
            st.markdown('<div class="complete-badge">✓ Group Stage Complete</div>', unsafe_allow_html=True)
        else:
            st.write("**Remaining fixtures:**")
            fixtures_html = '<div class="fixture-container">'
            for home_t, away_t in rem_fixtures:
                fixtures_html += f'<div class="fixture-badge">{home_t} vs {away_t}</div>'
            fixtures_html += '</div>'
            fixtures_html = re.sub(r'\n\s+', '', fixtures_html)
            st.markdown(fixtures_html, unsafe_allow_html=True)

