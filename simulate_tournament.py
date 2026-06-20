import copy
import itertools
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd

MODEL_FILE = "model.pkl"
FEATURE_COLUMNS_FILE = "feature_columns.pkl"
FEATURES_FILE = "wc_features_ranked.csv"
import os
_DIR = os.path.dirname(os.path.abspath(__file__))
MATCHES_FILE = os.path.join(_DIR, "all_matches.csv")
RANKINGS_FILE = os.path.join(_DIR, "fifa_ranking.csv")
MODEL_FILE = os.path.join(_DIR, "model.pkl")
FEATURE_COLUMNS_FILE = os.path.join(_DIR, "feature_columns.pkl")
DEFAULT_RANK = 100
RECENT_WINDOW = 10
SIMULATIONS = 10000
DRAW_LOW = 0.4
DRAW_HIGH = 0.6
RNG = np.random.default_rng(42)

# Official FIFA World Cup 2026 groups (December 2025 draw, playoffs resolved)
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

MATCH_NAME_MAP = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Cabo Verde": "Cape Verde Islands",
    "USA": "United States",
    "Korea Republic": "South Korea",
}

RANKING_NAME_MAP = {
    "South Korea": "Korea Republic",
    "United States": "USA",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "Cabo Verde": "Cabo Verde",
    "Curaçao": "Curacao",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cabo Verde",
    "Congo DR": "Congo DR",
    "Turkey": "Turkey",
}

DEFAULT_PROFILE = {
    "win_rate": 0.33,
    "goals_avg": 1.0,
    "conceded_avg": 1.0,
    "matches_played": 0,
    "win_rate_vs_top": 0.33,
}

def get_win_proba(model, features):
    proba = model.predict_proba([features])[0]
    classes = list(model.classes_)
    win_index = classes.index(1)
    return proba[win_index]

def to_match_name(team):
    return MATCH_NAME_MAP.get(team, team)


def to_ranking_name(team):
    return RANKING_NAME_MAP.get(team, team)


def load_match_history():
    matches = pd.read_csv(MATCHES_FILE)
    matches = matches.dropna(subset=["home_team", "away_team", "winner"])
    matches = matches[matches["winner"].isin(["home", "away", "draw"])]
    matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    return matches


def load_current_rankings():
    rankings = pd.read_csv(RANKINGS_FILE, usecols=["rank", "country_full", "rank_date"])
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    latest = rankings.sort_values("rank_date").groupby("country_full").tail(1)
    return dict(zip(latest["country_full"], latest["rank"]))


def build_team_profiles(matches, latest_rankings):
    profiles = {}
    all_teams = sorted({team for group in GROUPS.values() for team in group})

    for team in all_teams:
        match_name = to_match_name(team)
        team_games = []

        for row in matches.itertuples():
            if row.home_team == match_name:
                team_games.append(
                    {
                        "goals_for": float(row.home_goals),
                        "goals_against": float(row.away_goals),
                        "won": row.winner == "home",
                    }
                )
            elif row.away_team == match_name:
                team_games.append(
                    {
                        "goals_for": float(row.away_goals),
                        "goals_against": float(row.home_goals),
                        "won": row.winner == "away",
                    }
                )

        recent = team_games[-RECENT_WINDOW:]
        if recent:
            profile = {
                "win_rate": sum(game["won"] for game in recent) / len(recent),
                "goals_avg": np.mean([game["goals_for"] for game in recent]),
                "conceded_avg": np.mean([game["goals_against"] for game in recent]),
                "matches_played": len(team_games),
            }
        else:
            profile = DEFAULT_PROFILE.copy()

        ranking_name = to_ranking_name(team)
        profile["rank"] = float(latest_rankings.get(ranking_name, DEFAULT_RANK))
        profiles[team] = profile

    return profiles


def build_h2h_rates(matches):
    pair_history = defaultdict(list)

    for row in matches.itertuples():
        home = row.home_team
        away = row.away_team
        pair_key = tuple(sorted((home, away)))
        pair_history[pair_key].append((home, row.winner))

    h2h_lookup = {}
    for pair_key, games in pair_history.items():
        h2h_lookup[pair_key] = games
    return h2h_lookup


def h2h_home_win_rate(home_team, away_team, h2h_lookup):
    home_match_name = to_match_name(home_team)
    away_match_name = to_match_name(away_team)
    pair_key = tuple(sorted((home_match_name, away_match_name)))
    games = h2h_lookup.get(pair_key, [])

    if not games:
        return 0.5

    home_wins = 0
    for match_home, winner in games:
        if match_home == home_match_name and winner == "home":
            home_wins += 1
        elif match_home != home_match_name and winner == "away":
            home_wins += 1

    return home_wins / len(games)


def build_feature_vector(home_team, away_team, profiles, h2h_lookup, feature_columns):
    home = profiles[home_team]
    away = profiles[away_team]

    values = {
        "home_win_rate": home["win_rate"],
        "away_win_rate": away["win_rate"],
        "home_goals_avg": home["goals_avg"],
        "away_goals_avg": away["goals_avg"],
        "home_conceded_avg": home["conceded_avg"],
        "away_conceded_avg": away["conceded_avg"],
        "h2h_home_wins": h2h_home_win_rate(home_team, away_team, h2h_lookup),
        "home_rank": home["rank"],
        "away_rank": away["rank"],
        "rank_diff": away["rank"] - home["rank"],
        "home_win_rate_vs_top": home.get("win_rate_vs_top", 0.33),
        "away_win_rate_vs_top": away.get("win_rate_vs_top", 0.33),
    }
    return [values[column] for column in feature_columns]


def build_matchup_probabilities(model, feature_columns, profiles, h2h_lookup):
    all_teams = [team for group in GROUPS.values() for team in group]
    probabilities = {}

    for home_team in all_teams:
        for away_team in all_teams:
            if home_team == away_team:
                continue
            features = build_feature_vector(
                home_team, away_team, profiles, h2h_lookup, feature_columns
            )
            home_prob = get_win_proba(model, features)
            rev_features = build_feature_vector(away_team, home_team, profiles, h2h_lookup, feature_columns)
            away_prob = 1 - get_win_proba(model, rev_features)
            p = (home_prob + away_prob) / 2
            p = 0.5 + (p - 0.5) * 0.75
            probabilities[(home_team, away_team)] = p

    return probabilities


def play_match(home_team, away_team, probabilities, allow_draw):
    home_prob = probabilities[(home_team, away_team)]

    if allow_draw:
        evenness = 1 - abs(home_prob - 0.5) * 2
        draw_prob = 0.25 * evenness
        rand = RNG.random()
        if rand < draw_prob:
            return 0, 0, "draw"
        elif rand < draw_prob + (1 - draw_prob) * home_prob:
            return 1, 0, "home"
        else:
            return 0, 1, "away"

    if home_prob >= 0.5:
        home_goals = int(RNG.integers(1, 4))
        away_goals = int(RNG.integers(0, home_goals))
        return home_goals, away_goals, "home"
    else:
        away_goals = int(RNG.integers(1, 4))
        home_goals = int(RNG.integers(0, away_goals))
        return home_goals, away_goals, "away"

def simulate_group_match(home_team, away_team, probabilities):
    return play_match(home_team, away_team, probabilities, allow_draw=True)


def simulate_knockout_match(home_team, away_team, probabilities):
    home_prob = probabilities[(home_team, away_team)]
    return home_team if RNG.random() < home_prob else away_team

def init_group_table(teams):
    return {
        team: {
            "points": 0,
            "gf": 0,
            "ga": 0,
            "gd": 0,
            "mp": 0,
            "w": 0,
            "d": 0,
            "l": 0,
        }
        for team in teams
    }


def update_group_table(table, home_team, away_team, home_goals, away_goals, result):
    table[home_team]["gf"] += home_goals
    table[home_team]["ga"] += away_goals
    table[away_team]["gf"] += away_goals
    table[away_team]["ga"] += home_goals

    table[home_team]["mp"] += 1
    table[away_team]["mp"] += 1

    if result == "home":
        table[home_team]["points"] += 3
        table[home_team]["w"] += 1
        table[away_team]["l"] += 1
    elif result == "away":
        table[away_team]["points"] += 3
        table[away_team]["w"] += 1
        table[home_team]["l"] += 1
    else:
        table[home_team]["points"] += 1
        table[away_team]["points"] += 1
        table[home_team]["d"] += 1
        table[away_team]["d"] += 1

    for team in (home_team, away_team):
        table[team]["gd"] = table[team]["gf"] - table[team]["ga"]


def sort_group_table(table):
    return sorted(
        table.keys(),
        key=lambda team: (
            table[team]["points"],
            table[team]["gd"],
            table[team]["gf"],
        ),
        reverse=True,
    )


def third_place_rank_key(entry):
    group, team, stats = entry
    return (stats["points"], stats["gd"], stats["gf"])


def simulate_group_stage(probabilities):
    real_group_tables, played_matches = build_real_group_state()
    group_tables = {}
    group_order = {}

    for group_name, teams in GROUPS.items():
        table = real_group_tables[group_name]
        for home_team, away_team in itertools.combinations(teams, 2):
            if (home_team, away_team) in played_matches or (away_team, home_team) in played_matches:
                continue
            home_goals, away_goals, result = simulate_group_match(
                home_team, away_team, probabilities
            )
            update_group_table(table, home_team, away_team, home_goals, away_goals, result)

        group_tables[group_name] = table
        group_order[group_name] = sort_group_table(table)

    third_place = []
    for group_name, ordered in group_order.items():
        stats = group_tables[group_name][ordered[2]]
        third_place.append((group_name, ordered[2], stats))

    best_third = [
        team for _, team, _ in sorted(third_place, key=third_place_rank_key, reverse=True)[:8]
    ]

    qualifiers = []
    for group_name in sorted(GROUPS.keys()):
        qualifiers.append(group_order[group_name][0])
        qualifiers.append(group_order[group_name][1])
    qualifiers.extend(best_third)

    return qualifiers


_REAL_GROUP_STATE_CACHE = None


def build_real_group_state():
    global _REAL_GROUP_STATE_CACHE
    if _REAL_GROUP_STATE_CACHE is not None:
        group_tables, played_matches = _REAL_GROUP_STATE_CACHE
        return copy.deepcopy(group_tables), played_matches.copy()

    matches = pd.read_csv(MATCHES_FILE)
    real_2026 = matches[
        (matches["year"] == 2026) & 
        matches["winner"].notna() & 
        (matches["winner"] != "")
    ]
    
    total_found = len(real_2026)
    matched_count = 0
    
    group_tables = {}
    played_matches = set()
    
    for group_name, teams in GROUPS.items():
        table = init_group_table(teams)
        match_to_team = {to_match_name(team): team for team in teams}
        group_match_names = set(match_to_team.keys())
        
        for row in real_2026.itertuples():
            if row.home_team in group_match_names and row.away_team in group_match_names:
                home_team = match_to_team[row.home_team]
                away_team = match_to_team[row.away_team]
                
                update_group_table(
                    table,
                    home_team,
                    away_team,
                    int(row.home_goals),
                    int(row.away_goals),
                    row.winner
                )
                played_matches.add((home_team, away_team))
                matched_count += 1
                
        group_tables[group_name] = table
        
    print(f"Found {total_found} real 2026 matches. Matched {matched_count} to groups.")
    _REAL_GROUP_STATE_CACHE = (group_tables, played_matches)
    return copy.deepcopy(group_tables), played_matches.copy()


def get_groups_with_remaining_matches():
    group_tables, played_matches = build_real_group_state()
    remaining = {}
    for group_name, teams in GROUPS.items():
        remaining_matches = []
        for home_team, away_team in itertools.combinations(teams, 2):
            if (home_team, away_team) not in played_matches and (away_team, home_team) not in played_matches:
                remaining_matches.append((home_team, away_team))
        remaining[group_name] = remaining_matches
    return remaining


def simulate_knockout_round(teams, probabilities):
    current = teams[:]
    while len(current) > 1:
        next_round = []
        for index in range(0, len(current), 2):
            home_team = current[index]
            away_team = current[index + 1]
            winner = simulate_knockout_match(home_team, away_team, probabilities)
            next_round.append(winner)
        current = next_round
    return current[0]


def simulate_tournament(probabilities):
    qualifiers = simulate_group_stage(probabilities)
    bracket = qualifiers[:]
    RNG.shuffle(bracket)
    
    return simulate_knockout_round(bracket, probabilities)

def main():
    # wc_features_ranked.csv is the modelling dataset; match history comes from wc_all_matches.csv
    pd.read_csv(FEATURES_FILE)

    import train_model
    model, feature_columns = train_model.train_and_get_model()

    matches = load_match_history()
    latest_rankings = load_current_rankings()
    profiles = build_team_profiles(matches, latest_rankings)
    h2h_lookup = build_h2h_rates(matches)
    probabilities = build_matchup_probabilities(
        model, feature_columns, profiles, h2h_lookup
    )

    all_teams = [team for group in GROUPS.values() for team in group]
    win_counts = {team: 0 for team in all_teams}
    
    for _ in range(SIMULATIONS):
        champion = simulate_tournament(probabilities)
        win_counts[champion] += 1

    results = (
        pd.DataFrame(
            [
                {"team": team, "wins": wins, "probability": wins / SIMULATIONS * 100}
                for team, wins in win_counts.items()
            ]
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Simulated {SIMULATIONS:,} tournaments using official FIFA 2026 groups")
    print("Top 20 teams by tournament win probability:\n")
    for _, row in results.head(20).iterrows():
        print(f"  {row['team']:<22} {row['probability']:5.2f}%")


if __name__ == "__main__":
    group_tables, played_matches = build_real_group_state()
    for group_name, table in group_tables.items():
        print(f"\nGroup {group_name}:")
        for team in sort_group_table(table):
            stats = table[team]
            print(f"  {team:<25} pts={stats['points']} gd={stats['gd']:+d}")
    main()
