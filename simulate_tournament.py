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


def sort_group_table(table, h2h_results=None):
    """Sort group table by (points, gd, gf) descending, then break ties
    between adjacent teams using head-to-head results when available."""
    sorted_teams = sorted(
        table.keys(),
        key=lambda team: (
            table[team]["points"],
            table[team]["gd"],
            table[team]["gf"],
        ),
        reverse=True,
    )

    # Post-sort pass: swap adjacent pairs that are fully tied on (pts, gd, gf)
    # but have a head-to-head winner.
    if h2h_results:
        changed = True
        while changed:
            changed = False
            for i in range(len(sorted_teams) - 1):
                a = sorted_teams[i]
                b = sorted_teams[i + 1]
                stats_a = (table[a]["points"], table[a]["gd"], table[a]["gf"])
                stats_b = (table[b]["points"], table[b]["gd"], table[b]["gf"])
                if stats_a != stats_b:
                    continue
                # Check h2h — dict keys are (team_a, team_b) with sorted names
                pair_key = (a, b) if (a, b) in h2h_results else (b, a)
                winner = h2h_results.get(pair_key)
                if winner == b:
                    sorted_teams[i], sorted_teams[i + 1] = b, a
                    changed = True

    return sorted_teams


def third_place_rank_key(entry):
    group, team, stats = entry
    return (stats["points"], stats["gd"], stats["gf"])


_printed_group_status = False


def simulate_group_stage(probabilities):
    global _printed_group_status
    real_group_tables, played_matches, h2h_results = build_real_group_state()
    remaining = get_groups_with_remaining_matches()
    group_tables = {}
    group_order = {}

    locked_groups = []
    simulated_groups = []

    for group_name, teams in GROUPS.items():
        table = real_group_tables[group_name]
        group_rem_matches = remaining.get(group_name, [])

        if len(group_rem_matches) == 0:
            locked_groups.append(group_name)
            group_tables[group_name] = table
            group_order[group_name] = sort_group_table(table, h2h_results)
        else:
            simulated_groups.append(group_name)
            for home_team, away_team in group_rem_matches:
                home_goals, away_goals, result = simulate_group_match(
                    home_team, away_team, probabilities
                )
                update_group_table(table, home_team, away_team, home_goals, away_goals, result)
                h2h_results[(home_team, away_team)] = _h2h_winner_from_result(
                    home_team, away_team, result
                )
            group_tables[group_name] = table
            group_order[group_name] = sort_group_table(table, h2h_results)

    if not _printed_group_status:
        print(f"Locked groups (no simulation needed): {', '.join(locked_groups)}")
        print(f"Simulated groups (had remaining matches): {', '.join(simulated_groups)}")
        _printed_group_status = True

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


def _h2h_winner_from_result(home_team, away_team, result):
    """Return the winner's name from a match result, or None for draws."""
    if result == "home":
        return home_team
    elif result == "away":
        return away_team
    return None


def build_real_group_state():
    """Returns (group_tables, played_matches, h2h_results).
    
    h2h_results maps (team_a, team_b) tuples to the winner's name
    (or None if drawn / not played).
    """
    global _REAL_GROUP_STATE_CACHE
    if _REAL_GROUP_STATE_CACHE is not None:
        group_tables, played_matches, h2h_results = _REAL_GROUP_STATE_CACHE
        return copy.deepcopy(group_tables), played_matches.copy(), copy.deepcopy(h2h_results)

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
    h2h_results = {}
    
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
                h2h_results[(home_team, away_team)] = _h2h_winner_from_result(
                    home_team, away_team, row.winner
                )
                matched_count += 1
                
        group_tables[group_name] = table
        
    print(f"Found {total_found} real 2026 matches. Matched {matched_count} to groups.")
    _REAL_GROUP_STATE_CACHE = (group_tables, played_matches, h2h_results)
    return copy.deepcopy(group_tables), played_matches.copy(), copy.deepcopy(h2h_results)


def get_groups_with_remaining_matches():
    group_tables, played_matches, _h2h = build_real_group_state()
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


_REAL_KNOCKOUT_FIXTURES_CACHE = None


def get_real_knockout_fixtures():
    global _REAL_KNOCKOUT_FIXTURES_CACHE
    if _REAL_KNOCKOUT_FIXTURES_CACHE is not None:
        return copy.deepcopy(_REAL_KNOCKOUT_FIXTURES_CACHE)

    wc_matches_path = os.path.join(_DIR, "wc_matches.csv")
    df = pd.read_csv(wc_matches_path)
    df_last_32 = df[df["stage"] == "LAST_32"]

    match_to_group = {to_match_name(team): team for group in GROUPS.values() for team in group}

    fixtures = []
    confirmed_count = 0
    undetermined_count = 0

    for row in df_last_32.itertuples():
        home = row.home_team
        away = row.away_team

        home_name = None
        if isinstance(home, str) and not pd.isna(home) and home.strip() != "":
            home_name = match_to_group.get(home, home)

        away_name = None
        if isinstance(away, str) and not pd.isna(away) and away.strip() != "":
            away_name = match_to_group.get(away, away)

        fixtures.append({
            "date": row.date,
            "home_team": home_name,
            "away_team": away_name
        })

        if home_name is not None and away_name is not None:
            confirmed_count += 1
        else:
            undetermined_count += 1

    print(f"LAST_32 slots: {confirmed_count} fully confirmed, {undetermined_count} partially/fully undetermined")

    _REAL_KNOCKOUT_FIXTURES_CACHE = fixtures
    return copy.deepcopy(fixtures)


def predict_full_bracket(probabilities):
    # 1. Get the 32 qualifiers using simulate_group_stage()
    qualifiers = simulate_group_stage(probabilities)

    # 2. Build the R32 matchups the same way simulate_tournament() does
    fixtures = get_real_knockout_fixtures()
    bracket = [None] * 32
    placed_teams = set()

    # First pass: Place all confirmed real teams in their fixed slots
    for i, fixture in enumerate(fixtures):
        home = fixture["home_team"]
        away = fixture["away_team"]
        if home is not None:
            bracket[2 * i] = home
            placed_teams.add(home)
        if away is not None:
            bracket[2 * i + 1] = away
            placed_teams.add(away)

    # Find leftover qualifiers
    leftover_qualifiers = [q for q in qualifiers if q not in placed_teams]

    # Second pass: Fill undetermined slots with leftover qualifiers
    leftover_idx = 0
    for j in range(32):
        if bracket[j] is None:
            if leftover_idx < len(leftover_qualifiers):
                bracket[j] = leftover_qualifiers[leftover_idx]
                leftover_idx += 1
            else:
                for q in qualifiers:
                    if q not in bracket:
                        bracket[j] = q
                        break

    # 3. For each round starting with R32
    round_names = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
    rounds = []
    current_teams = bracket[:]

    for round_name in round_names:
        round_matches = []
        next_teams = []
        for index in range(0, len(current_teams), 2):
            team_a = current_teams[index]
            team_b = current_teams[index + 1]

            prob_a = probabilities[(team_a, team_b)]
            prob_b = probabilities.get((team_b, team_a), 1.0 - prob_a)

            winner = team_a if prob_a >= 0.5 else team_b

            match_dict = {
                "round": round_name,
                "team_a": team_a,
                "team_b": team_b,
                "prob_a": prob_a,
                "prob_b": prob_b,
                "winner": winner
            }
            round_matches.append(match_dict)
            next_teams.append(winner)

        rounds.append(round_matches)
        current_teams = next_teams

    champion = current_teams[0]
    return rounds, champion


def simulate_tournament(probabilities):
    # 1. Get the real knockout fixtures list
    fixtures = get_real_knockout_fixtures()

    # 2. Get the simulated qualifiers (32 team names, in their existing group_winner/runner_up/third_place order)
    qualifiers = simulate_group_stage(probabilities)

    # 3. Construct the hybrid bracket:
    # Confirmed real fixtures are used exactly, while undetermined slots use best-effort 
    # simulated qualifier assignment since FIFA's official 495-combination third-place 
    # mapping table isn't available as structured data.
    bracket = [None] * 32
    placed_teams = set()

    # First pass: Place all confirmed real teams in their fixed slots
    for i, fixture in enumerate(fixtures):
        home = fixture["home_team"]
        away = fixture["away_team"]
        if home is not None:
            bracket[2 * i] = home
            placed_teams.add(home)
        if away is not None:
            bracket[2 * i + 1] = away
            placed_teams.add(away)

    # Find leftover qualifiers (those in qualifiers that are not in placed_teams)
    leftover_qualifiers = [q for q in qualifiers if q not in placed_teams]

    # Second pass: Fill undetermined (None) slots with leftover qualifiers in order
    leftover_idx = 0
    for j in range(32):
        if bracket[j] is None:
            if leftover_idx < len(leftover_qualifiers):
                bracket[j] = leftover_qualifiers[leftover_idx]
                leftover_idx += 1
            else:
                # Fallback if leftover list is exhausted (should not happen normally)
                for q in qualifiers:
                    if q not in bracket:
                        bracket[j] = q
                        break

    # 4. Run simulate_knockout_round using this hybrid bracket
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
    group_tables, played_matches, h2h_results = build_real_group_state()
    for group_name, table in group_tables.items():
        print(f"\nGroup {group_name}:")
        for team in sort_group_table(table, h2h_results):
            stats = table[team]
            print(f"  {team:<25} pts={stats['points']} gd={stats['gd']:+d}")
    main()
