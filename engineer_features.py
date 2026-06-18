from collections import defaultdict

import pandas as pd

INPUT_FILE = "all_matches.csv"
RANKINGS_FILE = "fifa_ranking.csv"
OUTPUT_FILE = "all_features.csv"
MIN_PREVIOUS_MATCHES = 3
RECENT_WINDOW = 10
TOP_WINDOW = 20
TOP_DEFAULT_RATE = 0.33

FEATURE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_win_rate",
    "away_win_rate",
    "home_goals_avg",
    "away_goals_avg",
    "home_conceded_avg",
    "away_conceded_avg",
    "home_win_rate_vs_top",
    "away_win_rate_vs_top",
    "h2h_home_wins",
    "matches_played_diff",
    "home_rank",
    "away_rank",
    "rank_diff",
    "label",
]

TEAM_NAME_MAP = {
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "North Korea": "Korea DPR",
    "United States": "USA",
    "Ivory Coast": "Côte d'Ivoire",
    "Czech Republic": "Czechoslovakia",
    "West Germany": "Germany",
}


def recent_stats(matches):
    recent = matches[-RECENT_WINDOW:]
    count = len(recent)
    wins = sum(match["won"] for match in recent)
    goals_scored = sum(match["goals_for"] for match in recent) / count
    goals_conceded = sum(match["goals_against"] for match in recent) / count
    return wins / count, goals_scored, goals_conceded


def h2h_home_win_rate(past_matches):
    if not past_matches:
        return 0.5
    home_wins = sum(1 for home_won in past_matches if home_won)
    return home_wins / len(past_matches)


def normalize_team(team_name):
    return TEAM_NAME_MAP.get(team_name, team_name)


def attach_rank(matches, rankings, country_col, rank_col):
    ranked_matches = []

    for country, match_group in matches.groupby(country_col, sort=False):
        country_rankings = rankings[rankings["country_full"] == country].sort_values(
            "rank_date"
        )

        if country_rankings.empty:
            group = match_group.copy()
            group[rank_col] = pd.NA
            ranked_matches.append(group)
            continue

        merged = pd.merge_asof(
            match_group.sort_values("match_date"),
            country_rankings[["rank_date", "rank"]].rename(columns={"rank": rank_col}),
            left_on="match_date",
            right_on="rank_date",
            direction="backward",
            allow_exact_matches=False,
        )
        ranked_matches.append(merged)

    return pd.concat(ranked_matches).sort_index()


def win_rate_vs_top(matches):
    recent = matches[-TOP_WINDOW:]
    top_matches = [match for match in recent if pd.notna(match["opponent_rank"]) and match["opponent_rank"] <= 20]
    if len(top_matches) < MIN_PREVIOUS_MATCHES:
        return TOP_DEFAULT_RATE
    return sum(match["won"] for match in top_matches) / len(top_matches)


def record_match(team_history, team_counts, h2h_history, match):
    home_team = match["home_team"]
    away_team = match["away_team"]
    home_goals = int(match["home_goals"])
    away_goals = int(match["away_goals"])
    winner = match["winner"]

    team_history[home_team].append(
        {
            "goals_for": home_goals,
            "goals_against": away_goals,
            "won": winner == "home",
            "opponent_rank": match.get("away_rank_at_match"),
        }
    )
    team_history[away_team].append(
        {
            "goals_for": away_goals,
            "goals_against": home_goals,
            "won": winner == "away",
            "opponent_rank": match.get("home_rank_at_match"),
        }
    )
    team_counts[home_team] += 1
    team_counts[away_team] += 1

    pair_key = tuple(sorted((home_team, away_team)))
    h2h_history[pair_key].append(winner == "home")


def main():
    matches = pd.read_csv(INPUT_FILE)
    rankings = pd.read_csv(RANKINGS_FILE, usecols=["rank", "country_full", "rank_date"])
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    rankings = rankings.sort_values(["country_full", "rank_date"])

    matches["match_date"] = pd.to_datetime(matches["date"])
    matches["home_country"] = matches["home_team"].map(normalize_team)
    matches["away_country"] = matches["away_team"].map(normalize_team)
    matches = attach_rank(matches, rankings, "home_country", "home_rank_at_match")
    matches = attach_rank(matches, rankings, "away_country", "away_rank_at_match")
    matches = matches.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    team_history = defaultdict(list)
    team_counts = defaultdict(int)
    h2h_history = defaultdict(list)

    feature_rows = []
    kept = 0
    skipped = 0

    for _, match in matches.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        winner = match["winner"]

        insufficient_history = (
            team_counts[home_team] < MIN_PREVIOUS_MATCHES
            or team_counts[away_team] < MIN_PREVIOUS_MATCHES
        )

        if insufficient_history or winner == "draw" or winner not in {"home", "away"}:
            skipped += 1
        else:
            home_win_rate, home_goals_avg, home_conceded_avg = recent_stats(
                team_history[home_team]
            )
            away_win_rate, away_goals_avg, away_conceded_avg = recent_stats(
                team_history[away_team]
            )
            home_win_rate_vs_top = win_rate_vs_top(team_history[home_team])
            away_win_rate_vs_top = win_rate_vs_top(team_history[away_team])

            pair_key = tuple(sorted((home_team, away_team)))
            feature_rows.append(
                {
                    "date": match["date"],
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_win_rate": home_win_rate,
                    "away_win_rate": away_win_rate,
                    "home_goals_avg": home_goals_avg,
                    "away_goals_avg": away_goals_avg,
                    "home_conceded_avg": home_conceded_avg,
                    "away_conceded_avg": away_conceded_avg,
                    "home_win_rate_vs_top": home_win_rate_vs_top,
                    "away_win_rate_vs_top": away_win_rate_vs_top,
                    "h2h_home_wins": h2h_home_win_rate(h2h_history[pair_key]),
                    "matches_played_diff": team_counts[home_team] - team_counts[away_team],
                    "label": 1 if winner == "home" else 0,
                    "home_rank": float(match["home_rank_at_match"]) if pd.notna(match["home_rank_at_match"]) else 100.0,
                    "away_rank": float(match["away_rank_at_match"]) if pd.notna(match["away_rank_at_match"]) else 100.0,
                    "rank_diff": (float(match["away_rank_at_match"]) if pd.notna(match["away_rank_at_match"]) else 100.0) - (float(match["home_rank_at_match"]) if pd.notna(match["home_rank_at_match"]) else 100.0),
                }
            )
            kept += 1

        record_match(team_history, team_counts, h2h_history, match)

    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    features.to_csv(OUTPUT_FILE, index=False)

    print(f"Rows kept: {kept}")
    print(f"Rows skipped: {skipped}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
