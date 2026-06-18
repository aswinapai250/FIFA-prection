import pandas as pd

FEATURES_FILE = "all_features.csv"
RANKINGS_FILE = "fifa_ranking.csv"
OUTPUT_FILE = "all_features_ranked.csv"
DEFAULT_RANK = 100
MIN_MATCH_YEAR = 1994

TEAM_NAME_MAP = {
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "North Korea": "Korea DPR",
    "United States": "USA",
    "Ivory Coast": "Côte d'Ivoire",
    "Czech Republic": "Czechoslovakia",  # add this
    "West Germany": None,                # add this — will get default rank 100
}


def normalize_team(team_name):
    result = TEAM_NAME_MAP.get(team_name, team_name)
    return result if result is not None else team_name


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


def main():
    features = pd.read_csv(FEATURES_FILE)
    rankings = pd.read_csv(RANKINGS_FILE, usecols=["rank", "country_full", "rank_date"])
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    rankings = rankings.sort_values(["country_full", "rank_date"])

    starting_rows = len(features)

    features["match_date"] = pd.to_datetime(features["date"])
    features["year"] = features["match_date"].dt.year
    features["home_country"] = features["home_team"].map(normalize_team)
    features["away_country"] = features["away_team"].map(normalize_team)

    teams_with_ranking_history = set(rankings["country_full"])

    ranked_features = features[features["year"] >= MIN_MATCH_YEAR].copy()

    ranked_features = attach_rank(
        ranked_features, rankings, "home_country", "home_rank"
    )
    ranked_features = attach_rank(
        ranked_features, rankings, "away_country", "away_rank"
    )

    no_history_home = ~ranked_features["home_country"].isin(teams_with_ranking_history)
    no_history_away = ~ranked_features["away_country"].isin(teams_with_ranking_history)
    ranked_features.loc[no_history_home, "home_rank"] = DEFAULT_RANK
    ranked_features.loc[no_history_away, "away_rank"] = DEFAULT_RANK

    ranked_features = ranked_features.dropna(subset=["home_rank", "away_rank"])
    ranked_features["rank_diff"] = ranked_features["away_rank"] - ranked_features["home_rank"]

    output_columns = [
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
    ranked_features = ranked_features[output_columns].sort_values("date").reset_index(
        drop=True
    )
    ranked_features.to_csv(OUTPUT_FILE, index=False)

    kept = len(ranked_features)
    dropped = starting_rows - kept
    year_min = ranked_features["date"].str[:4].min()
    year_max = ranked_features["date"].str[:4].max()

    print(f"Rows kept: {kept}")
    print(f"Rows dropped: {dropped}")
    print(f"Year range kept: {year_min}-{year_max}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
