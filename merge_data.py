import pandas as pd

HISTORICAL_FILE = "wc_historical.csv"
MATCHES_FILE = "wc_matches.csv"
OUTPUT_FILE = "all_matches.csv"
OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "stage",
    "year",
    "winner",
]


def main():
    historical = pd.read_csv(HISTORICAL_FILE)
    matches = pd.read_csv(MATCHES_FILE)

    historical["year"] = pd.to_datetime(historical["date"]).dt.year
    historical = historical[historical["year"] >= 1993].copy()
    historical["stage"] = ""

    completed_2026 = matches[
        (matches["year"] == 2026) & (matches["winner"].notna()) & (matches["winner"] != "")
    ]

    combined = pd.concat([historical, completed_2026], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"],
        keep="first",
    )

    # 1. Split the combined dataframe into df_2026 and df_other
    df_2026 = combined[combined["year"] == 2026].copy()
    df_other = combined[combined["year"] != 2026].copy()

    # 2. Sort df_2026 so that rows with a non-null "stage" come FIRST
    df_2026["has_stage"] = df_2026["stage"].notna() & (df_2026["stage"] != "")
    df_2026 = df_2026.sort_values("has_stage", ascending=False)

    # 3. Drop duplicates based on (home_team, away_team) only
    initial_2026_count = len(df_2026)
    df_2026 = df_2026.drop_duplicates(
        subset=["home_team", "away_team"],
        keep="first",
    )
    removed_2026_dupes = initial_2026_count - len(df_2026)
    df_2026 = df_2026.drop(columns=["has_stage"])

    # 4. Concatenate df_other and the deduplicated df_2026 back together
    combined = pd.concat([df_other, df_2026], ignore_index=True)

    # 5. Sort by date, reset index
    combined = combined.sort_values("date").reset_index(drop=True)

    # 6. Print how many 2026 duplicate rows were removed in this pass
    print(f"2026 duplicate rows removed in this pass: {removed_2026_dupes}")


    combined = combined[OUTPUT_COLUMNS]
    combined.to_csv(OUTPUT_FILE, index=False)

    year_min = int(combined["year"].min()) if not combined.empty else None
    year_max = int(combined["year"].max()) if not combined.empty else None
    print(f"Total rows: {len(combined)}")
    print(f"Year range: {year_min} to {year_max}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
