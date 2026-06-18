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
    combined = combined.sort_values("date").reset_index(drop=True)

    combined = combined[OUTPUT_COLUMNS]
    combined.to_csv(OUTPUT_FILE, index=False)

    year_min = int(combined["year"].min()) if not combined.empty else None
    year_max = int(combined["year"].max()) if not combined.empty else None
    print(f"Total rows: {len(combined)}")
    print(f"Year range: {year_min} to {year_max}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
