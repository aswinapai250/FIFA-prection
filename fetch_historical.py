import csv
import io

import requests

SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
OUTPUT_FILE = "wc_historical.csv"
OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "year",
    "winner",
]


def determine_winner(home_goals, away_goals):
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def main():
    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()

    rows = []
    years = set()

    for row in csv.DictReader(io.StringIO(response.text)):
        tournament = row.get("tournament", "")
        if "friendly" in tournament.lower():
            continue
            
        year = int(row["date"][:4])
        if year <1993:
            continue
        if row["home_score"] == "NA" or row["away_score"] == "NA":
            continue

        home_goals = int(row["home_score"])
        away_goals = int(row["away_score"])

        rows.append(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "year": year,
                "winner": determine_winner(home_goals, away_goals),
            }
        )
        years.add(year)

    rows.sort(key=lambda match: (match["year"], match["date"]))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    year_list = sorted(years)
    print(f"Found {len(rows)} matches across {len(year_list)} World Cup years.")
    print(f"Years: {', '.join(str(year) for year in year_list)}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
