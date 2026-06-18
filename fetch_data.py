import csv
import time

import requests

BASE_URL = "https://api.football-data.org/v4/competitions/WC/matches"
COMPETITION_URL = "https://api.football-data.org/v4/competitions/WC"
REQUEST_DELAY = 6

# Insert your API key below
import os
API_KEY = os.environ.get("FOOTBALL_API_KEY", "")


HEADERS = {"X-Auth-Token": API_KEY}
CSV_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "stage",
    "year",
    "winner",
]


def fetch_json(url, params=None):
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if response.status_code == 403:
        return None
    response.raise_for_status()
    return response.json()


def get_wc_years():
    data = fetch_json(COMPETITION_URL)
    years = set()

    for season in data.get("seasons", []):
        start_date = season.get("startDate", "")
        if start_date:
            years.add(int(start_date[:4]))

    return sorted(years)


def parse_winner(score):
    winner = (score or {}).get("winner")
    if winner == "HOME_TEAM":
        return "home"
    if winner == "AWAY_TEAM":
        return "away"
    if winner == "DRAW":
        return "draw"
    return ""


def parse_match(match, year):
    score = match.get("score") or {}
    full_time = score.get("fullTime") or {}
    utc_date = match.get("utcDate") or ""

    return {
        "date": utc_date[:10],
        "home_team": (match.get("homeTeam") or {}).get("name", ""),
        "away_team": (match.get("awayTeam") or {}).get("name", ""),
        "home_goals": full_time.get("home", ""),
        "away_goals": full_time.get("away", ""),
        "stage": match.get("stage", ""),
        "year": year,
        "winner": parse_winner(score),
    }

def main():
    if not API_KEY:
        raise ValueError("Set your API key in the API_KEY variable before running.")

    years = get_wc_years()
    time.sleep(REQUEST_DELAY)

    all_matches = []
    for year in years:
        data = fetch_json(BASE_URL, params={"season": year})
        if data is None:
            print(f"Skipping {year} — not accessible on free tier")
            time.sleep(REQUEST_DELAY)
            continue
        for match in data.get("matches", []):
            all_matches.append(parse_match(match, year))
        time.sleep(REQUEST_DELAY)

    all_matches.sort(key=lambda row: (row["year"], row["date"]))

    with open("wc_matches.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_matches)

    print(f"Fetched {len(all_matches)} matches across {len(years)} World Cup years.")
    print(f"Saved to wc_matches.csv")


if __name__ == "__main__":
    main()
