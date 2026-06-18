import pandas as pd

# Define the rankings to add as specified
rankings_to_add = {
    "Argentina": 1,
    "Spain": 2,
    "France": 3,
    "England": 4,
    "Portugal": 5,
    "Brazil": 6,
    "Netherlands": 7,
    "Morocco": 8,
    "Belgium": 9,
    "Germany": 10,
    "Croatia": 11,
    "Senegal": 12,
    "Uruguay": 13,
    "Colombia": 14,
    "United States": 15,
    "Mexico": 16,
    "Switzerland": 17,
    "Japan": 18,
    "Denmark": 19,
    "Austria": 20,
    "South Korea": 21,
    "Turkey": 22,
    "Ecuador": 23,
    "Norway": 24,
    "Sweden": 25,
    "Paraguay": 26,
    "Scotland": 27,
    "Ghana": 28,
    "Czechia": 29,
    "Iran": 30,
    "Saudi Arabia": 31,
    "Bosnia and Herzegovina": 32,
    "Algeria": 33,
    "Egypt": 34,
    "Ivory Coast": 35,
    "Australia": 36,
    "Jordan": 37,
    "Tunisia": 38,
    "Ukraine": 39,
    "Poland": 40,
    "Cameroon": 41,
    "Nigeria": 42,
    "Serbia": 43,
    "Mali": 44,
    "Greece": 45,
    "Venezuela": 46,
    "Uzbekistan": 47,
    "Chile": 48,
    "Costa Rica": 49,
    "Romania": 50,
}

# Define the exact country_full name mappings to match existing data
mappings = {
    "South Korea": "Korea Republic",
    "United States": "USA",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina"
}

def main():
    # 1. Load fifa_ranking.csv
    file_path = "fifa_ranking.csv"
    df = pd.read_csv(file_path)
    
    # 2. Prepare the new rows
    new_rows = []
    for team, rank in rankings_to_add.items():
        mapped_name = mappings.get(team, team)
        # Use float for rank, keep all other column values (previous_points, rank_change, confederation) 
        # as 0 or empty. Keep total_points and country_abrv empty as well.
        new_row = {
            "rank": float(rank),
            "country_full": mapped_name,
            "country_abrv": "",
            "total_points": 0.0,
            "previous_points": 0.0,
            "rank_change": 0,
            "confederation": "",
            "rank_date": "2025-12-01"
        }
        new_rows.append(new_row)
        
    df_new = pd.DataFrame(new_rows)
    
    # Append the new rows
    df_updated = pd.concat([df, df_new], ignore_index=True)
    
    # 3. Save back to fifa_ranking.csv
    df_updated.to_csv(file_path, index=False)
    
    # Print how many rows were added
    print(f"Added {len(new_rows)} rows to {file_path}.")

if __name__ == "__main__":
    main()
