from database_connection import get_engine
import warnings
import pandas as pd

def load_data_test():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    engine = get_engine('Test')
    movies_df = pd.read_sql("""
        SELECT m.id AS movie_id, m.title, m.genre, m.release_date, m.running_time,
            m.budget, m.domestic, m.international, m.in_release, mp.rating AS mpaa_rating
        FROM movies m
        LEFT JOIN mpaa mp ON m.mpaa_id = mp.id
    """, con=engine)
    weeks_df = pd.read_sql("SELECT * FROM weekly_revenue", con=engine)
    weeks_pivot = weeks_df.pivot(index="movie_id", columns="week_number", values="revenue")
    weeks_pivot.columns = [f"week {int(c)}" for c in weeks_pivot.columns]

    df = movies_df.merge(weeks_pivot, on="movie_id", how="left")

    df.rename(columns={
        "title": "Title",
        "genre": "Genre",
        "release_date": "Release Date",
        "running_time": "Running Time",
        "budget": "Budget",
        "domestic": "Domestic",
        "international": "International",
        "in_release": "In Release",
        "mpaa_rating": "MPAA"
    }, inplace=True)

    static_cols = [
        "Title", "Domestic", "International", "Genre", "Release Date",
        "MPAA", "Running Time", "Budget", "In Release"
    ]
    week_cols = [col for col in df.columns if col.startswith("week ")]
    df = df[static_cols + week_cols]
    print("✅ Data Loaded")
    return df