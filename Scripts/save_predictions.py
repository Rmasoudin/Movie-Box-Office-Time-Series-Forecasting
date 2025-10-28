import sqlite3
import pandas as pd

def save_predictions(predicts, titles):
    results_df = pd.DataFrame({
    "Title": titles,
    "Prediction": predicts
})
    conn = sqlite3.connect("movie_predictions.db")


    results_df.to_sql("predictions", conn, if_exists="replace", index=False)

    conn.close()