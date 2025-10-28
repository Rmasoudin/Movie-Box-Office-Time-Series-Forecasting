import pandas as pd
import os
import warnings
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import pickle


warnings.filterwarnings("ignore", category=SyntaxWarning)

def preprocess(df, pipeline_flag):
    all_genre_cols = [
        "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
        "Documentary", "Drama", "Family", "Fantasy", "History", "Horror",
        "Music", "Musical", "Mystery", "News", "Romance", "Sci-Fi", "Short",
        "Sport", "Thriller", "War", "Western"
    ]


    genre_cols = [col for col in all_genre_cols if col in df.columns]

    if genre_cols:
        df = df[df[genre_cols].sum(axis=1) != 0]

    if "Budget" in df.columns:
        df = df.drop("Budget", axis=1)

    drop_genres = [
        'Animation', 'Biography', 'History', 'Horror', 'Music',
        'Musical', 'Mystery', 'Short', 'Sport', 'War', 'Western', 'News', 'Family'
    ]
    drop_genres_in_df = [col for col in drop_genres if col in df.columns]
    df = df.drop(columns=drop_genres_in_df)

    df.loc[df['Title'] == 'Mirrors', ['Thriller', 'Drama']] = 1
    df.loc[df['Title'] == '12 Mighty Orphans', ['Action', 'Drama']] = 1
    df.loc[df['Title'] == 'Cabin Fever', ['Comedy', 'Thriller']] = 1
    df.loc[df['Title'] == 'CyberWorld', ['Comedy']] = 1
    df.loc[df['Title'] == 'Darkness', ['Thriller']] = 1
    df.loc[df['Title'] == "Devil's Due", ['Thriller', 'Drama', 'Action']] = 1
    df.loc[df['Title'] == "Django Unchained", ['Comedy', 'Drama', 'Action']] = 1
    df.loc[df['Title'] == "Drag Me to Hell", ['Comedy', 'Thriller']] = 1
    df.loc[df['Title'] == "Evil Dead", ['Comedy', 'Thriller', 'Action']] = 1
    df.loc[df['Title'] == "Halloween", ['Thriller']] = 1
    df.loc[df['Title'] == "Hide and Seek", ['Thriller', 'Drama', 'Action']] = 1
    df.loc[df['Title'] == "It", ['Thriller', 'Action', 'Drama']] = 1
    df.loc[df['Title'] == "Jeepers Creepers", ['Thriller', 'Drama']] = 1
    df.loc[df['Title'] == "Lamb of God: The Concert Film", ['Drama']] = 1
    df.loc[df['Title'] == "Late Night with the Devil", ['Thriller']] = 1
    df.loc[df['Title'] == "One Missed Call", ['Thriller', 'Action', 'Crime']] = 1
    df.loc[df['Title'] == "Paranormal Activity", ['Thriller']] = 1
    df.loc[df['Title'] == "Paranormal Activity 2", ['Thriller']] = 1
    df.loc[df['Title'] == "Paranormal Activity 4", ['Thriller', 'Action']] = 1
    df.loc[df['Title'] == "Saw II", ['Thriller', 'Crime']] = 1
    df.loc[df['Title'] == "Scream", ['Thriller', 'Action', 'Drama', 'Comedy']] = 1
    df.loc[df['Title'] == "Scream 3", ['Thriller', 'Action', 'Drama', 'Comedy']] = 1
    df.loc[df['Title'] == "Scream 4", ['Thriller', 'Comedy']] = 1
    df.loc[df['Title'] == "Silent Hill", ['Thriller']] = 1
    df.loc[df['Title'] == "Skinamarink", ['Thriller']] = 1
    df.loc[df['Title'] == "Stigmata", ['Thriller']] = 1
    df.loc[df['Title'] == "Tarot", ['Thriller', 'Comedy']] = 1
    df.loc[df['Title'] == "Terrifier 2", ['Thriller']] = 1
    df.loc[df['Title'] == "The Blair Witch Project", ['Thriller', 'Drama']] = 1
    df.loc[df['Title'] == "The Crazies", ['Thriller', 'Drama']] = 1
    df.loc[df['Title'] == "The Dark and the Wicked", ['Thriller']] = 1
    df.loc[df['Title'] == "The Devil Inside", ['Thriller', ' Drama', 'Action', 'Documentary']] = 1
    df.loc[df['Title'] == "The Exorcist", ['Thriller', 'Drama', 'Action']] = 1
    df.loc[df['Title'] == "The Eye", ['Thriller', 'Drama']] = 1
    df.loc[df['Title'] == "The Ring", ['Thriller', 'Drama']] = 1
    df.loc[df['Title'] == "The Ring Two", ['Thriller', 'Drama']] = 1
    df.loc[df['Title'] == "The Strangers: Prey at Night", ['Thriller', 'Drama', 'Action', 'Crime']] = 1
    df.loc[df['Title'] == "The Texas Chainsaw Massacre", ['Thriller', 'Drama', 'Crime']] = 1
    df.loc[df['Title'] == "House of 1000 Corpses", ['Thriller', 'Crime', 'Documentary']] = 1
    df.loc[df['Title'] == "Jeepers Creepers 2", ['Thriller', 'Drama']] = 1
    df.loc[df['Title'] == "The Texas Chainsaw Massacre: The Beginning", ['Thriller', 'Documentary']] = 1
    df.loc[df['Title'] == "Lights Out", ['Thriller', 'Action', 'Drama']] = 1

    df = df.drop("days_in_release", axis = 1)
    df_movies = df.drop_duplicates(subset='Title').copy()
    all_mpaa_cols = ['MPAA_G', 'MPAA_NC-17', 'MPAA_PG', 'MPAA_PG-13', 'MPAA_R']
    mpaa_cols = [col for col in all_mpaa_cols if col in df_movies.columns]

    genre_cols = ['Action','Adventure','Comedy','Crime','Documentary','Drama','Fantasy','Romance','Sci-Fi','Thriller']


    impute_cols = [col for col in genre_cols + mpaa_cols if col in df_movies.columns]
    impute_df = df_movies[impute_cols]
    for col in all_mpaa_cols:
        if col not in df.columns:
            df[col] = 0

    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(impute_df)

    imputed_df = pd.DataFrame(imputed, columns=genre_cols + mpaa_cols)


    for col in mpaa_cols:
        if col in df_movies.columns:
            df_movies[col] = df_movies[col].astype("Int64")
    df_movies.loc[:, mpaa_cols] = imputed_df[mpaa_cols]
    df_movies.loc[df_movies['MPAA_nan'] == 1, 'MPAA_nan'] = 0


    df_updated = df.drop(columns=mpaa_cols + ['MPAA_nan']).merge(
        df_movies[['Title'] + mpaa_cols + ['MPAA_nan']], on='Title', how='left'
    )
    df_updated = df_updated.drop("MPAA_nan", axis = 1)
    df_updated['total_revenue'] = df_updated['Domestic'] + df_updated['International']
    df_updated = df_updated.drop("Domestic", axis = 1)
    df_updated = df_updated.drop("International", axis = 1)
    df_updated.loc[df_updated[' Drama'].notna(), 'Drama'] = 1
    df_updated = df_updated.drop(" Drama", axis = 1)
    df_updated = df_updated.dropna(subset = ["MPAA_R"])


    df_updated = df_updated.dropna(subset=["target"])
    df_updated = df_updated.drop(["next_week_revenue", "total_revenue", "Week", "Month", 'release_dayofweek', 'Day'], axis = 1)
    df_updated["movie_id"] = LabelEncoder().fit_transform(df_updated["Title"])

    df_updated["Year"] = LabelEncoder().fit_transform(df_updated["Year"])


    df_updated = df_updated.dropna(subset = ['rolling_std_2'])
    df_updated = df_updated.dropna(subset = ['target'])

    num_cols = ['total_minutes', 'Year']

    movie_ids = df_updated['movie_id'].unique()
    if pipeline_flag == "Train":
        df_updated = df_updated.drop('Title', axis = 1)
        train_ids, val_ids  = train_test_split(movie_ids, test_size=0.125, random_state=42)
        df_train = df_updated[df_updated['movie_id'].isin(train_ids)].copy()
        df_val   = df_updated[df_updated['movie_id'].isin(val_ids)].copy()
        X_train = df_train.drop(['target', 'movie_id'], axis = 1).copy()
        y_train = df_train["target"]

        X_val = df_val.drop(['target', 'movie_id'], axis = 1).copy()
        y_val = df_val["target"]

        scaler = StandardScaler()
        X_train[num_cols] = X_train[num_cols].astype(float)
        X_val[num_cols] = X_val[num_cols].astype(float)

        X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val.loc[:, num_cols] = scaler.transform(X_val[num_cols])

        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        X_train.to_csv('Data/X_train.csv')
        X_val.to_csv('Data/X_val.csv')
        y_train.to_csv('Data/y_train.csv')
        y_val.to_csv('Data/y_val.csv')
        print('✅ Preprocess Done')
        return X_train, y_train, X_val, y_val
    
    elif pipeline_flag == "Test":
        titles = df_updated['Title']
        df_updated = df_updated.drop('Title', axis = 1)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        X_test = df_updated.drop(['target', 'movie_id'], axis = 1).copy()
        y_test = df_updated["target"]

        X_test[num_cols] = X_test[num_cols].astype(float)

        X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])
        X_test.to_csv('Data/X_test.csv')
        y_test.to_csv('Data/y_test.csv')
        print('✅ Preprocess Done')
        return X_test, y_test, titles