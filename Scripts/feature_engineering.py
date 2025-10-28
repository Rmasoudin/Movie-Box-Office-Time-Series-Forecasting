import pandas as pd
from datetime import datetime
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def feature_engineer(df):
    df['Domestic'] = df['Domestic'].replace(r'[\$,]', '', regex=True).astype(int)
    df['International'] = df['International'].replace(r'[\$,]', '', regex=True).astype(int)
    df['Budget'] = (
        df['Budget']
        .replace(r'[\$,]', '', regex=True)
        .astype('Int64')
    )


    for i in range(1, 375):  
        col = f"week {i}"
        if col in df.columns:
            df[col] = df[col].replace(r'[\$,]', '', regex=True).astype('Int64')

    def extract_first_date(date_str):
        if not isinstance(date_str, str):
            return None
        match = re.search(r'[A-Za-z]{3,9} \d{1,2}, \d{4}', date_str)
        if match:
            try:
                return datetime.strptime(match.group(), '%b %d, %Y')
            except ValueError:
                return None
        return None
    release_date_index = df.columns.get_loc('Release Date')
    df['First Date'] = df['Release Date'].apply(extract_first_date)
    df.insert(release_date_index + 1, 'Month', df['First Date'].dt.strftime('%B'))
    df.insert(release_date_index + 2, 'Day', df['First Date'].dt.day.astype('Int64'))
    df.insert(release_date_index + 3, 'Year', df['First Date'].dt.year.astype('Int64'))
    df.drop(columns='First Date', inplace=True)
    df.drop(columns='Release Date', inplace=True)
    hour = df['Running Time'].str.extract(r'(\d+)\s*hr').astype('Int64').fillna(0)
    minute = df['Running Time'].str.extract(r'(\d+)\s*min').astype('Int64').fillna(0)
    Running_Time_index = df.columns.get_loc('Running Time')
    df.insert(Running_Time_index, 'total_minutes', hour * 60 + minute)
    df.drop(columns=['Running Time'], inplace=True)
    days_in_release = df['In Release'].str.extract(r'(\d+)\s*days').astype('Int64').fillna(0)
    in_release_index = df.columns.get_loc('In Release')
    df.insert(in_release_index + 1, 'days_in_release', days_in_release)
    df.drop(columns=['In Release'], inplace=True)

    df = df.iloc[: , :20]
    weeks_columns = ['week 1', 'week 2', 'week 3', 'week 4', 'week 5', 'week 6', 'week 7', 'week 8', 'week 9']
    df = df[df[weeks_columns].notna().sum(axis=1) == 9]

    week_cols = [f'week {i}' for i in range(1, 10)]

    df_long = df.melt(
        id_vars=['Title', 'Domestic', 'International', 'Genre', 'Month', 'Day', 'Year',
                'MPAA', 'total_minutes', 'Budget', 'days_in_release'],
        value_vars=week_cols,
        var_name='Week',
        value_name='Revenue'
    )

    df_long['Week'] = df_long['Week'].str.extract(r'(\d+)').astype(int)
    df_long = df_long.sort_values(['Title', 'Week'])

    df_long['Month'] = pd.to_datetime(df_long['Month'], format='%B').dt.month

    df_long['release_date'] = pd.to_datetime(dict(
        year=df_long['Year'],
        month=df_long['Month'],
        day=df_long['Day']
    ))

    df_long['release_dayofweek'] = df_long['release_date'].dt.dayofweek
    df_long = df_long.drop('release_date', axis = 1)
    for order in range(1,5):
        df_long[f'month_sin_{order}'] = np.sin(2 * np.pi * df_long['Month'] / 12 * order)
        df_long[f'month_cos_{order}'] = np.cos(2 * np.pi * df_long['Month'] / 12 * order)

    df_long.to_csv('Data/feature_engineer.csv')

    df_long = pd.get_dummies(df_long, columns=['MPAA'], drop_first=False, dummy_na=True)
    dummy_cols = [col for col in df_long.columns if col.startswith('MPAA_')]
    df_long[dummy_cols] = df_long[dummy_cols].astype(int)
    df_long['Genre'] = df_long['Genre'].fillna('')
    df_long['Genre_list'] = df_long['Genre'].str.split()
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(mlb.fit_transform(df_long['Genre_list']), columns=mlb.classes_, index=df_long.index)
    df_long = pd.concat([df_long, genre_dummies], axis=1)
    df_long.drop(['Genre', 'Genre_list'], axis=1, inplace=True)
    for order in range(1,5):
        df_long[f'week_sin_{order}'] = np.sin(2 * np.pi * df_long['Week'] / 7 * order)
        df_long[f'week_cos_{order}'] = np.cos(2 * np.pi * df_long['Week'] / 7 * order)


    df_long['Revenue'] = np.log(df_long['Revenue'])
    df_long["next_week_revenue"] = df_long.groupby("Title")["Revenue"].shift(-1)
    df_long["last_week_revenue"] = df_long.groupby("Title")["Revenue"].shift(+1)

    df_long["target"] = (df_long["next_week_revenue"])

    df_long['change_from_last_week'] = df_long.groupby('Title')['Revenue'].diff()
    df_long['rolling_std_2'] = df_long.groupby('Title')['target'].rolling(window=2).std().reset_index(level=0, drop=True)
    df_long['rolling_mean_2'] = df_long.groupby('Title')['target'].rolling(window=2).mean().reset_index(level=0, drop=True)

    print("✅ Feature Engineering done")
    return df_long