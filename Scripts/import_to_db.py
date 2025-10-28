import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sklearn.model_selection import train_test_split


os.makedirs('db', exist_ok=True)
os.makedirs('data', exist_ok=True)

csv_path = "data/welp.csv"
if not os.path.isfile(csv_path):
    print("CSV file not found. Please check the file path.")
    exit()

df = pd.read_csv(csv_path)

movie_ids = df['Title'].unique()
train_ids, test_ids = train_test_split(movie_ids, test_size=0.2, random_state=42)

train_df = df[df['Title'].isin(train_ids)].copy()
test_df = df[df['Title'].isin(test_ids)].copy()


Base = declarative_base()

class MPAA(Base):
    __tablename__ = 'mpaa'
    id = Column(Integer, primary_key=True)
    rating = Column(String, unique=True)
    movies = relationship("Movie", back_populates="mpaa")

class Movie(Base):
    __tablename__ = 'movies'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    genre = Column(String)
    release_date = Column(String)
    running_time = Column(String)  
    budget = Column(String)        
    domestic = Column(String)      
    international = Column(String)  
    in_release = Column(String)   
    mpaa_id = Column(Integer, ForeignKey('mpaa.id'))

    mpaa = relationship("MPAA", back_populates="movies")
    weeks = relationship("WeeklyRevenue", back_populates="movie")

class WeeklyRevenue(Base):
    __tablename__ = 'weekly_revenue'
    id = Column(Integer, primary_key=True)
    movie_id = Column(Integer, ForeignKey('movies.id'))
    week_number = Column(Integer)
    revenue = Column(String) 

    movie = relationship("Movie", back_populates="weeks")


def populate_database(df_split, db_path):
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    mpaa_dict = {}
    for rating in df_split["MPAA"].dropna().unique():
        mpaa_obj = MPAA(rating=rating)
        session.add(mpaa_obj)
        session.flush()
        mpaa_dict[rating] = mpaa_obj.id
    session.commit()

    for _, row in df_split.iterrows():
        mpaa_id = mpaa_dict.get(row["MPAA"], None)

        movie = Movie(
            title=row["Title"],
            genre=row["Genre"],
            release_date=row["Release Date"],
            running_time=str(row["Running Time"]),
            budget=row["Budget"],
            domestic=row["Domestic"],
            international=row["International"],
            in_release=str(row["In Release"]),
            mpaa_id=mpaa_id
        )
        session.add(movie)
        session.flush()

        for week in range(1, 375):
            week_col = f"week {week}"
            if week_col in row and not pd.isna(row[week_col]):
                revenue_entry = WeeklyRevenue(
                    movie_id=movie.id,
                    week_number=week,
                    revenue=row[week_col]
                )
                session.add(revenue_entry)

    session.commit()
    session.close()


populate_database(train_df, "db/train.db")
populate_database(test_df, "db/test.db")

print("✅ Databases created: db/train.db and db/test.db")
