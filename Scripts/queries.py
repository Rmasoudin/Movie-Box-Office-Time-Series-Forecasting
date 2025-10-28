from sqlalchemy import create_engine, func, desc, cast, Integer
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import label
from import_to_db import Base, Movie, MPAA, WeeklyRevenue

engine = create_engine("sqlite:///db/movies.db")
Session = sessionmaker(bind=engine)
session = Session()

print("\n1. Sample movie titles with MPAA ratings (limit 10):")
results = (
    session.query(Movie.title, MPAA.rating)
    .join(MPAA)
    .filter(MPAA.rating.isnot(None), MPAA.rating != "")
    .limit(10)
    .all()
)
for title, rating in results:
    print(f"{title} - {rating}")
print("\n2. Random movies with budget over $100 million (limit 10):")
budget_clean = func.replace(func.replace(Movie.budget, "$", ""), ",", "")
results = (
    session.query(Movie.title, Movie.budget)
    .filter(cast(budget_clean, Integer) > 100_000_000)
    .order_by(func.random())
    .limit(10)
    .all()
)
for title, budget in results:
    print(f"{title} - {budget}")
print("\n3. Number of movies per genre (limit 10):")
results = (
    session.query(Movie.genre, func.count(Movie.id).label("count"))
    .filter(Movie.genre.isnot(None), Movie.genre != "")
    .group_by(Movie.genre)
    .order_by(desc("count"))
    .limit(10)
    .all()
)
for genre, count in results:
    print(f"{genre}: {count} movies")

print("\n4. Random 5 movies with total revenue:")
revenue_clean = func.replace(func.replace(WeeklyRevenue.revenue, "$", ""), ",", "")
results = (
    session.query(
        Movie.title,
        label("total_revenue", func.sum(cast(revenue_clean, Integer)))
    )
    .join(WeeklyRevenue)
    .group_by(Movie.id)
    .having(func.sum(cast(revenue_clean, Integer)) > 0)
    .order_by(func.random())
    .limit(5)
    .all()
)
for title, total in results:
    print(f"{title} - ${total:,}")
print("\n5. Average number of weeks in release by MPAA rating (limit 10):")
results = (
    session.query(
        MPAA.rating,
        func.avg(
            session.query(func.count(WeeklyRevenue.id))
            .filter(WeeklyRevenue.movie_id == Movie.id)
            .correlate(Movie)
            .scalar_subquery()
        ).label("avg_weeks")
    )
    .join(Movie)
    .group_by(MPAA.rating)
    .limit(10)
    .all()
)
for rating, avg_weeks in results:
    print(f"{rating}: {avg_weeks:.2f} weeks")

session.close()
