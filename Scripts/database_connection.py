from sqlalchemy import create_engine
import os

def get_engine(pipeline_flag):
    if pipeline_flag == "Train":
        db_path = os.path.join(os.path.dirname(__file__), '..', 'db', 'train.db')
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
    elif pipeline_flag == "Test":
        db_path = os.path.join(os.path.dirname(__file__), '..', 'db', 'test.db')
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
    return engine