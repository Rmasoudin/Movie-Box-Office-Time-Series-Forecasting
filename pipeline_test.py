import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Scripts'))

from Scripts.load_data_test import load_data_test
from Scripts.feature_engineering import feature_engineer
from Scripts.preprocess import preprocess

from Scripts.make_prediction import make_prediction
from Scripts.save_predictions import save_predictions

from Scripts.mlflow_launch import launch_mlflow_dashboard

df = load_data_test()
df = feature_engineer(df)
X_test, y_test, titles = preprocess(df.copy(), 'Test')

predicts = make_prediction(X_test, y_test)

save_predicts = save_predictions(predicts, titles)


launch_mlflow_dashboard()