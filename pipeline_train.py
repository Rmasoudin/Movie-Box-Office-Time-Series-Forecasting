import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Scripts'))

from Scripts.load_data_train import load_data_train
from Scripts.feature_engineering import feature_engineer
from Scripts.preprocess import preprocess
from Scripts.train_model import train_model

df = load_data_train()
df = feature_engineer(df)
X_train, y_train, X_val, y_val = preprocess(df, 'Train')

df = train_model(X_train, y_train, X_val, y_val)