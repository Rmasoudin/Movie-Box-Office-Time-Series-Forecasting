from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np


def make_prediction(X_test, y_test):
    model = keras.models.load_model("models/nn_model.keras")

    y_pred = model.predict(X_test).flatten()
    y_true = np.array(y_test).flatten()
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"✅ MSE:  {mse:.4f}")
    print(f"✅ MAPE: {mape:.2%}")


    baseline_y_true = y_test.values
    baseline_y_pred = X_test["Revenue"].values

    baseline_mse = mean_squared_error(baseline_y_true, baseline_y_pred)
    baseline_mape = mean_absolute_percentage_error(baseline_y_true, baseline_y_pred)
    print(f"📉 Baseline MSE:  {baseline_mse:.4f}")
    print(f"📉 Baseline MAPE: {baseline_mape:.2%}")
    

    return y_pred