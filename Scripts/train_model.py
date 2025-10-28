import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pandas as pd
import numpy as np
import mlflow
import mlflow.tensorflow
import os

os.makedirs("models", exist_ok=True)  


def train_model(X_train, y_train, X_val, y_val):

    # Convert data
    X_train = X_train.to_numpy(dtype='float32')
    y_train = y_train.to_numpy(dtype='float32').ravel()
    X_val = X_val.to_numpy(dtype='float32')
    y_val = y_val.to_numpy(dtype='float32').ravel()

    # Enable MLflow autologging for TensorFlow
    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        # Define model
        model = models.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='sigmoid'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=[
                'mae',
                tf.keras.metrics.RootMeanSquaredError(),
                tf.keras.metrics.MeanAbsolutePercentageError()
            ]
        )

        print("X_train:", type(X_train), np.shape(X_train))
        print("y_train:", type(y_train), np.shape(y_train))

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            shuffle=True,
            callbacks=[
                callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=1
        )

        # Save locally
        model.save("models/nn_model.keras")

        print("✅ Model Trained and Logged to MLflow")

    return model
# def train_model(X_train, y_train, X_val, y_val):

#     model = models.Sequential([
#     layers.Input(shape=(X_train.shape[1],)),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.3),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10, activation='sigmoid'),
#     layers.Dropout(0.2),
#     layers.Dense(1)
#     ])
#     X_train = X_train.to_numpy(dtype='float32')
#     y_train = y_train.to_numpy(dtype='float32').ravel()

#     X_val = X_val.to_numpy(dtype='float32')
#     y_val = y_val.to_numpy(dtype='float32').ravel()
#     model.compile(

#         optimizer='adam',
#         loss='mse',
#         metrics=[
#             'mae', 
#             tf.keras.metrics.RootMeanSquaredError(),
#             tf.keras.metrics.MeanAbsolutePercentageError()
#         ]
#     )
#     print("X_train:", type(X_train), np.shape(X_train))
#     print("y_train:", type(y_train), np.shape(y_train))


#     history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=50,
#     batch_size=32,
#     shuffle = True,
#     callbacks=[
#         callbacks.EarlyStopping(patience=10, restore_best_weights=True)
#     ],
#     verbose=1
#     )

#     model.save("models/nn_model.keras")

#     print("✅Model Trained")