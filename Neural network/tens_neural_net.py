import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers

path = r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\data_sets\tensor_file.csv"

df_tens = pd.read_csv(path).to_numpy()
df_tens = np.delete(df_tens, 0, axis=1)


model = keras.Sequential(
    [
        layers.Dense(25, activation="sigmoid", name="Dense_1", input_shape=(420002,1)),
        layers.Dense(10, name="Dense_2"),
        layers.Dense(1, name="Dense_3"),
    ]
)

x = tf.ones((420002,1))
y = model(x)

model.summary()

x_sets = df_tens[:-1,:]
x_eval = x_sets[:,:102]
x_test = x_sets[:,102:].transpose()

y_sets = df_tens[-1,:]
y_eval= y_sets[:102]
y_test = y_sets[102:].transpose()

x_val = x_eval[:,89:].transpose()
x_train = x_eval[:,:89].transpose()
y_val = y_eval[89:].transpose()
y_train = y_eval[:89].transpose()

opt = keras.optimizers.Adam(learning_rate=0.001)
loss = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
mtr = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)



model.compile(
    optimizer = opt,  # Optimizer
    # Loss function to minimize
    loss = loss,
    # List of metrics to monitor
    metrics=[mtr],
)

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=60,
    epochs=200,
    validation_data=(x_val, y_val),
)

model.save(r'C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\Neural network\Model')


pass




