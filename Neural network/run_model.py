import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.load_model(r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\Neural network\Model")

path = r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\data_sets\tensor_file.csv"

df_tens = pd.read_csv(path).to_numpy()
df_tens = np.delete(df_tens, 0, axis=1)



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

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

print("Generate predictions for all samples")
predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)