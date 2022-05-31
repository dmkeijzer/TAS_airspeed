import os
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.load_model(r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\Neural network\Model_freq")

path = r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\data_sets\tensor_file.csv"

df_tens = pd.read_csv(path).to_numpy()
df_tens = np.delete(df_tens, 0, axis=1)
df_tens = df_tens.transpose()


x_sets = df_tens[:,:-1]
x_eval = x_sets[102:130,:]
x_test = x_sets[0:102,:]

y_sets = df_tens[:,-1]
y_eval= y_sets[102:130]
y_test = y_sets[0:102]

print("Evaluate on test data")
eval = model.evaluate(x_eval, y_eval, batch_size=128)
print("test loss, test acc:", eval)

print("Generate predictions for all samples")
predictions = model.predict(x_sets)
results = pd.DataFrame(predictions, columns=["estimated values"])
results["true speed"] = y_sets

print("predictions:", results)

plt.scatter(results["true speed"], results["estimated values"])
plt.xlabel("True speed")
plt.ylabel("Estimated value")
plt.show()

absolute_error = np.abs(results["estimated values"] - results["true speed"])
mean_absolute_error = np.sum(absolute_error)/len(absolute_error)
print("Mean absolute error of the entire data set: ", mean_absolute_error)