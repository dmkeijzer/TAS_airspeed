import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

model = keras.models.load_model(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\Keras_model\Model-5556")

path1 = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\tensor_file_raw_slice015.csv"
path2 = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\tensor_file_raw_balanced_slice015.csv"

df_tens = pd.read_csv(path2).to_numpy()
df_tens = np.delete(df_tens, 0, axis=1)


x_sets = df_tens[:,:-1]
x_eval = x_sets[1664:2080,:]
x_test = x_sets[0:1664,:]

y_sets = df_tens[:,-1]
y_eval= y_sets[1664:2080]
y_test = y_sets[0:1664]

print("Evaluate on test data")
eval = model.evaluate(x_eval, y_eval, batch_size=128)
print("test loss, test acc:", eval)

print("Generate predictions for all samples")
predictions = model.predict(x_sets)
results = pd.DataFrame(predictions, columns=["estimated values"])
results["true speed"] = y_sets

print("predictions:", results)

<<<<<<< HEAD
plt.scatter(results["true speed"], results["estimated values"])
plt.xlabel("True speed")
plt.ylabel("Estimated value")
=======
plt.scatter(results["true speed"], results["estimated values"], marker= "x" ,s = 30, c = "r")
plt.grid()
>>>>>>> 01393159cdbb32aa312038da783e4c7ec3561c67
plt.show()