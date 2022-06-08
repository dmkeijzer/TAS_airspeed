from turtle import pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
import h5py
import pandas as pd

model = keras.models.load_model(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\Keras_model\Model-2563")
model.summary()

for l in model.layers:
  try:
    print(l.activation)
  except: # some layers don't have any activation
    pass
print(f"Weight inspection\n"
f"________________________________________________________________________________________________________________________________________________________________________________\n"
f"\nAlpha layer 1 = {model.weights[0][-1,:]}\n"
f"\nEngine layer 1 = {model.weights[0][-2,:]}\n"
f"\nRandom sample layer 1 = {model.weights[0][3000,:]}\n")

def predict_some_values():
    path = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\tensor_file_raw_balanced_slice015.csv"
    ch_size = 1
    df_tens_gen = pd.read_csv(path, chunksize=ch_size)
    
    for chunk in df_tens_gen:
        data_point = chunk.to_numpy()[:,1:-1]
        run_config = chunk.to_numpy().flatten()[-2:]
        prediction = model.predict(data_point)
        print(f"\n#------------------------------------------------------------------ \n \
         prediction = {prediction[0][0]} value = {run_config[1]}  difference = {abs(run_config[1] - prediction[0][0])} alpha = {run_config[0]}\n\
#------------------------------------------------------------------\n")

# predict_some_values()