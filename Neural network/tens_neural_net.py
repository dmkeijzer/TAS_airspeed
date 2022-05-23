import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import layers
import h5py


if __name__ == "__main__":

    path = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\tensor_file_raw1.csv"

    df_tens = pd.read_csv(path).to_numpy()
    df_tens = np.delete(df_tens, 0, axis=1)
    df_tens = df_tens.transpose()


    model = keras.Sequential(
        [
            layers.Dense(30, name="Dense_2" , input_shape=(7002,)),
            layers.BatchNormalization(name="batch_norm"),
            layers.Dense(10, activation="sigmoid", name="Dense_1"), 
            layers.Dense(10, activation="sigmoid", name="Dense_4"), 
            layers.Dense(1, name="Dense_3"),
        ]
    )

    print("\n \nTest input\n #================================================================================================ \n")


    x = tf.ones((1, 7002)) # TODO fix the input of this and figure out what input it wants
    y = model(x)


    print(f"result of only ones = {y} \n #============================================================================================================== \n \n ")
    model.summary()
    print("\n \n")

    x_sets = df_tens[:,:-1]
    x_eval = x_sets[102:130,:]
    x_test = x_sets[0:102,:]

    y_sets = df_tens[:,-1]
    y_eval= y_sets[102:130]
    y_test = y_sets[0:102]




    opt = keras.optimizers.Adam(name="adam", learning_rate=0.001, beta_1= 0.9, beta_2 = 0.9)
    opt1 = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.7, beta_2=0.7, epsilon=1e-07, name="Nadam")
    opt2 = keras.optimizers.SGD(lr=1, decay=1e-6, momentum=0.2, nesterov=True)
    opt3 = keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta")
    loss = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    mtr = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    mtr1 = tf.keras.metrics.MeanAbsoluteError(name ="Mean_absolute_error")



    model.compile(
        optimizer = opt,  # Optimizer
        # Loss function to minimize
        loss = loss,
        # List of metrics to monitor
        metrics=[mtr1],
    )

    print("\nFit model on training data\n")
    history = model.fit(
        x_test,
        y_test,
        batch_size=15, #smaller batches seem to give better results so try that I guess
        epochs=2,
        shuffle= True,
        validation_split=0.2, #TODO figure out what inputs it wants
    )

    # Evaluate the model on the test data using `evaluate`
    print("\nEvaluate on test data\n")
    results = model.evaluate(x_eval, y_eval, batch_size=1)
    print("test loss, test metrics:", results)

    model.save(r'C:\Users\damie\OneDrive\Desktop\Damien\TAS\Keras_model\model_test5')






