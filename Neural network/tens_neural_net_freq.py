import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import h5py


if __name__ == "__main__":

    path = r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\data_sets\tensor_file_balanced_correct.csv"

    df_tens = pd.read_csv(path).to_numpy()
    df_tens = np.delete(df_tens, 0, axis=1)
    # df_tens = df_tens.transpose()


    def my_init(shape, dtype=None):
        a = np.zeros(shape, dtype="float32")
        a[-1] = 1
        a = tf.convert_to_tensor(a)
        return a

    model = keras.Sequential(
        [
            layers.BatchNormalization(name="batch_norm"),
            layers.Dense(5, name="Dense_1" , input_shape=(2391,), activation="sigmoid"),
            # layers.Dense(4, activation="sigmoid", name="Dense_2"),
            layers.Dense(2, activation="softmax" , name="Dense_3"), 
            layers.Dense(1, name="Dense_4"),
        ]
    )

    print("\n \nTest input\n #================================================================================================ \n")


    x = tf.ones((1, 2391)) # TODO fix the input of this and figure out what input it wants
    y = model(x)


    print(f"result of only ones = {y} \n #============================================================================================================== \n \n ")
    model.summary()
    print("\n \n")

    x_sets = df_tens[:,:-1]
    x_eval = x_sets[1664:2080,:]
    x_test = x_sets[0:1664,:]

    y_sets = df_tens[:,-1]
    y_eval= y_sets[1664:2080]
    y_test = y_sets[0:1664]




    opt = keras.optimizers.Adam(name="adam", learning_rate=0.003, beta_1= 0.95, beta_2 = 0.99)
    opt1 = keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.95, beta_2=0.999, epsilon=1e-07, name="Nadam")
    opt2 = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.2, nesterov=True)
    opt3 = keras.optimizers.Adadelta(learning_rate=0.05, rho=0.95, epsilon=1e-07, name="Adadelta")
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
        batch_size= 10, #smaller batches seem to give better results so try that I guess
        epochs= 300,
        validation_split=0.2, #TODO figure out what inputs it wants
    )

    # Evaluate the model on the test data using `evaluate`
    print("\nEvaluate on test data\n")
    results = model.evaluate(x_eval, y_eval, batch_size=10)
    print("test loss, test metrics:", results)

    def plot_history(history, key):
        plt.plot(history.history[key])
        plt.plot(history.history['val_'+key])
        plt.xlabel("Epochs")
        plt.ylabel(key)
        plt.legend([key, 'val_'+key])
        plt.show()

    plot_history(history, 'Mean_absolute_error')

    model.save(r'C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\Neural network\Model_freq_fatter_improved')






