import os
from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import h5py

df_tens = np.ones((1,15363))

if __name__ == "__main__":

    path = r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\data_sets\tensor_file_raw_slice015.csv"
    print("started opening")
    ch_size = 3000
    load_counter = 0
    df_tens_gen = pd.read_csv(path, chunksize=ch_size)
    
    for chunk in df_tens_gen:
        data_points = chunk.to_numpy()[:,1:]
        df_tens = np.append(df_tens, data_points, axis=0)
        load_counter += ch_size
        print(f"\n#------------------------------------------------------------------------------------------------------------------\n\
         {round(load_counter / 12672 * 100,2)} %\
          \n#------------------------------------------------------------------------------------------------------------------")
    
    df_tens = np.delete(df_tens,0, axis=0)
    np.random.shuffle(df_tens)



    def my_init(shape, dtype=None):
        a = np.random.randn(shape[0], shape[1]).astype("float32")
        a[-1] = 15
        a[-2] = 30
        a = tf.convert_to_tensor(a)
        return a


    model = keras.Sequential(
        [
            layers.Dense(50, name="Dense_1" , input_shape=(15362,), kernel_initializer=my_init), #Needs more parameters, it is not overfitting but oscillating
            layers.Dense(50, name="Dense_2", kernel_initializer=my_init),
            layers.BatchNormalization(name="batch_norm"),
            layers.Dense(20, activation= "sigmoid", name="Dense_3", kernel_initializer=my_init), 
            layers.Dense(20, activation = "sigmoid",  name="Dense_4", kernel_initializer=my_init), 
            layers.Dense(1, name="Dense_5"),
        ]
    )

    print("\n \nTest input\n #================================================================================================ \n")


    x = tf.ones((1, 15362)) # TODO fix the input of this and figure out what input it wants
    y = model(x)


    print(f"result of only ones = {y} \n #============================================================================================================== \n \n ")
    model.summary()
    print("\n \n")

    x_sets = df_tens[:,:-1]
    x_eval = x_sets[10138:12673,:]
    x_test = x_sets[0:10138,:]

    y_sets = df_tens[:,-1]
    y_eval= y_sets[10138:12673]
    y_test = y_sets[0:10138]




    opt = keras.optimizers.Adam(name="adam", learning_rate=0.001, beta_1= 0.2, beta_2 = 0.2)
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
        batch_size=5, #smaller batches seem to give better results so try that I guess
        epochs=5,
        shuffle= True,
        validation_split=0.2, #TODO figure out what inputs it wants
    )

    # Evaluate the model on the test data using `evaluate`
    print("\nEvaluate on test data\n")
    results = model.evaluate(x_eval, y_eval, batch_size=6)
    print("test loss, test metrics:", results)

    def plot_history(history, key):
        plt.plot(history.history[key])
        plt.plot(history.history['val_'+key])
        plt.xlabel("Epochs")
        plt.ylabel(key)
        plt.legend([key, 'val_'+key])
        plt.show()

    plot_history(history, 'Mean_absolute_error')

    model.save(r'C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\Neural network\Model_time')






