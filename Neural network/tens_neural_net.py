import os
import random
from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import h5py
from random import randint
from dask import dataframe as dd

df_tens = np.ones((1,15363))

if __name__ == "__main__":

    path = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\tensor_file_raw_balanced_slice015.csv"
    print("started opening")
    ch_size = 520
    load_counter = 0
    chunk = pd.read_csv(path, chunksize=ch_size)


    
    
    for chunk in chunk:
        data_points = chunk.to_numpy()[:,1:]
        df_tens = np.append(df_tens, data_points, axis=0)
        load_counter += ch_size
        print(f"\n#------------------------------------------------------------------------------------------------------------------\n\
         {round(load_counter / 2080 * 100,2)} %\
          \n#------------------------------------------------------------------------------------------------------------------")
    
    df_tens = np.delete(df_tens,0, axis=0)
    np.random.shuffle(df_tens)



    def my_init(shape, dtype=None):
        a = 7 * np.ones((shape[0], shape[1])) + np.random.randn(shape[0], shape[1])
        a[-1,:] = 30
        a[-2,:] = 4
        a = a.astype("float32")
        a = tf.convert_to_tensor(a)
        return a
    
    bias = tf.keras.initializers.RandomNormal(mean=6, stddev=2, seed=None)



    model = keras.Sequential(
        [
            layers.Dense(15, name="Dense_1" , input_shape=(15362,), kernel_initializer= my_init), #Needs more parameters, it is not overfitting but oscillating
            # layers.Dense(5, name="Dense_2"),
            layers.BatchNormalization(name="batch_norm"),
            layers.Dense(10, activation= "sigmoid", name="Dense_3"), 
            # layers.Dense(2, activation= "sigmoid", name="Dense_4"), 
            layers.Dense(7, activation = "softmax",  name="Dense_5"), 
            layers.Dense(1, name="Dense_6"),
        ]
    )

    print("\n \nTest input\n #================================================================================================ \n")


    x = tf.ones((1, 15362)) # TODO fix the input of this and figure out what input it wants
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




    opt = keras.optimizers.Adam(name="adam", learning_rate=0.035, beta_1= 0.93, beta_2 = 0.99)
    opt1 = keras.optimizers.Nadam(learning_rate=0.01, beta_1=0.8, beta_2=0.8, epsilon=1e-07, name="Nadam")
    opt2 = keras.optimizers.SGD(lr=1, decay=1e-6, momentum=0.2, nesterov=True)
    opt3 = keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta")
    opt4 = keras.optimizers.RMSprop(learning_rate= 0.001, rho=0.9, centered=True, name= 'RMSprop')
    opt5 = keras.optimizers.Adagrad(learning_rate= 0.001, initial_accumulator_value= 0.1, name= "Adagrad")
    opt6 = keras.optimizers.Ftrl(    learning_rate=0.001,learning_rate_power=-0.5,initial_accumulator_value=0.1, l1_regularization_strength=0.0,  \
    l2_regularization_strength=0.0,name="Ftrl",l2_shrinkage_regularization_strength=0.0,beta=0.0)

    loss1 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
    # loss2 = keras.losses.huber(delta=1.0)
    loss3 = keras.losses.MeanSquaredLogarithmicError(reduction="auto", name="mean_squared_logarithmic_error")


    mtr = keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    mtr1 = tf.keras.metrics.MeanAbsoluteError(name ="Mean_absolute_error")



    model.compile(
        optimizer = opt,  # Optimizer
        # Loss function to minimize
        loss = loss1,
        # List of metrics to monitor
        metrics=[mtr1],
    )

    print("\nFit model on training data\n")
    history = model.fit(
        x_test,
        y_test,
        batch_size=30, 
        epochs=1200,
        shuffle= True,
        validation_split=0.2, 
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

    random_int = str(randint(0,10000))
    print(random_int)

    plot_history(history, 'Mean_absolute_error')


    

    model.save(r'C:\Users\damie\OneDrive\Desktop\Damien\TAS\Keras_model\Model' + "-" + random_int)
    print(f"Saved as {random_int}")






