import os
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

path_model_damien = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\Keras_model\Model-2563"
model = keras.models.load_model(path_model_damien)

path_data_stijn = r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\data_sets\tensor_file_balanced_correct.csv"
path_data_damien = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\tens_file_time_balanced_correct.csv"
#path_test5 = r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\data_sets\tensor_file_5ms_evaluation.csv"
df_tens = pd.read_csv(path_data_damien).to_numpy()
df_tens = np.delete(df_tens, 0, axis=1)
# df_tens = df_tens.transpose()

x_sets = df_tens[:,:-1]
x_eval = x_sets[2217:2771,:]
x_test = x_sets[0:2217,:]

y_sets = df_tens[:,-1]
y_eval= y_sets[2217:2771]
y_test = y_sets[0:2217]

# df_tens_5 = pd.read_csv(path_test5).to_numpy()
# df_tens_5 = np.delete(df_tens_5, 0, axis=1)

# x_sets_5 = df_tens_5[:,:-1]


# y_sets_5 = df_tens_5[:,-1]



print("Evaluate on test data")
eval = model.evaluate(x_eval, y_eval, batch_size=128)
print("test loss, test acc:", eval)

print("Generate predictions for all samples")
predictions = model.predict(x_eval)
results = pd.DataFrame(predictions, columns=["estimated values"])
results["true speed"] = y_eval

print("predictions:", results)



absolute_error = np.abs(results["estimated values"] - results["true speed"])
alpha = df_tens[2217:2771,-2]

# plt.scatter(alpha, absolute_error)
# plt.xlabel("Angle of attack")
# plt.ylabel("Absolute error")
# plt.show()

X = np.linspace(0,90,100)
Y = np.linspace(0,16,100)
X, Y = np.meshgrid(X,Y)
Z = Y + 0*X

plt.scatter(results["true speed"], results["estimated values"], color ="red", marker="4", s=75)
plt.plot(Y,Z)
# plt.title('Estimated speed versus True speed')
plt.xlabel("True speed [m/s]")
plt.ylabel("Estimated value [m/s]")
plt.show()


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(alpha, results["true speed"], results["estimated values"],  color='red', label='Estimated data points', marker="4", s=100)
ax.plot_surface(X, Y, Z, alpha= 0.4)
# ax.scatter(mic2, alpha, Vel, color='green', label='Exact values 2')
# surf = ax.plot_trisurf(mic1, alpha, regressor1, label='Approximation')
# surf._edgecolors2d = surf._edgecolor3d
# surf._facecolors2d = surf._facecolor3d
# ax.legend()
# ax.set_title('Estimated speed versus Angle of Attack and True speed')
ax.set_xlabel('Angle of Attack [deg]')
ax.set_ylabel('True speed [m/s]')
ax.set_zlabel('Estimated values [m/s]')
plt.show()

standard_deviation = np.std(absolute_error)
relative_error = []
for i in range(0, len(absolute_error)):
    relative_error_it = absolute_error[i]/results.loc[i, "true speed"]*100
    if relative_error_it < 100:
        relative_error.append(relative_error_it)


mean_relative_error = np.sum(relative_error)/len(relative_error)

mean_absolute_error = np.sum(absolute_error)/len(absolute_error)
print("Mean absolute error of the test data set: ", mean_absolute_error, "\n Standard deviation of the test data set: ", standard_deviation , "\n Mean relative error of the test data set: ", mean_relative_error)