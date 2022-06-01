import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sp
from sklearn.metrics import mean_squared_error

# file_path = r"C:\Users\Tbeja\Desktop\CSVtest.csv"
file_path = r"C:\Users\Max Reinhard\Documents\BSc2 AE\Semester 2\Test, analysis and simulation\TAS_airspeed\data_sets\data_list_training.csv"
# file_loc = r"C:\Users\Max Reinhard\OneDrive\Documents\BSc2 AE\Test, analysis and simulation\TAS_airspeed\data_sets"
# regression = []
validation_path = r"C:\Users\Max Reinhard\Documents\BSc2 AE\Semester 2\Test, analysis and simulation\TAS_airspeed\data_sets\data_list_validation.csv"

op_test = r"C:\Users\Max Reinhard\Documents\BSc2 AE\Semester 2\Test, analysis and simulation\TAS_airspeed\data_sets\Op_validation.csv"

op_step = 2500
tr_step = 500000 # I got the best results with these coefficients. Just found them through trial and error


def extractFromFile(path):  # Function extracts the column the data from the csv in the form of a transpose dataframe
    data_set = pd.read_csv(path, sep=",", header=None)
    dataT = pd.DataFrame(data=data_set)
    dataT.transpose()
    return dataT


def sortDataFrame(x, y):
    # x_sort = np.sort(x)
    x_sort = sorted(x)
    y_sort = [i for _, i in sorted(zip(x, y))]
    return x_sort, y_sort


def PolyRegressor(x, a, b, c, d, e, f, g, h, i, j):
    # regressor (note that fo scipy.optimize.curve_fit to work, x has to be the first variable of the regressor,
    # followed by the coefficients
    return a * x[0] * x[0] * x[0] + b * x[1] * x[1] * x[1] + c * x[1] * x[1] * x[0] + d * x[1] * x[0] * x[0] + e * x[
        0] * x[1] + f * x[0] * x[0] + g * x[1] * x[1] + h * x[0] + i * x[1] + j


Data_main = extractFromFile(file_path)

Data_main = pd.DataFrame.to_numpy(Data_main)

# print(type(Data_main[0,0]))
Data_main = np.delete(Data_main, 0, 1)
Data_main = np.delete(Data_main, 0, 0)
check = True
scale = 1
i = 0
k = 0

alpha = pd.to_numeric(Data_main[0:, 1])
engine = pd.to_numeric(Data_main[0:, 0])
Vel = pd.to_numeric(Data_main[0:, 2])
mic1 = pd.to_numeric(Data_main[0:, 3])
mic2 = pd.to_numeric(Data_main[0:, 4])
mic = (mic1 + mic2) / (2 * tr_step)
# index = np.where(mic2 == max(mic2))
# print(alpha[index], engine[index], Vel[index])
inputarr1 = [mic1, alpha]
inputarr2 = [mic2, alpha]
inputarr3 = [mic, alpha]
weight1, pcov1 = sp.curve_fit(PolyRegressor, inputarr1, Vel)
weight2, pcov2 = sp.curve_fit(PolyRegressor, inputarr2, Vel)
weight3, pcov3 = sp.curve_fit(PolyRegressor, inputarr3, Vel)
# print(weight1)
# print(weight2)
# print(weight3)
regressor1 = PolyRegressor(inputarr1, weight1[0], weight1[1], weight1[2], weight1[3], weight1[4], weight1[5],
                           weight1[6], weight1[7], weight1[8], weight1[9])
regressor2 = PolyRegressor(inputarr2, weight2[0], weight2[1], weight2[2], weight2[3], weight2[4], weight2[5],
                           weight2[6], weight2[7], weight2[8], weight2[9])
regressor3 = PolyRegressor(inputarr3, weight3[0], weight3[1], weight3[2], weight3[3], weight3[4], weight3[5],
                           weight3[6], weight3[7], weight3[8], weight3[9])
print('MSE', np.sqrt(mean_squared_error(Vel, regressor1)), np.sqrt(mean_squared_error(Vel, regressor2)),
      (mean_squared_error(Vel, regressor3)))

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(mic1, alpha, Vel, color='red', label='Exact values 1')
# ax.scatter(mic2, alpha, Vel, color='green', label='Exact values 2')
# # surf = ax.plot_trisurf(mic1, alpha, regressor1, label='Approximation')
# # surf._edgecolors2d = surf._edgecolor3d
# # surf._facecolors2d = surf._facecolor3d
# ax.legend()
# ax.set_title('Regression - Engine 30 - Microphone 1')
# ax.set_xlabel('PSL Microphone 1')
# ax.set_ylabel('Angle of Attack')
# ax.set_zlabel('Velocity')
# # k = k + 1

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(mic2, alpha, Vel, color='red', label='Exact values')
# surf = ax.plot_trisurf(mic2, alpha, regressor2, label='Approximation')
# surf._edgecolors2d = surf._edgecolor3d
# surf._facecolors2d = surf._facecolor3d
# ax.legend()
# ax.set_title('Regression - Engine 30 - Microphone 2')
# ax.set_xlabel('PSL Microphone 2')
# ax.set_ylabel('Angle of Attack')
# ax.set_zlabel('Velocity')
# plt.show()
#

# MAIN PLOT BELOW. UNCOMMENT WHEN GETTING FINAL PLOTS
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(mic, alpha, Vel, color='red', label='Exact Velocities (m/s)', alpha=1)
# surf = ax.plot_trisurf(mic, alpha, regressor3, label='Calculated Velocities (m/s)', alpha=0.9)
# surf._edgecolors2d = surf._edgecolor3d
# surf._facecolors2d = surf._facecolor3d
# ax.legend()
# ax.set_title('Regression - All Engine Settings - Microphone Average')
# ax.set_xlabel('SPL')
# ax.set_ylabel('Angle of Attack(deg)')
# ax.set_zlabel('Velocity(m/s)')
# plt.show()

# VALIDATION
# Data_val = extractFromFile(validation_path)
#
# Data_val = pd.DataFrame.to_numpy(Data_val)
# Data_val = np.delete(Data_val, 0, 1)
# Data_val = np.delete(Data_val, 0, 0)
#
# Val_alpha = pd.to_numeric(Data_val[0:, 1])
# Val_engine = pd.to_numeric(Data_val[0:, 0])
# Val_Vel = pd.to_numeric(Data_val[0:, 2])
# Val_mic1 = pd.to_numeric(Data_val[0:, 3])
# Val_mic2 = pd.to_numeric(Data_val[0:, 4])
# Val_mic = (Val_mic1 + Val_mic2) / 2
#
# Val_input = [Val_mic, Val_alpha]
# Result = PolyRegressor(Val_input, weight3[0], weight3[1], weight3[2], weight3[3], weight3[4], weight3[5], weight3[6],
#                        weight3[7], weight3[8], weight3[9])
#
# Error = Result - Val_Vel
# STD = np.std(Error)
# Error = np.abs(Error)
# STD_abs = np.std(Error)
# print("STD:", STD, STD_abs)
# Avg_Error = np.average(Error)
# Avg_spd = np.average(Val_Vel)
# Norm_error = (Avg_Error / Avg_spd) * 100
# print(Avg_Error, Norm_error)
# E30 = np.where(Val_engine == 30)
# E0 = np.where(Val_engine == 0)
# E_1 = np.where(Val_engine == -1)
# Error30 = Error[E30]
# Error0 = Error[E0]
# Error_1 = Error[E_1]
#
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.scatter(Val_alpha[E30], Error30, s=100, label="engine 30%", marker="1")
# ax1.scatter(Val_alpha[E0], Error0, s=100, label="engine 0%", marker="2")
# ax1.scatter(Val_alpha[E_1], Error_1, s=100, label="no propeller", marker="3")
# ax1.legend()
# ax1.set_title("Error Versus Angle of Attack")
# ax1.set_xlabel("Angle of Attack (deg)")
# ax1.set_ylabel("Error (m/s)")
# plt.show()
#
# fig = plt.figure()
# ax2 = fig.add_subplot()
# ax2.scatter(Val_Vel[E30], Error30, s=100, label="engine 30%", marker="1")
# ax2.scatter(Val_Vel[E0], Error0, s=100, label="engine 0%", marker="2")
# ax2.scatter(Val_Vel[E_1], Error_1, s=100, label="no propeller", marker="3")
# ax2.legend()
# ax2.set_title("Error Versus Velocity")
# ax2.set_xlabel("Velocity (m/s)")
# ax2.set_ylabel("Error (m/s)")
# plt.show()

# TESTING THE MODEL WITH THE SHORT CLIPS

Data_op = extractFromFile(op_test)

Data_op = pd.DataFrame.to_numpy(Data_op)
Data_op = np.delete(Data_op, 0, 1)
Data_op = np.delete(Data_op, 0, 0)
print(Data_op)
op_alpha = pd.to_numeric(Data_op[0:, 2])
op_engine = pd.to_numeric(Data_op[0:, 1])
op_Vel = pd.to_numeric(Data_op[0:, 3])
op_mic1 = pd.to_numeric(Data_op[0:, 4])
op_mic2 = pd.to_numeric(Data_op[0:, 5])
op_mic = (op_mic1 + op_mic2) / (2*op_step)

op_input = [op_mic, op_alpha]
Result_op = PolyRegressor(op_input, weight3[0], weight3[1], weight3[2], weight3[3], weight3[4], weight3[5], weight3[6],
                          weight3[7], weight3[8], weight3[9])

Error_op = Result_op - op_Vel
STD_op = np.std(Error_op)
Error_op = np.abs(Error_op)
STD_op_abs = np.std(Error_op)
print("STD:", STD_op, STD_op_abs)
Avg_Error_op = np.average(Error_op)
print("Average", Avg_Error_op)
fig = plt.figure()
ax3 = fig.add_subplot()
ax3.scatter(op_alpha, Error_op, s=100, label="Operational test", marker="1")
ax3.legend()
ax3.set_title("Error Versus Angle of Attack")
ax3.set_xlabel("Angle of Attack (deg)")
ax3.set_ylabel("Error (m/s)")
plt.show()

fig = plt.figure()
ax4 = fig.add_subplot()
ax4.scatter(op_Vel, Error_op, s=100, label="Operational test", marker="1")
ax4.legend()
ax4.set_title("Error Versus Velocity")
ax4.set_xlabel("Velocity (m/s)")
ax4.set_ylabel("Error (m/s)")
plt.show()



fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')
ax.scatter(mic, alpha, Vel, color='red', label='Exact Velocities (m/s)', alpha=1)
surf = ax.plot_trisurf(op_mic, op_alpha, Result_op, label='Calculated Velocities (m/s)', alpha=0.9)
surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d
ax.legend()
ax.set_title('Regression - All Engine Settings - Short clips')
ax.set_xlabel('SPL')
ax.set_ylabel('Angle of Attack(deg)')
ax.set_zlabel('Velocity(m/s)')
plt.legend()
plt.show()