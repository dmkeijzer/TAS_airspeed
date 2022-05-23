import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sp
from sklearn.metrics import mean_squared_error

# file_path = r"C:\Users\Tbeja\Desktop\CSVtest.csv"
file_path = r'C:\Users\loxer\OneDrive\Documents\TAS_proj\.idea\data_list.csv'
# file_loc = r"C:\Users\Max Reinhard\OneDrive\Documents\BSc2 AE\Test, analysis and simulation\TAS_airspeed\data_sets"
# regression = []


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
    # regressor (note that fo scipy.optimize.curve_fit to work, x has to be the first variable of the regressor, followed by the coefficients
    return a*x[0]*x[0]*x[0] + b*x[1]*x[1]*x[1] + c*x[1]*x[1]*x[0] + d*x[1]*x[0]*x[0] +  e*x[0]*x[1] + f*x[0]*x[0] + g*x[1]*x[1] + h*x[0] + i*x[1] + j
# np.multiply(a, x**3) + np.multiply(b, y**3) + np.multiply(c, x**2, y) + np.multiply(d, y**2, x) +
# def linearReg(x, a, b):
#     return np.multiply(a,x) + b

Data_main = extractFromFile(file_path)

Data_main = pd.DataFrame.to_numpy(Data_main)

# print(type(Data_main[0,0]))
Data_main = np.delete(Data_main, 0, 1)
Data_main = np.delete(Data_main, 0, 0)
check = True
i = 0
k = 0

# Truth = (Data_main[0:, 1] == Data_main[i, 1]) * \
#             (Data_main[0:, 0] == Data_main[i, 0])
# index = [i for i, x in enumerate(Truth) if x]
# start = index[0]
# end = index[-1] + 1

a=0
b=0
g = int(input("Select your engine configuration:"))

if g == 0:
    a = 44
    b = 86
    # alpha = pd.to_numeric(Data_main[a:b, 1])
    # engine = pd.to_numeric(Data_main[a:b, 0])
    # x_set = pd.to_numeric(Data_main[a:b, 2])
    # y_set1 = pd.to_numeric(Data_main[a:b, 3])
    # y_set2 = pd.to_numeric(Data_main[a:b, 4])

if g == 30:
    a = 0
    b = 44
    # alpha = pd.to_numeric(Data_main[a:b, 1])
    # engine = pd.to_numeric(Data_main[a:b, 0])
    # x_set = pd.to_numeric(Data_main[a:b, 2])
    # y_set1 = pd.to_numeric(Data_main[a:b, 3])
    # y_set2 = pd.to_numeric(Data_main[a:b, 4])


if g == -1:
    a = 86
    b = 130
    # alpha = pd.to_numeric(Data_main[a:b, 1])
    # engine = pd.to_numeric(Data_main[a:b, 0])
    # x_set = pd.to_numeric(Data_main[a:b, 2])
    # y_set1 = pd.to_numeric(Data_main[a:b, 3])
    # y_set2 = pd.to_numeric(Data_main[a:b, 4])


alpha = pd.to_numeric(Data_main[a:b, 1])
engine = pd.to_numeric(Data_main[a:b, 0])
Vel = pd.to_numeric(Data_main[a:b, 2])
mic1 = pd.to_numeric(Data_main[a:b, 3])
mic2 = pd.to_numeric(Data_main[a:b, 4])

# alpha = pd.to_numeric(Data_main[0:, 1])
# engine = pd.to_numeric(Data_main[0:, 0])
# Vel = pd.to_numeric(Data_main[0:, 2])
# mic1 = pd.to_numeric(Data_main[0:, 3])
# mic2 = pd.to_numeric(Data_main[0:, 4])
inputarr1 = [pd.to_numeric(mic1), pd.to_numeric(alpha)]
inputarr2 = [pd.to_numeric(mic2), pd.to_numeric(alpha)]
# print(alpha[i], engine[i])
weight1, pcov1 = sp.curve_fit(PolyRegressor, inputarr1, Vel)
weight2, pcov2 = sp.curve_fit(PolyRegressor, inputarr2, Vel)
regressor1 = PolyRegressor(inputarr1, weight1[0], weight1[1], weight1[2], weight1[3], weight1[4], weight1[5], weight1[6], weight1[7], weight1[8], weight1[9])
regressor2 = PolyRegressor(inputarr2, weight2[0], weight2[1], weight2[2], weight2[3], weight2[4], weight2[5], weight2[6], weight2[7], weight2[8], weight2[9])
print('MSE', mean_squared_error(Vel, regressor1), mean_squared_error(Vel, regressor2))

# print(type(pd.to_numeric(x_set)))
# try:
#     weight1, pcov1 = sp.curve_fit(PolyRegressor, (pd.to_numeric(mic1), pd.to_numeric(alpha)), Vel)
#     weight2, pcov2 = sp.curve_fit(PolyRegressor, (pd.to_numeric(mic2), pd.to_numeric(alpha)), Vel)
#     regressor1 = PolyRegressor(mic1, alpha, weight1[0], weight1[1], weight1[2], weight1[3], weight1[4], weight1[5], weight1[6], weight1[7])
#     regressor2 = PolyRegressor(mic2, alpha, weight2[0], weight2[1], weight2[2], weight2[3], weight2[4], weight2[5], weight2[6], weight2[7])
# except:
#     weight1, pcov1 = sp.curve_fit(linearReg, pd.to_numeric(x_set), pd.to_numeric(y1_scale), maxfev=5000)
#     weight2, pcov2 = sp.curve_fit(linearReg, pd.to_numeric(x_set), pd.to_numeric(y2_scale), maxfev=5000)
#     weight1 = list(weight1)
#     weight2 = list(weight2)
#     weight1.append(0)
#     weight2.append(0)
#     regressor1 = linearReg(x_set, weight1[0], weight1[1])
#     regressor2 = linearReg(x_set, weight2[0], weight2[1])
# plt.plot(pd.to_numeric(x_set), pd.to_numeric(y_set1),
#          label=Data_main[i][1])
# print(row)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(mic1, alpha, Vel, color='red', label='Exact values')
surf = ax.plot_trisurf(mic1, alpha, regressor1, label='Approximation')
surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d
ax.legend()
ax.set_title('Regression - Engine ' + str(g) + ' - Microphone 1')
ax.set_xlabel('PSL Microphone 1')
ax.set_ylabel('Angle of Attack')
ax.set_zlabel('Velocity')
# k = k + 1
plt.show()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.scatter(mic2, alpha, Vel, color='red', label='Exact values')
surf = ax.plot_trisurf(mic2, alpha, regressor2, label='Approximation')
surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d
ax.legend()
ax.set_title('Regression - Engine ' + str(g) + ' - Microphone 2')
ax.set_xlabel('PSL Microphone 2')
ax.set_ylabel('Angle of Attack')
ax.set_zlabel('Velocity')
plt.show()

# i = end
# check = i < len(Data_main[0:,1])
# while check:
#
#     Truth = (Data_main[0:, 1] == Data_main[i, 1]) * \
#             (Data_main[0:, 0] == Data_main[i, 0])
#     index = [i for i, x in enumerate(Truth) if x]
#     start = index[0]
#     end = index[-1] + 1
#     alpha = pd.to_numeric(Data_main[0:, 1])
#     engine = pd.to_numeric(Data_main[0:, 0])
#     Vel = pd.to_numeric(Data_main[0:, 2])
#     mic1 = pd.to_numeric(Data_main[0:, 3])
#     mic2 = pd.to_numeric(Data_main[0:, 4])
#     dummy = Vel
#     Vel, mic1 = sortDataFrame(dummy, mic1)
#     Vel, mic2 = sortDataFrame(dummy, mic2)
#     if engine[i] == 40 or engine[i] == 50 or engine[i] == 60:
#         i = end
#         continue
#     inputarr = [pd.to_numeric(mic1), pd.to_numeric(alpha)]
#     # print(alpha[i], engine[i])
#     weight1, pcov1 = sp.curve_fit(PolyRegressor, inputarr, Vel)
#     weight2, pcov2 = sp.curve_fit(PolyRegressor, inputarr, Vel)
#     regressor1 = PolyRegressor(mic1, alpha, weight1[0], weight1[1], weight1[2], weight1[3], weight1[4])
#     regressor2 = PolyRegressor(mic2, alpha, weight2[0], weight2[1], weight2[2], weight2[3], weight2[4])
#     # print(type(pd.to_numeric(x_set)))
#     # try:
#     #     weight1, pcov1 = sp.curve_fit(PolyRegressor, (pd.to_numeric(mic1), pd.to_numeric(alpha)), Vel)
#     #     weight2, pcov2 = sp.curve_fit(PolyRegressor, (pd.to_numeric(mic2), pd.to_numeric(alpha)), Vel)
#     #     regressor1 = PolyRegressor(mic1, alpha, weight1[0], weight1[1], weight1[2], weight1[3], weight1[4], weight1[5], weight1[6], weight1[7])
#     #     regressor2 = PolyRegressor(mic2, alpha, weight2[0], weight2[1], weight2[2], weight2[3], weight2[4], weight2[5], weight2[6], weight2[7])
#     # except:
#     #     weight1, pcov1 = sp.curve_fit(linearReg, pd.to_numeric(x_set), pd.to_numeric(y1_scale), maxfev=5000)
#     #     weight2, pcov2 = sp.curve_fit(linearReg, pd.to_numeric(x_set), pd.to_numeric(y2_scale), maxfev=5000)
#     #     weight1 = list(weight1)
#     #     weight2 = list(weight2)
#     #     weight1.append(0)
#     #     weight2.append(0)
#     #     regressor1 = linearReg(x_set, weight1[0], weight1[1])
#     #     regressor2 = linearReg(x_set, weight2[0], weight2[1])
#     # plt.plot(pd.to_numeric(x_set), pd.to_numeric(y_set1),
#     #          label=Data_main[i][1])
#     # print(row)
#     plt.scatter3d(mic1, alpha, Vel)
#     plt.plot(Vel, regressor2)
#     k = k + 1
#     plt.show()
#
#     i = end
#     check = i < len(Data_main[0:,1])
#     # check = k < 2

# plt.legend()
# plt.yscale('log')
# plt.show()
# print("regression")
# print(regression)
#
# df = pd.DataFrame(regression, columns=["Engine", "alpha", "A1", "B1", "C1", "MSE1", "A2", "B2", "C2", "MSE2"])
# df.to_csv(file_loc + "\Regression_coef.csv", index_label= "index")