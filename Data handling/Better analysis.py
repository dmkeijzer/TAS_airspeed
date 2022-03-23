import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as sp
from sklearn.metrics import mean_squared_error

# file_path = r"C:\Users\Tbeja\Desktop\CSVtest.csv"
file_path = r"C:\Users\Max Reinhard\OneDrive\Documents\BSc2 AE\Test, analysis and simulation\data_list.csv"

regression = []


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


def expRegressor(x, a, b,
                 c):  # regressor (note that fo scipy.optimize.curve_fit to work, x has to be the first variable of the regressor, followed by the coefficients
    return a * np.exp(np.multiply(x, b)) + c


Data_main = extractFromFile(file_path)

Data_main = pd.DataFrame.to_numpy(Data_main)

# print(type(Data_main[0,0]))
Data_main = np.delete(Data_main, 0, 1)
Data_main = np.delete(Data_main, 0, 0)
check = True
i = 0
k = 0

while check:
    Truth = (Data_main[0:, 1] == Data_main[i, 1]) * \
            (Data_main[0:, 0] == Data_main[i, 0])
    index = [i for i, x in enumerate(Truth) if x]
    start = index[0]
    end = index[-1] + 1
    alpha = pd.to_numeric(Data_main[0:, 1])
    engine = pd.to_numeric(Data_main[0:, 0])
    x_set = pd.to_numeric(Data_main[start: end, 2])
    y_set1 = pd.to_numeric(Data_main[start: end, 3])
    y_set2 = pd.to_numeric(Data_main[start: end, 4])
    x_set, y_set1 = sortDataFrame(x_set, y_set1)
    x_set, y_set2 = sortDataFrame(x_set, y_set2)
    print(x_set)
    print(y_set1)
    print(type(pd.to_numeric(x_set)))

    weight1, pcov1 = sp.curve_fit(expRegressor, pd.to_numeric(x_set), pd.to_numeric(y_set1),maxfev=5000)
    weight2, pcov2 = sp.curve_fit(expRegressor, pd.to_numeric(x_set), pd.to_numeric(y_set2),maxfev=5000)

    # plt.plot(pd.to_numeric(x_set), pd.to_numeric(y_set1),
    #          label=Data_main[i][1])
    i = end
    # check = i < len(x_set)
    check = k < 2

    mse1 = mean_squared_error(y_set1, expRegressor(x_set, weight1[0], weight1[1], weight1[2]))
    mse2 = mean_squared_error(y_set2, expRegressor(x_set, weight2[0], weight2[1], weight2[2]))
    row = list(engine[k]) + list(alpha[k]) + list(weight1) + [mse1] + list(weight2) + [mse2]
    regression.append(row)
    k = k + 1

# plt.legend()
# plt.yscale('log')
# plt.show()
print(regression)
