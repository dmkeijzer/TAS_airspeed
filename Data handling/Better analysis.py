import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# file_path = r"C:\Users\Tbeja\Desktop\CSVtest.csv"
file_path = r"C:\Users\Max Reinhard\OneDrive\Documents\BSc2 AE\Test, analysis and simulation\data_list.csv"


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


def expRegressor(x, a, b):
    return a * np.exp(x) + b


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
    # print(Truth)
    x_set = pd.to_numeric(Truth * Data_main[0:, 2])
    y_set = pd.to_numeric(Truth * Data_main[0:, 3])
    x_set, y_set = sortDataFrame(x_set, y_set)
    #print(max(x_set))
    plt.plot(pd.to_numeric(x_set), pd.to_numeric(y_set),
             label=Data_main[i][1])
    i = np.nonzero(x_set == np.nanmax(x_set))[0][-1] + 1
    #print(i)
    #print(x_set)
    check = i < len(x_set)
    k = k + 1



plt.legend()
# plt.yscale('log')
plt.show()
