import scipy.optimize as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = r"C:\Users\Max Reinhard\OneDrive\Documents\BSc2 AE\Test, analysis and simulation\data_list.csv" #path of the file containing the parameters of interest
weights_file_path = r"C:\Users\Max Reinhard\OneDrive\Documents\BSc2 AE\Test, analysis and simulation\regression_weights.csv" #path of the csv file where you want to save the regression weights

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


def expRegressor(x, a, b, c):   #regressor (note that fo scipy.optimize.curve_fit to work, x has to be the first variable of the regressor, followed by the coefficients
    return a * np.exp(np.multiply(x, b)) + c


def transposeArray(array): #takes an array as input and transforms it into a dataframe equivalent
    array_df = pd.DataFrame(array).transpose()
    return array_df


file_csv = extractFromFile(file_path)

x_set1 = pd.to_numeric(file_csv[3][1:8])
y_set1 = pd.to_numeric(file_csv[4][1:8])

x_set2 = pd.to_numeric(file_csv[3][8:12])
y_set2 = pd.to_numeric(file_csv[4][8:12])

x_set2, y_set2 = sortDataFrame(x_set2, y_set2)
x_set1, y_set1 = sortDataFrame(x_set1, y_set1)

weight_1, pcov1 = sp.curve_fit(expRegressor, x_set1, y_set1)
weight_2, pcov2 = sp.curve_fit(expRegressor, x_set2, y_set2)
print(type(weight_1))
print(type(x_set1))
print(weight_1)
print(pcov1)

weight_df = pd.DataFrame()

weight_1_df = pd.DataFrame(weight_1).transpose()
weight_2_df = pd.DataFrame(weight_2).transpose()


weight_df = weight_df.append(weight_1_df, True)
weight_df = weight_df.append(weight_2_df, True)

weight_df.set_axis(['w0', 'w1', 'w2'], axis='columns', inplace=True)
print(weight_df)


weight_df.to_csv(weights_file_path, encoding= 'utf-8')

plt.scatter(x_set1, y_set1)
plt.plot(x_set1, expRegressor(x_set1, weight_1[0], weight_1[1], weight_1[2]), 'p--k')

plt.scatter(x_set2, y_set2)
plt.plot(x_set2, expRegressor(x_set2, weight_2[0], weight_2[1], weight_2[2]), 'p--g')

plt.yscale('log')
plt.show()

