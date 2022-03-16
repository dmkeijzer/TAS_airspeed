import csv
import numpy as np
import scipy.fft as scf
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import seaborn as sns


array = pd.read_csv(r"C:\Users\Stijn van Teylingen\OneDrive - Delft University of Technology\Test, Analysis & Simulation\Python_AI\TAS_airspeed\Interpolation\data_list.csv")
array2 = array.drop([42,43,44], axis=0)

array2["log_sum2"] = np.log10(array2["sum2"])

select_alpha0 = array2.loc[array2['alpha'] == 0.0]
select_alpha45 = array2.loc[array2['alpha'] == 45.0]
# print(array.to_string())

sns.pairplot(select_alpha0[["engine", "alpha", "log_sum2", "sum2", "stdev2", "v"]], hue="engine", corner=True, palette="bright", diag_kind="hist")
sns.pairplot(select_alpha45[["engine", "alpha", "log_sum2", "sum2", "stdev2", "v"]], hue="engine", corner=True, palette="bright", diag_kind="hist")
sns.pairplot(array2[["engine", "alpha", "sum3", "mean3", "stdev3", "v"]], hue="engine", corner=True, palette="deep", diag_kind="hist")
plt.show()