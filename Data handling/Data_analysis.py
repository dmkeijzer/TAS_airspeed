import csv
import numpy as np
import scipy.fft as scf
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
# import pylab as plt
from matplotlib.pyplot import cm

filename = r"C:\Users\Max Reinhard\OneDrive\Documents\BSc2 AE\Test, analysis and simulation\data_list.csv"
file = open(filename, encoding='utf-8')

csvreader = csv.reader(file)

header = []
header = next(csvreader)

rows = []
for row in csvreader:
    temp = []
    for i in range(1, len(row)):
        temp.append(float(row[i]))
    rows.append(temp)
file.close()
rows = np.array(rows)

engine = rows[:, 0]
AoA = rows[:, 1]
Spd = rows[:, 2]
Sum2 = rows[:, 3]
mean2 = rows[:, 5]
Sdev = rows[:, 7]
i = 0
j = 0
k = 0
Spd_plot = []
Sum2_plot = []
# color = iter(cm.rainbow(np.linspace(0, 1, k)))
#while engine[i] == engine[0]:
while AoA[i]==AoA[0]:
    i=i+1
    print(i)
Spd_plot = Spd[j:i]
Sum2_plot = Sum2[j:i]
print(AoA[j:i])

sort_speed = np.sort(Spd_plot)
sort_index = np.argsort(Spd_plot)
print(sort_speed, sort_index)


i = i+1
j = i
# print(AoA[j-2])
plt.plot(Spd_plot, Sum2_plot, label=AoA[j-2], )
Spd_plot = []
Sum2_plot = []





plt.yscale("log")
plt.legend(loc="upper left")
plt.xlabel("Velocity")
plt.ylabel("Integral response (logscale)")
plt.show()



# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(Spd_1, Sum2_1, color="blue")
# ax1.scatter(Spd_2, Sum2_2, color="pink")
# plt.show()
