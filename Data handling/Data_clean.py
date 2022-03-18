import csv
import numpy as np
import scipy.fft as scf
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

sys.path.append('D:\Aerospace Engineering\Bachelor Year 2\AE2223-I Test Analysis & Simulation\CSV')
# files = os.listdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\clean_data")
# os.chdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\clean_data")

class trial_data:
    
    """ This class allows convenient access to all data from chosen excel. Methods can be added accordingly. """
    
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.name = path
        self.mic_1 = self.df.loc[:, "m1"].values #voltage values from mic 1
        self.mic_2 = self.df.loc[:, "m2"].values #voltage values from mic 2
        self.mic_3 = self.df.loc[:, "m3"].values #voltage values from mic 3
        self.time_arr = self.df.loc[:,"m1_Time*"].values #series containing all time points
        self.x = scf.rfftfreq(len(self.time_arr), self.time_arr[1])
        self.y2 = np.abs(scf.rfft(self.mic_2))
        self.y3 = np.abs(scf.rfft(self.mic_3))
        self.y2 = abs(((self.y2 > 20) * (30 < self.x) * (self.x < 70))-1) * self.y2
        self.y3 = abs(((self.y3 > 20) * (30 < self.x) * (self.x < 70))-1) * self.y3
        for i in range(1, int(self.x[-1] / 50)):
            self.y2 = abs(((self.y2 > 20) * (50 * i - 5 < self.x) * (self.x < 50 * i + 5))-1) * self.y2
            self.y3 = abs(((self.y3 > 20) * (50 * i - 5 < self.x) * (self.x < 50 * i + 5))-1) * self.y3
        self.P_sum2 = np.sum(self.y2)
        self.P_sum3 = np.sum(self.y3)
        self.mean2 = np.mean(self.y2)
        self.mean3 = np.mean(self.y3)
        self.stdev2 = np.std(self.y2)
        self.stdev3 = np.std(self.y3)
        """ The following function let's you plot voltage vs time (vt). The index parameter
        let's you decide on which mic to plot. Will probs change since it is useless now"""
    def plot_vt(self, index , th_line = 0.6):
        mic = [self.mic_1, self.mic_2, self.mic_3]

        plt.plot(self.time_arr[0:10000], mic[index][0:10000], "-k", linewidth = th_line)
        plt.title(f"{self.name}") 
        plt.ylabel(f"voltage mic {index +1} [V]") 
        plt.xlabel("Time [s]")
        plt.show()

# run = trial_data(files[24])
#
# print(run.y2)
# def data_handling(filename):
    # run = trial_data(filename)

    # csvreader = csv.reader(file)
    #
    # header = []
    # header = next(csvreader)

    # rows = []
    # for row in csvreader:
    #     temp = []
    #     for i in range(1, len(row)):
    #         temp.append(float(row[i]))
    #     rows.append(temp)
    # file.close()
    # rows = np.array(rows)
    # t = rows[:, 0]
    # m2 = rows[:, 3]
    # m3 = rows[:, 5]
    # x = run.x
    # y2 = run.y2
    # y3 = run.y3
    # y2 = np.abs(y2)
    # y3 = np.abs(y3)

    # c = 0
    # d = len(x)
    # plt.plot(x[c:d], y2[c:d], color='blue')
    # plt.show()
    # plt.plot(x[c:d], y3[c:d], color='green')
    # plt.show()
    # y2_time = scf.irfft(y2)
    # y3_time = scf.irfft(y3)
    # plt.plot(t[:10000], y2_time[:10000], color='blue')
    # plt.plot(t[:10000], y3_time[:10000], color='green')
    # plt.show()

    # P_sum2 = np.sum(y2)
    # P_sum3 = np.sum(y3)
    # mean2 = np.mean(y2)
    # mean3 = np.mean(y3)
    # stdev2 = np.std(y2)
    # stdev3 = np.std(y3)
    # return [P_sum2, P_sum3], [mean2, mean3], [stdev2, stdev3]

directory = 'D:\Aerospace Engineering\Bachelor Year 2\AE2223-I Test Analysis & Simulation\CSV'


Englist = []
Alphalist = []
Vlist = []

for file in os.listdir(directory):
    # file = 'cleandata_run_noPropeller_60alpha_10ms_data_movedUAV.csv'
    f = os.path.join(directory, file)
    terms = file.split('_')
    print(terms)

    if terms[2] == 'noEngine':
        Englist.append(0)
    elif terms[2] == 'noPropeller':
        Englist.append(-1)
    else:
        Englist.append(int(terms[2][6:8]))

    try:
        Alphalist.append(int(terms[3][:2]))
    except:
        Alphalist.append(int(terms[3][:1]))

    if terms[4][2] == 'p':
        V = int(terms[4][:2]) + 0.5
    elif terms[4][:2].isdigit():
        V = int(terms[4][:2])
    elif terms[4][:1].isdigit():
        V = int(terms[4][:1])
    Vlist.append(V)

    if os.path.isfile(f):
        run = trial_data(f)
        print('Sums', [run.P_sum2, run.P_sum3])
        print('Means', [run.mean2, run.mean3])
        print('Stdev', [run.stdev2, run.stdev3])
print(Sumlist, Meanlist, Sdevlist, Alphalist, Vlist, Englist)


# # change working directory to csv files location, you will have to edit this path to suit your own OS. Make sure to keep the r'random//path' notation
# os.chdir(r"D:\Aerospace Engineering\Bachelor Year 2\AE2223-I Test Analysis & Simulation\CSV")
#
# # Create list of all files at path location, (Insert same path as used above)
# files = os.listdir(r"D:\Aerospace Engineering\Bachelor Year 2\AE2223-I Test Analysis & Simulation\CSV")
#
#
# class trial_data:
#     """ This class allows convenient access to all data from chosen excel. Methods can be added accordingly. """
#
#     def __init__(self, path):
#         self.df = pd.read_csv(path)
#         self.mic_1 = self.df.loc[:, "m1"]  # voltage values from mic 1
#         self.mic_2 = self.df.loc[:, "m2"]  # voltage values from mic 2
#         self.mic_3 = self.df.loc[:, "m3"]  # voltage values from mic 3
#         self.time_arr = self.df.loc[:, "m1_Time*"]  # series containing all time points
#
#         """ The following function let's you plot voltage vs time (vt). The index parameter
#         let's you decide on which mic to plot. Will probs change since it is useless now"""
#
#     def plot_vt(self, index, th_line=0.03):
#         mic = [self.mic_1, self.mic_2, self.mic_3]
#
#         plt.plot(self.time_arr, mic[index], "-k", linewidth=th_line)
#         plt.ylabel(f"voltage mic {index + 1} [V]")
#         plt.xlabel("Time [s]")
#         plt.show()
#
#
# # example of how to conventienly load a trial in
# run1 = trial_data(files[0])
#
# print(run1.mic_1)