import csv
import numpy as np
import scipy.fft as scf
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

file_path = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\clean_data"
files = os.listdir(file_path)
os.chdir(file_path)



class trial_data:
    
    """ This class allows convenient access to all data from chosen excel. Methods can be added accordingly. """
    
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.path = path.split("_")
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

        if self.path[2] == 'noEngine':
            self.engine = 0
        elif self.path[2] == 'noPropeller':
            self.engine = -1
        else:
            self.engine = int(self.path[2][6:8])

        try:
            self.alpha = int(self.path[3][:2])
        except:
            self.alpha = int(self.path[3][:1])

        if self.path[4][2] == 'p':
            self.v = int(self.path[4][:2]) + 0.5
        elif self.path[4][:2].isdigit():
            self.v = int(self.path[4][:2])
        elif self.path[4][:1].isdigit():
            self.v = int(self.path[4][:1])
        

        """ The following function let's you plot voltage vs time (vt). The index parameter
        let's you decide on which mic to plot. Will probs change since it is useless now"""
    def plot_vt(self, index , th_line = 0.6):
        mic = [self.mic_1, self.mic_2, self.mic_3]

        plt.plot(self.time_arr[0:10000], mic[index][0:10000], "-k", linewidth = th_line)
        plt.title(f"{self.path}") 
        plt.ylabel(f"voltage mic {index +1} [V]") 
        plt.xlabel("Time [s]")
        plt.show()
    

def create_data_file(file_location, limiter = False): 

    data = []

    for counter, file in enumerate(os.listdir(file_path)):
        if counter == limiter: 
            break

        run = trial_data(file)
        print(run.path)
        data.append([run.engine, run.alpha, run.v, run.P_sum2, run.P_sum3, run.mean2, run.mean3, run.stdev2, run.stdev3 ])

        print('Sums', [run.P_sum2, run.P_sum3])
        print('Means', [run.mean2, run.mean3])
        print('Stdev', [run.stdev2, run.stdev3])

    data = np.array(data)
    df = pd.DataFrame(data, columns= ["engine", "alpha", "v", "sum2", "sum3", "mean2", "mean3","stdev2", "stdev3"])
    df.to_csv(file_location + "\data_list.csv")

create_data_file(file_location=r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\TAS_airspeed\Data handling", limiter= 5)




