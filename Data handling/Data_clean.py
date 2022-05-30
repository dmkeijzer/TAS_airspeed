import csv
from venv import create
import numpy as np
import scipy.fft as scf
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from sklearn.preprocessing import normalize


file_path = r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\clean_data" #put the path to your data here
files = os.listdir(file_path)
os.chdir(file_path)



class trial_data:
    
    """ This class allows convenient access to all data from chosen excel. Methods can be added accordingly. """
    
    def __init__(self, path, transf = True):
        self.df = pd.read_csv(path)
        self.path = path.split("_")
        self.mic_1 = self.df.loc[:, "m1"].values #voltage values from mic 1
        self.mic_2 = self.df.loc[:, "m2"].values #voltage values from mic 2
        self.mic_3 = self.df.loc[:, "m3"].values #voltage values from mic 3
        self.time_arr = self.df.loc[:,"m1_Time*"].values #series containing all time points
        
        if transf:
        #switching to frequency domain
            self.x = scf.rfftfreq(len(self.time_arr), self.time_arr[1])
            self.y2 = np.abs(scf.rfft(self.mic_2))
            self.y3 = np.abs(scf.rfft(self.mic_3))
            #self.x = (self.x < 14000) * self.x
            #self.y2 = (self.x < 14000) * self.y2
            #self.y3 = (self.x < 14000) * self.y3
            
            #cleaning the weird peaks from the data
            
            self.y2 = abs(((self.y2 > 20) * (30 < self.x) * (self.x < 70))-1) * self.y2
            self.y3 = abs(((self.y3 > 20) * (30 < self.x) * (self.x < 70))-1) * self.y3
            self.y2 = (self.y2 < 100) * self.y2
            self.y3 = (self.y3 < 100) * self.y3
            
            radius = 10       
            for i in range(1, int(self.x[-1] / 50)):
                self.y2 = abs(((self.y2 > 20) * (50 * i - radius < self.x) * (self.x < 50 * i + radius))-1) * self.y2
                self.y3 = abs(((self.y3 > 20) * (50 * i - radius < self.x) * (self.x < 50 * i + radius))-1) * self.y3
            self.P_sum2 = np.sum(self.y2)
            self.P_sum3 = np.sum(self.y3)
            self.expect2 = np.dot(self.x, self.y2) / (np.sum(self.y2))
            self.expect3 = np.dot(self.x, self.y3) / (np.sum(self.y3))
            self.stdev2 = np.std(self.y2)
            self.stdev3 = np.std(self.y3)

        #recognizing what the input parameters are for current run

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
        if self.path[-1] == 'movedUAV.csv':
            self.state_uav = 1
        else: 
            self.state_uav = 0
        

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
    """Creates a csv file of some key parameters of all the runs. The amount
    of files can be limited with the limiter parameter"""

    data = np.ones((420003, 1))

    for counter, file in enumerate(os.listdir(file_path) , start=1):
        if counter == limiter: 
            break
            
        
        run = trial_data(file)
        print(run.path)
        
        arr = np.concatenate((run.y2[run.x < 14000], run.y3[run.x < 14000], [run.engine], [run.alpha], [run.v])).reshape(420003,1)
        
        data = np.append(data, arr, axis=1)

        print(np.shape(data))
    
    data = np.delete(data, 0 ,axis=1)
    data = np.array(data)

    #Writing it to csv

    df = pd.DataFrame(data)
    df.to_csv(os.path.realpath(file_location + "\\tensor_file.csv"), index_label= "index")

def create_raw_data_file(file_location, limiter= False):
    slice = 7680
    data = np.ones((1, int(2 * slice + 3)))

    for counter, file in enumerate(os.listdir(file_path) , start=1):
        start = 0

        if counter == limiter: 
            break

        for i in range(99):    
            run = trial_data(file, transf=False)
            arr = np.concatenate((run.mic_2[start:int(start + slice)], run.mic_3[start:int(start + slice)], [run.engine], [run.alpha], [run.v])).reshape(1,-1)
            data = np.append(data, arr, axis=0)
            start += slice
            
            print(run.path)
            print(np.shape(data))
            print(f"slice {start} to {start + slice}\n")
    
    data = np.delete(data, 0 , axis=0)
    data = np.array(data)

    #Writing it to csv

    df = pd.DataFrame(data)
    df.to_csv(os.path.realpath(file_location + "\\tensor_file_raw_slice015.csv"), index_label= "index")


def plot_frequency_domain():
    files.reverse()
    for i in files:
        path = i.split("_")
        print(path[3])
        print(path[2])
        if path[3] == "0alpha" and path[2].lower() == 'engine30':
            print("computing\n")
            run = trial_data(i)
            plt.plot(run.x, run.y2, label = str(run.v) + " [m/s]")
            continue
        print('skipped\n') 
        
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


create_raw_data_file(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data")




