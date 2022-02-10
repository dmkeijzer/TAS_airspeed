import pandas as pd
import matplotlib.pyplot as plt
import os

#change working directory to csv files location, you will have to edit this path to suit your own OS. Make sure to keep the r'random//path' notation
os.chdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\microphone_data")

#Create list of all files at path location, (Insert same path as used above)
files = os.listdir(r"C:\Users\damie\OneDrive\Desktop\Damien\TAS\data\microphone_data")



class trial_data:
    
    """ This class allows convenient access to all data from chosen excel. Methods can be added accordingly. """
    
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.mic_1 = self.df.loc[:, "m1"] #voltage values from mic 1
        self.mic_2 = self.df.loc[:, "m2"] #voltage values from mic 2
        self.mic_3 = self.df.loc[:, "m3"] #voltage values from mic 3
        self.time_arr = self.df.loc[:,"m1_Time*"] #series containing all time points


        """ The following function let's you plot voltage vs time (vt). The index parameter
        let's you decide on which mic to plot. Will probs change since it is useless now"""
    def plot_vt(self, index , th_line = 0.03):
        mic = [self.mic_1, self.mic_2, self.mic_3]

        plt.plot(self.time_arr, mic[index], "-k", linewidth = th_line) 
        plt.ylabel(f"voltage mic {index +1} [V]") 
        plt.xlabel("Time [s]")
        plt.show()

#example of how to conventienly load a trial in
run1 = trial_data(files[0])

print(run1.mic_1)




