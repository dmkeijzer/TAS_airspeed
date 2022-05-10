import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  "data_sets\plotting_data.csv")
df = pd.read_csv(path)
xtick = [0,15, 30,45 ,60, 75, 90]


def alpha_error_plot_ABC(ms= 5, col= "red"):
    fmt = "x"

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.54)
    fig.supylabel("Absolute error [m/s]")
    fig.supxlabel("Angle of Attack [deg]")
    ax2 = plt.subplot(131)
    plt.plot(df["true_alpha"], df["err_1"],fmt, color= col,  markersize= ms)
    plt.ylim([0,6])
    plt.xticks(xtick)
    plt.title("Model A")

    ax3 = plt.subplot(132)
    plt.plot(df["true_alpha"], df["err_18"], fmt, color= col,  markersize= ms)
    plt.title("Model B")
    plt.xticks(xtick)
    plt.ylim([0,6])

    ax4 = plt.subplot(133)
    plt.plot(df["true_alpha"], df["err_19"], fmt, color= col,  markersize= ms)
    plt.title("Model C")
    plt.xticks(xtick)
    plt.ylim([0,6])

    plt.show()


def v_error_plot_ABC(ms= 5, col= "red"):
    fmt = "x"

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3)
    
    ax2 = plt.subplot(131)
    plt.plot(df["true_v"], df["err_1"],fmt, color= col,  markersize= ms)
    plt.title("Model A")
    plt.ylabel("Absolute error [m/s]")
    plt.xlabel("Airspeed [m/s]")
    plt.ylim([0,6])
    ax3 = plt.subplot(132)
    plt.plot(df["true_v"], df["err_18"], fmt, color= col,  markersize= ms)
    plt.ylabel("Absolute error [m/s]")
    plt.xlabel("Airspeed [m/s]")
    plt.title("Model B")
    plt.ylim([0,6])
    ax4 = plt.subplot(133)
    plt.plot(df["true_v"], df["err_19"], fmt, color= col,  markersize= ms)
    plt.xlabel("Airspeed [m/s]")
    plt.ylabel("Absolute error [m/s]")
    plt.title("Model C")
    plt.ylim([0,6])

    plt.show()

    
def regress_data_set_plot(ms= 5, col= "red"):
    fmt = "x"

    fig = plt.figure()    
    ax = plt.subplot(121)
    plt.plot(df["alpha_18_regr"], df["err_18_regr"], fmt, color=col, markersize= ms)
    plt.ylabel("Absolute error [m/s]")
    plt.title("Model 18 dataset x")
    plt.ylim([0,7])
    ax1 = plt.subplot(122)
    plt.plot(df["v_18_regr"], df["err_18_regr"], fmt, color=col, markersize= ms)
    plt.ylabel("Absolute error [m/s]")
    plt.title("Model 18 dataset x")
    plt.ylim([0,7])

    plt.show()


alpha_error_plot_ABC()

