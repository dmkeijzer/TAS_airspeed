import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.path.join(os.path.dirname(os.path.dirname(__file__)),  "data_sets\plotting_data.csv")
df = pd.read_csv(path)

def alpha_error_plot(ms= 5, col= "red"):
    fmt = "x"

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3)
    ax1 = plt.subplot(221)
    plt.plot(df["alpha_18_regr"], df["err_18_regr"], fmt, color=col, markersize= ms)
    plt.ylabel("Absolute error [m/s]")
    plt.title("Model 18 dataset x")
    plt.ylim([0,6])
    ax2 = plt.subplot(222)
    plt.plot(df["true_alpha"], df["err_1"],fmt, color= col,  markersize= ms)
    plt.title("Model 1 dataset x")
    plt.ylim([0,6])
    ax3 = plt.subplot(223)
    plt.plot(df["true_alpha"], df["err_18"], fmt, color= col,  markersize= ms)
    plt.ylabel("Absolute error [m/s]")
    plt.xlabel("Angle of Attack [deg]")
    plt.title("Model 18 dataset x")
    plt.ylim([0,6])
    ax4 = plt.subplot(224)
    plt.plot(df["true_alpha"], df["err_19"], fmt, color= col,  markersize= ms)
    plt.xlabel("Angle of Attack [deg]")
    plt.title("Model 19 dataset x")
    plt.ylim([0,6])
    plt.show()


def v_error_plot(ms= 5, col= "red"):
    fmt = "x"

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3)
    ax1 = plt.subplot(221)
    plt.plot(df["v_18_regr"], df["err_18_regr"], fmt, color=col, markersize= ms)
    plt.ylabel("Absolute error [m/s]")
    plt.title("Model 18 dataset x")
    ax2 = plt.subplot(222)
    plt.plot(df["true_v"], df["err_1"],fmt, color= col,  markersize= ms)
    plt.title("Model 1 dataset x")
    ax3 = plt.subplot(223)
    plt.plot(df["true_v"], df["err_18"], fmt, color= col,  markersize= ms)
    plt.ylabel("Absolute error [m/s]")
    plt.xlabel("Airspeed [m/s]")
    plt.title("Model 18 dataset x")
    ax4 = plt.subplot(224)
    plt.plot(df["true_v"], df["err_19"], fmt, color= col,  markersize= ms)
    plt.xlabel("Airspeed [m/s]")
    plt.title("Model 19 dataset x")
    plt.show()

alpha_error_plot()







