import csv
import matplotlib.pyplot as plt
import numpy as np

data = []

file = input("File Path: ")

with open(f'./{file}.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([float(x) for x in row])

def plot_curve(arr_list, legend_list, color_list, ylabel, title=None):
    # set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    # PLEASE NOTE: Change the labels for different plots
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Episodes")
    ax.set_title(title)

    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err = 1.96 * arr_err
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3, color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.legend(handles=h_list)
    plt.show()
    
label = None
split_string = file.split("_", 1)
prefix = split_string[0]
if prefix == "ql":
    label = "Q-Learn"
elif prefix == "rand":
    label = "Random Policy"
elif prefix == "dqn":
    label = "Deep Q-Network"
else:
    raise NotImplementedError
title_plot = f"{label}"
if len(split_string) > 1:
    title_plot += " " + split_string[1]
plot_curve([np.array(data)], [label], ["Red"], "Rewards", title_plot)