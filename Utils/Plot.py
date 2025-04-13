import csv
import matplotlib.pyplot as plt
import numpy as np

data = []

file = input("File Path: ")

with open(f'./{file}.csv', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append([int(x) for x in row])

def plot_curve(arr_list, legend_list, color_list, ylabel):
    # set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    # PLEASE NOTE: Change the labels for different plots
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Episodes")

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

if file == "ql":
    label = "Q-Learn"
elif file == "rand":
    label = "Random Policy"
elif file == "dqn":
    label = "Deep Q-Network"
else:
    raise NotImplementedError
plot_curve([np.array(data)], [label], ["Red"], "Rewards")