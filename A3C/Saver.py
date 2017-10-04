
import matplotlib.pyplot as plt
import numpy as np

import parameters

RESULTS = []


def add_results(value):
    RESULTS.append(value)
    if len(RESULTS) % 200 == 0 and parameters.DISPLAY:
        disp()


def disp():
    plt.plot(RESULTS)
    x = [np.mean(RESULTS[max(i - 50, 1):i]) for i in range(2, len(RESULTS))]
    plt.plot(x)
    if parameters.DISPLAY:
        plt.show(block=False)
    else:
        plt.savefig("results/Reward.png")
