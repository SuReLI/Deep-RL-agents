
import matplotlib.pyplot as plt
import numpy as np

RESULTS = []

def add_results(value):
	RESULTS.append(value)
	if len(RESULTS) % 200 == 0:
		disp()

def disp():
	plt.plot(RESULTS)
	x = [np.mean(RESULTS[max(i-50, 1):i]) for i in range(2, len(RESULTS))]
	plt.plot(x)
	plt.show(block=False)