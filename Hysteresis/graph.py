# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

hdg = []
speed = []
stall_limit = []

def add(i, v, s_v):
	hdg.append(i)
	speed.append(v)
	stall_limit.append(s_v)

def disp():
	plt.plot(hdg)
	plt.plot(speed)
	plt.plot(stall_limit)
	plt.show()
