# BPSK demodulation
# 2021 Fall - Communications Lab. Project
# Multiagent Communications and Networking Lab.
# Dept. of Electronic and Electrical Engineering
# Ewha Womans University
# team project

import numpy as np
from numpy import cos
from numpy import pi
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import noise

# BPSK Modulation

encoded_data = np.loadtxt('./encoded_data.txt')
A = encoded_data

# parameters
f = 10
time_step = 0.01
num_sample = int(1/time_step)
t = np.arange(0, 1, time_step)
row = 100
col = 100

k = 0
D = noise.dem(A, 0)
AA = np.zeros((10000), dtype = 'float64')
for i in range(0, row):
    for j in range(0, col):
        AA[k] = D[i,j]
        k += 1
print(AA)
k = 0
BB = np.zeros((1000000), dtype = 'float64')
for i in range(0, 10000):
    for j in range(0, 100):
        BB[k] = cos(2*pi*f*t[j]+pi*AA[i])
        k += 1
print(BB)

n = row*col*num_sample

count = 0
for i in range(0, n):
    A[i] = round(A[i], 6)
    BB[i] = round(BB[i], 6)
    if A[i] != BB[i]:
        count += 1
        print(i)

print(count)