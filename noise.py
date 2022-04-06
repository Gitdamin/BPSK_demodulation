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


# import encoded_data(modulated data)
encoded_data = np.loadtxt('./encoded_data.txt')

# parameters
f = 10
time_step = 0.01
num_sample = int(1/time_step)
t = np.arange(0, 1, time_step) # 수정
row = 100
col = 100


# Q7 

A = encoded_data
n = row*col*num_sample

# define demodulation func.
def dem(AA, sigma):

    B = np.zeros(n, dtype = 'float64')
    j = 0
    for i in range(0, n):

        index = 2*pi*f*t[j]+pi
        result = AA[i]*cos(index)
        B[i] = result

        if j % num_sample == (num_sample-1):
            j = 0
        else:
            j += 1
    # print(B)
    T = num_sample
    nn = int(n/T)
    C = np.zeros(nn, dtype = 'float64')

    k = 0
    for i in range(0, n):
        C[k] += B[i]
        if i % T == (T-1):
            k += 1
    C = C/T
    C_r = np.reshape(C,(row,col))
    thres = 0
    D = np.zeros((row,col), dtype = 'int32')
    for i in range(0, row):
        for j in range(0, col):
            if C_r[i, j] > thres:
                D[i, j] = 1
            elif C_r[i, j] < thres:
                D[i, j] = 0

    return D


# main

# noisy matrix
AA = np.zeros(n, dtype = 'float64')
sigma = 4
noise = np.random.normal(0, sigma, n)

for i in range(0, n):
    AA[i] = A[i] + noise[i]

D_7 = dem(AA, sigma)
plt.imshow(D_7, cmap=cm.gray)
plt.title("image E with sigma = "+ str(sigma))
plt.show()


# Q8

# noiseless matrix
# to compare
F = dem(A, 0)

# repeat 10 times
repeat = 10
error = np.zeros((repeat, 10), dtype = 'int')
for q in range(0, repeat):
    for i in range(1, repeat+1):
        
        noise = np.random.normal(0, i, n)
        for j in range(0, n):
           AA[j] = A[j] + noise[j]
        # reshape matrix
        E = dem(AA, i)

        # num of bit error
        for j in range(0, row):
            for k in range(0, col):
                if F[j, k] != E[j, k]:
                    error[q, i-1] += 1


avg_e = np.zeros(10, dtype = 'int')
for k in range(0, 10):
    for j in range(0, repeat):
        avg_e[k] += error[j, k]

avg_e = avg_e/(repeat*row*col)
print(avg_e)
sigma = np.arange(1, 11) # 1 ~ 10
x = np.zeros(10, dtype = 'float')
for i in range(0, 10):
    x[i] = 1/sigma[i]

y = avg_e
plt.scatter(x, y)
plt.plot(x, y)
plt.xscale("log")
plt.yscale("log")
plt.title("plot BER as a func. of log10(1/sigma)")
plt.xlabel('log10(1/sigma)')
plt.ylabel('BER (log)')
plt.show()

