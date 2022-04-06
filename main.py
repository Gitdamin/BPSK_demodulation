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


encoded_data = np.loadtxt('./encoded_data.txt')

# parameters
f = 10
time_step = 0.01
num_sample = int(1/time_step)
# t = np.arange(0, 1-time_step, time_step)
t = np.arange(0, 1, time_step)
row = 100
col = 100
# print(t)


# Q1 : Show BPSK modulated signal where phase changes are observed.

n = row*col*num_sample

for Ax in range(0, row*col):
    if encoded_data[Ax*100] != encoded_data[Ax*100 + 100]:
        break

plt.xlim(Ax*100, Ax*100 +200)
plt.plot(encoded_data)
plt.xlabel("time (0.01*sec)")
plt.ylabel("voltage")
plt.title("plot A")
plt.show()


# Q2 : Show BPSK modulated signal where phase changes are observed.

B = np.array(encoded_data)
j = 0

for i in range(0, n):

    index = 2*pi*f*t[j]+pi
    B[i] = encoded_data[i]*cos(index)

    if j % num_sample == (num_sample-1):
        j = 0
    else:
        j += 1

print(B)
plt.xlim(Ax*100, Ax*100 + 200)
plt.plot(B)
plt.xlabel("time (0.01*sec)")
plt.ylabel("voltage")
plt.title("plot B")
plt.show()

# Q3 : Q2 * cos func.

T = num_sample
nn = int(n/T)
C = np.zeros(nn, dtype = 'float64')

k = 0
for i in range(0, n):
    C[k] += B[i]
    if i % T == (T-1):
        k += 1
C = C/100
C_r = np.reshape(C,(row,col))
plt.imshow(C_r, cmap = cm.gray)
plt.imshow(C_r, cmap = 'gray', vmin = 0, vmax = 1)
plt.title("image C")
plt.show()
print(C)


# Q4 : thresholding equation

def thres(serial_matrix):
    threshold = (min(serial_matrix) + max(serial_matrix))/2

    return threshold

thres = thres(C)
print(thres)
print(min(C))
print(max(C))


# Q5 : Reshape the signal

D = np.zeros((row,col), dtype = 'int32')
for i in range(0, row):
    for j in range(0, col):
        if C_r[i, j] > thres:
            D[i, j] = 1
        elif C_r[i, j] < thres:
            D[i, j] = 0

plt.imshow(D, cmap = cm.gray)
plt.imshow(D, cmap = 'gray', vmin = 0, vmax = 1)
plt.title("image D")
plt.show()
print(D)


# Q6 : Result

# Letter = E


# BPSK Modulation

k = 0
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

A = encoded_data
n = row*col*num_sample

count = 0
for i in range(0, n):
    A[i] = round(A[i], 6)
    BB[i] = round(BB[i], 6)
    if A[i] != BB[i]:
        count += 1
        print(i)

print(count)
