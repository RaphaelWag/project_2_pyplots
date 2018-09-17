#Author: Rapheal Wagner 16.09.2018

#TODO:
#plot cpp rotations
#plot time of both
#compute difference of python and cpp
#plot difference
#add unit tests

import numpy as np
import time
import matplotlib.pyplot as plt

cpp_results = np.loadtxt("runtime_rotations_.txt")
gridpoints = np.array([])
cpp_runtime = np.array([])
cpp_rotations = np.array([])

for k in range(int(len(cpp_results) / 3)):
    gridpoints = np. append(gridpoints, cpp_results[k * 3])
    cpp_runtime = np.append(cpp_runtime, cpp_results[(k * 3) + 1])
    cpp_rotations = np.append(cpp_rotations, cpp_results[(k * 3) + 2])

python_runtime = np.array([])

for n in gridpoints: #number of gridpoints

    N = int(n)

    #Eigenvalues with full matrix

    A = np.zeros(shape=(N,N))

    full_eigenvalues = np.zeros(shape=N)
    full_eigenvectors = np.zeros(shape=(N,N))

    for i in range(N):
        for j in range(N):
            if (i==j):
                A[i][j]=-2

            if ((i==j+1)or(i==j-1)):
                A[i][j] = 1

    start = time.time() #measures time in seconds

    full_eigenvalues, full_eigenvectors = np.linalg.eigh(A)

    end = time.time()

    runtime = end - start

    python_runtime = np.append(python_runtime, runtime)

