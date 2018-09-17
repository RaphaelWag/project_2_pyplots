#Author: Rapheal Wagner 16.09.2018

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

cpp_results = np.loadtxt("runtime_rotations_.txt")
gridpoints = np.array([])
cpp_runtime = np.array([])
cpp_rotations = np.array([])

for k in range(int(len(cpp_results) / 3)):
    gridpoints = np. append(gridpoints, cpp_results[k * 3])
    cpp_runtime = np.append(cpp_runtime, cpp_results[(k * 3) + 1])
    cpp_rotations = np.append(cpp_rotations, cpp_results[(k * 3) + 2])

print(gridpoints, cpp_runtime, cpp_rotations)

for N in gridpoints: #number of gridpoints

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

    full_eigenvalues, full_eigenvectors = np.linalg.eigh(A)

    print(full_eigenvalues,full_eigenvectors)

