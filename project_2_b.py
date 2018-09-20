#Author: Rapheal Wagner 16.09.2018

#TODO:
#plot cpp rotations
#plot time of both
#compute difference of python and cpp
#plot difference

import numpy as np
import time
import matplotlib.pyplot as plt

cpp_results = np.loadtxt("runtime_rotations_.txt")
gridpoints = np.array([],dtype=int)
cpp_runtime = np.array([])
cpp_rotations = np.array([])

for k in range(int(len(cpp_results) / 3)):
    gridpoints = np. append(gridpoints, int(cpp_results[k * 3]))
    cpp_runtime = np.append(cpp_runtime, cpp_results[(k * 3) + 1])
    cpp_rotations = np.append(cpp_rotations, cpp_results[(k * 3) + 2])

for n in gridpoints:

    N = int(n)  #number of gridpoints

    #initialize arrays

    A = np.zeros(shape=(N,N))

    eigenvalues = np.zeros(shape=N)
    eigenvectors = np.zeros(shape=(N, N))

    #set up Matrix

    for i in range(N):
        for j in range(N):
            if (i==j):
                A[i][j]=-2

            if ((i==j+1)or(i==j-1)):
                A[i][j] = 1


    eigenvalues, eigenvectors = np.linalg.eigh(A) #solve eigenvalue problem

    end = time.time()

##################################
### Read in data from cpp code ###
#################################

readin = np.array([],dtype=str)

for i in range(len(gridpoints)):
    readin = np.append(readin,"eigenvalues"+str(gridpoints[i])+".txt")

cpp_eigenvalues = np.zeros(shape=(int(len(gridpoints)),gridpoints[-1]))

for i in range(int(len(gridpoints))):
    temporary = np.loadtxt(readin[i])
    for k in range(len(temporary)):
        cpp_eigenvalues[i][k] = temporary[k]

    #cpp_eigenvalues[i].sort()

##################################################
### print runtime from ccp and python solution ###
##################################################

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(np.log10(gridpoints),cpp_runtime, "blue")
ax1.set_ylabel("log10(runtime)")
ax1.set_xlabel("log10(gridpoints)")
ax1.set_title("cpp runtime")

ax2.plot(np.log10(gridpoints), np.log10(cpp_rotations), "blue")
ax2.set_ylabel("log10(rotations)")
ax2.set_xlabel("log10(gridpoints)")
ax2.set_title("cpp rotations")

plt.show()