#Author: Rapheal Wagner 16.09.2018

#TODO:
#plot cpp rotations
#plot time of both
#compute difference of python and cpp
#plot difference

import numpy as np
import time
import matplotlib.pyplot as plt
#######################
###Initialize Arrays###
#######################

cpp_results = np.loadtxt("runtime_rotations_.txt")
gridpoints = np.array([],dtype=int)
cpp_runtime = np.array([])
cpp_rotations = np.array([])

for k in range(int(len(cpp_results) / 3)):
    gridpoints = np. append(gridpoints, int(cpp_results[k * 3]))
    cpp_runtime = np.append(cpp_runtime, cpp_results[(k * 3) + 1])
    cpp_rotations = np.append(cpp_rotations, cpp_results[(k * 3) + 2])

python_eigenvalues = np.zeros(shape=(len(gridpoints),gridpoints[-1]))

##################################################
### solve problem in python to compare results ###
##################################################

ii = 0 #loop variable

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

    for j in range(N):
        python_eigenvalues[ii][j] = eigenvalues[j]
    ii += 1

##################################
### Read in data from cpp code ###
#################################

readin = np.array([],dtype=str)

for i in range(len(gridpoints)):
    readin = np.append(readin,"eigenvalues"+str(gridpoints[i])+".txt")

cpp_eigenvalues = np.zeros(shape=(int(len(gridpoints)),gridpoints[-1]))
cpp_eigenvalues_sort = np.zeros(shape=(int(len(gridpoints)),gridpoints[-1]))

for i in range(int(len(gridpoints))):
    temporary = np.loadtxt(readin[i])
    for k in range(len(temporary)):
        cpp_eigenvalues[i][k] = temporary[k]

    cpp_eigenvalues_sort[i] = cpp_eigenvalues[i]
    cpp_eigenvalues_sort[i].sort()

##############################
### print runtime from ccp ###
##############################

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(np.log10(gridpoints),cpp_runtime, "blue")
ax1.set_ylabel("log10( runtime / nanoseconds )")
ax1.set_xlabel("log10(gridpoints)")
ax1.set_title("cpp runtime")

ax2.plot(np.log10(gridpoints), np.log10(cpp_rotations), "blue")
ax2.set_ylabel("log10(rotations)")
ax2.set_xlabel("log10(gridpoints)")
ax2.set_title("cpp rotations")

plt.show()

###########################################################
### compare python and cpp results with analytic results###
###########################################################

difference_cpp_pyhton_array = np.zeros(shape=(len(gridpoints)))
difference_cpp_python_matrix = np.zeros(shape=(len(gridpoints), gridpoints[-1]))

difference_cpp_analytic_array = np.zeros(shape=(len(gridpoints)))
difference_cpp_analytic_matrix = np.zeros(shape=(len(gridpoints), gridpoints[-1]))

difference_python_analytic_array = np.zeros(shape=(len(gridpoints)))
difference_python_analytic_matrix = np.zeros(shape=(len(gridpoints), gridpoints[-1]))



difference_cpp_python_matrix = abs(python_eigenvalues - cpp_eigenvalues_sort)

for i in range(len(gridpoints)):

    analytic_eigenvalues = np.zeros(shape=(gridpoints[i]))

    for j in range(gridpoints[i]):
        analytic_eigenvalues[j] = -2.0 + 2 * np.cos((j + 1) * np.pi / (gridpoints[i] + 1))

    analytic_eigenvalues.sort()

    for j in range(gridpoints[i]):

        difference_cpp_python_matrix[i][j] = difference_cpp_python_matrix[i][j] / abs(python_eigenvalues[i][j])
        difference_cpp_analytic_matrix[i][j] = abs((cpp_eigenvalues_sort[i][j]-analytic_eigenvalues[j])/(analytic_eigenvalues[j]))
        difference_python_analytic_matrix[i][j] = abs((python_eigenvalues[i][j]-analytic_eigenvalues[j])/(analytic_eigenvalues[j]))

    difference_cpp_pyhton_array[i] = max(difference_cpp_python_matrix[i])
    difference_cpp_analytic_array[i] = max(difference_cpp_analytic_matrix[i])
    difference_python_analytic_array[i] = max(difference_python_analytic_matrix[i])

f, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(np.log10(gridpoints),np.log10(difference_python_analytic_array), "blue")
ax1.set_ylabel("log10(difference)")
ax1.set_xlabel("log10(gridpoints)")
ax1.set_title("python-analytic")

ax2.plot(np.log10(gridpoints), np.log10(difference_cpp_analytic_array), "blue")
ax2.set_ylabel("log10(difference)")
ax2.set_xlabel("log10(gridpoints)")
ax2.set_title("cpp-analytic")

plt.show()