#!/usr/bin/env python2

import numpy
import matplotlib.pyplot as plt
import math

# define variables
r_part = 1
delta_t = 0.01
delta_r = 0.01
D = 1
c_0 = 0
c_s = 1
k = 1.0

# compute auxiliary variables
N = int(r_part / delta_r) + 1
beta = D * delta_t / delta_r**2

# construct arrays
r = numpy.linspace(0, r_part, num=N)
ae = numpy.zeros([N,])
aw = numpy.zeros([N,])
ap = numpy.zeros([N,])
alpha = numpy.zeros([N,])
A = numpy.zeros([N,N])
for i in range(1,N-1):
	ae[i] = beta * r[i+1]**2 / r[i]**2
	aw[i] = beta * r[i-1]**2 / r[i]**2
	ap[i] = 1 + ae[i] + aw[i]

# construct matrices
for i in range(1,N-1):
	A[i,i] = (1 + ae[i] + aw[i])
	A[i,i-1] = -ae[i]
	A[i,i+1] = -aw[i]

A[0,0] = 1
A[N-1,N-1] = 1

# construct production matrices
c_prev = numpy.zeros([N,]) * c_0
c_prev[N-1] = c_s

# populate solution vectors
for i in range(0,1000):
	c_prod = k * c_prev * delta_t
	c_prod[N-1] = 0
	c_new = numpy.linalg.solve(A,c_prev + c_prod)
	c_prev = c_new

# construct analytical solution
c_a = numpy.zeros([N,])
for i in range(1,N-1):
	c_a[i] = c_s * (r[i] / r_part) * math.sinh(r[i]*math.sqrt(k/D)) / math.sinh(r_part * math.sqrt(k/D))
c_a[N-1] = c_s

# plot results
plt.plot(r,c_new)
plt.plot(r,c_a)
plt.axis((0,1,0,1))
plt.show()