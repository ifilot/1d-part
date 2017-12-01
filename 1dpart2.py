#!/usr/bin/env python

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
r = numpy.linspace(delta_r / 2, r_part - delta_r / 2, num=N)

# construct matrices
A = numpy.zeros([N,N])
for i in range(1,N-1):
	A[i,i] = 1 + 2 * beta
	A[i,i-1] = - beta * (1 - delta_r / r[i])
	A[i,i+1] = - beta * (1 + delta_r / r[i])

# these follow from the boundary conditions
A[0,0] = 1 + 2 * beta
A[0,1] = - 2 * beta
A[N-1,N-1] = 1 + 2 * beta
A[N-1,N-2] = - beta * (1 - delta_r / r[i])

# construct production matrices
c_prev = numpy.ones([N,]) * c_0

# populate solution vectors
for i in range(0,1000):
	c_prod = -k * c_prev * delta_t
	c_prod[N-1] += beta * (1 + delta_r / r_part)
	c_new = numpy.linalg.solve(A,c_prev + c_prod)
	if numpy.linalg.norm(c_prev - c_new,2) < 1e-6:
		print "Convergence reached after %i steps" % i
		break
	c_prev = c_new

# construct analytical solution
c_a = numpy.zeros([N,])
r_a = numpy.linspace(0,1,N)
for i in range(1,N-1):
	c_a[i] = c_s * (r_part / r_a[i]) * math.sinh(r_a[i]*math.sqrt(k/D)) / math.sinh(r_part * math.sqrt(k/D))
c_a[N-1] = c_s
c_a[0] = math.sqrt(k/D) / math.sinh(r_part * math.sqrt(k/D)) * r_part # here we have to use a limit

# plot results
plt.plot(r,c_new,'--o',label='Numerical')
plt.plot(r_a,c_a,'--',label='Analytical')
#plt.axis((0,1,0,1))
plt.legend(loc=4)
plt.show()
