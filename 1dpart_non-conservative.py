#!/usr/bin/env python

#
# Purpose:
# Calculate 1D reaction-diffusion system over a spherical particle
# 
# Author: Ivo Filot <i.a.w.filot@tue.nl>
#

# This file is licensed under a
# Creative Commons Attribution 4.0 International License.
#
# You should have received a copy of the license along with this
# work. If not, see <http://creativecommons.org/licenses/by/4.0/>.

import numpy
import matplotlib.pyplot as plt
import math

# define variables
r_part = 1			# particle radius
delta_t = 0.01		# time step
delta_r = 0.002		# length increment
D = 1				# diffusion constant
c_0 = 0				# starting concentration
c_s = 1				# surface concentration
k = 1.0				# reaction rate constant

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
	c_prod[N-1] += beta * (1 + (delta_r/2.0) / r_part)
	c_new = numpy.linalg.solve(A,c_prev + c_prod)
	if numpy.linalg.norm(c_prev - c_new,2) < 1e-4:
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
