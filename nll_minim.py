"""
NumRep CP2.2    :   Negative Log Likelihood(NLL) Minimisation, a python script for finding the best estimation of Tau,
                    by minimising NLL.

Authors: Azid Harun

Date :  19/10/2018

"""

# Import required packages
import numpy as np
import math as m
import pylab as pl
import sys
from scipy import *
from Minimiser import Minimiser

# Define Negative Log Likelihood function
def nll(tau):
    pdf = 1/tau*exp(-t/tau)
    return np.sum(-np.log(pdf))

# Create list to store data
nll_list = []

# Load data from input file
t = np.loadtxt(sys.argv[1])

# Define initial tau and its range
tau = np.array([2.0])
tau_bnds = (1.0, 3.0)

# Create a minimiser class
minim = Minimiser(0.0, tau_bnds)

#====================================MINIMISING PROCESS=====================================

# Loop the process until difference between previous and next NLL value lower than threshold
diff = None
while not minim.isFinished(diff):

    # Calculate previous and next NLL value and also their difference.
    ini_nll = nll(tau)
    tau = minim.minimise(nll, tau)
    final_nll = nll(tau)
    diff = np.abs(ini_nll - final_nll)

#================================CREATING DATA FOR PLOTTING==================================

# Creating data around minimum NLL
tau_arr = np.arange(0.5, 2*tau + 1.2, 2*tau/200)
tau_arr = np.delete(tau_arr, 0)

for tau_val in tau_arr:
    nllval = nll(tau_val)
    nll_list.append(nllval)

#================================GENERATE AND DISPLAY RESULTS================================

# Calculate error for tau 
tau_error = minim.errorFinder(0.5, nll_list, final_nll, tau, tau_arr)

# Display the result 
print('-------------------------------------------------------------------------------')
print('Number of Muon Decay Event       :   {}'.format(len(t)))
print('Best Estimated Tau +- err(Tau)   :   {} +- {}'.format(tau[0], tau_error[0]))
print('-------------------------------------------------------------------------------')

#=========================================PLOTTING DATA======================================

#Plot the result
pl.title('Negative Log Likelihood Distribution', fontsize='x-large')
pl.plot(np.array(tau_arr), np.array(nll_list), 'b-', markersize=2.0)
pl.plot(tau, final_nll, 'ro')
pl.xlabel(r'$Tau\/(Model\/Parameter),\/\tau$')
pl.ylabel('Negative Log Likelihood')
pl.show()

"""
Comments:

- As the number of events increases, the error become smaller and the estimated lifetime get closer to the true value.

"""