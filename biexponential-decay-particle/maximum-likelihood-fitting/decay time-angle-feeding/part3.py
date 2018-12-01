"""
NumRep CP2.2    :   Negative Log Likelihood(NLL) Minimisation, a python script for finding the best estimation of Tau,
                    by minimising NLL.

Authors: Azid Harun

Date :  19/10/2018

"""

# Import required packages
import numpy as np
import pylab as pl
import sys
from scipy import *
from MinuitPart3 import Minuit
import scipy.integrate as integrate

# Define Negative Log Likelihood function
def nll(fraction, tau1, tau2):
    shape1 = lambda t, theta: (1+math.cos(theta)**2)*(math.exp(-t/tau1))
    shape2 = lambda t, theta: (3*math.sin(theta)**2)*(math.exp(-t/tau2))
    norm1 = integrate.dblquad( shape1, 0.0, 2*np.pi, lambda theta: 0.0, lambda theta: 10.0)[0]
    norm2 = integrate.dblquad( shape2, 0.0, 2*np.pi, lambda theta: 0.0, lambda theta: 10.0)[0]
    pdf1 = fraction*(1+np.cos(theta)**2)*(np.exp(-t/tau1))/norm1
    pdf2 = (1-fraction)*(3*np.sin(theta)**2)*(np.exp(-t/tau2))/norm2
    pdf = pdf1 + pdf2
    return np.sum(-np.log(pdf))

# Create list to store data
nll_list = []

# Read data from input file
t, theta = Minuit.readData(sys.argv[1])

# Define initial straight line parameters, m and c and their range
F_tau1_tau2 = np.array([0.5, 1.0, 2.0])
F_range = (0.0, 1)
tau1_range = (0.0, 5.0)
tau2_range = (0.0, 5.0)
fn_type = 'nll'

# Create a minimiser class
minim = Minuit(0.0, F_range, tau1_range, tau2_range, fn_type)

#====================================MINIMISING PROCESS=====================================

# # Loop the process until difference between previous and next NLL value lower than threshold
diff = None
ini_nll = nll(F_tau1_tau2[0], F_tau1_tau2[1], F_tau1_tau2[2])

while not minim.isFinished(diff):
    # Calculate previous and next NLL value and also their difference.
    m = minim.minimise(nll, F_tau1_tau2)
    F_tau1_tau2 = np.array([m.values['fraction'], m.values['tau1'], m.values['tau2']])
    final_nll = m.fval
    diff = np.abs(ini_nll - final_nll)
    ini_nll = final_nll

#===========================CREATING DATA FOR CALC SIMPLISTIC ERROR============================

# Creating data around minimum chi-squared
F_arr = np.arange(0, 2*F_tau1_tau2[0], 2*F_tau1_tau2[0]/200)
tau1_arr = np.arange(0.2, 2*F_tau1_tau2[1]+0.2, 2*F_tau1_tau2[1]/200)
tau2_arr = np.arange(0.2, 2*F_tau1_tau2[2]+0.2, 2*F_tau1_tau2[2]/200)

for F, tau1, tau2 in zip(F_arr, tau1_arr, tau2_arr):
    nllval = nll(F, tau1, tau2)
    nll_list.append(nllval)

# Calculate simplistic error for parameters
F_error = Minuit.simpleErrorFinder(0.5, nll_list, final_nll, F_tau1_tau2[0], F_arr)
tau1_error = Minuit.simpleErrorFinder(0.5, nll_list, final_nll, F_tau1_tau2[1], tau1_arr)
tau2_error = Minuit.simpleErrorFinder(0.5, nll_list, final_nll, F_tau1_tau2[2], tau2_arr)

#==============================CREATING DATA FOR CALC PROPER ERROR============================

proper = Minuit(0.5, F_range, tau1_range, tau2_range, fn_type)

F_perror = proper.properErrorFinder(nll, 0, F_tau1_tau2)
tau1_perror = proper.properErrorFinder(nll, 1, F_tau1_tau2)
tau2_perror = proper.properErrorFinder(nll, 2, F_tau1_tau2)

# ================================GENERATE AND DISPLAY RESULTS================================

# # Display the result 
print('===============================================================================')
print('Number of Muon Decay Event                   :   {}'.format(len(t)))
print('-------------------------------------------------------------------------------')
print('Best Estimated Fraction                      :   {0:0.4f}'.format(m.values['fraction']))
print('Best Estimated Tau 1                         :   {0:0.4f}'.format(m.values['tau1']))
print('Best Estimated Tau 2                         :   {0:0.4f}'.format(m.values['tau2']))
print('------------------------------SIMPLISTIC ERROR---------------------------------')
print('Simplistic error for F                       :   {0:0.4f}'.format(F_error))
print('Simplistic error for tau1                    :   {0:0.4f}'.format(tau1_error))
print('Simplistic error for tau2                    :   {0:0.4f}'.format(tau2_error))
print('-------------------------------PROPER ERROR------------------------------------')
print('MINUIT error for F                           :   {0:0.4f}'.format(m.errors['fraction']))
print('MINUIT error for tau1                        :   {0:0.4f}'.format(m.errors['tau1']))
print('MINUIT error for tau2                        :   {0:0.4f}\n'.format(m.errors['tau2']))
print('Calculated error for F                       :   {0:0.4f}'.format(F_perror))
print('Calculated error for tau1                    :   {0:0.4f}'.format(tau1_perror))
print('Calculated error for tau2                    :   {0:0.4f}'.format(tau2_perror))
print('-------------------------------------------------------------------------------')

# #=========================================PLOTTING DATA======================================

# #Plot the result
while True:
    type = (input('Around minimum point? (Y/N)'))

    if type == 'Y':
        pl.subplot(3, 1, 1)
        m.draw_profile('fraction')
        pl.xlabel(r'$Fraction\/F\/$')
        pl.ylabel('Negative Log Likelihood')

        pl.subplot(3, 1, 2)
        m.draw_profile('tau1')
        pl.xlabel(r'$First\/lifetime,\/\tau_{1}$')
        pl.ylabel('Negative Log Likelihood')

        pl.subplot(3, 1, 3)
        m.draw_profile('tau2')
        pl.xlabel(r'$Second\/lifetime,\/\tau_{2}$')
        pl.ylabel('Negative Log Likelihood')

        pl.subplots_adjust(hspace=0.6)
        pl.show()
        break

    elif type == 'N':
        pl.subplot(3, 1, 1)
        m.draw_profile('fraction', bound=(0,1))
        pl.plot(m.values['fraction'], m.fval, 'ro')
        pl.xlabel(r'$Fraction\/F\/$')
        pl.ylabel('Negative Log Likelihood')

        pl.subplot(3, 1, 2)
        m.draw_profile('tau1', bound=(0,5))
        pl.plot(m.values['tau1'], m.fval, 'ro')
        pl.xlabel(r'$First\/lifetime,\/\tau_{1}$')
        pl.ylabel('Negative Log Likelihood')

        pl.subplot(3, 1, 3)
        m.draw_profile('tau2', bound=(0,5))
        pl.plot(m.values['tau2'], m.fval, 'ro')
        pl.xlabel(r'$Second\/lifetime,\/\tau_{2}$')
        pl.ylabel('Negative Log Likelihood')

        pl.subplots_adjust(hspace=0.6)
        pl.show()
        break
    
    else:
        pass