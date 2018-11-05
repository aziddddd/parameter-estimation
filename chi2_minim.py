"""
Chi-Squared Minimisation, a python script for finding the best estimation of straight line parameter, m and c 
by minimising chi-squared.

Authors: Azid Harun

Date :  19/10/2018

"""

# Import required packages
import sys
import pylab as pl
import numpy as np
from Minimiser import Minimiser

# Define Chi-Squared function
def chi(param):
    m = param[0]
    c = param[1]
    model = m * x + c
    return sum((y - model/ y_err)**2)

# Create lists to store data
x_list, y_list, err_list, chi_list = [], [], [], []

# Read data from input file
with open(sys.argv[1], 'r') as f:
    content = []
    for line in f:
        x, y, err = map(float, line.split())
        x_list.append(x)
        y_list.append(y)
        err_list.append(err)

# Change the data from list to array
x = np.array(x_list)
y = np.array(y_list)
y_err = np.array(err_list)

# Define initial straight line parameters, m and c and their range
m_c = np.array([0, 0])
m_range = (-1.0, 0.0)
c_range = (0.0, 1.0)
bnds = (m_range, c_range)

# Create a minimiser class
minim = Minimiser(0.0, bnds)

#====================================MINIMISING PROCESS=====================================

# Loop the process until difference between previous and next chi-squared value lower than threshold
diff = None
while not minim.isFinished(diff):

    # Calculate previous and next chi-squared value and also their difference.
    ini_chi = chi(m_c)
    m_c = minim.minimise(chi, m_c)
    final_chi = chi(m_c)
    diff = np.abs(ini_chi - final_chi)

#================================CREATING DATA FOR PLOTTING==================================

# Creating data around minimum chi-squared
m_arr = np.arange(0.0, 2*m_c[0], 2*m_c[0]/200)
c_arr = np.arange(0.0, 2*m_c[1], 2*m_c[1]/200)

for m,c in zip(m_arr, c_arr):
    chival = chi((m, c))
    chi_list.append(chival)

#================================GENERATE AND DISPLAY RESULTS================================

# Calculate error for the straight line parameters, m and c
m_error = minim.errorFinder(1.0, chi_list, final_chi, m_c[0], m_arr)
c_error = minim.errorFinder(1.0, chi_list, final_chi, m_c[1], c_arr)

# Display the result 
print('Minimum Chi-Squared : {}'.format(final_chi))
print('Best estimated fit gradient, m +- err(m) : {} +- {}'.format(m_c[0], m_error))
print('Best estimated y-intercept, y +- err(y) : {} +- {}'.format(m_c[1], c_error))

#=========================================PLOTTING DATA======================================

#Plot the result
pl.subplot(1, 2, 1)
pl.title('Chi-Squared Distribution', fontsize='x-large')
pl.plot(np.array(m_arr), np.array(chi_list), 'b-', markersize=2.0)
pl.plot(m_c[0], final_chi, 'ro')
pl.ylabel(r'$Chi-Squared,\/\chi^{2}$')
pl.xlabel('Model Parameter, m')

pl.subplot(1, 2, 2)
pl.title('Chi-Squared Distribution', fontsize='x-large')
pl.plot(np.array(c_arr), np.array(chi_list), 'g-', markersize=2.0)
pl.plot(m_c[1], final_chi, 'ro')
pl.ylabel(r'$Chi-Squared,\/\chi^{2}$')
pl.xlabel('Model Parameter, c')
pl.show()
