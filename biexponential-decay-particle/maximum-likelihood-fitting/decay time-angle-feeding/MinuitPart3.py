"""
MinuitPart3, a class for for minimising the negative log likelihood (NLL) to find the best value of physics parameter
             F, first and second particle lifetime with both decay times and angle distribution of the given datafile.

Authors: Azid Harun

Date :  25/11/2018

"""

# Import required packages
import numpy as np
import iminuit as im
from iminuit import Minuit

class MinuitError(Exception):
    """ An exception class for Minuit """
    pass


class Minuit(object):
    """
    Class for minimising the NLL.
    
    Properties:
    threshold(float)             -   the minimising threshold value
    fraction_bnd(float, tuple)   -   fraction bound
    tau1_bnd(float, tuple)       -   tau1 bound
    tau1_bnd(float, tuple)       -   tau1 bound
    error_size(float)            -   the error of the calculated parameter is 1 unit if the function given increases by this value

    Methods:
    * minimise                   -    minimise the function
    * fix0minimise               -    minimise the function with fixed fraction
    * fix1minimise               -    minimise the function with fixed tau1
    * fix2minimise               -    minimise the function with fixed tau2
    * isFinished                 -    control the minimiser
    * isExceeded                 -    control the proper error finding process
    * errorFinder                -    find the parameter error
    * properErrorFinder          -    calculate the proper error of the parameter
    * simpleErrorFinder          -    calculate the simplistic error of the parameter
    * readData                   -    read the input decay time and angle distributions (t and theta)
    """

#========================================INITIALISER========================================

    def __init__(self, threshold, fraction_range, tau1_range, tau2_range, fn_type):
        self.threshold = threshold
        self.fraction_bnd = fraction_range
        self.tau1_bnd = tau1_range
        self.tau2_bnd = tau2_range
        if fn_type == 'nll':
            self.error_size = 0.5
        elif fn_type == 'chi':
            self.error_size = 1.0
        else:
            raise MinuitError("Invalid function type!")

#=========================================MINIMISER=========================================

    def minimise(self, f, x):
        m = im.Minuit(  f, 
                        fraction=x[0], 
                        tau1=x[1], 
                        tau2=x[2], 
                        limit_fraction = self.fraction_bnd, 
                        limit_tau1 = self.tau1_bnd, 
                        limit_tau2 = self.tau2_bnd,
                        error_fraction = self.error_size,
                        error_tau1 = self.error_size,
                        error_tau2 = self.error_size,
                        errordef = self.error_size,
                        print_level = 0, 
                        pedantic = False
                        )
        m.migrad()
        return m

    def fix0minimise(self, f, x):
        m = im.Minuit(  f, 
                        fraction = x[0], 
                        tau1 = x[1], 
                        tau2 = x[2],
                        fix_fraction = True, 
                        limit_fraction = self.fraction_bnd, 
                        limit_tau1 = self.tau1_bnd, 
                        limit_tau2 = self.tau2_bnd,
                        error_fraction = self.error_size,
                        error_tau1 = self.error_size,
                        error_tau2 = self.error_size,
                        errordef = self.error_size,
                        print_level = 0, 
                        pedantic = False
                        )
        m.migrad()
        return m

    def fix1minimise(self, f, x):
        m = im.Minuit(  f, 
                        fraction = x[0], 
                        tau1 = x[1], 
                        tau2 = x[2],
                        fix_tau1 = True, 
                        limit_fraction = self.fraction_bnd, 
                        limit_tau1 = self.tau1_bnd, 
                        limit_tau2 = self.tau2_bnd,
                        error_fraction = self.error_size,
                        error_tau1 = self.error_size,
                        error_tau2 = self.error_size,
                        errordef = self.error_size,
                        print_level = 0, 
                        pedantic = False
                        )
        m.migrad()
        return m

    def fix2minimise(self, f, x):
        m = im.Minuit(  f, 
                        fraction = x[0], 
                        tau1 = x[1], 
                        tau2 = x[2],
                        fix_tau2 = True, 
                        limit_fraction = self.fraction_bnd, 
                        limit_tau1 = self.tau1_bnd, 
                        limit_tau2 = self.tau2_bnd,
                        error_fraction = self.error_size,
                        error_tau1 = self.error_size,
                        error_tau2 = self.error_size,
                        errordef = self.error_size,
                        print_level = 0, 
                        pedantic = False
                        )
        m.migrad()
        return m

#==================================MINIMISER PROCESS CONTROL================================

    def isFinished(self,diff):
        if diff == None:
            pass

        elif diff > self.threshold:
            pass
            
        else:
            finish = 'Minimisation is finished'
            return finish

    def isExceeded(self,diff):
        if diff == None:
            pass

        elif diff < self.threshold:
            pass
            
        else:
            finish = 'Minimisation is finished'
            return finish

#==================================PARAMETER ERROR CALCULATOR================================
    
    def properErrorFinder(self, nll, idx, F_tau1_tau2):
        ini_nll = nll(F_tau1_tau2[0], F_tau1_tau2[1], F_tau1_tau2[2])
        best = F_tau1_tau2[idx]
        diff = None
        increment = 0
        while not self.isExceeded(diff):
            increment += 1
            delta = 0.000001 * increment
            F_tau1_tau2[idx] += delta
            # Calculate previous and next NLL value and also their difference.
            if idx == 0:
                m = self.fix0minimise(nll, F_tau1_tau2)
            elif idx == 1:
                m = self.fix1minimise(nll, F_tau1_tau2)
            elif idx == 2:
                m = self.fix2minimise(nll, F_tau1_tau2)
            F_tau1_tau2 = np.array([m.values['fraction'], m.values['tau1'], m.values['tau2']])
            final_nll = m.fval
            diff = np.abs(ini_nll - final_nll)
        return np.abs(m.values[idx] - best)

    @staticmethod
    def simpleErrorFinder(level, f_list, f_min, param, param_list):
        f_errline = min(f_list, key=lambda x:abs(x-(f_min+level)))
        index = f_list.index(f_errline)
        param_error =  np.abs(param - param_list[index])
        return param_error

    @staticmethod
    def readData(filename):
        with open(filename, 'r') as f:
            t_list = []
            theta_list = []
            for line in f:
                t, theta = map(float, line.split())
                t_list.append(t)
                theta_list.append(theta)
        return np.array(t_list), np.array(theta_list)

        