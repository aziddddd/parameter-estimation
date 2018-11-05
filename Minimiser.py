"""
NumRep CP2: Minimiser, a class for for minimising a function of a parameter to find the best value of parameter

Authors: Azid Harun

Date :  05/10/2018

"""

# Import required packages
import numpy as np
import iminuit as im
from iminuit import Minuit

class Minimiser(object):
    """
    Class for minimising a function.
    
    Properties:
    threshold(float)        -   the minimising threshold value
    bound(float, tuple)     -   parameter bound
    
    Methods:
    * minimise         -    minimise the function
    * isFinished       -    control the minimiser
    * errorFinder      -    find the parameter error
    """

#========================================INITIALISER========================================

    def __init__(self, threshold, bound):
        self.threshold = threshold
        self.bound = bound

#=========================================MINIMISER=========================================

    def minimise(self, f, x):
        new_param = im.minimize(f, x)
        # new_param = im.minimize(f, x, bounds=self.bound)
        return new_param.x

#==================================MINIMISER PROCESS CONTROL================================

    def isFinished(self,diff):
        if diff == None:
            pass

        elif diff > self.threshold:
            pass
            
        else:
            finish = 'Minimisation is finished'
            return finish

#==================================PARAMETER ERROR CALCULATOR================================

    @staticmethod
    def errorFinder(level, f_list, f_min, param, param_list):
        f_errline = min(f_list, key=lambda x:abs(x-(f_min+level)))
        index = f_list.index(f_errline)
        param_error =  np.abs(param - param_list[index])
        return param_error
