"""
MyPDF, a class for generating random events with decay time and angle distributions according to the PDF described in the report.

Authors: Azid Harun

Date :  25/11/2018

"""

import math
import numpy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys

class PDFError(Exception):
    """ An exception class for MyPDF """
    pass

#===============================================
# 2D pdf
class MyPDF:
    """
    Class for generating random events with decay time and angle distributions according to the PDF described in the report.
    
    Properties:
    lifetime1(float)       -  particle first lifetime
    lifetime2(float)       -  particle second lifetime

    t_lolimit(float)       -  lower limit of interval for decay time
    t_hilimit(float)       -  higher limit of interval for decay time

    theta_lolimit(float)   -  lower limit of interval for decay angle
    theta_hilimit(float)   -  higher limit of interval for decay angle

    shape1(function)       - PDF of first decay component, PDF1
    shape2(function)       - PDF of second decay component, PDF2

    max1(float)            - maximum value of PDF1
    max2(float)            - maximum value of PDF2

    fraction(float)        - fraction of PDF1 being the total PDF of the decay
    
    Methods:
    * maxVal               - return the maximum value of the total PDF of the decay
    * normalise            - normalise the given pdf
    * evaluate             - evaluate the normalised pdf at give decay time and angle 
    * next                 - draw N random number from distribution
    * drawSample           - draw a random sample of N events from a pdf using box method
    * plotShape            - plot histograms of the decay time and angle distributions of the generated data
    * writeData            - write out decay times and decay angles generated
    """
    
     # Constructor
    def __init__(self, t_lolim, t_hilim, theta_lolim, theta_hilim, lifetime1, lifetime2, fraction):
        self.lifetime1 = lifetime1
        self.lifetime2 = lifetime2
        self.t_lolimit = t_lolim
        self.t_hilimit = t_hilim
        self.theta_lolimit = theta_lolim
        self.theta_hilimit = theta_hilim
        self.shape1 = lambda t, theta: (1+math.cos(theta)**2)*(math.exp(-t/lifetime1))
        self.shape2 = lambda t, theta: (3*math.sin(theta)**2)*(math.exp(-t/lifetime2))
        self.max1 = self.shape1(t_lolim, theta_lolim)
        self.max2 = self.shape2(t_lolim, theta_lolim)
        self.fraction = fraction
  
    # Return the maximum value of the total PDF of the decay
    def maxVal( self ) :
        return self.fraction * self.max1 + (1-self.fraction) * self.max2
    
    def normalise( self, pdf ) :
        if pdf == 1:
            return integrate.dblquad( self.shape1, self.theta_lolimit, self.theta_hilimit, lambda theta: self.t_lolimit, lambda theta: self.t_hilimit)[0]
        elif pdf == 2:
            return integrate.dblquad( self.shape2, self.theta_lolimit, self.theta_hilimit, lambda theta: self.t_lolimit, lambda theta: self.t_hilimit)[0]
        else:
            raise PDFError('Invalid PDF')
 
    # Evaluate method (normalised)
    def evaluate( self, t, theta, norm1, norm2, pdf_type):
        pdf1 = self.fraction * (self.shape1(t, theta) / norm1)
        pdf2 = (1-self.fraction) * (self.shape2(t, theta) / norm2)
        if pdf_type == 'all':
            return pdf1 + pdf2
        elif pdf_type == '1':
            return pdf1
        elif pdf_type == '2':
            return pdf2
        else:
            raise PDFError('Invalid PDF type')

    # Draw N random number from distribution
    def next(self, nevents):
        data  = self.drawSample(self, self.t_lolimit, self.t_hilimit, self.theta_lolimit, self.theta_hilimit, nevents)
        return data

    @staticmethod
    # To draw a random sample of N events from a pdf using box method
    def drawSample(self, t_lolim, t_hilim, theta_lolim, theta_hilim, nevents):
        times = []
        thetas = []
        for i in range(nevents):
            ythrow = 1.
            yval=0.
            norm1, norm2 = self.normalise(1), self.normalise(2)
            while ythrow > yval:
                tthrow = numpy.random.uniform(t_lolim, t_hilim)
                thetathrow = numpy.random.uniform(theta_lolim, theta_hilim)
                ythrow = self.maxVal() * numpy.random.uniform()
                yval =  self.evaluate(tthrow, thetathrow, norm1, norm2, 'all')
            times.append(tthrow)
            thetas.append(thetathrow)
        return (times, thetas)

    @staticmethod
    # function to plot histograms of the decay time and angle distributions 
    def plotShape(data, t_lolim, t_hilim, theta_lolim, theta_hilim, nbins ):
        plt.subplot(2, 1, 1)
        plt.hist(data[0], bins=nbins, range=[t_lolim, t_hilim])
        plt.xlim(0, 10)
        plt.ylabel(r'$No.\/of\/entries$')
        plt.xlabel(r'$Decay\/Time,\/t\//\mu s$')
        plt.title(r'$t\/Distribution$', fontsize='x-large')

        plt.subplot(2, 1, 2)
        data_deg = data[1]
        data_deg = [ i/math.pi*180 for i in data_deg ]
        plt.hist(data_deg, bins=nbins, range=[theta_lolim/math.pi*180, theta_hilim/math.pi*180])
        plt.xlim(0, 360)
        plt.xticks(numpy.arange(0, 360, 90))
        plt.ylabel(r'$No.\/of\/entries$')
        plt.xlabel(r'$Decay\/Angle,\/\theta\//\degree$')
        plt.title(r'$\theta\/Distribution$', fontsize='x-large')

        # Display the subplots
        plt.subplots_adjust(hspace=0.6)
        plt.show()

    @staticmethod
    # function to write out decay times and decay angle generated
    def writeData(data, filename):
        with open(filename, 'a') as f:
            for time, angle in zip(data[0], data[1]):
                f.write('{0:0.16f} {1:0.16f}\n'.format(time, angle))

#===============================================
# Main code to generate and plot a single experiment

def singleToy( nevents):

    t_lolim, t_hilim            = 0., 10.
    theta_lolim, theta_hilim    = 0, 2 * math.pi
    lifetime1, lifetime2        = 1.0, 2.0
    fraction                    = 1.0       # Choose 0.0 / 0.5 / 1.0
 
    # Create the pdf
    pdf = MyPDF( t_lolim, t_hilim, theta_lolim, theta_hilim, lifetime1, lifetime2, fraction)
    norm1, norm2 = pdf.normalise(1), pdf.normalise(2)

    # Generate a single experiment
    data = pdf.next( nevents)

    # Write decay data in an output textfile
    # pdf.writeData(data, sys.argv[1])

    # Plot function and data
    pdf.plotShape( data, t_lolim, t_hilim, theta_lolim, theta_hilim, 100 )

#===============================================
#Main

def main():
    #Perform a single toy
    singleToy(10000)

main()