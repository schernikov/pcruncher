'''
Created on Jul 5, 2013

@author: schernikov
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt, numpy

def main():
    Axes3D.__doc__
    linearize()

def linearize():
    delta = 2
    zero = 273.15+20
    T = numpy.linspace(zero-delta, zero+delta, 20)
    T1, T2 = numpy.meshgrid(T, T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Z4 = numpy.power(T1, 4)-numpy.power(T2, 4)
    z = numpy.power(zero+1, 4)-numpy.power(zero, 4)
    Z1 = (T1-T2)*z
    Z = (Z4-Z1)/Z4*100
    #ax.plot3D(X, Y, Z, linestyle='', markersize=1, marker='.', color='grey', alpha=0.6)
    #ax.plot3D(T1, T2, Z, '-', color='grey', )
    #ax.plot_surface(T1, T2, Z4, linewidth=0, antialiased=False, alpha=0.5, color='red')
    #ax.plot_surface(T1, T2, Z1, linewidth=0, antialiased=False, alpha=0.5, color='green')
    #ax.plot_surface(T1, T2, Z4, linewidth=0, antialiased=False, alpha=0.5, color='grey')
    ax.plot_surface(T1, T2, Z, linewidth=0, antialiased=False, alpha=0.5, color='grey')
    #ax.plot3D(T1.flatten(), T2.flatten(), Z1.flatten(), '.', color='red', linestyle='', markersize=1, marker='.')
    plt.show()

if __name__ == '__main__':
    main()