'''
Created on Jul 5, 2013

@author: schernikov
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt, numpy

import extern
#import cruncher

def main():
    Axes3D.__doc__
    #linearize()
    #sin = cruncher.setup()
    #compare(loc, sin, 'AEPAN', 20006)
    loc = '/media/biggie/workspace/celeryprojects/buildsite/testing/Fu24L_Ox15L_760Tank_Sweep/output'
    shownode(loc, 'SEPTAE', 1)

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

def compare(loc, sin, mod, num):
    otms, ovals = extern.pullnode(loc, mod, num)
    nodes = sin.ext_nodes('sun')
    for nd in nodes:
        if nd.mod == mod and nd.num == num:
            break
    else:
        raise Exception("can not find '%s.%d'"%(mod, num))
    mult, tms, vals = nd.values(); mult
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(otms/3600.0, ovals, 'b.-', label='temps')
    ax.plot(tms/3600.0, vals, 'r.-', label='power')
    ax.legend()
    plt.show()

def shownode(loc, mod, num):
    otms, ovals = extern.pullnode(loc, mod, num)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(otms/3600.0, ovals, 'b.-', label='temps')
    ax.legend()
    plt.show()
    
if __name__ == '__main__':
    main()