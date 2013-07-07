'''
Created on Jun 25, 2013

@author: schernikov
'''

import sys, os, numpy, scipy.interpolate as inp

import matplotlib.pyplot as plt

libloc = os.path.join(os.path.dirname(__file__), '..', 'cython/build/lib.linux-x86_64-2.7')
sys.path.append(libloc)
numeroid = __import__('numeroid')

import extern
    
def main():
    sin = setup()
    #showgraphs(sin)
    compare(sin, 'AEPAN', 20006)

def setup():
    buildsite = '/media/biggie/workspace/celeryprojects/buildsite'
    sinfile = os.path.join(buildsite, "testing/Fu24L_Ox15L_760Tank_Sweep/Fu24L_Ox15L_760Tank_Sweep.sin")
    system = numeroid.PySystem(sinfile)
    return system

def showgraphs(sin):
    tp = 'sun'
    nodeset = sin.ext_nodes(tp)
    mods = {}
    for node in nodeset:
        md = mods.get(node.mod, None)
        if md is None:
            md = []
            mods[node.mod] = md
        md.append(node)
    print "\ngot %d '%s' modules with %d nodes"%(len(mods), tp, len(nodeset))
    for mod in mods.keys():
        nset = mods[mod]
        print "  %s[%d]"%(mod, len(nset))
    print
    for mod in mods.keys():
        nset = mods[mod]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('%s'%(mod))
        if len(nset) > 10:
            space = numpy.linspace(sin.start(), sin.stop(), 1000)
            newvals = numpy.zeros_like(space)
            for nd in sorted(nset, key=lambda n: n.num):
                mult, tms, vals = nd.values()
                lin = inp.interp1d(tms, vals)
                newvals += lin(space)*mult
            ax.plot(space/3600.0, newvals, 'b.-', label='aggregate of %d'%(len(nset)))
        else:
            for nd in sorted(nset, key=lambda n: n.num):
                mult, tms, vals = nd.values()
                ax.plot(tms/3600.0, vals, 'b.-', label='%d x%.2f'%(nd.num, mult))
        ax.legend()            
        plt.show()
    
def compare(sin, mod, num):
    loc = '/media/biggie/workspace/celeryprojects/buildsite/testing/Fu24L_Ox15L_760Tank_Sweep/output'
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
    
def cubic(tm, vals, space):
    tckp = inp.splrep(tm, vals, s=None)
    return inp.splev(space, tckp)

if __name__ == '__main__':
    main()