'''
Created on Jun 25, 2013

@author: schernikov
'''

import sys, os, numpy, scipy.interpolate as inp

import matplotlib.pyplot as plt

libloc = os.path.join(os.path.dirname(__file__), '..', 'cython/build/lib.linux-x86_64-2.7')
sys.path.append(libloc)
numeroid = __import__('numeroid')

def main():
    sin = setup()
    #showgraphs(sin)
    #showhist(sin)
    report(sin)

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
    
def showhist(sin):
    bins = 200
    xlog = False
    fig = plt.figure()
    ax = fig.add_subplot(311)
    capshist(sin, ax, bins, xlog=xlog)
    ax = fig.add_subplot(312)
    condhist(sin, ax, bins, xlog=xlog)
    ax = fig.add_subplot(313)
    numhist(sin, ax, bins, xlog=xlog)
    plt.show()

def mkbins(mx, bins):
    pw = int(numpy.log10(mx)+1)
    bset = [0]
    step = pw/0.75/bins
    b = [10**p for p in numpy.arange(pw, pw-bins*step, -step)]
    b.reverse()
    bset.extend(b)
    return bset

def capshist(sin, ax, bins, xlog=False):
    allcaps = []
    for md in sin.modnames():
        ndidx, caps = sin.modcaps(md); ndidx
        allcaps.extend(caps)
    ac = numpy.array(allcaps)
    nz = numpy.nonzero(ac)[0]
    lz = len(ac)-len(nz)
    nzac = ac[nz]
    zzac = numpy.zeros(lz)
    mx = ac.max()
    bset = mkbins(mx, bins) if xlog else bins
    rng = [ac.min(), mx]
    ax.hist([nzac, zzac], bset, range=rng, color=['green', 'red'], alpha=0.5, log=True, 
            label=['caps %d'%(len(nzac)), 'zeros %d'%(len(zzac))])
    #ax.hist(nzac, bins, range=rng, facecolor='green', alpha=0.5, log=True, label='caps %d'%(len(nzac)))    
    #ax.hist(zzac, bins, range=rng, facecolor='red', alpha=0.5, log=True, label='zeros %d'%(len(zzac)))
    #ax.plot([0, 0], [0, lz], 'r-', linewidth=5, label='zeros %d'%(lz))
    setax(ax, xlog, 'Capacity values (%d nodes)'%(len(ac)))

def setax(ax, xlog, title):
    ax.set_title(title)
    ax.set_ylabel('counts #')
    bot, top = ax.get_ylim(); bot
    ax.set_ylim(0.5, top)
    if xlog: ax.set_xscale('log')
    ax.legend()

def mkcons(modcons, tp):
    cons = None
    for mod in modcons:
        call = getattr(mod, tp)
        fidx, tidx, cc = call(); fidx, tidx
        if cc is None: continue
        if cons is None:
            cons = cc
        else:
            cons = numpy.append(cons, cc)
    return cons
            
def condhist(sin, ax, bins, xlog=False):
    modcons = sin.modconnects()
    lcons = mkcons(modcons, 'lins')
    rcons = mkcons(modcons, 'rads')
    sbconst = 5.67037321e-08
    abszero = 273.15
    scale = ((abszero+1)**4-(abszero)**4)*sbconst
    print "scale:",scale
    rcons *= scale
    mx = max(lcons.max(), rcons.max())
    bset = mkbins(mx, bins) if xlog else bins
    rng = [min(lcons.min(), rcons.min()), mx]
    ax.hist(lcons, bset, range=rng, facecolor='blue', alpha=0.5, log=True, 
            label='linear %d'%(len(lcons)))
    ax.hist(rcons, bset, range=rng, facecolor='red', alpha=0.5, log=True, 
            label='radiative %d'%(len(rcons)))
    setax(ax, xlog, 'Conductance values (%d connections, rad. scale: %.4f)'%(len(lcons)+len(rcons), scale))
    
def mkname(mname, num):
    return "%s.%d"%(mname, num)

def mktick(ticks, mname, num):
    nm = mkname(mname, num)
    val = ticks.get(nm, None)
    if val is None:
        ticks[nm] = 1
    else:
        ticks[nm] = val + 1

def mknums(modcons, tp):
    ticks = {}
    for mod in modcons:
        call = getattr(mod, tp)
        fidx, tidx, cc = call(); cc
        if cc is None: continue
        for idx in fidx: mktick(ticks, mod.fmod, idx)
        for idx in tidx: mktick(ticks, mod.tmod, idx)
    return numpy.array(ticks.values())

def numhist(sin, ax, bins, xlog=False):
    modcons = sin.modconnects()
    lvals = mknums(modcons, 'lins')
    rvals = mknums(modcons, 'rads')
    mx = max(lvals.max(), rvals.max())
    bset = mkbins(mx, bins) if xlog else bins
    rng = [min(lvals.min(), rvals.min()), mx]
    ax.hist(lvals, bset, range=rng, facecolor='blue', alpha=0.5, log=True, label='linear')
    ax.hist(rvals, bset, range=rng, facecolor='red', alpha=0.5, log=True, label='radiative')
    setax(ax, xlog, 'Number of connections per node')
    
def cubic(tm, vals, space):
    tckp = inp.splrep(tm, vals, s=None)
    return inp.splev(space, tckp)

def repext(sin, tp):
    s = set()
    for mname, num in sin.nodeset(tp):
        s.add(mkname(mname, num))
    return s
        
def report(sin):
    energy = repext(sin, 'energy')
    temper = repext(sin, 'temperature')
    sun = repext(sin, 'sun')
    one = repext(sin, 'one')
    approx = repext(sin, 'approximated')

    collects = {}
    for cap in ['heat', 'arithmetic', 'boundary', 'diffuse']:
        for mname, num in sin.nodeset(cap):
            nm = mkname(mname, num)
            flags = "["
            flags += ("E" if (nm in energy) else " ")
            flags += ("T" if (nm in temper) else " ")
            flags += ("S" if (nm in sun) else " ")
            flags += ("O" if (nm in one) else " ")
            flags += ("X" if (nm in approx) else " ")
            flags += "] %s"%cap

            col = collects.get(flags, None)
            if col is None:
                col = {}
                collects[flags] = col
            md = col.get(mname, None)
            if md is None:
                md = set()
                col[mname] = md
            md.add(num)
    for name, col in collects.items():
        print "%s"%(name)
        for mname, nodes in col.items():
            ns = ''
            mx = 10
            for n in sorted(nodes)[:mx]:
                ns += ' %d'%n
            if len(nodes) > mx:
                ns += ' ...'
            print "      %s[%d]%s"%(mname, len(nodes), ns)

if __name__ == '__main__':
    main()