# distutils: language = c++
# distutils: libraries = cruncher
# distutils: library_dirs = ../../cruncher/Library
# distutils: include_dirs = ../../cruncher/headers
# distutils: depends = ../../cruncher/Library/libcruncher.a

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair

import numpy as np
cimport cython
cimport numpy as np

ctypedef vector[string] NamesVec
ctypedef pair[string, long] NodePair
ctypedef vector[NodePair] NodesVec

cdef extern:
    int _import_array()
    int _import_umath()

def _dummy():
    "exists only to get rid of C++ compile warnings"
    cdef int tmp = 0
    if tmp:
        _import_umath()    
        _import_array()

cdef extern from "system.h":
    ctypedef double* (*DAllocator)(size_t size, void* f)
    ctypedef long* (*LAllocator)(size_t size, void* f)

    cdef cppclass ModConnects:
        string fr
        string to
        long lins
        long rads

    cdef cppclass WeightedNode:
        string nm
        long num
        double weight

    ctypedef vector[ModConnects] ModsVector
    ctypedef vector[WeightedNode] WeightedNodesVector

    cdef cppclass System:
        double start
        double stop
        
        System() except +
        void parsefile(const char* fname) except +
        void process() except +
        void report() except +
        void modnames(NamesVec& nms) except +
        void getnodes(string& mod, LAllocator alloc, void* f) except +
        void pulltemps(string& mod, LAllocator la, void* l, DAllocator da, void* d) except +
        void pullcaps(string& mod, LAllocator la, void* l, DAllocator da, void* d) except +
        void arraynodes(string& tp, NodesVec& nodes) except +
        double pullarray(string& tp, string& mod, long num, DAllocator alloc, void* f) except +
        void setnodes(string& tp, NodesVec& nodes)
        void modconns(ModsVector& vect)
        void consvals(string& tp, string& fmod, string& tmod, size_t size, LAllocator la, DAllocator da, void* f)
        void appox(string& mod, long num, WeightedNodesVector& vec)

@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef double* valscallback(size_t size, void *f):
    vals = (<object>f)
    cdef np.ndarray[np.double_t, ndim=1] arr = np.zeros(size, dtype=np.double)
    vals.append(arr)
    return <double*>arr.data

@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef long* numscallback(size_t size, void *f):
    vals = (<object>f)
    cdef np.ndarray[np.long_t, ndim=1] arr = np.zeros(size, dtype=np.long)
    vals.append(arr)
    return <long*>arr.data

class NodeBase(object):
    def __init__(self, mod, num):
        self.mod = mod
        self.num = num

class ArrayNode(NodeBase):
    def __init__(self, mod, num, onvals):
        super(ArrayNode, self).__init__(mod, num)
        self._onvalues = onvals
    def values(self):
        "Returns values for this node"
        return self._onvalues(self.mod, self.num)

class ConnVals(object):
    def __init__(self, nm, c):
        self.name = nm
        self.count = c
        self.vals = None
        if not self.count: self.vals = (None, None, None)
    def values(self, fmod, tmod, sstm):
        if self.vals: return self.vals
        self.vals = sstm._onconvals(self.name, fmod, tmod, self.count)
        return self.vals

class ModsConn(object):
    def __init__(self, parent, fr, to, l, r):
        self._parent = parent
        self.fmod = fr
        self.tmod = to
        self._lins = ConnVals('lins', l)
        self._rads = ConnVals('rads',r)
    def __repr__(self, *args, **kwargs):
        return '%s->%s lins:%d rads:%d'%(self.fmod, self.tmod, self._lins.count, self._rads.count)
    def lins(self):
        return self._lins.values(self.fmod, self.tmod, self._parent)
    def rads(self):
        return self._rads.values(self.fmod, self.tmod, self._parent)

class ApproxNode(NodeBase):
    def __init__(self, nm, num, refs):
        super(ApproxNode, self).__init__(nm, num)
        self.refs = tuple(refs)

cdef pullvalues(string& tp, System* sptr, string& mod, long num):
    vals = []
    mult = sptr.pullarray(tp, mod, num, valscallback, <void*>vals)
    return mult, vals[0], vals[1]
    
cdef class PySystem:
    cdef System *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, const char* fname):
        self.thisptr = new System()
        self.thisptr.parsefile(fname)
        self.thisptr.process()
    def __dealloc__(self):
        del self.thisptr
    def report(self):
        self.thisptr.report()
    def modnames(self):
        cdef NamesVec vect
        self.thisptr.modnames(vect)
        return vect
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    def modnodes(self, string mod):
        """Retrieves array of nodes for given module.
           
        Args:
            mod: module name
        Returns:
            numpy array of long type
        """
        vals = []
        self.thisptr.getnodes(mod, numscallback, <void*>vals)
        return vals[0] if vals else []
    
    def modconnects(self):
        cdef ModsVector vect
        self.thisptr.modconns(vect)
        stats = []
        for v in vect:
            stats.append(ModsConn(self, v.fr, v.to, v.lins, v.rads))
        return stats        
        
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    def nodeset(self, string& tp):
        """Retrieves set of nodes for requested type.
           
        Args:
            tp: request type. Available types: 
              based on capacity:
                 heat         - infinite capacity
                 arithmetic   - zero capacity
                 boundary     - infinite capacity
                 diffuse      - non-zero capacity
              based on references:
                 energy       - energy supplied from outside
                 temperature  - temperature is predefined
                 sun          - same as 'energy' but the sun is source of it
                 one          - one way conductor, lists receiving nodes
                 approximated - approximated from other nodes
                 undefined    - node is mentioned in connections, but type is not defined  
        Returns:
            list of tuples. 
        Raises:
            Exception: if tp is not valid. 
        """
        cdef NodesVec vect
        self.thisptr.setnodes(tp, vect)
        return vect
        
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    def modtemps(self, string& mod):
        """Default temperatures for given module.
           
        Args:
            mod: module name
        Returns:
            two tuple of numpy arrays: node indexes, default temperatures  
        Raises:
            Exception: if mod is not valid. 
        """
        return self._modvals(mod, True)
    
    def modcaps(self, string& mod):
        """Default capacities for given module.
           
        Args:
            mod: module name
        Returns:
            two tuple of numpy arrays: node indexes, default capacities  
        Raises:
            Exception: if mod is not valid. 
        """
        return self._modvals(mod, False)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    def _modvals(self, string& mod, temps):
        idxs = []
        vals = []
        if temps:
            self.thisptr.pulltemps(mod, numscallback, <void*>idxs, valscallback, <void*>vals)
        else:
            self.thisptr.pullcaps(mod, numscallback, <void*>idxs, valscallback, <void*>vals)
        if idxs and vals and len(idxs[0]) == len(vals[0]):
            return idxs[0], vals[0]
        if not idxs and not vals:
            return [], []
        raise Exception("No matching arrays in modtemps")

    def _onconvals(self, nm, fmod, tmod, size):
        vals = []
        self.thisptr.consvals(nm, fmod, tmod, size, numscallback, valscallback, <void*>vals)
        return tuple(vals)
    
    def ext_nodes(self, string tp):
        cdef NodesVec vect
        self.thisptr.arraynodes(tp, vect)
        def onvals(mod, num):
            return pullvalues(tp, self.thisptr, mod, num)
        nodes = []
        for mod, num in vect:
            nodes.append(ArrayNode(mod, num, onvals))
        return nodes
    def start(self):
        return self.thisptr.start
    def stop(self):
        return self.thisptr.stop
    
    def approximated(self):
        nset = self.nodeset('approximated')
        nodes = []
        for mname, num in nset:
            refs = self._pullapprox(mname, num)
            nodes.append(ApproxNode(mname, num, refs))
        return nodes

    def _pullapprox(self, mname, num):
        cdef WeightedNodesVector vect
        self.thisptr.appox(mname, num, vect)
        refs = []
        for v in vect:
            refs.append((v.nm, v.num, v.weight))
        return refs

