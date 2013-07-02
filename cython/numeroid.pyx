# distutils: language = c++
# distutils: libraries = cruncher
# distutils: library_dirs = ../../cruncher/Library
# distutils: include_dirs = ../../cruncher/headers

from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np
cimport cython
cimport numpy as np

ctypedef vector[string] NamesVec

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
    cdef cppclass System:
        System() except +
        void parsefile(const char* fname) except +
        void process() except +
        void report() except +
        void modnames(NamesVec& nms) except +
        long nodescount(string& mod) except +
        void getnodes(string& mod, long size, long* arr) except +
        void pulltemps(string& mod, long size, long* idxs, double* vals) except +
        void pullcaps(string& mod, long size, long* idxs, double* vals) except +

cdef class PySystem:
    cdef System *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self, const char* fname):
        self.thisptr = new System()
        self.thisptr.parsefile(fname)
    def __dealloc__(self):
        del self.thisptr
    def process(self):
        self.thisptr.process()
    def report(self):
        self.thisptr.report()
    def modnames(self):
        cdef NamesVec vect
        self.thisptr.modnames(vect)
        return vect
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    def nodes(self, string mod):
        cdef long size = self.thisptr.nodescount(mod)
        cdef np.ndarray[np.long_t, ndim=1] arr = np.zeros(size, dtype=np.long)
        self.thisptr.getnodes(mod, size, <long*>arr.data)
        return arr
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    def modtemps(self, string mod):
        cdef long size = self.thisptr.nodescount(mod)
        cdef np.ndarray[np.long_t, ndim=1] arridx = np.zeros(size, dtype=np.long)        
        cdef np.ndarray[np.double_t, ndim=1] arrvals = np.zeros(size, dtype=np.double)
        self.thisptr.pulltemps(mod, size, <long*>arridx.data, <double*>arrvals.data)
        return arridx, arrvals
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    def modcaps(self, string mod):
        cdef long size = self.thisptr.nodescount(mod)
        cdef np.ndarray[np.long_t, ndim=1] arridx = np.zeros(size, dtype=np.long)        
        cdef np.ndarray[np.double_t, ndim=1] arrvals = np.zeros(size, dtype=np.double)
        self.thisptr.pullcaps(mod, size, <long*>arridx.data, <double*>arrvals.data)
        return arridx, arrvals
