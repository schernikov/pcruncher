'''
Created on Jul 4, 2013

@author: schernikov
'''

import sys, os

projloc = os.path.join(os.path.dirname(__file__), '..', '..', 'flighttools', 'flightstuff', 'src')
sys.path.append(projloc)
flight_pb2 = __import__('flight_pb2')
process = __import__('process.misc')
fbsubst = __import__('fbsubst.data')

def pullnode(loc, mod, num):
    fname = os.path.join(loc, mod, '%d.simnode'%(num))
    with open(fname) as f:
        nd = process.misc.pullFlightData(f.read(), flight_pb2.Node)
        return fbsubst.data.getnoderaw(nd)