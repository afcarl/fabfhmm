from fabfhmm import FactorialHiddenMarkovModel, protect, saveModel, loadModel
from cfabfhmm import CollapsedFactorialHiddenMarkovModel
import numpy as np
import os
import sys
import datetime
        
configs = [ dict( name="CFAB", layers=3, doPrune=True, doShrink=True, maxIter=6,
                  initKs=10, convTolerance=5e-7, minPriorAlpha=0.5, maxPriorAlpha=0.5,
                  noshrinkRounds=0, useFirstOrderApproxH=False, 
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20  ),
            dict( name="FAB", layers=3, doPrune=True, doShrink=True, maxIter=6,
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20 ),
            dict( name="ML", layers=3, doPrune=True, doShrink=False, maxIter=6, 
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20 ),
          ]

basedir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join( basedir, 'data/' )

N = 3

fileTemplate = "3d-2,2,3-{}.csv"
#fileTemplate = "3d-2,3-{}.csv"

trainingFiles = []
testFiles = []
for n in xrange(N):
    trainingFiles.append( fileTemplate.format( n+1 ) )
# one sequence for test
for n in xrange( N, N + 2 ):
    testFiles.append( fileTemplate.format( n+1 ) )

X = []
Xlen = 0

shorten = False

for datafile in (trainingFiles):
    datafile = os.path.join(datadir, datafile)
    fp = open(datafile)
    Xn = []
    for line in fp:
        fields = [ float(field) for field in line[:-1].split(',') ]
        Xn.append(fields)
    fp.close()
    
    if shorten:
        Xn = np.array( Xn[0:1000] )
    else:
        Xn = np.array( Xn )
    
    protect(Xn)
    X.append( Xn )
    Xlen += len(Xn)
    #X.append( np.array( Xn ) )   

X2 = []
X2len = 0

for datafile in (testFiles):
    datafile = os.path.join(datadir, datafile)
    fp = open(datafile)
    Xn = []
    for line in fp:
        items = [float(item) for item in line[:-1].split(',')]
        Xn.append(items)
    fp.close()
    Xn = np.array( Xn )
    protect(Xn)
    X2.append( Xn )
    X2len += len(Xn)

algconfig = configs[0]

if 'varShrink' not in algconfig or not algconfig['varShrink']:
    algconfig['baseShrinkRate'] = 1
    
algconfig['data'] = X

if algconfig['name'] == 'CFAB':
    fabfhmm = CollapsedFactorialHiddenMarkovModel(**algconfig)
else:
    fabfhmm = FactorialHiddenMarkovModel(**algconfig)

varDist, FIC = fabfhmm.fabIterate( X )
saveModel(fabfhmm, "test.bin")
fabfhmm = loadModel("test.bin")

fabfhmm.useZeroOrderApproxH  = True
fabfhmm.useFirstOrderApproxH = False
fabfhmm.iterUseZeroOrderApproxH  = True
fabfhmm.iterUseFirstOrderApproxH = False
fabfhmm.iterDoShrink = True

print

partialFIC2 = fabfhmm.calcPartialFIC(X2)
kls_test = np.zeros(10)
print "Test sequence V-step 1"
varDist2, kls_test[0] = fabfhmm.VStep( X2, None )
for j in xrange(1, 10):
    print "Test sequence V-step %d" %(j+1)
    varDist2, kls_test[j] = fabfhmm.VStep( X2, varDist2 )
    
pll_test = np.max(kls_test) + partialFIC2

print "Test sequences log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )

fabfhmm.useZeroOrderApproxH  = True
fabfhmm.useFirstOrderApproxH = True
fabfhmm.iterUseZeroOrderApproxH  = True
fabfhmm.iterUseFirstOrderApproxH = True
fabfhmm.iterDoShrink = True

print

partialFIC2 = fabfhmm.calcPartialFIC(X2)
kls_test = np.zeros(10)
print "Test sequence V-step 1"
varDist2, kls_test[0] = fabfhmm.VStep( X2, None )
for j in xrange(1, 10):
    print "Test sequence V-step %d" %(j+1)
    varDist2, kls_test[j] = fabfhmm.VStep( X2, varDist2 )
    
pll_test = np.max(kls_test) + partialFIC2

print "Test sequences log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )

