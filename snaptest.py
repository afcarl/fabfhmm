from fabfhmm import FactorialHiddenMarkovModel, protect, saveModel, loadModel
from cfabfhmm import CollapsedFactorialHiddenMarkovModel
import numpy as np
import os
import sys
import datetime

class Tee(object):
    def __init__(self, name):
        self.filename = name
        self.file = open(name, "wb", buffering=0)
        print "Logging into %s" %(name)
        self.stdout = sys.stdout
        sys.stdout = self
    def close(self):
        sys.stdout = self.stdout
        self.file.close()
        print "End of logging into %s" %(self.filename)
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)


configs = [ dict( name="CFAB", layers=3, doPrune=True, doShrink=True, maxIter=300,
                  initKs=10, convTolerance=5e-7, minPriorAlpha=0.5, maxPriorAlpha=0.5,
                  noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20  ),
            dict( name="FAB", layers=3, doPrune=True, doShrink=True, maxIter=300,
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20 ),
            dict( name="ML", layers=3, doPrune=True, doShrink=False, maxIter=300,
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20 ),

            dict( name="CFAB", layers=3, doPrune=True, doShrink=True, maxIter=600,
                  initKs=10, convTolerance=5e-7, minPriorAlpha=0.5, maxPriorAlpha=0.5,
                  noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=False  ),
            dict( name="FAB", layers=3, doPrune=True, doShrink=True, maxIter=600,
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=False ),
            dict( name="ML", layers=3, doPrune=True, doShrink=False, maxIter=600,
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=False )
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
    Xn = np.array( Xn ) #Xn[0:500] )
    protect(Xn)
    X2.append( Xn )
    X2len += len(Xn)

if len(sys.argv) != 2:
    raise getopt.GetoptError("")

snapshotPath = sys.argv[1]
if snapshotPath.startswith("CFAB"):
    algconfig = configs[0]
if snapshotPath.startswith("FAB"):
    algconfig = configs[1]
if snapshotPath.startswith("ML"):
    algconfig = configs[2]

algconfig['data'] = X

if algconfig['name'] == 'CFAB':
    fabfhmm = CollapsedFactorialHiddenMarkovModel(**algconfig)
else:
    fabfhmm = FactorialHiddenMarkovModel(**algconfig)

fabfhmm = loadModel(snapshotPath)
#fabfhmm.loadParams(1)

fabfhmm.useZeroOrderApproxH  = True
fabfhmm.useFirstOrderApproxH = True
fabfhmm.iterUseZeroOrderApproxH  = True
fabfhmm.iterUseFirstOrderApproxH = True
fabfhmm.iterDoShrink = False

pll_test = np.zeros(30)
print "Test sequence V-step 1"
varDist2, kls_test = fabfhmm.VStep( X2, None )
partialFIC2 = fabfhmm.calcPartialFIC(X2)
pll_test[0] = kls_test + partialFIC2

for j in xrange(1, 30):
    print "Test sequence V-step %d" %(j+1)
    varDist2, kls_test = fabfhmm.VStep( X2, varDist2 )
    partialFIC2 = fabfhmm.calcPartialFIC(X2)
    pll_test[j] = kls_test + partialFIC2
    
best_pll_test = np.max(pll_test)
print "Test sequences log likelihood: %f. Avg: %f" %( best_pll_test, best_pll_test/X2len )
