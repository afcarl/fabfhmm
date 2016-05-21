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
        
        
configs = [ dict( name="CFAB", layers=2, doShrink=True, isDiag=True, maxIter=3000,
                  initKs=3, convTolerance=5e-7, minPriorAlpha=0.01, maxPriorAlpha=0.03,
                  noShrinkRounds=20, useFirstOrderApproxH=True, doNormalizeW=False,
                  approxVIterNum=5, miniBatchSize=500, stocUpdateForgetRate=0.7 ),
            dict( name="FAB", layers=2, doShrink=True, isDiag=True, maxIter=1000,
                  initKs=3, convTolerance=5e-7, noShrinkRounds=20, 
                  useFirstOrderApproxH=True, doNormalizeW=False,
                  approxVIterNum=0, miniBatchSize=500, stocUpdateForgetRate=0.7 ),
            dict( name="ML", layers=2, doShrink=False, isDiag=True, maxIter=1000, 
                  initKs=3, convTolerance=5e-7, useFirstOrderApproxH=True )
          ]

basedir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join( basedir, 'data/' )

N = 1

#fileTemplate = "3d-2,2,3-{}.csv"
fileTemplate = "3d-2,3-{}.csv"

trainingFiles = []
testFiles = []
for n in xrange(N):
    trainingFiles.append( fileTemplate.format( n+1 ) )
# 1 sequence for testing
for n in xrange( N, N + 1 ):
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

loadSnapshotPath = "approxv.bin"

# repeat certain times
for i in xrange(4):    
    for algconfig in configs[1:2]:    
        #np.random.seed(500000)
        
        startTime = datetime.datetime.now()
        logname = "%s-%d-%02d%02d%02d.log" %( algconfig['name'], i+1, 
                        startTime.hour, startTime.minute, startTime.second )
        tee = Tee( os.path.join( datadir, "log", logname ) )
        
        print "%s iteration %d starts at %02d:%02d:%02d" %( algconfig['name'], 
                        i+1, startTime.hour, startTime.minute, startTime.second )
        
        algconfig['data'] = X
        algconfig['shrinkThres'] = Xlen * 0.5 / algconfig['initKs']
#       algconfig['test_sequence'] = X2
        #algconfig['maxIter'] = 1
        
        if algconfig['name'] == 'CFAB':
            fabfhmm = CollapsedFactorialHiddenMarkovModel(**algconfig)
        else:
            fabfhmm = FactorialHiddenMarkovModel(**algconfig)
        
        if loadSnapshotPath:
            fabfhmm = loadModel(loadSnapshotPath)
            fabfhmm.loadParams(1)
            fabfhmm.noShrinkRounds = 0
            fabfhmm.doShrink = True
            fabfhmm.approxVIterNum = 3
            varDist = fabfhmm.varDist
            if fabfhmm.miniBatchSize == 0:
                varDist, FIC = fabfhmm.fabIterate( X, varDist )
            else:
                varDist, FIC = fabfhmm.fabIterate( fabfhmm.X2, varDist )
            #varDist, FIC = fabfhmm.fabIterate( X )
        else:
            if fabfhmm.miniBatchSize == 0:
                varDist, FIC = fabfhmm.fabIterate( X )
            else:
                varDist, FIC = fabfhmm.fabIterate( fabfhmm.X2, varDist )
        
        fabfhmm.forward_backward( X, varDist )
        
        fabfhmm.dumpAbnormal = False
        # MLEst is set to ignore the state scale factors, 
        # these factors are computed according to X, therefore useless to X2
        varDist2, pll_test = fabfhmm.VStep( X2, None, MLEst=True )
        print "Test sequences FAB log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )
        # first update varDist using ML
        # then reestimate the parameters using ML
        # experiments show the improvement is slight 
        fabfhmm.lastNegKL = -np.Inf
        varDist, pll_training = fabfhmm.VStep( X, varDist, MLEst=True )
        # it's still FAB, since the parameters are the FAB ones
        print "Training sequences FAB log likelihood: %f. Avg: %f" %( pll_training, pll_training/Xlen )        
        # Parameters updated now. Disable shrinkage, 
        # otherwise the shrunk states are not removed from varDist2,
        # and the dimensionality of varDist2 might mismatch Sigma & W, leading to an error
        fabfhmm.iterDoShrink = False
        FIC = fabfhmm.MStep( X, varDist )
        # this is just to calc negKL
        varDist, pll_training = fabfhmm.VStep( X, varDist, MLEst=True )
        print "Training sequences ML log likelihood: %f. Avg: %f" %( pll_training, pll_training/Xlen )        
        varDist2, pll_test = fabfhmm.VStep( X2, varDist2, MLEst=True )
        print "Test sequences ML log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )
        
    #    pll_test = model.calcNegKLDiv( X2, MLEst=False )
    #    print "Test sequences FAB log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )
        
        #hidden_states = model.viterbi_algorithm()
        #hidden_states2 = model.viterbi_algorithm( X, variationalDist )
        
        #print hidden_states
        #print hidden_states2
        endTime = datetime.datetime.now()
        print "%s iteration %d ends at %02d:%02d:%02d" %( algconfig['name'], 
                        i+1, endTime.hour, endTime.minute, endTime.second )
        print "Duration %d seconds\n" %( (endTime - startTime).seconds )
        
        tee.close()

    