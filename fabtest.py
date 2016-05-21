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
        
        
configs = [ dict( name="CFAB", layers=3, doShrink=True, maxIter=300,
                  initKs=10, convTolerance=5e-7, minPriorAlpha=0.5, maxPriorAlpha=0.5,
                  noshrinkRounds=0, useFirstOrderApproxH=False, 
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20  ),
            dict( name="FAB", layers=3, doShrink=True, maxIter=300,
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20 ),
            dict( name="ML", layers=3, doShrink=False, maxIter=300, 
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=True, baseShrinkRate=20 ),
                  
            dict( name="CFAB", layers=3, doShrink=True, maxIter=600,
                  initKs=10, convTolerance=5e-7, minPriorAlpha=0.5, maxPriorAlpha=0.5,
                  noshrinkRounds=0, useFirstOrderApproxH=False, 
                  approxVIterNum=0, VStepFirst=True, varShrink=False  ),
            dict( name="FAB", layers=3, doShrink=True, maxIter=600,
                  initKs=10, convTolerance=5e-7, noshrinkRounds=0, useFirstOrderApproxH=False,
                  approxVIterNum=0, VStepFirst=True, varShrink=False ),
            dict( name="ML", layers=3, doShrink=False, maxIter=600, 
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

loadSnapshotPath = None #"approxv.bin"

# repeat certain times
for i in xrange(5):
    for idx, algconfig in enumerate(configs):
        if i == 0 and idx < 2:
            continue
                
        #np.random.seed(500000)
        
        if 'varShrink' not in algconfig or not algconfig['varShrink']:
            algconfig['baseShrinkRate'] = 1
            
        startTime = datetime.datetime.now()
        logname = "%s-%d-%d-%02d%02d%02d.log" %( algconfig['name'], algconfig['baseShrinkRate'], i+1, 
                        startTime.hour, startTime.minute, startTime.second )
        tee = Tee( os.path.join( datadir, "log", logname ) )
        
        print "%s iteration %d starts at %02d:%02d:%02d" %( algconfig['name'], 
                        i+1, startTime.hour, startTime.minute, startTime.second )
        
        algconfig['data'] = X
        
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
            varDist, FIC = fabfhmm.fabIterate( X, varDist )
            #varDist, FIC = fabfhmm.fabIterate( X )
        else:
            varDist, FIC = fabfhmm.fabIterate( X )
        
        #fabfhmm.forward_backward( X, varDist )
        
        modelDumpname = "%s-%d-%d.bin" %( algconfig['name'], algconfig['baseShrinkRate'], i + 1 )
        saveModel(fabfhmm, modelDumpname)
        
        fabfhmm.lastNegKL = -np.Inf
        fabfhmm.useZeroOrderApproxH  = True
        fabfhmm.useFirstOrderApproxH = True
        fabfhmm.iterUseZeroOrderApproxH  = True
        fabfhmm.iterUseFirstOrderApproxH = True
        fabfhmm.iterDoShrink = algconfig['doShrink']

        partialFIC = fabfhmm.calcPartialFIC(X)
        varDist, kl_training = fabfhmm.VStep( X, varDist )
        pll_training = partialFIC + kl_training
        print "Training sequences log likelihood: %f. Avg: %f\n" %( pll_training, pll_training/Xlen ) 
        
        partialFIC2 = fabfhmm.calcPartialFIC(X2)
        kls_test = np.zeros(30)
        print "Test sequence V-step 1"
        varDist2, kls_test[0] = fabfhmm.VStep( X2, None )
        for j in xrange(1, 30):
            print "Test sequence V-step %d" %(j+1)
            varDist2, kls_test[j] = fabfhmm.VStep( X2, varDist2 )
        
        pll_test = np.max(kls_test) + partialFIC2
        
        print "Test sequences log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )
               
        #=======================================================================
        # MLEst is set to ignore the state scale factors, 
        # these factors are computed according to X, therefore useless to X2
        # first update varDist using ML
        # then reestimate the parameters using ML
        # experiments show the improvement is slight 
        # # Parameters updated now. Disable shrinkage, 
        # # otherwise the shrunk states are not removed from varDist2,
        # # and the dimensionality of varDist2 might mismatch Sigma & W, leading to an error
        # fabfhmm.iterDoShrink = False
        # partialFIC = fabfhmm.MStep( X, varDist )
        # # this is just to calc negKL
        # varDist, pll_training = fabfhmm.VStep( X, varDist, MLEst=True )
        # pll_training += partialFIC
        # print "Training sequences ML log likelihood: %f. Avg: %f" %( pll_training, pll_training/Xlen )        
        # varDist2, pll_test = fabfhmm.VStep( X2, varDist2, MLEst=True )
        # print "Test sequences ML log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )
        #=======================================================================
        
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

    