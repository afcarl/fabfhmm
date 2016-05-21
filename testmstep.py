from fabfhmm import FactorialHiddenMarkovModel
from fabfhmm import VariationalDist
import numpy as np
import os
import re
import sys

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
        
def readArrayLine(fp, delim=", ", dtype=float):
    line = fp.readline()
    nums = line.split(", ")
    nums = [ dtype(n) for n in nums ]
    return nums

def loadGroundZ(stateFiles, M, T, Ks):
    varDist = VariationalDist( M, len(stateFiles), T, Ks )
    
    for n, stateFile in enumerate(stateFiles):
        stateFile_path = os.path.join(datadir, stateFile)
        fp = open(stateFile_path)
        t = 0

        sum_Z = [ np.zeros(K) for K in Ks ]
    
        # initialized arbitrarily to satisfy the interpreter
        qZmnt = [0] * M
        old_qZmnt = [0] * M
        
        for line in fp:
            Znt = [int(item) for item in line[:-1].split(',')]
            
            for m in range(M):
                qZmnt[m] = np.zeros( Ks[m] )
                qZmnt[m][ Znt[m] ] = 1
                sum_Z[m][ Znt[m] ] += 1
                varDist.qZ[m][n][t] = np.copy( qZmnt[m] )
                if t>=1:
                    varDist.qZZ[m][n][t-1] = old_qZmnt[m].reshape(-1, 1) * qZmnt[m]
                old_qZmnt[m] = qZmnt[m]
            t += 1
    
        fp.close()
        print "State file: ", stateFile, ". Sum: ", sum_Z
        
    return varDist

basedir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join( basedir, 'data/' )
#configfile = "param-3d-2,2,3.cfg"
configfile = "param-3d-2,3.cfg"
observationTemplate = "3d-2,3-{}.csv"
meanTemplate = "mean-3d-2,3-{}.csv"
trueStateTemplate = "state-3d-2,3-{}.csv"

Sigma = []
W = []
alpha = []
beta = []

configfile = os.path.join( datadir, configfile )
try:
    with open(configfile) as configfp: pass
except IOError as e:
    raise RuntimeError( 'Cannot open "%s". Exit' %(configfile) )

print "Loading config file {}".format(configfile)

configfp = open(configfile)
# D: 4, M: 3, K: [2, 3, 4]
header_re = re.compile( r"D: (\d+), M: (\d+), K: \[([\d, ]+)\]")
header = configfp.readline()
header_match = header_re.match(header)
D = int( header_match.group(1) )
M = int( header_match.group(2) )
Kspec = header_match.group(3)
Ks = Kspec.split(", ")
Ks = [ int(d) for d in Ks ]

configfp.readline()
configfp.readline()

for d in xrange(D):
    Sigma.append( readArrayLine(configfp) )

wholeW = []

for m in xrange(M):
    configfp.readline()
    configfp.readline()
    configfp.readline()
    Wm = []
    for k in xrange(Ks[m]):
        W_row = readArrayLine(configfp)
        Wm.append( W_row )
        wholeW.append( W_row )
        
    configfp.readline()
    alpha.append( readArrayLine(configfp) )
    if np.sum( alpha[-1] ) != 1:
        raise RuntimeError( "unnormalized alpha{}".format(m) )
    configfp.readline()
    beta_m = []
    for k in xrange(Ks[m]):
        beta_m.append( readArrayLine(configfp) )
        if np.sum( beta_m[-1] ) != 1:
            raise RuntimeError( "unnormalized beta{}-{}".format(m,k) )
        
    W.append(Wm)
    beta.append(beta_m)

wholeW = np.array(wholeW)

configs = [ dict( name="FAB", layers=2, doShrink=False, isDiag=True, maxIter=1000,
                  convTolerance=5e-7 ),
            dict( name="ML", layers=2, doShrink=False, isDiag=True, maxIter=1000, 
                  convTolerance=5e-7 )
         ]

observationFiles = []
meanFiles = []
stateFiles = []
testObFiles = []
testStateFiles = []

observationFiles.append( observationTemplate.format(1) )

for i in xrange(1,2):
    meanFiles.append( meanTemplate.format(i) )
    stateFiles.append( trueStateTemplate.format(i) )

testObFiles.append( meanTemplate.format(i+1) )
testStateFiles.append( trueStateTemplate.format(i+1) )

X2 = []
X2len = 0
mean = []
meanLen = 0 
testMean = []
testMeanLen = 0

for datafile in (observationFiles):
    datafile = os.path.join(datadir, datafile)
    fp = open(datafile)
    Xn = []
    for line in fp:
        items = [float(item) for item in line[:-1].split(',')]
        Xn.append(items)
    fp.close()
    Xn = np.array( Xn ) #Xn[0:500] )
    X2.append( Xn )
    X2len += len(Xn)

for meanfile in (meanFiles):
    meanfile = os.path.join(datadir, meanfile)
    fp = open(meanfile)
    mean_n = []
    for line in fp:
        items = [float(item) for item in line[:-1].split(',')]
        mean_n.append(items)
    fp.close()
    mean_n = np.array( mean_n ) #Xn[0:500] )
    mean.append( mean_n )
    meanLen += len(mean_n)

for testObFile in (testObFiles):
    testObFile = os.path.join(datadir, testObFile)
    fp = open(testObFile)
    testMean_n = []
    for line in fp:
        items = [float(item) for item in line[:-1].split(',')]
        testMean_n.append(items)
    fp.close()
    testMean_n = np.array( testMean_n ) #Xn[0:500] )
    testMean.append( testMean_n )
    testMeanLen += len(testMean_n)

sum_X = []
for Xn in X2:
     sum_X.append( np.sum( Xn, axis = 0 ) )

sum_mean = []
for mean_n in mean:
    sum_mean.append( np.sum( mean_n, axis = 0 ) )

dupRandVarDist = VariationalDist( M, len(stateFiles), [ len(mean_n) for mean_n in mean ], Ks )
trainVarDist = loadGroundZ( stateFiles, M, [ len(mean_n) for mean_n in mean ], Ks )
testVarDist = loadGroundZ( testStateFiles, M, [ len(mean_n) for mean_n in testMean ], Ks )

sum_qZ = []
for qZ_m in trainVarDist.qZ:
    sum_qZ_m = []
    for qZ_mn in qZ_m:
        sum_qZ_m.append( np.sum( qZ_mn, axis = 0 ) )
    sum_qZ.append( sum_qZ_m )           
           
algconfig = configs[0]    

algconfig['initKs'] = Ks
#algconfig['shrink_threshold'] = X2len * 0.75 / 5
algconfig['data'] = X2

fabfhmm = FactorialHiddenMarkovModel(**algconfig)

fabfhmm.updateSigmaW( mean, dupRandVarDist, mean, wholeW )

fabfhmm.updateSigmaW( mean, trainVarDist, mean, wholeW )

SumK = np.sum( [ layer.K for layer in fabfhmm.layers ] )
test_Sum_meanqZ, test_Sum_qZqZ = fabfhmm.sum_xqz_qzqz(SumK, testMean, testVarDist.qZ)
test_wholeW = np.dot( test_Sum_meanqZ, np.linalg.pinv( test_Sum_qZqZ ) )
error = test_Sum_meanqZ - np.dot( fabfhmm.wholeW, test_Sum_qZqZ )

#    for m,layer in enumerate(fabfhmm.layers):
#        layer.alpha = np.array( alpha[m] )
#        layer.beta = np.array( beta[m] )
#        layer.W = np.array( W[m] )
#        layer.WT = layer.W.T
#    fabfhmm.Sigma = np.array( Sigma )
#    fabfhmm.invSigma = np.linalg.inv( Sigma )

varDist2 = fabfhmm.forward_backward( X2, trainVarDist, MLEst=True )

# MLEst is set to ignore the state scale factors, 
# these factors are computed according to X, and so useless to X2
trainVarDist, pll_test = fabfhmm.calcNegKLDiv( X2, trainVarDist, MLEst=True )
print "Ground sequences FAB log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )
# first update trainVarDist using ML
# then reestimate the parameters using ML
trainVarDist, pll_test = fabfhmm.VStep( trainVarDist, X2, MLEst=True )
# it's still FAB, since the parameters are the FAB ones
print "Ground sequences FAB log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )        
# parameters updated now
FIC = fabfhmm.MStep( X2, trainVarDist )
# this is just to calc negKL
trainVarDist, pll_test = fabfhmm.VStep( trainVarDist, X2, MLEst=True )
print "Ground sequences ML log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )        
