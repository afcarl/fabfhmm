from fabfhmm import FactorialHiddenMarkovModel
from cfabfhmm import CollapsedFactorialHiddenMarkovModel
import numpy as np
import os
import re
import sys
import pdb

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

basedir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join( basedir, 'data/' )
configfile = "param-3d-2,2,3.cfg"
fileTemplate = "3d-2,2,3-{}.csv"
#configfile = "param-3d-2,3.cfg"
#fileTemplate = "3d-2,3-{}.csv"

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
print "Ks: %s" %Ks

configfp.readline()
configfp.readline()

for d in xrange(D):
    Sigma.append( readArrayLine(configfp) )

for m in xrange(M):
    configfp.readline()
    configfp.readline()
    configfp.readline()
    Wm = []
    for k in xrange(Ks[m]):
        Wm.append( readArrayLine(configfp) )
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

configs = [ 
            dict( name="CFAB", layers=3, doShrink=False, isDiag=True, maxIter=1000,
                  convTolerance=5e-7 ),
            dict( name="FAB", layers=3, doShrink=False, isDiag=True, maxIter=1000,
                  convTolerance=5e-7 ),
            dict( name="ML", layers=3, doShrink=False, isDiag=True, maxIter=1000,
                  convTolerance=5e-7 )
         ]

testFiles = []
# two sequences for test
for n in xrange( 4, 6 ):
    testFiles.append( fileTemplate.format(n) )

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
    X2.append( Xn )
    X2len += len(Xn)

logname = "groundlike.log"
tee = Tee( os.path.join( datadir, "log", logname ) )

for algconfig in configs:
        #np.random.seed(500000)

    algconfig['initKs'] = Ks
    #algconfig['shrink_threshold'] = X2len * 0.75 / 5
    algconfig['data'] = X2

    if algconfig['name'] == 'CFAB':
        fabfhmm = CollapsedFactorialHiddenMarkovModel(**algconfig)
    else:
        fabfhmm = FactorialHiddenMarkovModel(**algconfig)
        
    fabfhmm.iterUseZeroOrderApproxH = True
    fabfhmm.iterUseFirstOrderApproxH = True
    fabfhmm.iterDoShrink = False
    fabfhmm.approxKLCalc = True
    
    for m,layer in enumerate(fabfhmm.layers):
        layer.alpha = np.array( alpha[m] )
        layer.beta = np.array( beta[m] )
        layer.W = np.array( W[m] )
        layer.WT = layer.W.T
    fabfhmm.Sigma = np.array( Sigma )
    fabfhmm.invSigma = np.linalg.inv( Sigma )
    
    #pdb.set_trace()
    
    sumK2 = 0
    for m in xrange(fabfhmm.M):
        K_m = fabfhmm.layers[m].K
        # each W[m] is K_m*D, instead of D*K_m.
        fabfhmm.wholeW[ sumK2 : sumK2 + K_m ] = np.copy( fabfhmm.layers[m].W )
        sumK2 += K_m

    #varDist = fabfhmm.iniVarDist( X2, Ks )
    #varDist = fabfhmm.forward_backward( X2, varDist )

    #varDist, pll_test = fabfhmm.calcNegKLDiv( X2, varDist )
    #print "Ground sequences FAB log likelihood: %f. Avg: %f" %( pll_test, pll_test/X2len )
    # first update varDist
    # then reestimate the parameters using ML
    varDist, kls_test = fabfhmm.VStep( X2, None )
    fabfhmm.varDist = varDist
    
    for i in xrange(2):
        varDist, kls_test = fabfhmm.VStep( X2, varDist )
        
    # it's still FAB, since the parameters are the FAB ones
    partialFIC = fabfhmm.calcPartialFIC(X2)
    pll_test = kls_test + partialFIC
    
    print "Ground sequences FAB log likelihood: %f. Avg: %f\n" %( pll_test, pll_test/X2len )

    # parameters updated now
    FIC = fabfhmm.MStep( X2, varDist )
    # this is just to calc negKL
    varDist, kls_test = fabfhmm.VStep( X2, varDist )
    partialFIC = fabfhmm.calcPartialFIC(X2)
    pll_test = kls_test + partialFIC
    
    print "Ground sequences ML log likelihood: %f. Avg: %f\n" %( pll_test, pll_test/X2len )

tee.close()
