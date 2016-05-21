from fabfhmm import FactorialHiddenMarkovModel
import numpy as np
import os
import re
import sys

def readArrayLine(fp, delim=", ", dtype=float):
    line = fp.readline()
    nums = line.split(", ")
    nums = [ dtype(n) for n in nums ]
    return nums

basedir = os.path.abspath(os.path.dirname(__file__))
datadir = os.path.join( basedir, 'data/' )
#configfile = "param-3d-2,2,3.cfg"
configfile = "param-3d-2,3.cfg"

Sigma = []
W = []
alpha = []
beta = []

if 'configfile' in locals():
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
        
else:    
    D = 3
    #seed = 100
    M = 3
    Ks = [2,2,3]
    #np.random.seed(seed)

fabfhmm = FactorialHiddenMarkovModel( layers=M, initKs=Ks, data=np.zeros((1,1,D)), iniBySampling=True )

if 'configfile' in locals():
    for m,layer in enumerate(fabfhmm.layers):
        layer.alpha = np.array( alpha[m] )
        layer.beta = np.array( beta[m] )
        layer.W = np.array( W[m] )
        layer.WT = layer.W.T
    fabfhmm.Sigma = np.array( Sigma )
    fabfhmm.invSigma = np.linalg.inv( Sigma )

samples_means, states = fabfhmm.sample(5, 2000)

#sys.exit()

strKs = ",".join( [ str(e) for e in Ks ] )
paramfile = os.path.join( datadir, "param-{}d-{}.csv".format( D, strKs ) )

pfp = open(paramfile, "w")

pfp.write( "D: {}, M: {}, K: {}\n\n".format( D, M, Ks ) )
pfp.write("Sigma:\n")
for d in xrange(D):
    pfp.write( ", ".join( str(Sigma_dc) for Sigma_dc in fabfhmm.Sigma[d] ) )
    pfp.write("\n")
pfp.write("\n")

for m,layer in enumerate(fabfhmm.layers):
    pfp.write( "LAYER {}:\n".format(m) )
    
    pfp.write("W:\n")
    for k in xrange(layer.K):
        pfp.write( ", ".join( str(w_kd) for w_kd in layer.W[k] ) )
        pfp.write("\n")
        
    pfp.write("alpha:\n")
    pfp.write( ", ".join( str(alpha_k) for alpha_k in layer.alpha ) )
    pfp.write("\n")
        
    pfp.write("beta:\n")
    for i in xrange(layer.K):
        pfp.write( ", ".join( str(beta_ij) for beta_ij in layer.beta[i] ) )
        pfp.write("\n")
        
    pfp.write("\n")
pfp.close()

for n,sequenceXMu in enumerate(samples_means):
    print "Writing sequence {}".format(n)
    
    Xfile = os.path.join( datadir, "{}d-{}-{}.csv".format( D, strKs, n+1 ) )
    xfp = open(Xfile, "w")
    
    meanfile = os.path.join( datadir, "mean-{}d-{}-{}.csv".format( D, strKs, n+1 ) )
    mfp = open(meanfile, "w")
    
    statefile = os.path.join( datadir, "state-{}d-{}-{}.csv".format( D, strKs, n+1 ) )
    zfp = open(statefile, "w")

    for t,x_mu in enumerate(sequenceXMu):
        x = x_mu[0]
        mu = x_mu[1]
        zs = states[n,t]
        
        line = ",".join( str(x_d) for x_d in x )
        xfp.write(line + "\n")
        
        line = ",".join( str(mu_d) for mu_d in mu )
        mfp.write(line + "\n")
        
        line = ",".join( str(z) for z in zs )
        zfp.write(line + "\n")
            
    xfp.close()
    mfp.close()
    zfp.close()
    
print "Done."

