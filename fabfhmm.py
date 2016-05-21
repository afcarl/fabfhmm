import datetime
import numpy as np
import copy
import pickle
from operator import itemgetter
from vardist import VariationalDist

# protect() is to make the array arguments read-only,
# so as to detect accidental changes to them due to bugs
# farg: formal argument
def protect( farg, *args ):
    farg.setflags(write=False)
    for arg in args:
        arg.setflags(write=False)

def unprotect( farg, *args ):
    farg.setflags(write=True)
    for arg in args:
        arg.setflags(write=True)

def normalize(data):
    return data / np.sum(data)

def saveModel(obj, filepath):
    model_fh = open(filepath, "wb")
    pickle.dump(obj, model_fh)
    model_fh.close()

def loadModel(filepath):
    model_fh = open(filepath, "rb")
    model = pickle.load(model_fh)
    model_fh.close()
    return model

# this matrix is singular: all rows/columns sum to 0
def jacobdQzdh(qZ_mnt):
    K = qZ_mnt.shape[0]
    # Eq(143) j != k
    J = - qZ_mnt.reshape(-1,1) * qZ_mnt
    # Eq(143) j == k
    J[ range(K), range(K) ] = qZ_mnt * ( 1 - qZ_mnt )
    return J

def calcPruneThres(T, initK, K):
    thres = T * 0.7 / np.sqrt(K*initK)
    return thres

class FHMMLayer:
    def __init__(self, M, K, X, iniBySampling):
        # initially component number is set to K. No shrinkage has occurred
        self.K = K
        self.D = X[0][0].shape[0]
        self.compComplexities = np.zeros(K)

        while True:
            self.alpha = np.random.rand(K)
            self.alpha /= np.sum(self.alpha)
            if iniBySampling:
                self.alpha = np.round( self.alpha, 1 )
                self.alpha[-1] = 1 - np.sum( self.alpha[:-1] )
                if not (self.alpha == 0).any():
                    break
            else:
                break

        self.beta = np.zeros( (K,K) )
        for k in xrange(K):
            while True:
                self.beta[k] = np.random.rand(K)
                self.beta[k] /= np.sum(self.beta[k])

                if iniBySampling:
                    self.beta[k] = np.round( self.beta[k], 1 )
                    self.beta[k][-1] = 1 - np.sum( self.beta[k][:-1] )
                    if not (self.beta[k] == 0).any():
                        break
                else:
                    break

        Xconcat = np.concatenate(X)
        sumT = len(Xconcat)
        # W: K*D
        # weight matrix W is initialized as 1/M of random points in X
        # [ ... for ... ] produces a list of np.array. so we need to convert
        # it to np.array to use matrix operations
        if sumT > K:
            self.W = np.array( [ Xconcat[ np.random.randint(sumT) ] for _ in xrange(K) ] )
            self.W /= M
        else:
            self.W = np.random.randint( -4, 4, size=(K, self.D) )
            #self.W = np.round( self.W, 1 )

        # division by M is to ensure the sum of M layers is near a point
        # save the transpose in advance to increase efficiency
        self.WT = self.W.T

        self.logStateScaleFactors = np.ones( (2, K) ) * np.log( 1.0 / K )
        self.stateScaleFactorsNorm = np.array( [ K, K ] )

        self.epsilon = 1e-10
        self.h = []
        self.logh = []
        self.oldH = []
        self.oldLogh = []

        self.params = [ dict() for i in xrange(3) ]

    def saveParams(self, i):

        self.params[i] = dict( alpha=self.alpha, beta=self.beta,
                                      W=self.W, WT=self.WT, h=self.h, logh=self.logh,
                                      logStateScaleFactors=self.logStateScaleFactors,
                                      stateScaleFactorsNorm=self.stateScaleFactorsNorm
                          )

    def loadParams(self, i):
        self.alpha = self.params[i]['alpha']
        self.beta = self.params[i]['beta']
        self.W = self.params[i]['W']
        self.WT = self.params[i]['WT']
        self.h = self.params[i]['h']
        self.logh = self.params[i]['logh']
        self.logStateScaleFactors = self.params[i]['logStateScaleFactors']
        self.stateScaleFactorsNorm = self.params[i]['stateScaleFactorsNorm']

    def MStep( self, varDist, m ):
        # the variational distribution on the m-th layer
        qZm = varDist.qZ[m]
        qZZm = varDist.qZZ[m]

        # update initial probability
        self.alpha = self.epsilon + np.sum( [ qZmn[0] for qZmn in qZm ], axis=0 )
        self.alpha /= np.sum(self.alpha)

        # update transition probability
        self.beta = np.sum( qZZm[0], axis=0 ) + self.epsilon
        for qZZmn in qZZm[1:]:
            self.beta += np.sum( qZZmn, axis=0 )
        for k in xrange(self.K):
            self.beta[k] /= np.sum( self.beta[k] )

        protect( self.alpha, self.beta )

    def forward_backward( self, N, T, varDist, m ):

        qZ = varDist.qZ[m]
        qZZ = varDist.qZZ[m]
        forward = varDist.forward[m]
        backward = varDist.backward[m]
        norm = varDist.norm[m]

        for n in xrange(N):
            unprotect( norm[n], qZ[n], qZZ[n] )

            hn = self.h[n]

            # forward pass
            forward[n][0] = hn[0] * self.alpha
            norm[n][0] = np.sum( forward[n][0] )
            forward[n][0] /= norm[n][0]

            forward[n][1:] = np.copy( hn[1:] )
            for t in xrange(1, T[n]):
                forward[n][t] *= np.dot( forward[n][t-1], self.beta )
                norm[n][t] = np.sum(forward[n][t])
                try:
                    forward[n][t] /= norm[n][t]
                except FloatingPointError:
                    pdb.set_trace()

            # backward pass
            backward[n][-1] = np.ones( self.K )
            hbbeta = np.zeros( ( T[n]-1, self.K, self.K ) )
            # t: T[n]-2, ..., 0
            for t in xrange( T[n] - 2, -1, -1 ):
                hb_t1 = hn[ t+1 ] * backward[n][ t+1 ]
                for k in xrange(self.K):
                    hbbeta[ t, k ] = np.copy( hb_t1 )

                try:
                    hbbeta[t] *= self.beta
                except FloatingPointError:
                    pdb.set_trace()

                backward[n][t] = np.sum( hbbeta[t], axis=1 )
                try:
                    backward[n][t] /= norm[n][ t+1 ]
                except FloatingPointError:
                    pdb.set_trace()

            # update qZ & qZZ
            qZ[n] = forward[n] * backward[n]
            qZZ[n] = hbbeta
            for t in xrange(T[n]-1):
                for k in xrange(self.K):
                    qZZ[n][t,k] *= forward[n][t,k]
                qZZ[n][t] /= norm[n][t+1]

            for t in xrange(T[n]):
                if np.abs( np.sum( qZ[n][t] ) - 1 ) > 1e-5:
                    raise RuntimeError("qZ unnormalized")

            for t in xrange(T[n]-1):
                if np.abs( np.sum( qZZ[n][t] ) - 1 ) > 1e-5:
                    raise RuntimeError("qZZ unnormalized")

            protect( qZ[n], qZZ[n], norm[n] )
        # for n

    def approxQzUpdate( self, N, T, varDist, m ):
        # only varDist.qZ is updated.
        # varDist.{qZZ,norm,forward,backward} is intact, since VStep doesn't involve it
        oldQz_m = varDist.oldQz[m] = varDist.qZ[m]
        qZ_m = copy.deepcopy( varDist.qZ[m] )

        minFrac = 0.3
        maxoff = 1 - minFrac

        for n in xrange(N):
            for t in xrange( T[n] ):
                dlogh = self.logh[n][t] - self.oldLogh[n][t]
                J = jacobdQzdh( oldQz_m[n][t] )
                dqZ = np.dot( J, dlogh )
                # gamma in Eq(144)
                negk = dqZ < -1e-5
                gamVec = np.ones(self.K)
                # dqZ[negk] < 0, so gamVec[negk] >= 0
                gamVec[negk] = - maxoff * oldQz_m[n][t][negk] / dqZ[negk]
                gam = np.min(gamVec)
                # Eq(144)
                qZ_m[n][t] = oldQz_m[n][t] + gam * dqZ

                # there should be no need to do normalization for qZ_m[n][t]
                # just in case, check if they sum to 1
                #if np.sum( np.abs( qZ_m[n][t] - oldQz_m[n][t] ) ) > 1e-6:
                #    pass
                if gam > 1:
                    pdb.set_trace()

                if np.abs( np.sum(qZ_m[n][t]) - 1 ) > 1e-6:
                    pdb.set_trace()

        varDist.qZ[m] = qZ_m

    def updateMass(self, varDist, m):
        # varDist.qZ[m,0]: a subarray of T,K. axis=0: adding up over T
        self.mass = np.sum( varDist.qZ[m][0], axis=0 ) # sum_{n,t} qZ(n,t)
        # MUST MAKE A COPY!
        # otherwise the summation later will change the original qZ
        self.endingMass = np.copy( varDist.qZ[m][0][-1] )

        totalLen = 0

        for qZmn in varDist.qZ[m][1:]:
            self.mass += np.sum(qZmn, axis=0)
            self.endingMass += qZmn[-1]
            totalLen += len(qZmn)

        if np.sum(self.mass) < totalLen:
            print "Bug: layer " + str(m) + " mass is missing"

        # update midMass
        self.midMass = self.mass - self.endingMass

        # update component complexities
        self.compComplexities = -0.5 * ( self.K - 1 ) * ( np.log( self.midMass ) - 1 )
        self.compComplexities -= 0.5 * self.D * ( np.log( self.mass ) - 1 )

        #protect( self.mass, self.midMass, self.endingMass )

    def breakSymmetry(self, active):
        masses = []
        for k in xrange(self.K):
            if active[k]:
                masses.append( [ k, self.mass[k] ] )

        masses = sorted( masses, key=itemgetter(1) )
        
        #unprotect( self.mass )

        for k in xrange( 1, len(masses) ):
            # two possibly identical states. break the symmetry randomly
            if np.isclose( masses[k-1][1], masses[k][1], rtol=1e-3 ):
                # scale factor is uniform in [-0.99,1.01]. the expectation is 0, unbiased
                scale = np.random.uniform(-1, 1) * 0.01
                delta = masses[k-1][1] * scale
                idx = masses[k-1][0]
                self.mass[idx] += delta
                self.midMass[idx] += delta

        #protect( self.mass )
        
    # calculate the log of \delta in Eq.(8.3), for the m-th layer
    def updateStateScaleFactors( self, pruneThres, shrinkRate ):
        self.stateScaleFactorsNorm = np.zeros( 2 )
        self.logStateScaleFactors = np.zeros( ( 2, self.K ) )

        masses = []
        for k in xrange(self.K):
            masses.append( [ k, self.mass[k] ] )

        masses = sorted( masses, key=itemgetter(1) )

        active = np.ones(self.K, dtype=bool)
        
        # each time the total mass to be set inactive should not exceed pruneThres
        for k in xrange(self.K):
            if masses[k][1] <= pruneThres:
                active[ masses[k][0] ] = False
                pruneThres -= masses[k][1]
            else:
                break
        
#        active = self.mass > pruneThres

        if (active == False).all():
            raise RuntimeError( "All mass {} < pruneThres {}".format(self.mass, pruneThres) )

        self.breakSymmetry(active)

        for k in xrange(self.K):
            if active[k]:
                if self.mass[k] > 0 and self.midMass[k] > 0:
                    # the factors for all but the ending point in the sequence
                    self.logStateScaleFactors[ 0, k ] = -0.5 * shrinkRate * (
                            self.D / self.mass[k] + ( self.K - 1 ) / self.midMass[k] )
                    # the factor for the ending point in the sequence
                    self.logStateScaleFactors[ 1, k ] = -0.5 * shrinkRate * (
                            self.D / self.mass[k] )
                    self.stateScaleFactorsNorm += np.exp( self.logStateScaleFactors[ :, k ] )

        if (self.stateScaleFactorsNorm == 0).any():
            pdb.set_trace()

        for i in range(2):
            self.logStateScaleFactors[i] -= np.log( self.stateScaleFactorsNorm[i] )

        protect( self.stateScaleFactorsNorm, self.logStateScaleFactors )

        return active

    def sampleState( self, lastState=-1 ):
        # t=0
        if lastState == -1:
            return np.random.choice( self.K, p= self.alpha )
        return np.random.choice( self.K, p= self.beta[lastState] )

class FactorialHiddenMarkovModel:
    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

        self.initKs = kwargs.get('initKs')
        #self.pruneThres= kwargs.get( 'pruneThres', 100 )
        self.convTolerance= kwargs.get( 'convTolerance', 5e-7 )
        self.maxIter = kwargs.get( 'maxIter', 1000 )

        # M: the number of layers
        self.M = M = kwargs.get('layers')

        self.doShrink = kwargs.get( 'doShrink', True )
        self.doPrune = kwargs.get( 'doPrune', True )

        self.MLEst = not self.doShrink

        # First do so many rounds without shrinkage
        self.noshrinkRounds = kwargs.get( 'noshrinkRounds', 0 )

        # Decreasing degree of shrinkage
        self.varShrink = kwargs.get( 'varShrink', False )
        if self.varShrink:
            self.baseShrinkRate = kwargs.get( 'baseShrinkRate', 10.0 )
            self.shrinkRate = self.baseShrinkRate
        else:
            self.shrinkRate = 1.0

        # whether the covariance matrix is diagonal
        self.isDiag = kwargs.get( 'isDiag', True )
#        self.testSeq = kwargs.get('testSequence', None)

        # must be provided, even if a fake data (e.g. in the sample mode)
        self.X = X = kwargs.get('data')
        self.iniBySampling = iniBySampling = kwargs.get('iniBySampling', False)
        self.VStepFirst = kwargs.get('VStepFirst', False)

        self.useFirstOrderApproxH = kwargs.get('useFirstOrderApproxH', False)

        if self.useFirstOrderApproxH:
            print "First order approximation of H is used"
        
        self.useZeroOrderApproxH  = kwargs.get('useZeroOrderApproxH', True)

        #self.zeroOrderRounds = kwargs.get('zeroOrderRounds', 1)
        self.doNormalizeW = kwargs.get('doNormalizeW', True)
        # 0 to disable the fast approximate V iterations
        self.approxVIterNum = kwargs.get('approxVIterNum', 0)

        # the old parameter format
        # all layers have Ks(singular) state numbers
        if type(self.initKs) == int:
            self.sumK = self.initKs * M
            self.Ks = [self.initKs] * M
            self.initKs = [self.initKs] * M
            
        # new format. different layers may have different numbers of states
        else:
            if len(self.initKs) != M:
                raise RuntimeError("Ks array length does not match M")

            self.sumK = np.sum(self.initKs)
            self.Ks = self.initKs

        # in case of generating samples, we should provide an example data point in X
        # e.g. D==5, then X = np.zeros((1,1,5))
        self.D = X[0][0].shape[0]
        # initial Sigma is always diagonal
        self.Sigma = np.diag( [ 0.5 ] * self.D )
        self.invSigma = np.linalg.inv( self.Sigma )

        self.layers = [ FHMMLayer( M, self.Ks[m], X, iniBySampling ) for m in xrange(M) ]

        self.wholeW = np.zeros( (self.sumK, self.D) )
        sumK2 = 0

        for m in xrange(self.M):
            K_m = self.layers[m].K
            # each W[m] is K_m*D, instead of D*K_m.
            self.wholeW[ sumK2 : sumK2 + K_m ] = np.copy( self.layers[m].W )
            sumK2 += K_m

        self.N = len(X)
        self.T = np.array( [ x.shape[0] for x in X ], dtype=int )

        # historical sum log norm, part of negKL. for debugging purpose
        self.lastSumLogNorm = -np.inf

        self.VStepNum = 0

        self.epsilon = 1e-8

        # to satisfy the python interpreter
        # before first use, it is uninitialized.
        # first call to zeroOrderApproxH will initialize it.
        # After that, first order approx is used, which requires it's initialized
        self.hs = []
        self.loghs = []
        self.oldHs = []
        self.oldLoghs = []

        self.loghUpperbound = 5
        self.loghLowerbound = -5
            
        self.lastNegKL = -np.Inf
        self.mstepPruned = False
        self.dumpAbnormal = False
        self.approxKLCalc = True
        
        # catch abnormal floating point operations
        np.seterr(all='raise')

        # 0: backup of the active/conventional params
        # 1: params of the last iteration
        # 2: params with W normalized
        self.params = [ dict() for i in xrange(3) ]

    def saveParams(self, i):

        self.params[i] = dict( wholeW=self.wholeW, Sigma=self.Sigma,
                                invSigma=self.invSigma,
                                hs=self.hs, loghs=self.loghs,
                                varDist=self.varDist
                            )

        for m in xrange(self.M):
            self.layers[m].saveParams(i)

    def loadParams(self, i):
        self.wholeW = self.params[i]['wholeW']
        self.Sigma = self.params[i]['Sigma']
        self.invSigma = self.params[i]['invSigma']
        self.hs = self.params[i]['hs']
        self.loghs = self.params[i]['loghs']
        self.varDist = self.params[i]['varDist']

        for m in xrange(self.M):
            self.layers[m].loadParams(i)


    def fabIterate(self, X, varDist=None):
        if varDist is None:
            varDist = self.iniVarDist( X, self.Ks )

        T = np.array( [ x.shape[0] for x in X ], dtype=int )

        # the total length of all sequences
        sumT = np.sum( T )
        convTolerance = sumT * self.convTolerance

        startTime = datetime.datetime.now()
        lastTime = startTime

        self.VStepNum = 0
        smallFICDecCounter = 0
        FIC = -np.Inf
        oldFIC = -np.Inf
        oldFIC2 = -np.Inf

        self.iterDoShrink = False
        self.iterUseFirstOrderApproxH = self.useFirstOrderApproxH
        # first iteration must be zero order
        self.iterUseZeroOrderApproxH = True

        # Doing one V-step before the EM iterations is extremely important
        # It lets the optimization avoids the local minimum with
        # many similar components
        if self.VStepFirst:
            varDist, negKL = self.VStep( X, varDist )

        for i in xrange(self.maxIter):
            now = datetime.datetime.now()
            elapsed = ( now - startTime ).seconds
            if i == 0:
                avgIterTime = 0
                dur = 0
            else:
                avgIterTime = elapsed * 1.0 / i
                dur = ( now - lastTime ).seconds + ( now - lastTime ).microseconds * 0.000001
            print "EM iteration %d starts at %02d:%02d:%02d. %d seconds elapsed. Dur: %.1f, avg: %.1f" %( i,
                    now.hour, now.minute, now.second, elapsed, dur, avgIterTime )
            lastTime = now

            if i < self.noshrinkRounds:
                self.iterDoShrink = False
            else:
                self.iterDoShrink = self.doShrink
                if i == self.noshrinkRounds:
                    print "Shrikage begins."
                    #saveModel(self, "shrink.bin")
                    #exit()

            if self.varShrink:
                self.shrinkRate = self.baseShrinkRate * min( 1, 300.0 / self.baseShrinkRate / (i + 1) )
                if self.iterDoShrink:
                    print "Shrink rate: %.2f" %(self.shrinkRate)

            partialFIC = self.MStep( X, varDist )
            varDist, negKL = self.VStep( X, varDist )
            FIC = partialFIC + negKL

            # when no shrinkage, V-step only will result in the same h and varDist
            # so no sense to iterate
            if self.VStepNum >= self.noshrinkRounds and self.approxVIterNum > 0 \
                    and not self.mstepPruned \
                    and FIC > oldFIC and np.abs(FIC - oldFIC) < 0.1:
                print "Approximate V-step iteration begins"
                #saveModel(self, "approxv.bin")
                iterNum = 0
                while iterNum < self.approxVIterNum:
                    #and FIC > oldFIC and np.abs(FIC - oldFIC) < 0.1:
                    oldFIC = FIC
                    # disable shrinkage during the approximate iterations
                    #self.iterDoShrink = False
                    partialFIC = self.MStep( X, varDist )
                    varDist, negKL = self.VStep( X, varDist, isExact=False )

                    for m in xrange(self.M):
                        print "%d: %s" %( m, str( self.layers[m].mass ) )

                    FIC = partialFIC + negKL
                    print "Iter %d FIC: %f" %(iterNum, FIC)
                    iterNum = iterNum + 1
                    if FIC < oldFIC:
                        print "********* abnormal increase of FIC *********"

                oldFIC = FIC
                # exact update to ensure accuracy
                varDist, negKL = self.VStep( X, varDist )
                FIC = partialFIC + negKL

            if FIC < oldFIC:
                if self.mstepPruned:
                    print "Shrinkage caused increase of FIC. Normal"
                elif FIC > oldFIC2:
                    print "Cyclic steps, but still moving forward"
                else:
                    print "********* abnormal increase of FIC *********"
                    #if not self.useFirstOrderApproxH or not self.useZeroOrderApproxH:
                    #    print "Zero and First order approximations will be used together"
                    #    self.iterUseFirstOrderApproxH = True
                    #    self.iterUseZeroOrderApproxH  = True
            else:
                self.iterUseFirstOrderApproxH = self.useFirstOrderApproxH
                self.iterUseZeroOrderApproxH = self.useZeroOrderApproxH

            print i, "FIC: ", FIC, self.Ks

            # Convergence.
            # abs() here is necessary, as FIC seldom decreases
            if FIC > oldFIC and abs(FIC - oldFIC) < convTolerance:
                break
            elif abs(FIC - oldFIC) < convTolerance:
                smallFICDecCounter += 1
                if smallFICDecCounter > 10:
                    break

            if FIC > oldFIC2 and abs(FIC - oldFIC2) < convTolerance:
                break
            elif abs(FIC - oldFIC2) < convTolerance:
                smallFICDecCounter += 1
                if smallFICDecCounter > 10:
                    break
    
            oldFIC2 = oldFIC
            oldFIC = FIC

        return varDist, FIC

    def sum_xqz_qzqz(self, SumK, X, qZ):
        Sum_qZqZ = np.zeros( ( SumK, SumK ) )
        Sum_xqZ  = np.zeros( ( self.D, SumK ) )

        for n,Xn in enumerate( X ):
            for t,Xnt in enumerate( Xn ):
                # we couldn't write qZ[m,n,t], since qZ is a list
                # only qZ[m][n] is an np.array
                # don't use ravel here. ravel returns an array with dtype=object
                qZnt = [ qZ[m][n][t] for m in xrange(self.M) ]
                qZnt_flat = np.concatenate( qZnt )
                # reshape(-1,1) makes it a column vector, and "*" does
                # element-wise multiplication, produces a SumK*SumK matrix
                qZqZ = qZnt_flat.reshape(-1, 1) * qZnt_flat
                # the diagonal blocks (in different sizes) should be replaced
                dim = 0
                for m in xrange(self.M):
                    Km = self.layers[m].K
                    # diagonalize the vector qZ to be the diagonal blocks of qZqZ
                    diag_qZmnt = np.diag( qZnt[m] )
                    qZqZ[ dim : dim + Km, dim : dim + Km ] = diag_qZmnt
                    dim += Km
                Sum_qZqZ += qZqZ

                xnt = Xnt.reshape(-1, 1)
                Sum_xqZ += xnt * qZnt_flat
        return Sum_xqZ, Sum_qZqZ

#===============================================================================
#     # F = -Tr(W'Y)+ l*|WA-Y|^2
#     # A = Sum_qZqZ, Y = Sum_xqZ
#     def evalF(self, W, A, Y, ell):
#         F = - np.trace( np.dot(W.T, Y) ) + ell * np.linalg.norm( np.dot(W, A) - Y )**2
#         return F
#
#     # Armijo linear search to find W that maximizes Tr(W'Y)
#     # Proven to be useless
#     def findWByLinearSearch(self, A, Y):
#         # initialize W using Eq.(8.11)
#         W3 = np.dot( Y, np.linalg.pinv( A ) )
#         W = np.random.rand(Y.shape[0], A.shape[0])
#         ell = 1.0
#
#         while ell < 100000000:
#             F = self.evalF( W, A, Y, ell )
#             dW = - np.dot( self.invSigma, Y )
#             dW += ell * np.dot( W, np.dot( A, A.T ) )
#             dW -= ell * np.dot( Y, A.T )
#             dWsize = np.sum( np.abs(dW) )
#             # dF/dW ~= 0, already at maximum. so proceed to next iteration
#             if dWsize < 1e-2:
#                 ell *= 1.6
#                 continue
#
#             # a very big number to pass the condition check at the first time
#             F2 = 1e50
#             step = 1.0
#             while F2 > F:
#                 W2 = W - step * dW
#                 F2 = self.evalF( W2, A, Y, ell )
#                 step = step / 2
#             W = W2
#             ell *= 1.6
#
#         f1 = - np.trace( np.dot(W.T, Y) )
#         f2 = - np.trace( np.dot(W3.T, Y) )
#         return W
#===============================================================================

    #===========================================================================
    # # only for square matrices
    # def pinvBySVD(self, A):
    #
    #     U, S, V = np.linalg.svd( A )
    #     S1 = np.zeros(A.shape[0])
    #     # more strict than numpy's pinv
    #     #if (np.abs(S) < 0.1).any():
    #     #    pass
    #
    #     normalSV = S > 1e-5
    #     S1[normalSV] = 1 / S[normalSV]
    #     #S1 = 1 / S
    #     S1 = np.diag(S1)
    #     A1 = np.dot( V.T, np.dot( S1, U.T) )
    #     return A1
    #===========================================================================

    def updateSigmaW(self, X, varDist, mean=None, wholeW=None):
        # varDist: N, T, M, K
        # wholeW: ground truth W. To test the accuracy of the algorithm

        # the hidden states of each layer, layer.K, could be smaller than
        # the initial K, self.sumK, after shrinkage

        SumK = np.sum( [ layer.K for layer in self.layers ] )

        qZ = varDist.qZ

        Sum_xqZ, Sum_qZqZ = self.sum_xqz_qzqz(SumK, X, qZ)
        inv_qZqZ = np.linalg.pinv( Sum_qZqZ, 5e-2 )
        #inv_qZqZ = self.pinvBySVD(Sum_qZqZ)

        W = np.dot( Sum_xqZ, inv_qZqZ )
        # weight of the perturbation
        #=======================================================================
        # weight = 1
        # # adding perturbation, to make the rows different (break symmetry)
        # # random matrix entries in [-2,2]
        # randB = 4 * np.random.random_sample( W0.shape ) - 2
        # W = W0 + weight * np.dot( randB,
        #             np.identity(Sum_qZqZ.shape[0]) - np.dot( Sum_qZqZ, inv_qZqZ ) )
        #=======================================================================

        error = Sum_xqZ - np.dot( W, Sum_qZqZ )

        #=======================================================================
        # f1 = - np.trace( np.dot(W.T, Sum_xqZ) )
        # f2 = - np.trace( np.dot(wholeW, Sum_xqZ) )
        # IAAiY = np.dot( np.identity(Sum_qZqZ.shape[0]) -
        #                 np.dot( Sum_qZqZ, np.linalg.pinv( Sum_qZqZ ) ), Sum_xqZ.T )
        #
        # if mean != None:
        #     Sum_meanqZ, Sum_qZqZ = self.sum_xqz_qzqz(SumK, mean, qZ)
        #     W2 = np.dot( Sum_meanqZ, np.linalg.pinv( Sum_qZqZ ) )
        #     error = Sum_xqZ - np.dot( wholeW.T, Sum_qZqZ )
        #     error1 = Sum_meanqZ - np.dot( W, Sum_qZqZ )
        #     Sum_xqZ[0][1] = Sum_meanqZ[0][1]
        #     W3 = np.dot( Sum_xqZ, np.linalg.pinv( Sum_qZqZ ) )
        #     error3 = Sum_xqZ - np.dot( W3, Sum_qZqZ )
        #=======================================================================

#        error = np.sum( np.abs( Sum_xqZ - np.dot( W, Sum_qZqZ ) ) )
#        print "Pseudo inv Error: %f" %(error)

        W = W.T
        self.wholeW = W

        # distribute W into different layers
        dim = 0
        for m in xrange(self.M):
            Km = self.layers[m].K

            # each W[m] is Km*D, instead of D*Km. so transpose before use
            self.layers[m].W = np.copy( W[ dim : dim + Km ] )
            # cache the transposes to improve efficiency
            self.layers[m].WT = self.layers[m].W.T

            #protect( self.layers[m].W, self.layers[m].WT )

            dim += Km

        #self.calcNegKLDiv( X, varDist )

        Sum_residue = np.zeros( ( self.D, self.D ) )
        Sum_Tn = 0

        for n,Xn in enumerate(X):
            for t,Xnt in enumerate(Xn):
                qZnt = [ qZ[m][n][t] for m in xrange(self.M) ]
                # Eq.(8.11)
                Sum_residue += Xnt.reshape(-1, 1) * Xnt
                for m in xrange( self.M ):
                    # WT: D*Km, qZnt[m]: 1*Km. dot produces D*1
                    Sum_residue -= np.dot( self.layers[m].WT, qZnt[m].reshape(-1,1) ) * Xnt
            Sum_Tn += Xn.shape[0]

        Sigma2_diag = Sum_residue / Sum_Tn
        # the first diag extract the diagonal as an array,
        # the second builds a matrix with that array as the diagonal
        if self.isDiag:
            newSigma = np.diag( np.diag( Sigma2_diag ) )
        else:
            diagVec = np.diag( Sigma2_diag ).copy()
            newSigma = Sigma2_diag * 0.5
            for d in xrange(self.D):
                newSigma[d,d] = diagVec[d]
            newSigma = ( self.Sigma + self.Sigma.T ) * 0.5

        for i in xrange(self.D):
            if( newSigma[i,i] <= -1e-10 ):
                raise RuntimeError('Sigma is not positive definite')

        self.Sigma = newSigma

        self.invSigma = np.linalg.inv( self.Sigma )

        protect( self.Sigma, self.invSigma )

    def calcLogHResidue(self, X, loghs, varDist):
        # loghs: m, n, t
        # Eq.(92)

        T = np.array( [ x.shape[0] for x in X ], dtype=int )

        qZ = varDist.qZ
        Ks = self.Ks
        halfLambda = []
        V = []
        C1W = [ np.dot( self.invSigma, self.layers[m].WT ) for m in xrange(self.M) ]
        sum_residue = 0

        for layer in self.layers:
            # different \Lambda_i has different length.
            # so we couldn't use np.array to store them
            # calculate 1/2 * \Lambda, since these values are repeatedly used
            halfLambda.append( 0.5 * np.diagonal(
                            np.dot( np.dot( layer.W, self.invSigma ), layer.WT ) ) )
            V.append([])

            for layer2 in self.layers:
                V[-1].append( np.dot( np.dot( layer.W, self.invSigma ), layer2.WT ) )

        for n, Xn in enumerate(X):
            for t in xrange( T[n] ):
                # qZ[p][n][t]: 1*Kp, Ws[p]: Kp*D, qW: 1*D
                if t < T[n] - 1:
                    isEnd = 0
                else:
                    isEnd = 1

                for m in xrange(self.M):
                    qV = []
                    for p in xrange(self.M):
                        # the summation excludes subscript m
                        if p == m:
                            continue

                        qV.append( np.dot(qZ[p][n][t], V[p][m]) )
                    sum_qV = np.sum( qV, axis=0 )

                    K_m = Ks[m]
                    c_m = np.zeros( K_m )
                    if self.iterDoShrink and not self.MLEst:
                        c_m += self.layers[m].logStateScaleFactors[isEnd]
                    c_m += np.dot( Xn[t], C1W[m] )
                    c_m -= halfLambda[m]

                    residue = loghs[m][n][t] + sum_qV - c_m
                    sum_residue += np.sum( np.abs(residue) )

        return sum_residue

    # scale between [ self.loghLowerbound, self.loghUpperbound ]
    # If the maximum element in logh > self.loghUpperbound, then subtract the extra amount from all elements
    def normalizeLogH(self, logh):
        
        valRange = np.max(logh) - np.min(logh)
        if valRange > self.loghUpperbound - self.loghLowerbound:
            logh2 = logh / valRange * (self.loghUpperbound - self.loghLowerbound)
        else:
            logh2 = np.copy(logh)
                
        logh_max = np.max(logh2)
        logh_min = np.min(logh2)
        
        if logh_max > self.loghUpperbound:
            logh2 -= logh_max - self.loghUpperbound
        if logh_min < self.loghLowerbound:
            logh2 -= logh_min - self.loghLowerbound
            
        if np.min(logh2) < self.loghLowerbound - 0.1 or np.max(logh2) > self.loghUpperbound + 0.1:
            pdb.set_trace()
            
        return logh2

    def calcH( self, X, varDist, loghs, method=0 ):
        if method == 1:
            return self.firstOrderApproxH( X, varDist, loghs )
        else:
            return self.zeroOrderApproxH( X, varDist )

    # compute the variational parameters h. Eq.(8.10)
    # the computed h is distributed into M layers.
    # in each layer, dimensions: N, T, K. The same as X
    def zeroOrderApproxH( self, X, varDist ):

        T = np.array( [ x.shape[0] for x in X ], dtype=int )

        halfLambda = []

        for layer in self.layers:
            # different \Lambda_i has different length.
            # so we couldn't use np.array to store them
            # calculate 1/2 * \Lambda, since these values are repeatedly used
            halfLambda.append( 0.5 * np.diagonal(
                            np.dot( np.dot( layer.W, self.invSigma ), layer.WT ) ) )
            #layer.oldh = copy.deepcopy(layer.h)

        qZ = varDist.qZ

        # M elements of [], to back up layers.h
        hs = [ [] for m in xrange(self.M) ]
        loghs = [ [] for m in xrange(self.M) ]

        for n, Xn in enumerate(X):
            # allocate the M arrays. numpy.array needs pre-specify the array size
            try:
                h_Mn = [ np.zeros( ( T[n], layer.K ) ) for layer in self.layers ]
            except IndexError:
                pdb.set_trace()

            log_h_Mn = [ np.zeros( ( T[n], layer.K ) ) for layer in self.layers ]

            for t in xrange( T[n] ):
                # qZ[p][n][t]: 1*Kp, Ws[p]: Kp*D, qW: 1*D
                for m in xrange(self.M):
                    if qZ[m][n][t].shape[0] != self.layers[m].W.shape[0]:
                        raise RuntimeError("Matrices are not aligned")

                qW = [ np.dot( qZ[m][n][t], self.layers[m].W ) for m in xrange(self.M) ]
                sum_qW = np.sum( qW, axis=0 )

                if t < T[n] - 1:
                    isEnd = 0
                else:
                    isEnd = 1

                for m in xrange(self.M):
                    residue = Xn[t] - sum_qW + qW[m]
                    log_h_ntm = np.dot( np.dot( residue, self.invSigma ), self.layers[m].WT )
                    log_h_ntm -= halfLambda[m]

                    #===========================================================
                    # if MLEst:
                    #     # sometimes log_h is very small,
                    #     # thus going to local optimum immediately. this is to mitigate
                    #     # the issue
                    #     logh_bound = 3
                    #     # this bounding is before shrinkage, to ensure
                    #     # it could be effected
                    #     for k in xrange( len(log_h_ntm) ):
                    #         if log_h_ntm[k] < -logh_bound:
                    #             log_h_ntm[k] = -logh_bound
                    #         elif log_h_ntm[k] > logh_bound:
                    #             log_h_ntm[k] = logh_bound
                    #===========================================================

                    # if not adding the scale factors, this alg falls back to
                    # Zoubin's original scheme
                    if self.iterDoShrink and not self.MLEst:
                        log_h_ntm += self.layers[m].logStateScaleFactors[isEnd]

                    log_h_ntm = self.normalizeLogH(log_h_ntm)
                    log_h_Mn[m][t] = log_h_ntm
                    h_Mn[m][t] = np.exp( log_h_ntm )

            for m in xrange(self.M):
                hs[m].append( h_Mn[m] )
                loghs[m].append( log_h_Mn[m] )

        return hs, loghs

    # compute the variational parameters h. Eq.(97)
    # the computed h is distributed into M layers.
    # in each layer, dimensions: N, T, K. The same as X
    def firstOrderApproxH( self, X, varDist, oldLoghs ):

        T = np.array( [ x.shape[0] for x in X ], dtype=int )

        halfLambda = []
        V = []
        Ks = self.Ks

        for layer in self.layers:
            # different \Lambda_i has different length.
            # so we couldn't use np.array to store them
            # calculate 1/2 * \Lambda, since these values are repeatedly used
            halfLambda.append( 0.5 * np.diagonal(
                            np.dot( np.dot( layer.W, self.invSigma ), layer.WT ) ) )
            V.append([])

            for layer2 in self.layers:
                V[-1].append( np.dot( np.dot( layer.W, self.invSigma ), layer2.WT ) )

        C1W0 = np.dot( self.invSigma, self.wholeW.T )
        C1W = [ np.dot( self.invSigma, self.layers[m].WT ) for m in xrange(self.M) ]

        qZ = varDist.qZ

        # M elements of [], to store layers.h
        hs = [ [] for m in xrange(self.M) ]
        loghs = [ [] for m in xrange(self.M) ]

        for n, Xn in enumerate(X):
            # allocate the M arrays. numpy.array needs pre-specify the array size
            h_Mn = [ np.zeros( ( T[n], layer.K ) ) for layer in self.layers ]
            log_h_Mn = [ np.zeros( ( T[n], layer.K ) ) for layer in self.layers ]

            for t in xrange( T[n] ):
                # qZ[p][n][t]: 1*Kp, Ws[p]: Kp*D, qW: 1*D
                J = []
                JW = np.zeros(( self.sumK, self.D ))
                v = np.zeros(self.sumK)
                if t < T[n] - 1:
                    isEnd = 0
                else:
                    isEnd = 1

                sumK2 = 0
                for m in xrange(self.M):
                    if qZ[m][n][t].shape[0] != self.layers[m].W.shape[0]:
                        raise RuntimeError("Matrices are not aligned")

                    K_m = Ks[m]
                    J.append( jacobdQzdh(qZ[m][n][t]) )
                    # J^m W^m'
                    JW[ sumK2 : sumK2 + K_m ] = np.dot( J[-1], self.layers[m].W )
                    sumK2 += K_m

                # \mathcal{J} in Eq.(97)
                JV = np.dot( JW, C1W0 )
                sumK2 = 0
                for m in xrange(self.M):
                    K_m = Ks[m]
                    JV[ sumK2 : sumK2 + K_m, sumK2 : sumK2 + K_m ] = np.identity(K_m)
                    sumK2 += K_m

                sumK2 = 0
                # Eq.(97), q'-log h J
                for m in xrange(self.M):
                    q_loghJV = []
                    for p in xrange(self.M):
                        # the summation excludes subscript m
                        if p == m:
                            continue

                        q_loghJ_p = qZ[p][n][t] - np.dot( oldLoghs[p][n][t], J[p])

                        q_loghJV.append( np.dot(q_loghJ_p, V[p][m]) )
                    sum_q_loghJV = np.sum( q_loghJV, axis=0 )

                    K_m = Ks[m]
                    c_m = np.zeros( K_m )
                    if self.iterDoShrink and not self.MLEst:
                        c_m += self.layers[m].logStateScaleFactors[isEnd]
                    c_m += np.dot( Xn[t], C1W[m] )
                    c_m -= halfLambda[m]

                    v[ sumK2 : sumK2 + K_m ] = c_m - sum_q_loghJV
                    sumK2 += K_m

                # Eq.(98)
                log_h_nt = np.dot( v, np.linalg.pinv( JV, 5e-2 ) )

                sumK2 = 0
                for m in xrange(self.M):
                    log_h_Mn[m][t] = self.normalizeLogH( log_h_nt[ sumK2 : sumK2 + Ks[m] ] )
                    try:
                        h_Mn[m][t] = np.exp( log_h_Mn[m][t] )
                    except FloatingPointError:
                        pdb.set_trace()
                    sumK2 += Ks[m]

            for m in xrange(self.M):
                hs[m].append( h_Mn[m] )
                loghs[m].append( log_h_Mn[m] )

        return hs, loghs

    # Caller has to make sure one VStep has been done before calling forward_backward()
    def forward_backward( self, X, varDist, isExact=True ):

        N = len(X)
        T = np.array( [ x.shape[0] for x in X ], dtype=int)

        for m in xrange(self.M):
            layer = self.layers[m]
            if isExact:
                layer.forward_backward( N, T, varDist, m )
            else:
                layer.approxQzUpdate( N, T, varDist, m )

            layer.updateMass(varDist, m)
            layerPruneThres = calcPruneThres( np.sum(T), self.initKs[m], layer.K)
            layer.updateStateScaleFactors( layerPruneThres, self.shrinkRate )

        return varDist

    # Same structured approximation. Finding exact optimal is exponential to M
    def viterbiDecode(self, X=None, varDist=None):
        # decodes the training sequences
        if X is None:
            N = self.N
            T = self.T
        else:
            N = len(X)
            T = [ x.shape[0] for x in X ]
            if varDist is None:
                raise RuntimeError('Variational distribution should be given if X is given')
            self.VStep( X, varDist )

        state_seq = []
        for m in xrange(self.M):
            hm = self.layers[m].h
            layer = self.layers[m]
            stateSeq_m = []

            for n in xrange(N):
                hmn = hm[n]
                log_hmn = np.log(hmn)

                bestProb = np.zeros( (T[n], layer.K) )
                bestTrans = np.zeros( (T[n], layer.K), dtype=int )
                stateSeq_m.append( np.zeros(T[n]) )

                for t in xrange(T[n]):
                    if t == 0:
                        bestProb[0] = np.log( layer.alpha ) + log_hmn[0]
                    else:
                        for k in xrange(layer.K):
                            forwardProb =  bestProb[t - 1] + np.log( layer.beta[:, k] ) + log_hmn[t]
                            bestProb[t, k] = np.max( forwardProb )
                            bestTrans[t, k] = np.argmax( forwardProb )

                t = T[n] - 1
                nextState = stateSeq_m[n][t] = np.argmax( bestProb[t, k] )
                for t in xrange(T[n] - 2, -1, -1):
                    nextState = stateSeq_m[n][t] = bestTrans[ t+1, nextState ]

            state_seq.append( stateSeq_m )
        # for m

        return state_seq

    # if MLEst is set, use ML estimation
    # used when calc'ing predictive log-likelihood
    # in this case we calc -KL(q||p), instead of -KL(q||\tilde{p}).
    # i.e., scaling factors don't participate
    def calcNegKLDiv( self, X, varDist ):

        if self.VStepNum == 0:
            print "Skip -KL calc in the first VM iteration"
            return -np.Inf

        if X is None:
            raise RuntimeError('X could not be None')

        N = len(X)
        T = np.array( [ x.shape[0] for x in X ], dtype=int )

        C1W = []
        WC1W = []
        Lambda = []
        WC1_sump_qW = []

        for layer in self.layers:
            # different \Lambda_i has different length.
            # so we couldn't use np.array to store them
            # K^-1 W, D*Km
            C1W.append( np.dot( self.invSigma, layer.WT ) )
            # W' K^-1 W, Km*Km
            WC1W.append( np.dot( layer.W, C1W[-1] ) )
            # \Lambda = diag{ W' K^-1 W }, column vector, 1*Km
            # instead of the row vector in the paper
            Lambda.append( np.diagonal( WC1W[-1] ) )
            WC1_sump_qW.append( np.zeros( layer.K ) )

        qZ = varDist.qZ
        qZZ = varDist.qZZ

        detSigma = np.linalg.det( self.Sigma )
        logdetSigma = np.log( detSigma )

        #invSigma = np.diag( self.invSigma )

        if self.approxKLCalc:
        # approximate calculation of total_residue
            total_residue = 0
            total_misc = 0
            for n in xrange(N):
                for t in xrange( T[n] ):
                    # q1'W1', q2'W2', ..., each is 1*D
                    qW = [ np.dot( qZ[m][n][t], self.layers[m].W ) for m in xrange(self.M) ]
                    # transpose of \sum_l W^l q^l
                    sum_qW = np.sum( qW, axis=0 )
    
                    xC1x = np.dot( np.dot( X[n][t], self.invSigma ), X[n][t] )
    
                    summ_qWC1_sump_qW = 0
    
                    for m in xrange(self.M):
                        WC1_sump_qW[m] = np.dot( sum_qW - qW[m], C1W[m] )
                        summ_qWC1_sump_qW += np.dot( WC1_sump_qW[m], qZ[m][n][t] )
    
                    residue_nt = summ_qWC1_sump_qW - xC1x
                    total_residue += residue_nt
    
                total_misc -= 0.5 * T[n] * ( self.D * np.log( 2 * np.pi ) + logdetSigma )
    
            sumLogNorm = 0
            for norm_m in varDist.norm:
                for norm_mn in norm_m:
                    sumLogNorm += np.sum( np.log( norm_mn ) )
    
            negKL = 0.5 * total_residue + total_misc + sumLogNorm
    
            print "Approx negKL: %f. residue %f, detSigma %f, sumLogNorm %f" %(negKL, 0.5 * total_residue, detSigma, sumLogNorm)

        # exact calculation of total_residue
        total_residue = 0
        total_misc = 0

        for n in xrange(N):
            for t in xrange( T[n] ):
                # q1'W1', q2'W2', ..., each is 1*D
                qW = [ np.dot( qZ[m][n][t], self.layers[m].W ) for m in xrange(self.M) ]
                # transpose of \sum_l W^l q^l
                sum_qW = np.sum( qW, axis=0 )

                xC1x = np.dot( np.dot( X[n][t], self.invSigma ), X[n][t] )

                summ_qWC1_sump_qW = 0
                sum_zlogh = 0
                sum_zlogscale = 0
                sum_tr_WC1Wq = 0

                if t < T[n] - 1:
                    isEnd = 0
                else:
                    isEnd = 1

                for m,layer in enumerate(self.layers):
                    sum_zlogh += np.dot( qZ[m][n][t], layer.logh[n][t] )
                    if self.iterDoShrink and not self.MLEst:
                        sum_zlogscale += np.dot( qZ[m][n][t], layer.logStateScaleFactors[isEnd] )
                    WC1_sump_qW[m] = np.dot( sum_qW - qW[m], C1W[m] )
                    summ_qWC1_sump_qW += np.dot( WC1_sump_qW[m], qZ[m][n][t] )
                    tr_WC1Wq = np.dot( Lambda[m], qZ[m][n][t])
                    sum_tr_WC1Wq += tr_WC1Wq

                sum_qW_C1 = np.dot( np.sum( qW, axis=0 ), self.invSigma )
                xC1sum_Wq = np.dot( sum_qW_C1, X[n][t] )

                residue_nt = -sum_zlogh + sum_zlogscale - 0.5 * (
                              xC1x - 2 * xC1sum_Wq + summ_qWC1_sump_qW + sum_tr_WC1Wq )

                total_residue += residue_nt
            # end of for t
            total_misc -= 0.5 * T[n] * ( self.D * np.log( 2 * np.pi ) + logdetSigma )

        sumLogNorm = 0
        for norm_m in varDist.norm:
            for norm_mn in norm_m:
                sumLogNorm += np.sum( np.log( norm_mn ) )

        # total_misc & sumLogNorm are the same as the approx calculation
        # total_residue has been halved. no "0.5" coeff here
        negKL = total_residue + total_misc + sumLogNorm

        if self.MLEst:
            print "ML negKL: %f. residue %f, detSigma %f, sumLogNorm %f" %(negKL, total_residue, detSigma, sumLogNorm)
        else:
            print "FAB negKL: %f. residue %f, detSigma %f, sumLogNorm %f" %(negKL, total_residue, detSigma, sumLogNorm)

        # E_q [ log p(x,z|theta) ]
        total_residue = 0
        total_misc = 0
        total_init = 0
        total_transition = 0

        for n in xrange(N):
            for t in xrange( T[n] ):
                # q1'W1', q2'W2', ..., each is 1*D
                qW = [ np.dot( qZ[m][n][t], self.layers[m].W ) for m in xrange(self.M) ]
                # transpose of \sum_l W^l q^l
                sum_qW = np.sum( qW, axis=0 )

                xC1x = np.dot( np.dot( X[n][t], self.invSigma ), X[n][t] )

                summ_qWC1_sump_qW = 0
                sum_zlogh = 0
                sum_zlogscale = 0
                sum_tr_WC1Wq = 0

                for m,layer in enumerate(self.layers):
                    WC1_sump_qW[m] = np.dot( sum_qW - qW[m], C1W[m] )
                    summ_qWC1_sump_qW += np.dot( WC1_sump_qW[m], qZ[m][n][t] )
                    tr_WC1Wq = np.dot( Lambda[m], qZ[m][n][t])
                    sum_tr_WC1Wq += tr_WC1Wq

                sum_qW_C1 = np.dot( np.sum( qW, axis=0 ), self.invSigma )
                xC1sum_Wq = np.dot( sum_qW_C1, X[n][t] )

                residue_nt = - 0.5 * (
                              xC1x - 2 * xC1sum_Wq + summ_qWC1_sump_qW + sum_tr_WC1Wq )

                total_residue += residue_nt

                if t > 1:
                    qZZlogb = [ np.dot( qZZ[m][n][t-1], np.log( self.layers[m].beta ) ) for m in xrange(self.M) ]
                    tr_qZZlogb = [ np.trace( qZZlogb[m] ) for m in xrange(self.M) ]

                    total_transition += np.sum( tr_qZZlogb )

            # end of for t
            qloga = [ np.dot( qZ[m][n][0], np.log( self.layers[m].alpha ) ) for m in xrange(self.M) ]
            total_init += np.sum( qloga )
            total_misc -= 0.5 * T[n] * ( self.D * np.log( 2 * np.pi ) + logdetSigma )
        # end of for n

        Eq_logpxz = total_residue + total_misc + total_init + total_transition
        print "E_q [ log p(x,z|theta) ]: %f" %( Eq_logpxz )

        return negKL

    def calcPartialFIC(self, X):

        N = len(X)
        T = np.array([ x.shape[0] for x in X ], dtype=int)
        layercomplex = 0
        sumdelta = 0
        trivia = 0

        # update layers
        for m in xrange(self.M):
            layer = self.layers[m]
            layercomplex += np.sum( layer.compComplexities )

            if self.iterDoShrink:
                for n in xrange( len(X) ):
                    #FIC += sum(log(varDist.norm[n])) # log p(x_n^1, ..., x_n^T)
                    # \sum log \delta_t^{n,(m)} item in the FIC
                    sumdelta += ( T[n] - 1 ) * np.log( layer.stateScaleFactorsNorm[0] )
                    sumdelta += np.log( layer.stateScaleFactorsNorm[1] )

            # log of the normalization constant of the quadratic form of \alpha
            trivia -= 0.5*( layer.K - 1 ) * np.log(N)

        for n in xrange( len(X) ):
            trivia -= 0.5 * self.D * np.log( 0.5 * T[n] )

        # *** This term is ignored as being asymptotically small ***
        # c_d is the diagonal element of Sigma^-1, so "+" here
        #trivia += 0.5 * (self.sumK - 2) * np.sum( np.log( np.diag(self.Sigma) ) )

        partialFIC = layercomplex + sumdelta + trivia

        print "Partial FIC: %f. Layers complexity: %f, Sum_Delta: %f, Trivia: %f" %(
                                                partialFIC, layercomplex, sumdelta, trivia)

        return partialFIC

    def sample( self, N, T ):
        # samples_states[n,t]: observation, mean, hidden state 1, ..., hidden state M
        samples_means = np.zeros( (N, T, 2, self.D) )
        states = np.zeros( (N, T, self.M), dtype=int )

        for n in xrange(N):
            print "Generating sample sequence {}".format(n)

            mean = 0
            for m in xrange(self.M):
                states[ n,0,m ] = self.layers[m].sampleState()
                mean += self.layers[m].W[ states[ n,0,m ] ]
            samples_means[ n,0,0 ] = np.random.multivariate_normal( mean, self.Sigma )
            samples_means[ n,0,1 ] = mean

            for t in xrange(1,T):
                mean = 0
                for m in xrange(self.M):
                    states[ n,t,m ] = self.layers[m].sampleState( states[ n,t-1,m ] )
                    mean += self.layers[m].W[ states[ n,t,m ] ]
                samples_means[ n,t,0 ] = np.random.multivariate_normal( mean, self.Sigma )
                samples_means[ n,t,1 ] = mean

        return samples_means, states

    def statLogH(self, X, loghs):

        T = np.array( [ x.shape[0] for x in X ], dtype=int )
        maxAbsLogh = 0
        sumAbsLogh = 0

        for m in xrange(self.M):
            for n in xrange( len(loghs[m]) ):
                for t in xrange(T[n]):
                    if maxAbsLogh < np.abs( loghs[m][n][t] ).max():
                        maxAbsLogh = np.abs( loghs[m][n][t] ).max()
                        maxpos = [m,n,t]
                    sumAbsLogh += np.sum( np.abs( loghs[m][n][t] ) )

        return maxAbsLogh, maxpos

    def boundH(self, X, loghs, logUpperbound):
        hs2 = []
        loghs2 = []
        T = np.array( [ x.shape[0] for x in X ], dtype=int )

        for m in xrange(self.M):
            loghs2.append([])
            hs2.append([])
            for n in xrange( len(loghs[m]) ):
                loghs2[m].append( np.zeros( ( T[n], self.Ks[m] ) ) )
                hs2[m].append( np.zeros( ( T[n], self.Ks[m] ) ) )
                for t in xrange(T[n]):
                    loghmnt = np.copy(loghs[m][n][t])
                    bigValueIndices = loghs[m][n][t] > logUpperbound
                    loghmnt[bigValueIndices] = logUpperbound
                    bigNegValueIndices = loghs[m][n][t] < -logUpperbound
                    loghmnt[bigNegValueIndices] = -logUpperbound
                    loghs2[m][n][t] = loghmnt
                    hs2[m][n][t] = np.exp(loghmnt)

        return hs2, loghs2

    #===========================================================================
    # def interpolateH(self, X, loghs0, loghs1, w0):
    #
    #     T = np.array( [ x.shape[0] for x in X ], dtype=int )
    #
    #     hs01 = []
    #     loghs01 = []
    #
    #     w1 = 1 - w0
    #
    #     for m in xrange(self.M):
    #         loghs01.append([])
    #         hs01.append([])
    #         for n in xrange( len(loghs0[m]) ):
    #             loghs01[m].append( np.zeros( ( T[n], self.Ks[m] ) ) )
    #             hs01[m].append( np.zeros( ( T[n], self.Ks[m] ) ) )
    #             for t in xrange(T[n]):
    #                 loghmnt = loghs0[m][n][t] * w0 + loghs1[m][n][t] * w1
    #                 loghs01[m][n][t] = loghmnt
    #                 hs01[m][n][t] = np.exp(loghmnt)
    #
    #     return hs01, loghs01
    #===========================================================================

    def assignLayersH(self, hs, loghs):

        # backup the old values for use in layer.approxQzUpdate()
        for m in xrange(self.M):
            self.layers[m].oldH = self.layers[m].h
            self.layers[m].oldLogh = self.layers[m].logh
            self.layers[m].h = hs[m]
            self.layers[m].logh = loghs[m]

        self.oldHs = self.hs
        self.oldLoghs = self.loghs
        self.hs = hs
        self.loghs = loghs

    def testBound(self, X, varDist, loghs, bound ):
        hs1, loghs1 = self.boundH(X, loghs, bound)
        self.assignLayersH(hs1, loghs1)
        varDist1 = copy.deepcopy(varDist)
        varDist1 = self.forward_backward( X, varDist1 )
        loghResidue1 = self.calcLogHResidue(X, loghs1, varDist1 )
        negKL1 = self.calcNegKLDiv( X, varDist1 )
        print "Bound %d: negKL: %f, log residue: %f" %( bound, negKL1, loghResidue1 )

    def VStep(self, X, varDist, isExact=True):

        self.VStepNum += 1

        if varDist is None:
            N = len(X)
            T = np.array( [ x.shape[0] for x in X ], dtype=int)
            Ks = [ layer.K for layer in self.layers ]
            varDist = VariationalDist( self.M, N, T, Ks )

        bestVarDist = varDist
        bestHs = self.oldHs
        bestLoghs = self.oldLoghs
        bestLoghResidue = np.Inf
        bestNegKL = -np.Inf
        bestMethod = -1
        #bestBound = 0

        logUpperbound = 60
        # use zero order method
        #for logUpperbound in reversed(xrange(20,31,5)):
        #=======================================================================
        # maxAbsLogh, maxpos = self.statLogH(X, loghs0)
        # if maxAbsLogh > 50:
        #     self.zeroOrderApproxH(X, varDist, MLEst)
        #     pass
        #
        #=======================================================================

        if self.iterUseZeroOrderApproxH:
            varDist0 = copy.deepcopy(varDist)
            hs0, loghs0 = self.calcH( X, varDist0, self.loghs, 0 )
            hs0, loghs0 = self.boundH(X, loghs0, logUpperbound)
            self.assignLayersH(hs0, loghs0)
            varDist0 = self.forward_backward( X, varDist0, isExact )
            loghResidue0 = self.calcLogHResidue(X, loghs0, varDist0)
            negKL0 = self.calcNegKLDiv( X, varDist0 )
            print "Zero order residue/negKL: %f, %f." %(
                                        loghResidue0, negKL0)
            print

            if negKL0 >= bestNegKL:
                bestHs = hs0
                bestLoghs = loghs0
                bestVarDist = varDist0
                bestLoghResidue = loghResidue0
                bestNegKL = negKL0
                bestMethod = 0
                #bestBound = logUpperbound

        #=======================================================================
        # # use first order method
        # hs1, loghs1 = self.calcH( X, varDist1, self.loghs, logUpperbound, MLEst, 1 )
        # self.assignLayersH(hs1, loghs1)
        # varDist1 = self.forward_backward( X, varDist1, MLEst )
        # loghResidue1 = self.calcLogHResidue(X, loghs1, varDist1, MLEst)
        #=======================================================================

        if self.iterUseFirstOrderApproxH and isExact:
            varDist1 = copy.deepcopy(bestVarDist)
            hs1, loghs1 = self.calcH( X, varDist1, bestLoghs, 1 )
            #for logUpperbound in reversed(xrange(20,31,5)):
            hs1, loghs1 = self.boundH(X, loghs1, logUpperbound)
            #===================================================================
            # maxAbsLogh, maxpos = self.statLogH(X, loghs1)
            # if maxAbsLogh > 50:
            #     self.firstOrderApproxH(X, varDist1, bestLoghs, MLEst)
            #     pass
            #===================================================================

            self.assignLayersH(hs1, loghs1)
            varDist1 = self.forward_backward( X, varDist1, isExact )
            loghResidue1 = self.calcLogHResidue(X, loghs1, varDist1)
            negKL1 = self.calcNegKLDiv( X, varDist1 )
            print "First order residue/negKL: %f, %f." %(
                                        loghResidue1, negKL1)
            print

            if negKL1 >= bestNegKL:
                bestHs = hs1
                bestLoghs = loghs1
                bestVarDist = varDist1
                bestLoghResidue = loghResidue1
                bestNegKL = negKL1
                #bestBound = logUpperbound
                bestMethod = 1

        if self.dumpAbnormal and self.VStepNum > 1 and abs(self.VStepNum - self.noshrinkRounds) > 1 \
                and not self.mstepPruned and bestNegKL < self.lastNegKL - 30:
            maxAbsLogh, maxpos = self.statLogH(X, bestLoghs)
            self.saveParams(0)
            self.loadParams(1)
            oldNegKL2 = self.calcNegKLDiv( X, varDist )
            self.loadParams(0)
            saveModel(self, "abnormal.bin")
            #exit()

    #===========================================================================
    #     varDist01 = copy.deepcopy(bestVarDist)
    #     for w0 in xrange(2,8,1):
    #         w0 /= 10.0
    #         hs01, loghs01 = self.interpolateH(X, loghs0, loghs1, w0)
    #         self.assignLayersH(hs01, loghs01)
    #         varDist01 = self.forward_backward( X, varDist01, MLEst )
    #         loghResidue1 = self.calcLogHResidue(X, loghs01, varDist01, MLEst)
    #         negKL = self.calcNegKLDiv( X, varDist01, MLEst )
    #         print "%.1f order residue/negKL: %f, %f." %(
    #                                     w0, loghResidue1, negKL)
    #
    #         if negKL > bestNegKL:
    #             bestHs = hs01
    #             bestLoghs = loghs01
    #             bestVarDist = varDist01
    #             bestLoghResidue = loghResidue1
    #             bestNegKL = negKL
    #             #bestBound = logUpperbound
    #             bestMethod = w0
    #===========================================================================

        if self.iterUseZeroOrderApproxH and self.iterUseFirstOrderApproxH and isExact:
            print "Best method: %.1f. Residue/negKL %f, %f.\n" %(
                            bestMethod, bestLoghResidue, bestNegKL)

        hs, loghs, varDist = bestHs, bestLoghs, bestVarDist
        self.assignLayersH(hs, loghs)

        negKL = bestNegKL
        self.lastNegKL = negKL

        #=======================================================================
        # all_hs = []
        # all_loghs = []
        # kls = []
        # old_sum_logh, hs, loghs = self.calcH( X, varDist, MLEst )
        # # collect them, and then find the one that minimizes KL div
        # all_hs.append( hs )
        # all_loghs.append( loghs )
        #
        # kls.append( negKL )
        #=======================================================================

        # *** the extra iterations are disabled. they don't bring much improvement ***
        # set to 0 to disable this iteration
        # at MLE stage, do refined optimization for h
        #=======================================================================
        # if MLEst:
        #    print "H iter 1 sum_logh: %f" %( old_sum_logh )
        #    maxIter = 10
        # else:
        #    maxIter = 0
        #=======================================================================

        #=======================================================================
        # maxIter = 0
        #
        # hs = []
        #
        # for i in xrange(maxIter):
        #     sum_logh, hs, loghs = self.calcH( X, varDist, MLEst )
        #     all_hs.append( hs )
        #     all_loghs.append( loghs )
        #     varDist = self.forward_backward( X, varDist, MLEst )
        #     varDist, negKL = self.calcNegKLDiv( X, varDist, MLEst )
        #     kls.append( negKL )
        #
        #     diff = abs( sum_logh - old_sum_logh )
        #     print "H iter %d sum_logh: %f, diff: %f" %( i+2, sum_logh, diff )
        #     if diff < sum_T * self.M * 1e-08:
        #         break
        #     if diff > 50:
        #         pass
        #     old_sum_logh = sum_logh
        #
        # # disabled when no extra iterations are executed
        # if len(kls) > 1:
        #     best_i = np.argmax( kls )
        #     hs = all_hs[best_i]
        #     loghs = all_loghs[best_i]
        #     # distribute them back to layers.h & layers.logh
        #     for m in xrange(self.M):
        #         self.layers[m].h = hs[m]
        #         self.layers[m].logh = loghs[m]
        #     print "Choose best iter %d, negKL: %f" %( best_i + 1, kls[best_i] )
        #     varDist = self.forward_backward( X, varDist, MLEst )
        #     varDist, negKL = self.calcNegKLDiv( X, varDist, MLEst )
        #=======================================================================

        # we have to return varDist, since sometimes this arg is set to None
        # and a new varDist is generated
        return varDist, negKL

    def normalizeW(self, i):

        self.saveParams(0)

        sum_row1_ex_M = np.zeros(self.D)
        sumK2 = 0

        self.wholeW = np.zeros( (self.sumK, self.D) )

        for m in xrange(self.M):
            layer = self.layers[m]
            K_m = self.Ks[m]

            if m < self.M - 1:
                sum_row1_ex_M += layer.W[0]
                layer.W = layer.W - layer.W[0]
            else:
                layer.W = layer.W + sum_row1_ex_M

            layer.WT = layer.W.T
            self.wholeW[ sumK2:sumK2+K_m ] = np.copy( layer.W )
            sumK2 += K_m

        self.saveParams(i)
        self.loadParams(0)

    def MStep( self, X, varDist ):

        N = len(X)
        T = np.array([ x.shape[0] for x in X ], dtype=int)

        # save varDist in a member variable, so that it will be saved by pickle
        self.varDist = varDist
        self.saveParams(1)

        self.mstepPruned = False

        qZ = varDist.qZ
        qZZ = varDist.qZZ
        forward = varDist.forward
        backward = varDist.backward

        # whether shrinkage happens during the iteration below
        iterPruned = False

        for m in xrange(self.M):
            # shrinkage operation
            layer = self.layers[m]
            layer.updateMass(varDist, m)
            h = layer.h
            logh = layer.logh

            layerPruneThres = calcPruneThres( np.sum(T), self.initKs[m], layer.K)

            if self.doPrune:
                # update logStateScaleFactors. remove states whose mass < self.pruneThres
                active = layer.updateStateScaleFactors( layerPruneThres, self.shrinkRate )
            else:
                # all states are kept
                active = layer.updateStateScaleFactors( 0, self.shrinkRate )

            print "%d mass : %s" %( m, str( layer.mass ) )
            print "%d scale: %s" %( m, str( layer.logStateScaleFactors[0] ) )
            
            oldLayerK = layer.K
            layer.K = np.sum(active)
            pruned = (layer.K != oldLayerK)
            if pruned:
                # update varDist. keep only active submatrices
                # varDist: M, N, T, K
                for n in xrange(N):
                    unprotect( qZ[m][n], qZZ[m][n] )

                    qZ[m][n] = qZ[m][n][:,active]
                    qZZ[m][n] = qZZ[m][n][:,active][:,:,active]
                    # this is just to reshape the forward and backward arrays
                    # the values are always temporary in f-b alg
                    # and useless elsewhere
                    forward[m][n] = forward[m][n][:,active]
                    backward[m][n] = backward[m][n][:,active]

                    # h[n][t]
                    h[n] = h[n][:,active]
                    logh[n] = logh[n][:,active]

                    for t in xrange( T[n]-1 ):
                        qZZ[m][n][t] = normalize( qZZ[m][n][t] )
                        qZ[m][n][t] = np.sum( qZZ[m][n][t], axis=1 )

                    qZ[m][n][ T[n]-1 ] = np.sum( qZZ[m][n][ T[n]-2 ], axis=0 )

                    protect( qZ[m][n], qZZ[m][n] )
                # end of inner for
                # update mass, endingMass
                layer.updateMass(varDist, m)

                layerPruneThres = calcPruneThres( np.sum(T), self.initKs[m], layer.K)
                print "Layer %d prune threshold adjusted to %f" %(m, layerPruneThres)
                # update logStateScaleFactors and stateScaleFactorsNorm
                active = layer.updateStateScaleFactors( layerPruneThres, self.shrinkRate )
                iterPruned = True
                self.mstepPruned = True
                layer.oldH = layer.h
                layer.oldLogh = layer.logh
        # end of outer for

        self.Ks = [ layer.K for layer in self.layers ]

        self.sumK = np.sum( self.Ks )

        # update Sigma, and W of all layers
        self.updateSigmaW( X, varDist )

        if self.doNormalizeW:
            # normalized W overwrites the active W
            self.normalizeW(0)
        else:
            # only store the normalized W
            self.normalizeW(2)

        # update layers
        for m in xrange(self.M):
            layer = self.layers[m]
            # update \alpha and \beta for each layer
            # also layer.compComplexities
            layer.MStep( varDist, m )

        partialFIC = self.calcPartialFIC(X)

        print "Sigma:"
        for d in xrange(self.D):
            print str(self.Sigma[d])

        for m,layer in enumerate(self.layers):
            for k in xrange(layer.K):
                print "W{}-{}: {}".format( m, k, layer.W[k] )

        print

        return partialFIC

    def iniVarDist(self, X, Ks):
        N = len(X)
        # elements in T are the lengths of different sequences
        T = np.array([ x.shape[0] for x in X ], dtype=int)

        if type(Ks) == int:
            return VariationalDist( self.M, N, T, [Ks]*self.M, "dirichlet" )
        return VariationalDist( self.M, N, T, Ks, "dirichlet" )
