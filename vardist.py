import numpy as np

class VariationalDist:
    def __init__(self, M, N, T, Ks, distType="rand", dirichletAlpha=0.8):
        # T is an array of sequence lengths
        self.qZ = []
        self.qZZ = []
        self.forward = []
        self.backward = []
        self.norm = []
        self.oldQz = [ [] for m in xrange(M) ]
        
        for m in range(M):
            qZm = []
            qZZm = []
            forward_m = []
            backward_m = []
            norm_m = []
            K = Ks[m]
            
            # we have to append the variational distributions of each m and n 
            # one by one, because later after shrinkage at different layers,
            # layer.K will be different, and np.array couldn't have different 
            # shapes on different sub-arrays. But under the same layer m and sequence n, 
            # sub-arrays have the same shape. So the first two dimensions are m and n
            for n in range(N):
                qZm.append( np.zeros(( T[n], K )) )
                qZZm.append( np.zeros(( T[n]-1, K, K )) )
                forward_m.append( np.zeros(( T[n], K )) )
                backward_m.append( np.zeros(( T[n], K )) )
                # avoid log(0) when doing the first M-step before the first V-step
                norm_m.append( np.ones(( T[n] )) )
                
                qZZmnt = np.random.rand( K, K )
                # for algorithm test only.
                qZZm[n][0] = qZZmnt / np.sum(qZZmnt)
                qZm[n][0] = np.sum(qZZm[n][0], axis=1)
                                
                for t in xrange( 1, T[n] - 1 ):
                    # all states are initialized with equal probabilities.
                    # for algorithm test only. should not be used in applications
                    if distType == "uniform":
                        qZZmnt = np.ones( (K, K) )
                    else:
                        if distType == "dirichlet":
                            for i in xrange(K):
                                qZZmnt[i] = np.random.dirichlet( [ dirichletAlpha ] * K )
                        else:
                            qZZmnt = np.random.rand( K, K )
                        if distType == "dupRand" and K > 2:
                            qZZmnt[:,0] = qZZmnt[:,1]

                    for i in xrange(K):
                        # the i-th row of qZZm[n][t] is normalized to have the same sum as 
                        # the i-th column of qZZm[n][t-1] (both are equal to qZm[n][t])
                        # to make the probabilities consistent between consecutive time
                        qZZm[n][t][i] = qZZmnt[i] * np.sum(qZZm[n][t-1][:,i]) / np.sum(qZZmnt[i])
                    qZm[n][t] = np.sum(qZZm[n][t], axis=1)
                
                qZm[n][ T[n]-1 ] = np.sum( qZZm[n][ T[n]-2 ], axis=0 )
                
            self.qZ.append(qZm)
            self.qZZ.append(qZZm)
            self.forward.append( forward_m )
            self.backward.append( backward_m )
            self.norm.append( norm_m )
