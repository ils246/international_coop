import time
import numpy as np
from scipy.stats import pearsonr
# from ThreadedMap import staticMap

def createLMj(IN,sharedIn,sharedOut):
    ID,NUM,Xm,Ym,N,tau,E,LMN = IN
    partial_LMj= np.zeros((2,2,N))
    for j in range(ID,N,NUM):
        # neighborhood search
        # We are interating over each xm, and finding the nn for that xm
        n1,d1 = neighborSearch(Xm[j,:],Xm,num=N)
        n2,d2 = neighborSearch(Ym[j,:],Ym,num=N)

        # LM

        LMn1=n1[(n1 != j)]
        LMn2=n2[(n2 != j)]
        LMd1=d1[(n1 != j)]
        LMd2=d2[(n2 != j)]

        susXY = [np.arange(len(LMn1))[(LMn1 == x)][0] for x in LMn2[:LMN]]
        susYX = [np.arange(len(LMn2))[(LMn2 == x)][0] for x in LMn1[:LMN]]

        sum1=np.sum(LMd1)/(N-1);
        sum2=np.sum(LMd2)/(N-1);

        partial_LMj[:,:,j] = [[(N/2-np.sum(susXY)/LMN)/(N/2-(LMN+1)/2) ,
                           (sum1-np.sum(LMd1[susXY])/LMN)/(sum1-np.sum(LMd1[:LMN])/LMN)],
                      [(N/2-np.sum(susYX)/LMN)/(N/2-(LMN+1)/2) ,
                            (sum2-np.sum(LMd2[susYX])/LMN)/(sum2-np.sum(LMd2[:LMN])/LMN)]]
    sharedOut[ID] = partial_LMj


def neighborSearch(v,M,num):
    # compares euclidean distance between rows of M
    # and vector v. Returns num vectors ordered in
    # increasing distance to v.
    #distances = np.array([distFunct(v,M[i,:]) for i in range(M.shape[0])])
    #print(distances)
    distances = np.linalg.norm(np.tile(v,(M.shape[0],1))-M,  axis=1)
    order = np.argsort(distances)
#     if num is None:
#         num = len(distances)
    return order[:num],distances[order[:num]]


def parCorrs(IN,sharedIn,sharedOut):

    """
    Maybe computes partial correlation of each ts??
    """
    ID,NUM,dat,N,Xm,Ym,SugiN,X,Y,T,verbose,directed = IN
    partial_SugiX=np.zeros(N)
    partial_SugiY=np.zeros(N)
    for ii in range(dat+ID,N,NUM):
        # reproduce Ym based on Xm....no?
        n2s,d2s = neighborSearch(Ym[ii,:],Ym[(ii-dat):ii,:],num=SugiN)

        u2s=np.exp(-d2s/d2s[0])
        w2s=np.nan_to_num(u2s/np.sum(u2s)) # calculated from Y
        x = np.array([X[z] for z in n2s+T-1+ii-dat])
        # x = X[n2s+T-1+ii-dat]
        partial_SugiX[ii]= np.dot(w2s,x) #np.dot(np.nan_to_num(w2s),x)

        if not directed:
            n1s,d1s = neighborSearch(Xm[ii,:],Xm[(ii-dat):ii,:],num=SugiN)
            u1s=np.exp(-d1s/d1s[0])
            w1s=np.nan_to_num(u1s/np.sum(u1s)) # calculated from X
            y = np.array([Y[z] for z in n1s+T-1+ii-dat])
            # y = Y[n1s+T-1+ii-dat]
            partial_SugiY[ii]= np.dot(w1s,y) #np.dot(np.nan_to_num(w1s),y)
        sharedOut[ID] = (partial_SugiX,partial_SugiY)

def CCM(X,Y,tau=1,E=2,LMN=None,numPar=1,reportLM=False,verbose=False,directed=False):
    # Input:
    # tau: step size in lagged time series
    # E: maximum number of points to include in lagged time series
    # LMN: number of neighbors for L and M methods. (defaults to E+1)
    #
    # Outputs:
    # SugiCorr - correlation between the CCM estimation of original data and original data the important thing is sugicorr because is thde Y one
    # Ps are the p values of the pearson correlation
    # SugiR    - sqrt((sum((originaldata-CCMestimaleddata).^2)/numel(originaldata)))/std(origaldata)
    # LM - results for L and M methods
    # SugiY, SugiX - the CCM estimate of original data
    # origY, origX - original data
    # sugiR - the error of the timeseries with themselves so sugix with original x and sugiy with originaly

    # defaults from Sugi paper
    # http://www.nature.com/articles/srep14750
    # http://science.sciencemag.org/content/338/6106/496
    if LMN is None:
        LMN = E+1

    # size of time series data
    L=len(X)
    # size of shadow manifold
    T = 1+(E-1)*tau
    # number of shadow manifolds that can be made with data available
    N = L-T
    # minimum number of points needed for a bounding simplex in an E-dimensional space
    SugiN=E+1

    dat=int(np.floor(N/2.0))
    if verbose:
        print("tau = %d, E = %d, LMN = %d, N = %d, L = %d, dat = %d" % (tau,E,LMN,N,L,dat))
    Xm=np.zeros((N,E))
    Ym=np.zeros((N,E))

    ## RECONTRUCTIONS OF ORIGINAL SYSTEMS
    if verbose:
        tic = time.clock()
    count = 0
    for t in range(1+(E-1)*tau,L):
        Xm[count,:] = X[t-(E-1)*tau:t+tau:tau]
        Ym[count,:] = Y[t-(E-1)*tau:t+tau:tau]
        count += 1
    Xm = np.fliplr(Xm)
    Ym = np.fliplr(Ym)
    if verbose:
        print("manifold creation time: %f" % (time.clock()-tic))

    LMj= np.zeros((2,2,N))

    SugiX=np.zeros(N)
    SugiY=np.zeros(N)

    if reportLM:
        if numPar > 1:
            OUT = staticMap(createLMj,[(i,numPar,Xm,Ym,N,tau,E,LMN) for i in range(numPar)],{}).values()
        else:
            OUT = {}
            createLMj((0,1,Xm,Ym,N,tau,E,LMN),{},OUT)
            OUT = OUT.values()
        for out in OUT:
            LMj += out


    if verbose:
        start = time.time()
    if numPar > 1:
        OUT = staticMap(parCorrs,[(i,numPar,dat,N,Xm,Ym,SugiN,X,Y,T,verbose,directed) for i in range(numPar)],{}).values()
    else:
        OUT = {}
        parCorrs((0,1,dat,N,Xm,Ym,SugiN,X,Y,T,verbose,directed),{},OUT)
        OUT = OUT.values()
    for out in OUT:
        #the reconstruction of X based on CCM
        SugiX += out[0]
        SugiY += out[1]
    if verbose:
        print("Calculation time: %f" % (time.time()-start))


    origX=X[T:]
    origX=origX[dat:N]
    SugiX=SugiX[dat:N]


    # We want it not to be directed because this returns the sugicorr
    if not directed:
        origY=Y[T:]
        origY=origY[dat:N]
        SugiY=SugiY[dat:N]
        # maybe change calculation to include p-values
        SugiR = np.zeros(2)
        # how well does Y|M_x reproduce Y?
        #SugiCorr1 = np.corrcoef(origY,SugiY);
        C1,p1 = pearsonr(origY,SugiY)
        SugiCorr = np.zeros(2)
        Ps = np.zeros(2)
        SugiCorr[1] = C1
        Ps[1] = p1#SugiCorr1[0,1]

        # how well does X|M_y reproduce X?
        #SugiCorr2 = np.corrcoef(origX,SugiX);
        C2,p2 = pearsonr(origX,SugiX)
        SugiCorr[0] = C2
        Ps[0] = p2#SugiCorr2[0,1]
        #     SugiR = np.zeros(2)
        #     SugiCorr = [] #np.zeros(2)
        #     R,p = pearsonr(origX,SugiX)
        #     SugiCorr.append((R,p))
        #     R,p = pearsonr(origY,SugiY)
        #     SugiCorr.append((R,p))

        # alternatively, what is the error between the time series?
        SugiR[1] = np.sqrt(np.sum(((origY-SugiY)**2))/len(origY)) / np.std(origY)
        SugiR[0] = np.sqrt(np.sum(((origX-SugiX)**2))/len(origX)) / np.std(origX)
        LM = [x for x in np.mean(LMj,axis=2)]# if len(x) > 1]

        # return SugiCorr,SugiR,Ps,LM,SugiX,SugiY,origX,origY
        return SugiCorr,Ps
    else:
        SugiR = np.sqrt(np.sum(((origX-SugiX)**2))/len(origX)) / np.std(origX)
        R,p = pearsonr(origX,SugiX)
        return R,SugiR,p,SugiX,origX
#
# X = range(1,200)
# Y = range(100,500)
# tau=1
# E=20

# print(CCM(X,Y,1,E,LMN=None,numPar=1,reportLM=False,verbose=False,directed=False))


if __name__ == "__main__" :

    X = range(1,200)
    Y = range(100,500)
    tau=1
    E=20
    print(CCM(X,Y,1,E,LMN=None,numPar=1,verbose=False,directed=False))
