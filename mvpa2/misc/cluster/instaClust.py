#!/usr/bin/env python
from mvpa2.suite import *
import sys
from mvpa2.measures.RSA import dsm
from scipy.cluster.hierarchy import *
from scipy.spatial.distance import *
from scipy import stats
import time
import numpy as numpy
import scipy as scipy
import gc
import pickle
from joblib import Parallel,delayed
from subprocess import call

def save_to_csv(mat,prefix='data',delim=",",suffix='.csv'):
    x,y = mat.shape
    fn=prefix+suffix
    f = open(fn,'w')
    
    for i in range(x):
        line=''
        for j in range(y):
            line = line + '%s%s'%(mat[i,j],delim)
        line = line[:len(line)-1]
        f.write(line+"\n")
    f.close()
        

def get_xsub_corr_for_all_nodes(data):
    """Return the 3 X n-features cross-subject correlation data. The first
    row is the mean cross subject correlation, followed by the t-statistic for
    mean r > 0, and the p-values. Iternally this funection iterates over all
    features calling get_xsub_corr_for_node.
    """
    data = np.array(data)
    nsubs,nnod,ldist = data.shape
    cors = np.zeros((3,nnod))
    for n in range(nnod):
        if n%100==0:
            sys.stdout.write("Calculating cross-subject correlations: %s/%s \r"%(n,nnod))
            sys.stdout.flush()
        cors[:,n] = get_xsub_corr_for_node(data,n)
    cors[np.isnan(cors)] = 0
    print '\n'
    return cors

def get_xsub_corr_for_node(data,node,mean_sample=True):
    """Returns the cross-subject correlation for patterns associated with a
    specified feature ("node"), where subjects are stacked on the first
    (samples, x) dimension, features on the second (y), and patterns on the third (z)
    dimension. Returns: (1) mean cross-subject correlation (when mean_sample==True, 
    default), (2) the cross-subject t-score for for r > 0 (dof = nsubjects-1), and 
    (3) the p-value associated with the t-stat. When mean_samples == False, the
    first return value is an n-subjects length list of containing the
    mean correlation for each subject's pattern with that of every other
    subject.
    """
    data = np.array(data)
    data = data[:,node,:]
    m,n = data.shape
    cors =[]
    for i in range(m):
        d = data[np.array(range(m))!=i,:]
        cor = np.corrcoef(d,data[i,:])
        cors.append(np.mean(cor[m-1,0:m-1]))
    tt = stats.ttest_1samp(cors,0)
    if mean_sample:
        cors = np.mean(cors)
    return cors,tt[0],tt[1]

def save_masks(data,prefix='ico32_lh_',suffix='_instaclust_masks.niml.dset'):
    fn = prefix+"th%s"%th+suffix
    ds = dict(data=data,node_indices=np.array(range(len(data))))
    print 'saving masks in %s'%fn
    afni_niml_dset.write(fn,ds,form='text')
        
      
def node_frequencies_over_thr_range(I,start=5,stop=10.5,step=.5,overlap=False):
    results = {}
    for i in np.arange(start,stop,step):
        masks,nodes = I.get_cluster_masks(i,xsubp=.0001,overlap=overlap)
        for n in nodes:
            if results.has_key(n):
                results[n] = results[n] + 1
            else:
                results[n] = 1
    return results

def get_patterns_for_nodes(ds,nodes):
    """Return the patterns for a list of nodes.
    """
    ds = np.array(ds)
    dists = None
    for node in nodes:
        if dists is None:
            dists = ds[:,node,:]
        else:
            dists = np.vstack((dists,ds[:,node,:]))
    return dists


def get_connectivity_matrices(ds):
    """For a data set of shape n-subjects X n-voxels X n-pattern-dimensions
    return Dataset containing an n-subjects X n-voxels X n-voxels set of
    connectivity matrices. Based on the feature-wise pairwise correlations
    between pattern vectors. 
    """
    data = np.array(ds)
    nsubs,nnod,ldist = data.shape

    print "allocating a lot of memory..."
    cm = numpy.zeros((nsubs,nnod,nnod),dtype='float32')
    
    print "so far so good..."
    for i in range(nsubs):
        sys.stdout.write("Computing connectivity matrix %s of %s \r"%(i+1,nsubs))
        sys.stdout.flush()
        cm[i,:,:] = np.float32(np.corrcoef(data[i,:,:]))
    return Dataset(cm)

def get_block_indices(n,nblocks):
    """Helper function to break up n-features into blocks.
    Returns a list of n start and end indices.
    """
    blocks = []
    block_size = n/nblocks
    for i in range(nblocks-1):
        blocks.append((i*block_size,i*block_size+block_size))
    blocks.append(((nblocks-1)*block_size,n))
    return blocks

def target_featurewise_correlation(group_ds,target):
    """For an n-subjects X n-features X n-pattern-dimension Dataset, this function
    returns the n-subjects X n-features Dataset of feature-wise
    correlations with a target model of length n-pattern-dimensions
    """
    x,y,z = group_ds.shape
    result = np.zeros((x,y))
    for s in range(x):
        for f in range(y):
            result[s,f] = np.corrcoef(group_ds.samples[s,f,:],target)[0,1]
    result[np.isnan(result)] = 0.
    return Dataset(result)

def group_ttest(ds,null_hyp=0):
    """For a Datatset with n-subjects X n-features X n-pattern-dimension
    (optoinal), this function returns 2 X n-features X n-dimension (if exists)
    Dataset where the (sample) volume is the t-statistic and the second volume
    is the p statistic computed across subjects (axis=0) for a group result for
    all values significantly different from null-hypothesis value null_hyp (default = 0).
    """
    ds = np.array(ds)
    if len(ds.shape)==2:
        m,n = ds.shape
        results = np.zeros((3,n))
        tt,p = scipy.stats.ttest_1samp(ds,null_hyp,axis=0)
        results[0,:] = np.mean(ds,0)
        results[1,:] = tt
        results[2,:] = p
    else:
        m,n,y = ds.shape
        results = np.zeros((3,n,y))
        tt,p = scipy.stats.ttest_1samp(ds,null_hyp,axis=0)
        results[0,:,:] = np.mean(ds,0)
        results[1,:,:] = tt
        results[2,:,:] = p
    return Dataset(results)
       
def get_tt(data,nblocks=1000,ncpus=8,parallel=False):
    """In order compute a group t-statistics on large 3-d dataset, it is
    necessary to break up the job into smaller pieces and then put the pieces
    back together. This is what this function does, calling 'group_ttest' block
    by block. You can specify the number of pieces (nblocks) and the number of
    cpus (ncpus) which is only used if parallel is True.
    """
    data = np.array(data)
    x,y,z = data.shape
    tt = np.float32(np.zeros((y,z)))
    tp = np.float32(np.zeros((y,z)))
    ncpus = np.min((nblocks,ncpus))
    blocks = get_block_indices(z,nblocks)
    if parallel:
        results = Parallel(n_jobs=ncpus)(delayed(group_ttest)(data[:,:,c[0]:c[1]]) 
                            for c in blocks)
    else:
        results = []
        for i,c in enumerate(blocks):
            sys.stdout.write("Calculating t-tests block %s of %s\r"%(i+1,nblocks))
            sys.stdout.flush()
            results.append(group_ttest(data[:,:,c[0]:c[1]]))
    for i,c in enumerate(blocks):
        z = len(range(c[0],c[1]))
        tt[:,c[0]:c[1]] = np.float32(results[i].samples[0,:,:].reshape(y,z))
        tp[:,c[0]:c[1]] = np.float32(results[i].samples[1,:,:].reshape(y,z))

    # Now fix up any unwanted infinities and NaNs 
    tp[np.diag_indices(len(tp))] = 0.
    tp[np.isnan(tp)] = 1. 
    tt[numpy.isnan(tt)] = 0
    tt[numpy.isinf(tt)] = 0
    tt[np.diag_indices(len(tt))] = np.max(tt,1)
    return tt,tp

def get_min_max_range_for_plotting(ds,I_vol=0,T_vol=1,T=3.1,rnd=2,nothresh=False):
    """
    Helper function to get proper range for plotting functional maps.

    Parameters
    ----------
    ds : dataset to be plotted
    I_vol : Intensity volume contains the values to be plotted by color (default=0)
    T_vol : Threshold volume contains the measure to use for thresholding (default=1)
    T : threshold value (default=3.1 corresponds to T(11) = 3.1, p< .01)
    rnd : number of digits to round results to (default = 2)

    Returns
    -------
    (mn,mx) values for minimum and maximum for range to be used to set colorbar

    """
    ds = np.array(ds)
    if not nothresh:
        ds_I = ds[I_vol,ds[T_vol,:]>T]
    else:
        ds_I = ds[I_vol,:]
    return np.round(np.min(ds_I),rnd),np.round(np.max(ds_I),rnd)



def fdr(P,q):
    """Given an array of P values from multiple tests
    """
    c_V = sum(1./np.arange(1.,len(P)+1))
    th = (np.arange(1.,len(P)+1)*q)/(len(P)*c_V)
    a = np.sort(P)
    for i in range(len(P)):
        if a[i]>th[i]: break
    return a[i-1]

def group_insta_clust(ds,th=90,crit='pct',overlap=False,
                score='xsub_max',xsubp=.0001,min_nodes=100,max_leftover=100,econ=True):
    """
    "InstaClust" Clustering algorithm useful for creating common group ROIs.

    The InstaClust algorithm uses the statistical (T) map based on the 
    group analysis of connectivity maps for  sataset (ds) which contains all 
    subjects (aligned in a common space) to select a set of clusters based on 
    the seed features with the highest 'scores'. 
    
    Scores are defined by default as the mean cross-subject
    correlation between data vectors at each feature (see param "score"). For 
    example, in the present case, the mean correlation across subjects for the
    searchlight dissimiliarty matrix at each node. Data vectors can be from any 
    source (i.e., raw-time series, Beta coefficients, etc.). 

    At each iteration, the feature with the highest score becomes the "seed" for
    the group connectivity analysis. All features above the threshold criterion
    based on group T-map are selected for inclusion in the cluster for that
    iteration. By default, the threshold is the top 10 percent of elgible
    (unclustered) features (see param "th").

    After a feature has been clustered it is no longer eligble to be used as a
    seed. By default, clusters also may not overlap, i.e., no two clusters
    may share a common feature (see param "overlap").


    Parameters
    ----------
    ds : Dataset 
        This is a 3-dimensional dtaset with subjects X features X feature-specific 
        patterns (e.g.searchlight DSMs)
    th : int or float
        Threshold to be applied in calculating the scores for each node
    crit : str 
        The criterion for thresholding the tmap to form clusters.
        Options:
            'pct' (default) for percentile, thresholds the tmap for a chosen
                seed keeping just the features with score >= the score at the top th
                percentile (where 0 < th 100)
            't' Threshold the tmap using the t-values >= th (some t-value)
            'p' Threshold tmap using p-values <= th (0 < th < 1)
            'q' Threshold using false discover rate as calculated by fdr
                function with pmap and th as input
    overlap : bool 
        If False (default) clusters may not overlap, else, clusters may overlap
    singlevol : bool
        If True (default) return results in a single volume. If overlap is true
        singlevol will be ignored and multiple volumes are returned.
    score : str
        Method used to determine the score for each feature. At every iteration,
        the feature with the maximum score is used as a seed for next cluster.
        Options:
            'xsub_max' (default) 
                This option chooses the feature with the maximum across-subject
                correlation value in xsub_map
            'instacorr_max'
                Chooses the feature with the highest cumulative sum of the
                t-values associated with it, i.e., the feature with the highest
                overall connectivity. Does not depend on common activation
                profiles across-subjects  
    xsubp : float 
        Maximum p-value for a node to be considered as a possible seed
        (default: 0.0001). 
    min_nodes : int 
        The minumim number of nodes a cluster needs to have. (default: 100)
    max_leftover : int
        Stopping criterion. When the number of unclustered features falls below
        this number the algorithm returns, leaving the rest of the features
        unclustered. (default: 100)
    econ : Boolean
        If True (default) returns "economy size" results with just the dataset 
        containing clusters. If False returns the large matrices containing the
        statistics for the group analysis on connectivity matrices ('tmap' and
        'pmap') as well as the map of mean cross subject pattern correlations 
        for each node ('xsub_corr'). 

    Returns
    -------
    Dataset
        The dataset returned contains a binary samples matrix with one cluster
        mask per row, and where the columns are the features (nodes,voxels). The
        indices of the seeds corresponding to each cluster are stored as the sample
        attributes "seeds". Only this return when option 'econ' is True
        (default).

    or

    (Dataset, tmap, pmap, xsub_map)
        When option 'econ' if False, returns in addition to the clusters, the
        statistical maps needed to perform analysis. One may save tmap, pmap, 
        and xsub_map as eponymous feature attributes in the original dataset. 
        Calling the function again with these values saved thus in the dataset 
        will speed up subsequent analyses. 


    N.B.:
    Although the running duration of this function is somewhat longer than instant, the
    name derives from the group "InstaCorr" functionality built into the
    AFNI/SUMA (Cox,1996) software packages for exploring functional connectivity maps. In
    particular the algorithm instantiated here was inspired by the 3dGroupInCorr
    program that instantiates the group-instantaneous correlation mapping
    tool. However, the relationship between the AFNI/SUMA software and this software
    ends there. The authors of AFNI/SUMA should not be held responsible for the
    functionality (or lack thereof) of the present software. The name given to
    this function is an homage to the awesomeness of InstaCorr, and we
    hereby profusely thank Bob Cox and Ziad Saad providing us with it. For more
    information on InstaCorr check out AFNI/SUMA. And:
    http://afni.nimh.nih.gov/pub/dist/edu/latest/afni_handouts/instastuff.pdf

    References
    ----------
    Robert R. Cox (1996) AFNI: Software for Analysis and Visualization of
        Functional Magnetic Resonance Neuroimages. Computers and Biomedical Research 
        vol. 9(3), pp.162-173. 
        http://dx.doi.org/10.1006/cbmr.1996.0014 
        http://afni.nimh.nih.gov/
 


 
    """
    if ds.fa.has_key('tmap') and ds.fa.has_key('pmap'):
        print "<> Using stored pmap and tmap from feature attributes of DS."
        tmap = np.copy(ds.fa['tmap'].value)
        pmap = np.copy(ds.fa['pmap'].value)
    else:
        print "Calculating Full connectivity matrices and Group T-stats"
        tmap,pmap = get_tt(get_connectivity_matrices(ds))
    nnod = len(tmap)
    if ds.fa.has_key('xsub_map'):
        print "<> Using stored xsub_map from feature attributes of DS." 
        xsub_map = np.copy((ds.fa['xsub_map'].value).transpose())
    else:
        xsub_map = get_xsub_corr_for_all_nodes(ds)
    if not econ:
        tmap_copy = np.copy(tmap)
        pmap_copy = np.copy(pmap)
        xsub_map_copy  = np.copy(xsub_map)
    pmap[tmap<0] = 1. # this excludes all negative t-scores 
    nodes = [] 
    masks = None
    mask_count = 1
    tmap[xsub_map[2,:]>xsubp,:] = 0 # Discard all nodes > xsubp 
    tmap[tmap<0] = 0

    while sum(np.sum(tmap,1)>0) > max_leftover:
        mask = np.zeros((1,nnod))
        nodes_left = sum(np.sum(tmap,1)>0)
        sys.stdout.write("%s/%s nodes left \r"%(nodes_left,nnod))
        sys.stdout.flush()
        if score=='instacorr_max':
            maxnode = list(np.sum(tmap,1)==np.max(np.sum(tmap,1))).index(True)
        if score=='xsub_max':
            maxnode = list(xsub_map[0,:]==np.max(xsub_map[0,:])).index(True)
            #print maxnode, np.max(xsub_map[:,0])
        print "\n"+str(maxnode)
        if crit=='t':
            nodes_in_mask = tmap[maxnode,:]>=th
        elif crit=='q':
            qth =  fdr(pmap[maxnode,:],th)
            nodes_in_mask = pmap[maxnode,:]<=qth
        elif crit=='p':
            nodes_in_mask = pmap[maxnode,:]<=th
        elif crit=='pct':
            tm = tmap[maxnode,:]
            tm = tm[tm>0]
            if tm.shape[0]==0:
                break
            print "\n\nLength of tm: "+str(tm.shape)+"\n\n"
            pct_th =scipy.stats.scoreatpercentile(tm,th)
            nodes_in_mask = tmap[maxnode,:]>=scipy.stats.scoreatpercentile(tm,th)
        else:
            print "!! Warning: Invalid threshold criterion, assuming crit=='pct', and th=90"
            nodes_in_mask = pmap[maxnode,:]<=scipy.stats.scoreatpercentile(pmap[maxnode,:],90)
               
        if sum(nodes_in_mask)>=min_nodes:
            c = xsub_map[:,maxnode]
            if c[0] > 0 and c[2]<xsubp:
                print c[2]
                print "number of nodes in mask == %s"%sum(nodes_in_mask)   
                print nodes_in_mask.shape
                print mask.shape
                
                mask[0,nodes_in_mask] = mask_count
                mask_count = mask_count + 1
                print "found node: %s, %s nodes, xsub_corr=%s"%(maxnode,sum(nodes_in_mask),c[0])
                nodes.append(maxnode)
                if masks is None:
                    masks = mask
                else:
                    masks = np.vstack((masks,mask))
                if not overlap:
                    tmap[:,nodes_in_mask] = 0
                    pmap[:,nodes_in_mask] = 1.
            
        tmap[nodes_in_mask,:] = 0
        xsub_map[:,nodes_in_mask] = -2.
        xsub_map[:,maxnode] = -2.   
   
    if not overlap:
        masks = np.vstack((np.sum(masks,0).reshape((1,nnod)),masks)) #For viewing purposes
    ds = Dataset(masks)
    ds.a['seeds'] = nodes
    
    if econ:
        return ds
    else:
        return ds,tmap_copy,pmap_copy,xsub_map_copy

def save_sims_at_node(data,node,infix=''):
    nnod,ldist,nsubs = data.shape
    n = np.floor(np.sqrt(2*ldist))+1 
    dists = np.zeros((n,n,nsubs))
    for s in range(nsubs):
        x = data[node,:,s]
        xmin = np.min(x)
        xmax = np.max(x)
        i = xmax-xmin
        x = x-xmin             
        x = x/i
        dists[:,:,s] = squareform(x)
    scipy.io.savemat('node%s_%s_dists.mat'%(node,infix),{'dists':dists})
    
def derivative(x,step):
    d_x = []
    for i in range(len(x)-1):
        d_x.append((x[i+1]-x[i])/step)
    d_x.append(d_x[-1])
    return np.array(d_x)

def find_fourier_inflection(I,node,freq_cutoff):
    nv = []
    for t in np.arange(0,20,.01):
        print t
        nv.append(np.sum(I.get_mask(node,t)))
    nv_hat = np.fft.fft(nv,len(nv))
    nv_hat[freq_cutoff:] = 0
    nv_smooth = np.fft.ifft(nv_hat,len(nv_hat)).real
    pyplot.figure()
    pyplot.plot(nv)
    pyplot.plot(nv_smooth)
    pyplot.title(str(freq_cutoff))


def find_inflection(I,node):
    nvox = []
    for t in np.arange(1,20,.5):
        nvox.append(sum(I.get_mask(node,t)))
    print "for node %s"%node
    pyplot.plot(np.arange(1,20,.5),nvox)
    dd_nvox = derivative(derivative(nvox,.5),.5)
    pyplot.plot(np.arange(1,20,.5),dd_nvox)
    print "Negatives in second derivative:"
    for i,t in enumerate(np.arange(1,20,.5)):
        if dd_nvox[i]<0:
            print t
    print dd_nvox


