# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support for clustering

WiP
TODO unit test"""


import numpy as np
from mvpa2.misc.neighborhood import HollowSphere
from mvpa2.measures.base import Measure
from mvpa2.base.types import is_datasetlike
from mvpa2.base.dataset import vstack
from mvpa2.datasets.base import Dataset
from mvpa2.base.state import ConditionalAttribute
from mvpa2.clfs.stats import *

def dataset_neighbors(ds, nn=1, space='voxel_indices'):
    '''Computes the neighbors of features in a dataset
    
    Parameters
    ----------
    ds: dataset
        Dataset for which neighbors are computed. This is typically
        an fmri_dataset.
    nn: int 
        Nearest neighbour level. For a spatial dataset in n dimensions
        it is required that 1 <= nn <= n. With a dataset in three dimensions 
        it means that features are considered neighbors if they have a 
        common side (nn=1), edge (nn=2) or corner (nn=3).
    space: str
        space in which neighbors are computed. It is required that 
        ds.fa[space].value is an array of shape (n, ds.nfeatures), where n
        is the number of spatial dimensions
    
    Returns
    -------
    nbrs: dict
        A mapping from feature ids to sets of neighbouring feature ids.
        nbrs[i]==set([j1, ..., jN]) means that feature with index i has 
        neighbors with feature indices j1, ..., jN.
        
    Notes
    -----
    Adopted from surfing_clusterize.m (Nikolaas N. Oosterhof, Oct 2010)
    http://surfing.sourceforge.net
    '''

    if not space in ds.fa:
        raise ValueError("Cannot find space '%s'" % space)

    idxs = ds.fa[space].value
    random_idx = idxs[0]
    ndim = len(random_idx) # number of dimensions (usually 3)

    if not nn in range(1, ndim + 1):
        raise ValueError("nn should be in 1..%d" % ndim)

    # define neighborhood
    eps = .0001 # avoid rounding issues
    nbrhood = HollowSphere(float(nn) ** .5 + eps, 0)
    offsets = np.asarray(nbrhood((0,) * ndim))

    # mapping from tupled indices to feature indices
    tup_idx2feature = dict((tuple(idx), fi) for fi, idx in enumerate(idxs))
    tup_idxs = set(tup_idx2feature)

    nbrs = dict()
    for fi, idx in enumerate(idxs):
        tup_nbr = set(map(tuple, offsets + idx))
        tup_nbr_inside = set.intersection(tup_nbr, tup_idxs)
        nbrs[fi] = set(tup_idx2feature[n] for n in tup_nbr_inside)

    return nbrs

def find_clusters(ds, neighbors, iso_value=True):
    '''Finds clusters of non-zero values in a dataset.
    
    Parameters
    ----------
    ds: Dataset or numpy.array
        dataset in which clusters are to be formed
    neighbors: dict
        A mapping from feature ids to sets of neighbouring feature ids.
        nbrs[i]==set([j1, ..., jN]) means that feature with index i has 
        neighbors with feature indices j1, ..., jN. neighbors can be 
        computed, for example using dataset_neighbors, or for a surface s
        using s.neighbors.
    iso_value: True or False (default: True)
        If True then two elements are part of the same cluster if they
        have the same value and are connected. If False then it is only 
        required that both elements are non-zero and connected.
        In this context 'connected' is the transitive closure
        of neighbourhood-ness of non-zero (iso_value=False) or identically
        valued  (iso_value=True) elements.
    
    Returns
    -------
    clusters: dict or set
        If iso_value=True then a dictionary is returned. In this array, 
        clusters[k]=vs, where vs is a set of arrays v1,...vN, means that
        the clusters with feature indices in the arrays v1, ..., vN
        all have the value k.
        If iso_value=False then a set with clusters vs.  
    '''


    ds_like = is_datasetlike(ds)
    if len(ds.shape) == 1:
        ns = 1
        nf = ds.shape[0]
    else:
        ns, nf = ds.shape

    if not np.array_equal(np.unique(neighbors), np.arange(nf)):
        raise ValueError("neighbors should have keys range(%d)" % (nf - 1))


    if ns > 1:
        return [find_clusters(d, neighbors, iso_value=iso_value)
                    for d in ds]

    ys = ds
    if ds_like:
        ys = ys.samples

    if len(ys.shape) != 1:
        ys = np.ravel(ys)

    if not (iso_value or ys.dtype is np.bool_):
        # convert to boolean
        ys = np.asarray(ys, dtype=np.bool_)

    clusters = dict() if iso_value else list() # space for output
    queue = list() # feature indices of candidate

    visited = set()

    for pos in xrange(nf):
        if pos in visited:
            # already part of a cluster, continue
            continue

        y = ys[pos]

        if y: # non-zero value; start a new cluster
            current_cluster = set([pos])
            queue.append(pos)

            while queue:
                qpos = queue.pop(0) # process all candidates
                current_cluster.add(qpos)
                visited.add(qpos)

                for nbr in neighbors[qpos]:
                    # is a nieghbour and has the correct value
                    if not nbr in queue and ys[nbr] == y and not nbr in visited:
                        queue.append(nbr)

            current_cluster_arr = np.asarray(list(current_cluster), dtype=np.int)
            if iso_value:
                if not y in clusters:
                    clusters[y] = list()
                clusters[y].append(current_cluster_arr)
            else:
                clusters.append(current_cluster_arr)

    return clusters

class TFCEMeasure(Measure):
    '''Measure of Threshold Free Cluster Enhancement
    
    References
    ----------
    Stephen M. Smith, Thomas E. Nichols, Threshold-free
        cluster enhancement: Addressing problems of smoothing, threshold 
        dependence and localisation in cluster inference, NeuroImage, 
        Volume 44, Issue 1, 1 January 2009, Pages 83-98.
    '''

    null_prob = ConditionalAttribute(enabled=False)
    null_t = ConditionalAttribute(enabled=False)

    # indicate it is always trained
    is_trained = True

    def __init__(self, neighbors, feature2size=None, nt=None, b=.5, e=2., **kwargs):
        '''
        Parameters
        ----------
        neighbors: dict
            Mapping from feature indices to indices of neighbouring features
        feature2size: dict or np.ndarray
            Mapping from feature index to spatial size measure for each feature.
            If omitted then all sizes are set to unity.
        nt: int or float
            If a positive int, then the number of steps in computing TFCE 
            integral. If a positive float, then the step size in computing this 
            integral. If negative, then a TFCE value is computed only for
            the threshold value -nt (allowing to to use this class
            for traditional 'single threshold' clustering). 
            None is equivalent to 100.
        b: float
            TFCE parameter 'B'.
        e: float
            TFCE parameter 'E'.
        '''

        Measure.__init__(self, **kwargs)
        self._neighbors = neighbors

        if feature2size is None:
            feature2size = np.ones((len(neighbors),))

        if len(feature2size) != len(neighbors):
            raise ValueError("size mismatch: %d != %d" %
                                (len(feature2size), len(neighbors)))

        if nt is None:
            nt = 100

        self._feature2size = feature2size

        self._nt = nt
        self._b = b
        self._e = e

    def __repr__(self, prefixes=None):
        prefixes_ = prefixes or []
        prefixes_ += ['neighbors=%r' % self._neighbors]
        return \
            super(TFCEMeasure, self).__repr__(prefixes=prefixes)

    def _call(self, ds):
        '''Computes the measure
        
        Parameters
        ----------
        ds: Dataset
            Dataset for which the TFCE scores are computed
        
        Returns
        -------
        tfce_ds: Dataset
            Dataset with computed TFCE values (for each sample separately)
        '''

        ns, nf = ds.shape
        neighbors = self._neighbors
        if nf != len(neighbors):
            raise ValueError("%d neighbors, but %d features" %
                                    (nf, len(neighbors)))
        if ns > 1:
            # compute per sample using recursion
            stacked = vstack((self._call(ds[i]) for i in xrange(ns)), a='all')
            stacked.sa.update(ds.sa.copy())
            return stacked

        ys = ds.samples.ravel()
        assert len(ys) == nf
        if np.any(ys < 0):
            # both positive and negative values in input;
            # compute for each separately
            def _copy_signed(sgn, ds=ds):
                # make a copy using only positive or negative values
                # from the input. the output has always positive values only
                c = ds.copy()
                y = c.samples * sgn
                y[y < 0] = 0
                c.samples = y
                return c

            tfce_neg = TFCEMeasure._call(self, _copy_signed(-1))
            tfce_pos = TFCEMeasure._call(self, _copy_signed(1))

            tfce_pos.samples -= tfce_neg
            return tfce_pos

        ys[ys < 0] = 0
        ymax = np.max(ys)

        nt = self._nt

        if nt < 0:
            ts = [-nt]
        else:
            if type(nt) is int:
                dt = ymax / float(nt)
            else:
                dt = nt
            # values for t
            ts = np.arange(0, ymax, dt) + dt

        # TFCE parameters and mapping from feature to size
        b = self._b
        e = self._e
        f2s = self._feature2size

        # allocate space
        tfce = np.zeros((1, nf))

        for t in ts:
            msk = ys >= t
            clusters = find_clusters(msk, neighbors, iso_value=False)
            if not clusters:
                break

            if __debug__:
                debug('TFCE', "t=%s, %d clusters", (t, len(clusters)))

            for cluster in clusters:
                # apply TFCE formula
                tfce[0, cluster] += np.sum(f2s[cluster]) ** b * t ** e * dt

        return Dataset(tfce, fa=ds.fa.copy(), a=ds.a.copy())


def mri_dataset_tfce_measure(ds, nn=1, nt=None, b=.5, e=2., **kwargs):
    '''TFCE measure for MRI dataset
    
    Parameters
    ----------
    ds: Dataset
        fmri_dataset-like Dataset
    nn: int or dict.
        If a dictionary then this is assumed to be a mapping from features
        to its neighbors. If an int then the neighbors are computed on the
        fly using dataset_neighbous, and nn indicates the nearest neighbour 
        level. With a dataset in  three dimensions  it means that features 
        are considered neighbors if they have a common side (nn=1), 
        edge (nn=2) or corner (nn=3).
    nt: int
        Number of steps in computing TFCE integral.
        If None then a reasonable value (currently 100) is taken.
    b: float
        TFCE parameter 'B'.
    e: float
        TFCE parameter 'E'.
        
    Returns
    -------
    measure: TFCEMeasure
        TFCE measure that can be applied to an mri dataset.
    '''
    if type(nn) is int:
        nbrs = dataset_neighbors(ds=ds, nn=nn)
    elif has_attr(nn, shape):
        nbrs = nn
    else:
        raise ValueError("nn option not understood")


    elvol = 1. # default value for voxel size

    eldim = ds.a.get('voxel_eldim', None)
    if eldim is not None:
        # compute the volume of a single voxel
        for e in eldim.value:
            elvol *= e

    nf = ds.nfeatures
    feature2size = np.ones((nf,)) * elvol

    m = TFCEMeasure(nbrs, feature2size, nt=nt, b=b, e=e, **kwargs)
    return m


def surface_dataset_tfce_measure(surface_anatomy, nt=None, b=.5, e=2., **kwargs):
    '''TFCE measure for MRI dataset
    
    Parameters
    ----------
    surface_anatomy: surf.Surface
        Surface anatomy.
    nt: int
        Number of steps in computing TFCE integral.
        If None then a reasonable value (currently 100) is taken, and
        the integral increment dt is set to max(ds.samples[i,:])/nt
        for the i-th sample. 
    b: float
        TFCE parameter 'B'.
    e: float
        TFCE parameter 'E'.
        
    Returns
    -------
    measure: TFCEMeasure
        TFCE measure that can be applied to a surface dataset
    '''
    nbrs = surface_anatomy.neighbors
    feature2size = surface_anatomy.node_areas
    m = TFCE_Measure(nbrs, feature2size, nt=nt, b=b, e=e, **kwargs)
    return m

class SecondLevelBootstrapPValueTFCEMeasure(TFCEMeasure):
    '''Computes p value for second level analysis
    XXX how to set ConditionalAttribute null probabilities etc?
    '''
    def __init__(self, neighbors, h0mean=0., niter=1000, measure=np.mean,
                        feature2size=None, nt=None, b=.5, e=2., **kwargs):
        '''        
        Parameters
        ----------
        neighbors: dict
            Contains neighbors for each feature
        h0mean: float
            Expected mean value under null hypothesis
            
        TODO more documentation
        TODO check one tailed/two tailed
        '''

        TFCEMeasure.__init__(self, neighbors=neighbors,
                             feature2size=feature2size, nt=nt,
                             b=b, e=e, **kwargs)
        self._h0mean = h0mean
        self._niter = niter
        self._measure = measure

    def _call(self, dsets):
        '''
        Parameters
        ----------
        dsets: list of Dataset
            Datasets used for group analysis
            
        Returns
        -------
        ps: Dataset
            p-values for finding the observed data or more extreme value 
            under the null hypothesis
        
        '''
        for dset0 in dsets:
            break

        if not all(dset0.shape == dset.samples.shape for dset in dsets):
            raise ValueError("Not all datasets have the same shape")

        ns, nf = dset0.shape
        ndsets = len(dsets)
        if dset0.nsamples > 1:
            # compute for each sample
            stacked = vstack([self._call([ds[i] for ds in dsets])
                                    for i in xrange(ns)], a='all')
            stacked.sa.update(ds.sa.copy())
            return stacked

        # get the data
        ysdset = vstack([dset.samples for dset in dsets])
        ys_orig = np.copy(ysdset.samples)

        measure = self._measure
        h0mean = self._h0mean
        niter = self._niter

        max_values = np.zeros((niter,))
        min_values = np.zeros((niter,))

        ys_random = ys_orig
        for i in xrange(niter + 1):
            ys_demeaned = measure(ys_random - h0mean, axis=0)

            # overwrite samples in ysdset - still have the data in ys_orig
            ysdset.samples = np.reshape(ys_demeaned, (1, -1))

            # compute TFCE
            tfce = TFCEMeasure._call(self, ysdset)

            if i == 0:
                tfce_orig = tfce
            else:
                # keep track of extreme values under H0
                max_values[i - 1] = np.max(tfce.samples)
                min_values[i - 1] = np.min(tfce.samples)

            idxs = np.random.randint(ndsets, size=ndsets) # sample with replacement
            sgns = np.random.randint(2, size=ndsets) * 2 - 1 # invert sign randomly

            ys_random = ys_orig[idxs] * np.reshape(sgns, (ndsets, 1))

        np_maxmin = Nonparametric(np.hstack((max_values, min_values)))
        null_dist = FixedNullDist(np_maxmin)

        # compute the p values
        # Make this a conditional attribute? How?
        ps = null_dist.p(tfce_orig)

        return ps








