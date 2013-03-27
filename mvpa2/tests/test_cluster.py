# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA classifier cross-validation"""

from mvpa2.testing.tools import assert_equal, ok_, assert_array_equal

from mvpa2.testing import *
from mvpa2.testing.datasets import pure_multivariate_signal, get_mv_pattern
from mvpa2.testing.clfs import *
from mvpa2.support.nibabel import surf

from mvpa2.misc.cluster import neighborhood_clustering as cl
from mvpa2.datasets.base import Dataset

class ClusterTests(unittest.TestCase):
    def test_clustering_surface(self):
        s = surf.generate_plane((0, 0, 0), (0, 1, 0), (1, 0, 0), 4, 4)

        nv = s.nvertices
        ds = Dataset(np.reshape(np.arange(nv), (1, -1)))
        clusters = cl.find_clusters(ds, s.neighbors)

        for k in xrange(nv):
            if k:
                assert_true(k in clusters)
                cluster = clusters.pop(k)
                assert_array_equal(k, cluster)

        assert_true(len(clusters) == 0)

        ds.samples[:, ds.samples < 6] = 0
        ds.samples[:, ds.samples > 0] = 1

        clusters = cl.find_clusters(ds, s.neighbors)
        assert_true(len(clusters) == 1)
        assert_true(1 in clusters)
        cluster = clusters.pop(1)
        assert_true(len(cluster) == 1)
        assert_array_equal(cluster[0].ravel(), np.arange(6, nv))

    def test_clustering_volume(self):
        if not externals.exists('nibabel'):
            return

        import nibabel as nb
        from mvpa2.datasets.mri import fmri_dataset

        side = 4
        data = np.reshape(np.mod(np.arange(side ** 3), 3), (side,) * 3)

        img = nb.Nifti1Image(data, np.eye(4))
        ds = fmri_dataset(img)

        space = 'voxel_indices'
        for nn in (1, 2, 3):
            nbrs = cl.dataset_neighbors(ds, space=space, nn=nn)
            max_nbrs = {1:6, 2:18, 3:26}[nn]

            assert_equal(set(nbrs.keys()), set(range(ds.nfeatures)))
            for i, idxs in enumerate(ds.fa[space].value):
                arr_idxs = np.asarray(idxs)
                ln = len(nbrs[i])

                on_edge = any(arr_idxs == 0) or any(arr_idxs == side - 1)
                assert_equal(on_edge, len(nbrs[i]) < max_nbrs)

        ds.samples = np.mod(ds.samples, side + 1)

        nbrs = cl.dataset_neighbors(ds, space=space, nn=1)
        clusters = cl.find_clusters(ds, nbrs)
        # TODO: more tests

def suite():
    return unittest.makeSuite(ClusterTests)


if __name__ == '__main__':
    import runner
