# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Regressions"""

from mvpa.datasets.splitters import NFoldSplitter, OddEvenSplitter

from mvpa.misc.errorfx import CorrErrorFx

from mvpa.clfs.meta import SplitClassifier
from mvpa.clfs.transerror import TransferError
from mvpa.misc.exceptions import UnknownStateError
from mvpa.misc.attrmap import AttributeMap
from mvpa.mappers.fx import mean_sample
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError

from tests_warehouse import *
from tests_warehouse_clfs import *

class RegressionsTests(unittest.TestCase):

    @sweepargs(ml=clfswh['regression_based'] + regrswh[:])
    def test_non_regressions(self, ml):
        """Test If binary regression-based  classifiers have proper tag
        """
        self.failUnless(('binary' in ml.__tags__) != ml.__is_regression__,
            msg="Inconsistent tagging with binary and regression features"
                " detected in %s having %r" % (ml, ml.__tags__))

    @sweepargs(regr=regrswh['regression'])
    def test_regressions(self, regr):
        """Simple tests on regressions
        """
        ds = datasets['chirp_linear']
        # we want numeric labels to maintain the previous behavior, especially
        # since we deal with regressions here
        ds.sa.labels = AttributeMap().to_numeric(ds.labels)

        cve = CrossValidatedTransferError(
            TransferError(regr, CorrErrorFx()),
            splitter=NFoldSplitter(),
            postproc=mean_sample(),
            enable_states=['training_confusion', 'confusion'])
        corr = cve(ds).samples.squeeze()

        self.failUnless(corr == cve.states.confusion.stats['CCe'])

        splitregr = SplitClassifier(
            regr, splitter=OddEvenSplitter(),
            enable_states=['training_confusion', 'confusion'])
        splitregr.train(ds)
        split_corr = splitregr.states.confusion.stats['CCe']
        split_corr_tr = splitregr.states.training_confusion.stats['CCe']

        for confusion, error in (
            (cve.states.confusion, corr),
            (splitregr.states.confusion, split_corr),
            (splitregr.states.training_confusion, split_corr_tr),
            ):
            #TODO: test confusion statistics
            # Part of it for now -- CCe
            for conf in confusion.summaries:
                stats = conf.stats
                self.failUnless(stats['CCe'] < 0.5)
                self.failUnlessEqual(stats['CCe'], stats['Summary CCe'])

            s0 = confusion.asstring(short=True)
            s1 = confusion.asstring(short=False)

            for s in [s0, s1]:
                self.failUnless(len(s) > 10,
                                msg="We should get some string representation "
                                "of regression summary. Got %s" % s)
            if cfg.getboolean('tests', 'labile', default='yes'):
                self.failUnless(error < 0.2,
                            msg="Regressions should perform well on a simple "
                            "dataset. Got correlation error of %s " % error)

            # Test access to summary statistics
            # YOH: lets start making testing more reliable.
            #      p-value for such accident to have is verrrry tiny,
            #      so if regression works -- it better has at least 0.5 ;)
            #      otherwise fix it! ;)
            #if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(confusion.stats['CCe'] < 0.5)

        # just to check if it works fine
        split_predictions = splitregr.predict(ds.samples)

        # To test basic plotting
        #import pylab as P
        #cve.confusion.plot()
        #P.show()

    @sweepargs(clf=clfswh['regression'])
    def test_regressions_classifiers(self, clf):
        """Simple tests on regressions being used as classifiers
        """
        # check if we get values set correctly
        clf.states.change_temporarily(enable_states=['estimates'])
        self.failUnlessRaises(UnknownStateError, clf.states['estimates']._get)
        cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(),
            enable_states=['confusion', 'training_confusion'])
        ds = datasets['uni2small'].copy()
        # we want numeric labels to maintain the previous behavior, especially
        # since we deal with regressions here
        ds.sa.labels = AttributeMap().to_numeric(ds.labels)
        cverror = cv(ds)

        self.failUnless(len(clf.states.estimates) == ds[ds.chunks == 1].nsamples)
        clf.states.reset_changed_temporarily()



def suite():
    return unittest.makeSuite(RegressionsTests)


if __name__ == '__main__':
    import runner
