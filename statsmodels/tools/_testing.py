"""Testing helper functions

Warning: current status experimental, mostly copy paste

Warning: these functions will be changed without warning as the need
during refactoring arises.

The first group of functions provide consistency checks

"""
import os
import sys
from packaging.version import Version, parse
import numpy as np
from numpy.testing import assert_allclose, assert_
import pandas as pd

class PytestTester:

    def __init__(self, package_path=None):
        f = sys._getframe(1)
        if package_path is None:
            package_path = f.f_locals.get('__file__', None)
            if package_path is None:
                raise ValueError('Unable to determine path')
        self.package_path = os.path.dirname(package_path)
        self.package_name = f.f_locals.get('__name__', None)

    def __call__(self, extra_args=None, exit=False):
        try:
            import pytest
            if not parse(pytest.__version__) >= Version('3.0'):
                raise ImportError
            if extra_args is None:
                extra_args = ['--tb=short', '--disable-pytest-warnings']
            cmd = [self.package_path] + extra_args
            print('Running pytest ' + ' '.join(cmd))
            status = pytest.main(cmd)
            if exit:
                print(f'Exit status: {status}')
                sys.exit(status)
        except ImportError:
            raise ImportError('pytest>=3 required to run the test')

def check_ftest_pvalues(results):
    """
    Check that the outputs of `res.wald_test` produces pvalues that
    match res.pvalues.

    Check that the string representations of `res.summary()` and (possibly)
    `res.summary2()` correctly label either the t or z-statistic.

    Parameters
    ----------
    results : Results

    Raises
    ------
    AssertionError
    """
    pass

def check_predict_types(results):
    """
    Check that the `predict` method of the given results object produces the
    correct output type.

    Parameters
    ----------
    results : Results

    Raises
    ------
    AssertionError
    """
    pass