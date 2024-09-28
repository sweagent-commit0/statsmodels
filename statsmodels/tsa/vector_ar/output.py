from statsmodels.compat.python import lzip
from io import StringIO
import numpy as np
from statsmodels.iolib import SimpleTable
mat = np.array
_default_table_fmt = dict(empty_cell='', colsep='  ', row_pre='', row_post='', table_dec_above='=', table_dec_below='=', header_dec_below='-', header_fmt='%s', stub_fmt='%s', title_align='c', header_align='r', data_aligns='r', stubs_align='l', fmt='txt')

class VARSummary:
    default_fmt = dict(data_fmts=['%#15.6F', '%#15.6F', '%#15.3F', '%#14.3F'], empty_cell='', colsep='  ', row_pre='', row_post='', table_dec_above='=', table_dec_below='=', header_dec_below='-', header_fmt='%s', stub_fmt='%s', title_align='c', header_align='r', data_aligns='r', stubs_align='l', fmt='txt')
    part1_fmt = dict(default_fmt, data_fmts=['%s'], colwidths=15, colsep=' ', table_dec_below='', header_dec_below=None)
    part2_fmt = dict(default_fmt, data_fmts=['%#12.6g', '%#12.6g', '%#10.4g', '%#5.4g'], colwidths=None, colsep='    ', table_dec_above='-', table_dec_below='-', header_dec_below=None)

    def __init__(self, estimator):
        self.model = estimator
        self.summary = self.make()

    def __repr__(self):
        return self.summary

    def make(self, endog_names=None, exog_names=None):
        """
        Summary of VAR model
        """
        pass