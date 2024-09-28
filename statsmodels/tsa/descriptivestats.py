"""Descriptive Statistics for Time Series

Created on Sat Oct 30 14:24:08 2010

Author: josef-pktd
License: BSD(3clause)
"""
import numpy as np
from . import stattools as stt

class TsaDescriptive:
    """collection of descriptive statistical methods for time series

    """

    def __init__(self, data, label=None, name=''):
        self.data = data
        self.label = label
        self.name = name