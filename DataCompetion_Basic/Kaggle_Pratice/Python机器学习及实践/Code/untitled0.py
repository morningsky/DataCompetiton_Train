# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 20:37:27 2018

@author: sky
"""

import pandas as pd
import numpy as np
data_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header = None)
data_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header = None)

X_train = data_train[np.]