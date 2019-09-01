import pandas as pd
import numpy as np
import json
import os
import sys
import simplejson
import traceback

from ..transformer.continous_encoding import *
from ..transformer.category_encoding import *
from ..transformer.base_encoding import *
from ..transformer.feature_gen import *
from ..filter.base_filter import *

from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.naive_bayes import *
from sklearn.decomposition import *
from sklearn.random_projection import *
from sklearn.covariance import *
from sklearn.manifold import *
from sklearn.gaussian_process import *
from sklearn.calibration import *
from sklearn.dummy import DummyClassifier
from sklearn.semi_supervised import *
from sklearn.neighbors import *
from sklearn.neural_network import *
from sklearn.manifold import *
from sklearn.kernel_ridge import *
from sklearn.kernel_approximation import *
from sklearn.isotonic import *
from sklearn.discriminant_analysis import *
from sklearn.multiclass import *
from sklearn.multioutput import *

from xgboost.sklearn import XGBClassifier

from ...algorithms import *


def instantiate(method, method_params=None):
    """
    字符串形式类名的实例化
    :param method:
    :param method_params:
    :return:
    """
    if method_params is None:
        method_params = {}
    try:
        enc = eval(method)(**method_params)
    except Exception as e:
        print(traceback.format_exc())
        if method.endswith('.py'):
            path = method
        else:
            path = os.path.dirname(get_rootpath()) + '/udf/' + method + '.py'
        print(path)
        dirpath = os.path.dirname(path)
        classname = os.path.basename(path).split('.')[0]
        sys.path.append(dirpath)

        mod = __import__(classname)
        enc = getattr(mod, classname, None)(**method_params)

    return enc


def get_rootpath():
    path = os.path.realpath(__file__)
    subpath = path.split('mlearn', 1)[0] + 'mlearn/'
    if os.path.exists(subpath + 'mlearn'):
        utilspath = subpath + 'mlearn'
    else:
        utilspath = subpath + 'mlearndev'
    return utilspath
