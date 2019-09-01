import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
import os
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from ..transformer.base_encoding import HistogramEncoder
import simplejson
from ..base_utils import simplejsonEncoder


# TODO: 以下三个函数等待删除
def get_max_same_count(c):
    try:
        return c.value_counts().iloc[0]
    except:
        return len(c)


def desc_df(df_origin):
    df = df_origin.copy()

    df_desc = pd.DataFrame(df.isnull().sum(axis=0), columns=['null_num'])
    df_desc['notnull_num'] = df.shape[0] - df_desc['null_num']
    df_desc['notnull_ratio'] = df_desc['notnull_num'] / df.shape[0]

    nunique_value = df.apply(lambda c: c.nunique())
    df_desc['diff_values_num'] = nunique_value

    same_value = df.apply(get_max_same_count)
    df_desc['most_value_num'] = same_value
    df_desc['same_ratio'] = same_value / df.shape[0]

    return df_desc


def cover_stats(df, feature_map_dict=None):
    data_fg_r = (df.notnull().sum(axis=1) > 0).sum() / df.shape[0]

    t = desc_df(df)
    t = t[['notnull_ratio']]
    t.columns = ['覆盖率']

    t['饱和度'] = t['覆盖率'] / data_fg_r
    if feature_map_dict:
        t.index = pd.MultiIndex.from_tuples([(i, feature_map_dict[i]) for i in t.index])
        t1 = pd.DataFrame(columns=['覆盖率', '饱和度'])
        t1.ix[('dataset', '数据集'),] = [data_fg_r, np.nan]
        t = pd.concat([t1, t], axis=0)
    else:
        t1 = pd.DataFrame(columns=['覆盖率', '饱和度'])
        t1.loc['数据集'] = [data_fg_r, np.nan]
        t = pd.concat([t1, t], axis=0)

    return t


def target_stats(df_origin, target_col, time_col):
    """
    样本起始时间和结束时间，逾期率
    :param df_origin:
    :param target_col:
    :param time_col:
    :return: dict

    Examples
    --------
    >>> target_stats(df, 'target', 'apply_risk_created_at')
    {'apply_risk_created_at_end': '2018-05-31 23:22:57', 'apply_risk_created_at_start': '2018-03-01 16:30:24.0', 'sample_num': 14999, 'target': 0.0}
    """
    # TODO: 考虑增加按天逾期率变化图数据
    dic = {}
    df = df_origin.copy()
    dic[target_col] = df[target_col].astype(int).mean()
    dic[f'{time_col}_start'] = df[time_col].min()
    dic[f'{time_col}_end'] = df[time_col].max()
    dic['sample_num'] = df.shape[0]

    return dic


def get_freq_stats(df):
    """
    得到众数及众数占比
    :param df:
    :return: pd.DataFrame

    Examples
    --------
    >>> get_freq_stats(df)
                                        count	unique	top	            freq	freq_ratio
    paydayloanlevelonechannelname_woe	13500	11	    -0.468033807992	7094	0.525481
    individual_gender_woe	            13500	2	    0.0710246970594	10494	0.777333
    """
    t = df.astype(str).describe().T
    t['freq_ratio'] = t['freq'] / t['count']
    return t


def get_desc_stats(df):
    """
    得到描述性统计量
    :param df:
    :return: pd.DataFrame

    Examples
    --------
    >>> get_desc_stats(df)
    	                                count	mean	    std	        min	        25%	        50%	        75%	        max
    paydayloanlevelonechannelname_woe	13500.0	-0.087283	0.913581	-20.000000	-0.468034	-0.468034	0.574691	0.574691
    individual_gender_woe	            13500.0	-0.005581	0.143138	-0.273014	0.071025	0.071025	0.071025	0.071025
    """
    if not df.dtypes.map(is_numeric_dtype).any():
        return pd.DataFrame(columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    return df.describe().T


def get_missing_stats(df):
    """
    得到缺失率、饱和率
    :param df:
    :return: pd.DataFrame

    Examples
    --------
    >>> get_missing_stats(df)
	                                    missing	cover	saturation
    paydayloanlevelonechannelname_woe	0.0	    1.0	    1.0
    individual_gender_woe	            0.0	    1.0	    1.0

    """
    saturation_ratio = (df.notnull().sum(axis=1) > 0).sum() / df.shape[0]
    t = pd.DataFrame(df.isnull().mean(), columns=['missing'])
    t['cover'] = 1 - t['missing']
    t['saturation'] = t['cover'] / saturation_ratio
    return t


def _series_distribution_stats(dfs, cate_convert_thr=20, cate_thr=20, cont_thr=20, cont_low_percent=5,
                               cont_high_percent=95):
    """
    得到pd.Series的直方图分布，连续型会先做分箱，会对尾部做归并处理
    :param dfs: pd.Series
    :param cate_convert_thr:
    :param cate_thr:
    :param cont_thr:
    :param cont_low_percent:
    :param cont_high_percent:
    :return: json

    Examples
    --------
    >>> _series_distribution_stats(dfs)
    '{"code0":6694,"code108":827,"code109":581,"missing":6897}'
    """
    print('start', dfs.name, '--------------')
    if (not is_numeric_dtype(dfs)) | (dfs.nunique() < cate_convert_thr):
        if (dfs.nunique() < cate_convert_thr):
            tmp = dfs.value_counts()
        else:
            tmp = dfs.value_counts()[:cate_thr]
            tmp['others+'] = dfs.value_counts()[cate_thr:].sum()
        null_num = dfs.isnull().sum()
        if null_num > 0:
            tmp.ix['missing'] = null_num
    else:
        t = dfs[dfs.notnull() & (dfs != np.inf) & (dfs != -np.inf)]

        low_thr = np.percentile(t, q=cont_low_percent)
        high_thr = np.percentile(t, q=cont_high_percent)
        min_thr = t.min()
        max_thr = t.max()

        t2 = t[(t > low_thr) & (t < high_thr)]

        if t2.shape[0] > 0:
            tmp = pd.cut(t2, bins=cont_thr).value_counts().sort_index()
            tmp.index = tmp.index.categories.to_native_types().astype(object)
        else:
            tmp = pd.Series(name=dfs.name)

        null_num = dfs.isnull().sum()
        inf_num = (dfs == np.inf).sum()
        neg_inf_num = (dfs == -np.inf).sum()

        low_num = (t <= low_thr).sum()
        high_num = (t >= high_thr).sum()

        if low_num > 0:
            pd.Series(low_num, ['[%s,%s]' % (min_thr, low_thr)]).append(tmp)
        if neg_inf_num > 0:
            pd.Series(neg_inf_num, ['-inf']).apppend('tmp')
        if high_thr > 0:
            tmp.ix['[%s,%s]' % (high_thr, max_thr)] = high_num
        if inf_num > 0:
            tmp.ix['inf'] = inf_num
        if null_num > 0:
            tmp.ix['missing'] = null_num

    return tmp.to_json()


def distribution_stats(df, cate_convert_thr=20, cate_thr=20, cont_thr=20, cont_low_percent=5, cont_high_percent=95):
    """
    得到df每个字段直方图分布
    :param df:
    :param cate_convert_thr:
    :param cate_thr:
    :param cont_thr:
    :param cont_low_percent:
    :param cont_high_percent:
    :return: pd.DataFrame

    Examples
    --------
    >>> distribution_stats(df)
                            type	distribution
    overdue_days	        int64	{"0":11554,"1":3445}
    channelname	            object	{"\u5782\u76f4\u6e20\u9053":7856,"APP_Android"...
    individual_gender	    int64	{"1":11665,"0":3334}
    baidu_panshi_prea_score	float64	{"(507.908, 512.6]":162,"(512.6, 517.2]":259,"...
    cn_gt_0	                float64	{"(10.872, 17.4]":994,"(17.4, 23.8]":1031,"(23...
    """
    t = pd.DataFrame(df.dtypes.astype(str), columns=['type'])
    t['distribution'] = np.nan
    for c in t.index:
        t.loc[c, 'distribution'] = _series_distribution_stats(df[c], cate_convert_thr=cate_convert_thr,
                                                              cate_thr=cate_thr, cont_thr=cont_thr,
                                                              cont_low_percent=cont_low_percent,
                                                              cont_high_percent=cont_high_percent)

    return t


def data_stats(df_origin, cate_cols=None, error='ignore'):
    """
    df描述性统计汇总
    :param df_origin:
    :param cate_cols:
    :param error:
    :return: pd.DataFrame

    Examples
    --------
    >>> data_stats(df)
                        type    count	unique	top	    freq	freq_ratio	mean	    std	        min	        25%	        50%	        75%	        max	        missing	    cover	    saturation	distribution
    equipmentos	        float64	14999	3	    0.0	    9732	0.648843	0.240222	0.427235	0.000000	0.000000	0.000000	0.000000	1.000000	0.146010	0.853990	0.853990	{"0.0":9732,"1.0":3077,"missing":2190}
    fbi_score	        float64	14999	3719	nan	    8651	0.576772	6489.798677	1581.774192	4000.000000	5055.500000	6391.000000	7827.000000	9999.000000	0.576772	0.423228	0.423228	{"(4179.1, 4429.0]":381,"(4429.0, 4674.0]":353...
    individual_gender	int64	14999	2	    1	    11665	0.777719	0.777719	0.415793	0.000000	1.000000	1.000000	1.000000	1.000000	0.000000	1.000000	1.000000	{"1":11665,"0":3334}
    province	        object	14999	31	    广东省	1303	0.0868725	NaN	        NaN	        NaN	        NaN	        NaN	        NaN	        NaN	        0.000000	1.000000	1.000000	{"\u5e7f\u4e1c\u7701":1303,"\u56db\u5ddd\u7701...
    user_gray       	object	14999	6	    True	9000	0.60004	    NaN	        NaN	        NaN	        NaN	        NaN	        NaN	        NaN	        0.145343	0.854657	0.854657	{"true":9269,"false":3550,"missing":2180}
    """
    df = df_origin.copy()
    if cate_cols:
        cols = [c for c in cate_cols if c in df.columns]
        error_cols = [c for c in cate_cols if c not in df.columns]
        if not error_cols:
            if error == 'ignore':
                print('-------', error_cols, 'are not in data!', '-----------------')
            elif error == 'raise':
                raise Exception('---------', error_cols, 'are not in data!', '----------')
            else:
                raise KeyError('error params must be "ignore" or "raise"')

        df['cols'] = df[cols].astype(str)

    t0 = df.dtypes.astype(str)
    t0.name = 'type'
    t1 = get_freq_stats(df)
    t2 = get_desc_stats(df)
    del t2['count']
    t3 = get_missing_stats(df)
    t4 = distribution_stats(df)
    del t4['type']
    t = pd.concat([t0, t3, t1, t2, t4], axis=1)
    t.index.name = 'feature_name'
    t = t.reset_index()
    t = t.set_index(['feature_name', 'type'])
    return t


class DescStats(HistogramEncoder):
    def transform(self, X):
        df = X.copy()

        t0 = df.dtypes.astype(str)
        t0.name = 'type'
        t1 = get_freq_stats(df)
        t2 = get_desc_stats(df)
        del t2['count']
        t3 = get_missing_stats(df)
        t4 = pd.DataFrame(df.dtypes.astype(str), columns=['type'])
        t4['distribution'] = np.nan
        tmp = super().transform(X)
        for c in t4.index:
            t4.loc[c, 'distribution'] = simplejson.dumps(tmp[c], cls=simplejsonEncoder)
        del t4['type']
        t = pd.concat([t0, t3, t1, t2, t4], axis=1)
        t.index.name = 'feature_name'
        t = t.reset_index()
        t = t.set_index(['feature_name', 'type'])

        return t
