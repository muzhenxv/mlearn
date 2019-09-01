import pandas as pd
import numpy as np
from ..data_service import get_data
import json
import os
from ..reporter import sampler_report
from copy import deepcopy

def _get_min_sample(sample_num, group_ratio, thr=0.5):
    """

    :param sample_num:
    :param group_ratio:
    :param thr: float or int. if float, 抽样后样本量不能小于样本*thr， if int，抽样后样本量不能小于thr。
    :return:
    """
    total_num = sum(sample_num.values())
    sample_num_copy = sample_num.copy()
    group_ratio_copy = deepcopy(group_ratio)
    for i in sample_num.keys():
        if i not in group_ratio['ratio']:
            sample_num_copy.pop(i)

    for i in group_ratio['ratio'].keys():
        if i not in sample_num_copy:
            group_ratio_copy['ratio'].pop(i)

    j = max(sample_num_copy, key=sample_num_copy.get)
    for i in group_ratio_copy['ratio'].keys():
        if (sample_num_copy[i] / sample_num_copy[j]) < (group_ratio_copy['ratio'][i] / group_ratio_copy['ratio'][j]):
            num = sample_num_copy[i] / (group_ratio_copy['ratio'][i] / total_num)
            if thr <= 1:
                if num < total_num * thr:
                    continue
            else:
                if thr > total_num:
                    raise ValueError(
                        'param thr must be not larger than the records of origin dataset which is %s.' % total_num)
                elif num < thr:
                    continue
            j = i
    base_num = sample_num_copy[j]
    dic = dict()
    for i in group_ratio_copy['ratio'].keys():
        dic[i] = int(base_num * (group_ratio_copy['ratio'][i] / group_ratio_copy['ratio'][j]))
    return dic


def _get_data(**kwargs):
    """
    返回一个df， 包含一列group_key
    """
    return


def data_sampler(ds, report_dst, group_key, group_key_level, sort_values, group_ratio, thr=0.5, group_num=10,
                 base_df_key='group_key', base_df=None, get_group_data=None):
    """
    分组抽样
    ds: 数据源
    group_key: 分层字段，业务场景中为模型分或level
    group_key_level: bool， if true，group_key为level，不需要分箱
    sort_values: 排序字段, 业务场景中为进件时间或者应还时间
    group_ratio: 分层比例， 人工指定或者根据基准数据集计算, if None,根据基准数据集计算
                 if group_key_level is true, group_ratio like {'ratio': {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.2}}
                 else group_ratio like {'ratio': {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.2}, 'cut_points': {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.2}, 'lower': 0.01}
    group_num: 分层数
    base_df: 基准数据集， if None，根据get_group_data获取基准数据集
    get_group_data: 通过pyhive获得基准数据集
    """
    # 之所以要做astype(str), 因为dict做json化过程中key即使是int也会被转成string，因此一律在内部做string处理，避免冲突
    if group_ratio is None:
        group_ratio = {}
        if base_df is None:
            base_df = _get_data(**get_group_data)
        else:
            base_df = get_data(base_df)
        if group_key_level:
            group_ratio['ratio'] = base_df[base_df_key].astype(str).value_counts().to_dict()
        else:
            base_df[base_df_key], bins = pd.qcut(base_df[base_df_key], q=group_num, retbins=True, duplicates='drop')
            group_ratio['ratio'] = base_df[base_df_key].value_counts().sort_index().reset_index()[
                base_df_key].to_dict()
            group_ratio['cut_points'] = dict(zip(range(len(bins[1:])), bins[1:]))
            group_ratio['lower'] = bins[0]
            group_ratio = json.loads(json.dumps(group_ratio))

    data = get_data(ds)
    if not group_key_level:
        bins = [group_ratio['lower']] + list(group_ratio['cut_points'].values())
        data[group_key] = pd.cut(data[group_key], bins=bins, labels=group_ratio['cut_points'].keys())
    data[group_key] = data[group_key].astype(str)
    sample_num = data[group_key].value_counts().to_dict()
    dic = _get_min_sample(sample_num, group_ratio, thr)

    df = pd.DataFrame()
    for k, v in dic.items():
        data = data.sort_values(sort_values, ascending=False)
        tmp = data[data[group_key] == k].iloc[:v, :]
        df = pd.concat([tmp, df], axis=0)

    sampler_report(sample_num, dic, group_ratio, report_dst)
    return df


if __name__ == '__main__':
    group_ratio = {'ratio': {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.2}}

    df = pd.read_pickle('../../data/test_data.pkl')
    df['level'] = np.random.randint(0, 4, size=df.shape[0])

    df = data_sampler(df, None, 'level', True, 'apply_risk_created_at', group_ratio)
