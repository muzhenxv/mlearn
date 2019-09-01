import pandas as pd
import numpy as np
import os
import simplejson
from collections import defaultdict
import pickle
import traceback
from . import data_reporter 
from ...service.base_utils import *
from ..data_service import get_data


def _get_common_col(df1, df2):
    """获取col的交集"""
    return sorted([fea for fea in df1.columns if fea in df2.columns])


def _get_common_idx(df1, df2, idx=None):
    """idx的交集，可以为index，也可以为某一给定的col"""
    idx1 = df1.index.tolist() if idx is None else df1[idx].tolist()
    idx2 = df2.index.tolist() if idx is None else df2[idx].tolist()
    
    if (len(idx1) != len(set(idx1))) or (len(idx2) != len(set(idx2))):
        msg = 'index of dataframe is not unique!' if idx is None else f'{idx} of dataframe is not unique!'
        raise Exception(msg)
    return sorted(set(idx1).intersection(set(idx2)))


def get_common_data(df1, df2, columns=None, index=None, precison=3):
    """获取df1和df2的交集数据"""
    com_col = _get_common_col(df1, df2) if columns is None else columns
    com_idx = _get_common_idx(df1, df2, index)
    
    if index is not None:
        df_1 = df1.set_index(index)
        df_2 = df2.set_index(index)
        com_col = [col for col in com_col if col != index]
    else:
        df_1 = df1.copy()
        df_2 = df2.copy()

    df11 = df_1.loc[com_idx][com_col].apply(pd.to_numeric, errors='ignore').round(precison)
    df22 = df_2.loc[com_idx][com_col].apply(pd.to_numeric, errors='ignore').round(precison)
    return df11, df22
    

def get_consistency_ratio(df1, df2, fillna_val=None):
    """对比两个dataframe的一致率，并对比用某些值填充后的一致率
    :param df1: 
    :param df2: 
    :param fill_na_val:
    :return:
    """

    if fillna_val is None:
        return pd.DataFrame((df1 == df2).mean(), columns=['实际一致率'])
    else:
        tmp1 = pd.DataFrame((df1 == df2).mean(), columns=['实际一致率'])
        tmp2 = pd.DataFrame((df1.fillna(fillna_val) == df2.fillna(fillna_val)).mean(), columns=[f'空值填{fillna_val}的一致率'])
        return tmp1.join(tmp2)


def get_difference_values(df1, df2, name1='online_data', name2='offline_data', idx='apply_risk_id'):
    """对比df1和df2，并列出不同的值
    :param df1: 
    :param df2: 
    :param name1:
    :param name1:
    :param idx:
    :return:
    """
    from collections import defaultdict
    df_diff = pd.DataFrame(df1 == df2)
    df_diff2 = df_diff.copy()
    idx_dic = defaultdict(list)
    for col in df_diff.columns:
        idx_dic[col] = df_diff.query(f'{col} == 0').index.tolist()
        
    for col, idx_list in idx_dic.items():
        df_tmp = pd.DataFrame()
        df_tmp[idx] = idx_list
        df_tmp[name1] = df1.loc[idx_list][col].tolist()
        df_tmp[name2] = df2.loc[idx_list][col].tolist()
        df_tmp['diff_data'] = name1 + ':' + df_tmp[name1].astype(str) + ', ' + name2 + ':' + df_tmp[name2].astype(str)
        df_tmp.set_index(idx, inplace=True)
        df_diff[col].loc[idx_list] = df_tmp['diff_data'].loc[idx_list]
    return df_diff, df_diff2


def get_desc_report(df1, df2, label=None):
    """调用desc_report中的func
    :param df1:
    :param df2:
    :param label:
    """
    return data_reporter._gen_desc_report(df1, df2, label, '')


def sort_app_names(df, col_name='equipment_app_names_v2', sep='|', drop_duplicates=False):
    """对app的那列分割后重新排序
    :param df:
    :param col_name:
    :param sep:
    :param drop_duplicates:
    """
    def _sort_app_names(x, sep='|', drop_duplicates=False):
        xx = str(x).split(sep)
        xx2 = sep.join(sorted(set(xx))) if drop_duplicates else sep.join(sorted(xx))
        return xx2
    df2 = df.copy()
    df2[col_name] = df2[col_name].map(lambda x : _sort_app_names(x, sep=sep, drop_duplicates=drop_duplicates))
    return df2


def gen_compare_report(df1, df2, columns=None, index=None, precison=3, fillna_val=999,
                       app_col=None, app_sep='|', app_drop_dup=False):
    """获取df1和df2的对比报告，需保证index的唯一性，若index为None，则默认使用dataframe的index作为关联主键
    :param df1: 
    :param df2: 
    :param columns:
    :param index:
    :param precison:
    :param fillna_val:
    :param app_col: 
    :param app_sep:
    :param app_drop_dup:
    :return:
    """
    df11, df22 = get_common_data(df1, df2, columns=columns, index=index, precison=precison)
    if app_col is not None:
        df11 = sort_app_names(df11, col_name=app_col, sep=app_sep, drop_duplicates=app_drop_dup)
        df22 = sort_app_names(df22, col_name=app_col, sep=app_sep, drop_duplicates=app_drop_dup)
        
    df_consis_report = get_consistency_ratio(df11, df22, fillna_val=fillna_val)
    df_diff_report1, df_diff_report2 = get_difference_values(df11, df22)
    df_desc_report = get_desc_report(df11, df22)
    return df_consis_report, df_diff_report1, df_diff_report2, df_desc_report


def dfs_to_excel(df_dic, file_name=None, index=True):
    """将dic中的df写入同一个excel
    :param df_dic:
    :param file_name:
    :param index:
    """
    import time, xlsxwriter
    if file_name is None:
        file_name = time.strftime("%Y%m%d", time.localtime()) + '_Report.xlsx'

    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')
    for name, df in df_dic.items():
        df.to_excel(writer, sheet_name=name,index=index)
    writer.save()
