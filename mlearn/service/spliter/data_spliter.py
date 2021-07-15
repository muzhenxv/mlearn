from sklearn.model_selection import train_test_split
import os
from ..data_service import get_data
from ..reporter import spliter_report
import pandas as pd

def data_spliter(df_src, label, report_dst, time_col='biz_report_expect_at', index_col='apply_risk_id', label_col='overdue_days', test_size=0.2, method='oot', random_state=7, group_key=None, drop_cols=None):
    """
    数据分割功能，支持oot分割，随机分割，分组oot分割（每一组按照oot标准分割）
    :param df_src:
    :param label:
    :param report_dst:
    :param time_col:
    :param index_col:
    :param label_col:
    :param test_size:
    :param method:
    :param random_state:
    :param group_key:
    :param drop_cols: None or list
    :return:
    """
    df = get_data(df_src)
    if index_col in df.columns:
        df = df.set_index(index_col)
    if df[label_col].nunique() > 2:
        df[label] = (df[label_col].astype(int) > int(label[:-1])).astype(int)
    else:
        df[label] = df[label_col]

    # # TODO: 强行写死group_key,等中台可以正确从sampler回传group_key后，删除本段代码
    # if group_key is None:
    #     if 'level' in df.columns:
    #         group_key = 'level'


    if method == 'oot':
        # 凡group_key is not None,就是分层oot。不想分层oot，key is None
        if group_key is not None:
            df = df.sort_values(time_col, ascending=False)
            train = pd.DataFrame()
            test = pd.DataFrame()
            for k in df[group_key].unique():
                tmp = df[df[group_key] == k]

                train = pd.concat([train, tmp.iloc[int(tmp.shape[0] * test_size):, ]], axis=0)
                test = pd.concat([test, tmp.iloc[:int(tmp.shape[0] * test_size), ]], axis=0)
            del train[group_key], test[group_key]
        else:
            df = df.sort_values(time_col, ascending=False)
            test = df.iloc[:int(df.shape[0] * test_size), ]
            train = df.iloc[int(df.shape[0] * test_size):, ]
    elif method == 'random':
        if group_key is not None:
            del df[group_key]
        train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        raise Exception(f'method {method} is not defined.')

    if drop_cols is not None:
        for c in drop_cols:
            if c in train.columns:
                del train[c], test[c]

    if label != label_col:
        del train[label_col], test[label_col]

    # spliter_report(train, test, time_col, label, report_dst, covariate_shift_eva=False)
    del train[time_col], test[time_col]


    return train, test