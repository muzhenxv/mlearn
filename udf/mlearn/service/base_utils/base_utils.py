import simplejson
import pandas as pd
import numpy as np
import json
import os


def convert_df_type(df, threshold=5, retsize=False):
    """
    dataframe 判断特征类型。首先根据类型判断，然后对数值型做附加判断
    :param df:
    :param threshold: 取值数小于thr，视为离散。
    :param retsize:
    :return:
    """
    df = df.apply(pd.to_numeric, errors='ignore')
    cols = df.nunique()[df.nunique() < threshold].index.values
    df[cols] = df[cols].astype(str)

    cate_cols = df.select_dtypes(include=['object'])
    cont_cols = df.select_dtypes(exclude=['object'])
    # bool convert to int by producting 1
    df[cont_cols] = df[cont_cols] * 1
    if retsize:
        feature_size = df[cate_cols].nunique().sum() + len(cont_cols)
        return df, feature_size
    return df


def get_feature_importances(estimator, norm_order=1):
    """
    由于此处对于线模型，直接使用系数绝对值作为特征重要性的度量指标。因此对于线模型，需要要求入模特征标准化
    :param estimator:
    :param norm_order:
    :return:
    """
    """Retrieve or aggregate feature importances from estimator"""
    importances = getattr(estimator, "feature_importances_", None)

    coef_ = getattr(estimator, "coef_", None)
    if importances is None and coef_ is not None:
        if estimator.coef_.ndim == 1:
            importances = np.abs(coef_)

        else:
            importances = np.linalg.norm(coef_, axis=0,
                                         ord=norm_order)

    elif importances is None:
        raise ValueError(
            "The underlying estimator %s has no `coef_` or "
            "`feature_importances_` attribute. Either pass a fitted estimator"
            " to SelectFromModel or call fit before calling transform."
            % estimator.__class__.__name__)

    return importances


def table_to_dict(df_origin):
    df = df_origin.copy()
    try:
        df.columns = df.columns.levels[0]
    except:
        pass
    df.index.name = 'feature'
    df = df.reset_index()
    return df.to_dict(orient='records')


def merge_multi_df(dic, axis=1, sort_f=True):
    l = []
    for k, v in dic.items():
        tmp = pd.DataFrame(v)
        if axis == 1:
            d_tr_t = tmp.T
            d_tr_t['dataset'] = k
            d_tr_t = d_tr_t.set_index('dataset', append=True)
            tmp = d_tr_t.T
        else:
            tmp['dataset'] = k
            tmp = tmp.set_index('dataset', append=True)
        l.append(tmp.copy())
    t = pd.concat(l, axis=axis)

    if sort_f:
        t = t.sort_index(axis=axis, ascending=False)

    return t

import string


def _index_to_alphabet(i):
    l = list(string.ascii_uppercase)
    j = i % len(l)
    k = i // len(l)
    ja = str(l[j])
    if k > len(l):
        ka = _index_to_alphabet(k)
    elif k == 0:
        ka = ''
    else:
        ka = str(l[k - 1])
    return ka + ja


def randomcolor(seed):
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    np.random.seed(seed)
    indices = np.random.randint(0, 14, size=6)
    return '#' + ''.join(np.array(colorArr)[indices])


def _format_color_index(l, control_color_num=2):
    if (type(control_color_num) != int) | (control_color_num < 1):
        raise ValueError('control_color_num must be int and larger than 0!')
    t = pd.Series(l)
    t = t.replace(t.unique(), range(t.nunique()))

    i = 1
    while control_color_num < t.max():
        t = t.replace(range(control_color_num * i, control_color_num * (i + 1)), range(control_color_num))
        i += 1
    return list(t)


def _describe_excel_format(writer, df, sheetname, axis=0, format_index=-1, control_color_num=2, sparkline_f=True):
    if (type(control_color_num) != int) | (control_color_num < 1):
        raise ValueError('control_color_num must be int and larger than 0!')

    df_copy = df.copy()
    if sparkline_f:
        df_copy.insert(0, 'sparkline', value=np.nan)
    df_copy.to_excel(writer, sheetname)
    workbook = writer.book
    worksheet = writer.sheets[sheetname]

    dic_format = {}
    for i in range(control_color_num):
        if i == 0:
            dic_format.update({i: workbook.add_format({'bg_color': '#C5D9F1'})})
        elif i == 1:
            dic_format.update({i: workbook.add_format({'bg_color': '#D9D9D9'})})
        else:
            dic_format.update({i: workbook.add_format({'bg_color': randomcolor(i)})})

    if axis == 0:
        # 对于dataframe，当column非multiindex时，不管column有没有name，打入excel都不会保留name，
        # 但是如果行index有名字，则会和column这一表头平齐。换言之，挤掉本来column name应该在的位置。
        # 如果是multiindex，则name会正常保留，并且会在column下多出一个空行，如果行index存在name，则会在该行对应位置。
        # 从上面的说明，也可以看到行index名字的存储处理方式。
        start_i = len(df_copy.columns.names)
        if start_i > 1:
            start_i += 1
        start_j0 = len(df_copy.index.names)
        start_j = len(df_copy.index.names) + df_copy.shape[1]
        index_list = _format_color_index(list(df_copy.index.labels[format_index]), control_color_num)

        for row in range(df_copy.shape[0]):
            worksheet.set_row(row + start_i, cell_format=dic_format[index_list[row]])
            if sparkline_f:
                dic_row = json.loads(df_copy['distribution'].iloc[row])
                worksheet.write_row(row + start_i, start_j, list(dic_row.values()))
                range_col = _index_to_alphabet(start_j) + str(row + start_i + 1) + ':' + _index_to_alphabet(
                    start_j + len(dic_row) - 1) + str(row + start_i + 1)
                worksheet.add_sparkline(row + start_i, start_j0, {'range': range_col,
                                                                  'type': 'column',
                                                                  'style': index_list[row]+12})
        # 如果隐藏，sparkline出不来
        # worksheet.set_column(_index_to_alphabet(start_j) + ':XFD', None, None, {'hidden': True})

    elif axis == 1:
        start_i = len(df_copy.index.names)
        index_list = _format_color_index(list(df_copy.columns.labels[format_index]), control_color_num)

        for row in range(df_copy.shape[1]):
            worksheet.set_column(row + start_i, row + start_i, cell_format=dic_format[index_list[row]])
    else:
        raise ValueError('axis param must be 1 or 0!')
        # workbook.close()


def merge_excel(report_src, report_dst, format_index=0, control_color_num=2):
    writer = pd.ExcelWriter(report_dst)
    for root, dirs, files in os.walk(report_src):
        if root != report_src:
            for file in files:
                if file.endswith('report'):
                    path = os.path.join(root, file)
                    sheetname = os.path.basename(os.path.dirname(root)) + '_' + file[:-7]
                    df = pd.read_pickle(path)
                    if 'data_stats' in file:
                        _describe_excel_format(writer, df, sheetname, 0, format_index, control_color_num)
                    elif 'woe_eva' in file:
                        _describe_excel_format(writer, df, sheetname, 0, format_index, control_color_num, sparkline_f=False)
                    else:
                        df.to_excel(writer, sheet_name=sheetname,
                                    encoding='gbk')
                if file.startswith('level_report'):
                    workbook = writer.book
                    worksheet = writer.sheets[sheetname]
                    worksheet.insert_image('B18', os.path.join(report_src, 'trainer/report/trainer_eva_report.png'))
    df = pd.DataFrame()
    df.to_excel(writer, sheet_name='transformer_feature_plot')
    # workbook = writer.book
    worksheet = writer.sheets['transformer_feature_plot']
    worksheet.insert_image('B02', os.path.join(report_src, 'transformer/report/train_feature_plot.png'),
                           {'x_scale': 0.5, 'y_scale': 0.5})
    worksheet.insert_image('S02', os.path.join(report_src, 'transformer/report/test_feature_plot.png'),
                           {'x_scale': 0.5, 'y_scale': 0.5})
    workbook.close()

    writer.save()
    writer.close()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class simplejsonEncoder(simplejson.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
