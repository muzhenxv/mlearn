import pandas as pd
import numpy as np
import os
from .visualize import feature_plot, eva_plot
from . import feature_evaluation
from .feature_explore import *
import simplejson
from itertools import chain
from collections import defaultdict
from ..monitor import data_monitor as dm
import pickle
import traceback
from ..base_utils import *
# from ...service.base_utils import *
from ..data_service import get_data
from ..monitor.psi import cmpt_psi
from .feature_explore import DescStats


# TODO: report各函数df_test应该作为可选参数
# TODO: 计算psi时目前只考虑了两个数据集、其中之一为base的情况，对于多数据集情况没有考虑和适应。



def _update_dict(s, k1, k2):
    """
    json to dict，dict值转成频率，外部套上对应数据集名称
    :param s:
    :param k1:
    :param k2:
    :return:
    """
    dic = json.loads(s[k2])
    total = sum(dic.values())
    dic = {k: v / total for k, v in dic.items()}
    return {s[k1]: dic}


def _merge_dict(s):
    """
    不同字典key值对齐，不存在的key填0
    :param s:
    :return:
    """
    dic = {}
    keys_list = []
    for i in s:
        for k, dic_i in i.items():
            keys_list += list(dic_i.keys())
    keys_list = set(keys_list)

    for i in s:
        for outk, dic_i in i.items():
            ii = {k: dic_i.get(k, 0) for k in keys_list}
            i[outk] = ii
        dic.update(i)
    return json.dumps(dic)


def _get_single_plot_data(s, k1, k2):
    return json.dumps(json.loads(s[k2])[s[k1]])


def _get_distribution_psi(s):
    return cmpt_psi(list(json.loads(s[0]).values()), list(json.loads(s[1]).values()))


def _gen_plot_data(df, k, set_index=False):
    tmp = df.reset_index()

    tmp[k] = tmp.apply(lambda s: _update_dict(s, df.index.names[-1], k), axis=1)
    tmp2 = tmp.groupby(df.index.names[0])[k].agg(_merge_dict)
    tmp2 = pd.DataFrame(tmp2)
    tmp2.columns = ['plot']
    tmp = df.join(tmp2, how='outer')
    tmp = tmp.reset_index()
    tmp['distribution'] = tmp.apply(lambda s: _get_single_plot_data(s, df.index.names[-1], 'plot'), axis=1)
    tmp = tmp.set_index(df.index.names)

    t = tmp.groupby(level=0)['distribution'].agg(_get_distribution_psi)
    t.name = 'H-PSI'
    tmp = tmp.join(t, how='outer')
    col_list = list(tmp.columns)
    col_list = col_list[-1:] + col_list[:-3] + col_list[-3:-1]
    tmp = tmp[col_list]

    if set_index:
        tmp = tmp.set_index('plot', append=True).swaplevel(-2, -1)
    return tmp


def _gen_desc_report(df_train_origin, df_test_origin, label, report_dst, decimals=4, gen_report=False):
    df_train = df_train_origin.copy()
    df_test = df_test_origin.copy()
    # data_stats
    enc = DescStats()
    enc.fit(df_train)
    c_tr = enc.transform(df_train)
    c_te = enc.transform(df_test)
    df = merge_multi_df({'train': c_tr, 'test': c_te}, axis=0).round(decimals)
    df = _gen_plot_data(df, 'distribution')
    # TODO: plot列应该放到索引中以图片形式呈现，因为暂时没有解决excel画图插入cell问题，先删除
    del df['plot']

    if gen_report:
        df.to_pickle(os.path.join(report_dst, 'data_stats.report'))
        dic = {'train': table_to_dict(c_tr), 'test': table_to_dict(c_te)}
        simplejson.dump(dic, open(os.path.join(report_dst, 'data_stats.json'), 'w'), ignore_nan=True,
                        cls=simplejsonEncoder)
    return df


def _gen_woe_report(df_train_origin, df_test_origin, label_col, report_dst, decimals=4, gen_report=False):
    df_train = df_train_origin.copy()
    df_test = df_test_origin.copy()
    # woe_report
    wr = feature_evaluation.WOEReport()
    d_tr = wr.fit_transform(df_train, label_col)
    d_te = wr.transform(df_test, label_col)
    df = merge_multi_df({'train': d_tr, 'test': d_te}, axis=1)

    df['group_sample_ratio'] = df['group_sample_ratio'].fillna(0).astype(float)
    df['total_sample_num'] = df['total_sample_num'].fillna(0).astype(float)
    df['overdue_sample_num'] = df['overdue_sample_num'].fillna(0).astype(float)
    df['overdue_ratio'] = df['overdue_ratio'].fillna(0).astype(float)

    t = df.iloc[:, [i for i, v in enumerate(list(df.columns.labels[0])) if v == list(df.columns.levels[0]).index('IV')]]
    del df['IV']
    df = t.join(df, how='outer')
    df.insert(0, 'L-PSI', value=np.nan)
    df.insert(0, 'V-PSI', value=np.nan)
    for i in df.index.levels[0]:
        # TODO： 以下写法在train和test数据集index有重复时会报错,why?
        # df.loc[i, 'V-PSI'] = cmpt_psi(df.loc[i, 'group_sample_ratio']['train'],
        #                               df.loc[i, 'group_sample_ratio']['test'])
        # df.loc[i, 'L-PSI'] = cmpt_psi(df.loc[i, 'overdue_ratio']['train'],
        #                               df.loc[i, 'overdue_ratio']['test'])
        df.loc[i, 'V-PSI'] = cmpt_psi(df.loc[i]['group_sample_ratio']['train'],
                                      df.loc[i]['group_sample_ratio']['test'])
        df.loc[i, 'L-PSI'] = cmpt_psi(df.loc[i]['overdue_ratio']['train'],
                                      df.loc[i]['overdue_ratio']['test'])
    df.round(decimals)

    if gen_report:
        df.to_pickle(os.path.join(report_dst, 'woe_stats.report'))
    return df


def _gen_eva_report(df_train_origin, df_test_origin, label, report_dst, decimals=4, gen_report=False):
    df_train = df_train_origin.copy()
    df_test = df_test_origin.copy()
    # feature_indice
    train_data = feature_evaluation.feature_evaluation(df_train, [label])
    test_data = feature_evaluation.feature_evaluation(df_test, [label])

    df = merge_multi_df({'train': train_data, 'test': test_data}, axis=1).round(decimals=decimals)

    if gen_report:
        df.to_pickle(os.path.join(report_dst, 'feature_indices.report'))

        text = {'train': table_to_dict(train_data), 'test': table_to_dict(test_data)}
        simplejson.dump(text, open(os.path.join(report_dst, 'feature_indices.json'), 'w'), ignore_nan=True,
                        cls=simplejsonEncoder)
    return df


def _gen_woe_eva_report(df_train_origin, df_test_origin, label, report_dst, decimals=4, gen_report=True):
    df1 = _gen_woe_report(df_train_origin, df_test_origin, label, report_dst)
    df2 = _gen_eva_report(df_train_origin, df_test_origin, label, report_dst)
    df2.columns = df2.columns.droplevel(1)
    del df2['IV']
    df2.index.name = 'feature_name'
    df = df2.join(df1, how='outer').round(decimals)

    df = df.reset_index()
    tmp = df[['feature_name', 'IV']]
    tmp.columns = ['_'.join(col).strip() for col in tmp.columns.values]
    tmp = tmp[['feature_name_', 'IV_test']].drop_duplicates().sort_values('IV_test', ascending=False)
    columns = tmp.feature_name_

    df_res = pd.DataFrame()
    for c in columns:
        tmp = df[df['feature_name'] == c]
        try:
            tmp.level = tmp.level.replace('missing', np.nan).astype(float)
        except:
            pass
        tmp = tmp.sort_values([('IV', 'train'), 'feature_name', 'level'], ascending=[False, True, True])
        tmp.level = tmp.level.replace(np.nan, 'missing')
        tmp = tmp.set_index(['feature_name', 'level', 'bin'])
        df_res = pd.concat([df_res, tmp], axis=0)

    if gen_report:
        df_res.to_pickle(os.path.join(report_dst, 'woe_eva_report.report'))
    return df_res


def zeros_deal(test_value, train_value, smooth=0.01):
    difference_rate = 1 - (test_value + smooth) / (train_value + smooth)
    return difference_rate


def stable_different(df, process_name, threshold):
    df_diff = df[df['psi'] >= threshold]
    df_result = pd.DataFrame()
    if df_diff.empty == False:

        for feat_name in df_diff.index.tolist():
            df_result_m = pd.DataFrame()
            df_result_m['feature_name'] = [feat_name]
            df_result_m['process_name'] = process_name
            df_result_m['warning_indice'] = 'PSI'
            df_result_m['warning_reason'] = "数据集的PSI为" + str(df_diff[df_diff.index == feat_name].psi.values[0])
            df_result_m['detailed_data'] = "PSI : " + str(df_diff[df_diff.index == feat_name].psi.values[0])
            df_result = pd.concat([df_result_m, df_result], axis=0)
    return df_result


def woe_different(df, process_name, threshold, woe_columns=['KS', 'AUC', 'IV']):
    df_result_fin = pd.DataFrame()
    for woe_column in woe_columns:
        df[woe_column + '_difference_rate'] = zeros_deal(df[woe_column]['test'], df[woe_column]['train'], smooth=0.01)
        df_diff = df[
            (df[woe_column + '_difference_rate'] >= threshold) | (df[woe_column + '_difference_rate'] <= -threshold)]
        df_result = pd.DataFrame()
        if df_diff.empty == False:
            for feat_name, feat_name_df in df_diff.reset_index().groupby('feature_name'):
                df_result_m = pd.DataFrame()
                df_result_m['feature_name'] = [feat_name]
                df_result_m['process_name'] = process_name
                df_result_m['warning_indice'] = woe_column
                df_result_m['warning_reason'] = "train数据集和test数据集" + woe_column + "差异率为" + format(
                    feat_name_df[woe_column + '_difference_rate'].values[0], '.2%')
                df_result_m['detailed_data'] = woe_column + " train : " + str(
                    feat_name_df[woe_column]['train'].values[0]) + " " + woe_column + " test : " + str(
                    feat_name_df[woe_column]['test'].values[0])
                df_result = pd.concat([df_result_m, df_result], axis=0)
        df_result_fin = pd.concat([df_result, df_result_fin], axis=0)
        df_result_fin
    return df_result_fin


def desc_different(df, process_name, threshold, desc_columns=['cover', '50%']):
    df = df.sort_index(level='feature_name')
    df_result_fin = pd.DataFrame()
    for desc_column in desc_columns:
        diff_rate_m = pd.DataFrame(index=df.index)
        train_i = pd.Series(df.loc[(df.index.levels[0].tolist(), slice(None), ['train']), desc_column])
        test_i = pd.Series(df.loc[(df.index.levels[0].tolist(), slice(None), ['test']), desc_column])
        diff_rate_m['test_difference_rate'] = test_i
        diff_rate_m = diff_rate_m.fillna(method='ffill')
        diff_rate_m['train_difference_rate'] = train_i
        diff_rate_m = diff_rate_m.fillna(method='bfill')
        diff_rate_m[desc_column + '_difference_rate'] = zeros_deal(diff_rate_m['test_difference_rate'],
                                                                   diff_rate_m['train_difference_rate'])
        df_m = df.join(diff_rate_m, how='inner')

        df_diff = df_m[(df_m[desc_column + '_difference_rate'] >= threshold) | (
            df_m[desc_column + '_difference_rate'] <= -threshold)]
        df_result = pd.DataFrame()
        if df_diff.empty == False:
            for feat_name, feat_name_df in df_diff.groupby('feature_name'):
                df_result_m = pd.DataFrame()
                df_result_m['feature_name'] = [feat_name]
                df_result_m['process_name'] = process_name
                df_result_m['warning_indice'] = desc_column
                df_result_m['warning_reason'] = "train数据集和test数据集" + desc_column + "差异率为" + format(
                    feat_name_df[desc_column + '_difference_rate'][0], '.2%')

                df_result_m['detailed_data'] = desc_column + " train : " + str(
                    feat_name_df.loc[(feat_name_df.index.levels[0].tolist(), slice(None), ['train']), desc_column][
                        0]) + " " + desc_column + " test : " + str(
                    feat_name_df.loc[(feat_name_df.index.levels[0].tolist(), slice(None), ['test']), desc_column][0])
                df_result = pd.concat([df_result_m, df_result], axis=0)
        df_result_fin = pd.concat([df_result, df_result_fin], axis=0)

    return df_result_fin


def _gen_warning_report(report_src, report_dst, gen_report=True):
    if os.path.exists(report_src) == False:
        raise ValueError('%s is not exists!' % report_src)
    else:
        df_result = pd.DataFrame()
        for root, dirs, files in os.walk(report_src):
            if root != report_src:
                for file in files:
                    if file.endswith('report'):
                        path = os.path.join(root, file)
                        sheetname = os.path.basename(os.path.dirname(root)) + '_' + file[:-7]
                        if 'woe_eva' in file:
                            df_woe = pd.read_pickle(path)
                            df_diff_m = woe_different(df_woe, sheetname, 0.2)
                        elif 'stable_test' in file:
                            df_stable = pd.read_pickle(path)
                            df_diff_m = stable_different(df_stable, sheetname, 0.2)
                        elif 'data_stats' in file:
                            df_desc = pd.read_pickle(path)
                            df_diff_m = desc_different(df_desc, sheetname, 0.2)
                        else:
                            df_diff_m = pd.DataFrame()
                        df_result = pd.concat([df_diff_m, df_result], axis=0)
        df_result = df_result.sort_values('feature_name')
        df_result = df_result.reset_index(drop=True)
    if gen_report:
        if not os.path.exists(os.path.join(report_dst, 'reporter')):
            os.makedirs(os.path.join(report_dst, 'reporter'))
        df_result.to_pickle(os.path.join(report_dst, 'reporter', 'warning_report.report'))
    return df_result


def _gen_stable_report(df_train_origin, df_test_origin, label, report_dst, decimals=4, gen_report=True,
                       covariate_shift_eva=True):
    df_train_cp = df_train_origin.copy()
    df_test_cp = df_test_origin.copy()
    # stable_test
    dic, mcc, weights = dm.consistent_dataset_test(df_train_cp, df_test_cp, label, report_dst=None,
                                                   covariate_shift_eva=covariate_shift_eva)

    df = pd.DataFrame(dic).T
    df = pd.concat([df, pd.DataFrame({'mcc': {'All_Dataset': mcc}})], axis=0).round(decimals)

    if gen_report:
        df.to_pickle(os.path.join(report_dst, 'stable_test.report'))
        stable_test = {'single_feature_test': dic, 'mcc': mcc}
        simplejson.dump(stable_test, open(os.path.join(report_dst, 'stable_test.json'), 'w'), ignore_nan=True,
                        cls=simplejsonEncoder)
        pickle.dump(weights, open(os.path.join(report_dst, 'weights.pkl'), 'wb'))
    return df


def _gen_eda_report(df_train_origin, df_test_origin, label, report_dst, decimals=4, gen_report=True,
                    covariate_shift_eva=True):
    tmp = _gen_woe_eva_report(df_train_origin, df_test_origin, label, report_dst, decimals=decimals,
                              gen_report=gen_report)
    tmp = _gen_desc_report(df_train_origin, df_test_origin, label, report_dst, decimals=decimals, gen_report=gen_report)
    tmp = _gen_stable_report(df_train_origin, df_test_origin, label, report_dst, decimals=decimals,
                             gen_report=gen_report,
                             covariate_shift_eva=covariate_shift_eva)
    return


def reporter_report(report_src, report_dst=None):
    """

    :param report_dst: 工作流的根目录
    :return:
    """
    if report_dst is None:
        report_dst = os.path.join(report_src, 'reporter')
        if not os.path.exists(report_dst):
            os.makedirs(report_dst)
    tmp = _gen_warning_report(report_src, report_dst)
    merge_excel(report_src, os.path.join(report_dst, 'all_report.xlsx'))
    return


def sampler_report(sample_num, dic, group_ratio, report_dst):
    """
    报告分组基准，分组前后样本分布对比
    :param sample_num:
    :param dic:
    :param group_ratio:
    :param report_dst:
    :return:
    """
    simplejson.dump(sample_num, open(os.path.join(report_dst, 'sample_num.json'), 'w'), ignore_nan=True,
                    cls=simplejsonEncoder)
    simplejson.dump(dic, open(os.path.join(report_dst, 'dic.json'), 'w'), ignore_nan=True, cls=simplejsonEncoder)
    simplejson.dump(group_ratio, open(os.path.join(report_dst, 'group_ratio.json'), 'w'), ignore_nan=True,
                    cls=simplejsonEncoder)


def spliter_report(X_train, X_test, time_col, label_col, report_dst, gen_report=True, covariate_shift_eva=True):
    """
    覆盖率、饱和度统计，数据描述性统计和直方图，目标分布统计
    :param df_train:
    :param df_test:
    :param time_col:
    :param label_col:
    :param report_dst:
    :return:
    """
    df_train = X_train.copy()
    df_test = X_test.copy()

    # # TODO: 等待删除，暂时前端需要该输出
    # c_tr = cover_stats(df_train)
    # c_te = cover_stats(df_test)
    # c_tr.to_pickle(os.path.join(report_dst, 'c_tr.pkl'))
    # c_te.to_pickle(os.path.join(report_dst, 'c_te.pkl'))
    #
    # dic = {'train': table_to_dict(c_tr), 'test': table_to_dict(c_te)}
    # simplejson.dump(dic, open(os.path.join(report_dst, 'cover_stats.json'), 'w'), ignore_nan=True,
    #                 cls=simplejsonEncoder)


    d_tr = target_stats(df_train, label_col, time_col)
    d_te = target_stats(df_test, label_col, time_col)
    dic = {'train': d_tr, 'test': d_te}
    df = pd.DataFrame(dic).T
    df.to_pickle(os.path.join(report_dst, 'target_stats.report'))
    simplejson.dump(dic, open(os.path.join(report_dst, 'target_stats.json'), 'w'), ignore_nan=True,
                    cls=simplejsonEncoder)

    del df_train[time_col], df_test[time_col]

    _gen_eda_report(df_train, df_test, label_col, report_dst, gen_report=gen_report,
                    covariate_shift_eva=covariate_shift_eva)


def transformer_report(df_train_origin, df_test_origin, label, report_dst, gen_report=True, covariate_shift_eva=True):
    """
    单变量分析、直方排序图分析、train/test特征一致性分布分析
    :param df_train:
    :param df_test:
    :param label:
    :param report_dst:
    :return:
    """
    df_train = df_train_origin.copy()
    df_test = df_test_origin.copy()

    _gen_eda_report(df_train_origin, df_test_origin, label, report_dst, gen_report=gen_report,
                    covariate_shift_eva=covariate_shift_eva)

    # feature_plot
    df_train.fillna(-999, inplace=True)
    train_plot_data = feature_plot(df_train, label, path=os.path.join(report_dst, 'train_feature_plot.png'))
    df_test.fillna(-999, inplace=True)
    test_plot_data = feature_plot(df_test, label, path=os.path.join(report_dst, 'test_feature_plot.png'))

    train_plot_data = {k: {'train': v} for k, v in train_plot_data.items()}
    test_plot_data = {k: {'test': v} for k, v in test_plot_data.items()}
    plot_data = defaultdict(dict)
    for k, v in chain(train_plot_data.items(), test_plot_data.items()):
        plot_data[k].update(v)
    simplejson.dump(plot_data, open(os.path.join(report_dst, 'plot_data.json'), 'w'), ignore_nan=True,
                    cls=simplejsonEncoder)


def filter_report(enc, df_train, df_test, label, report_dst, gen_report=True, covariate_shift_eva=True):
    """
    过滤指标得分输出
    :param enc:
    :param score_func:
    :param report_dst:
    :return:
    """
    data = {}
    for encoder_dict in enc.st:
        df = encoder_dict['enc'].features_report
        data[encoder_dict['method']] = table_to_dict(df)

    simplejson.dump(data, open(os.path.join(report_dst, 'feature_indices.json'), 'w'), ignore_nan=True,
                    cls=simplejsonEncoder)

    df = pd.DataFrame()
    for encoder_dict in enc.st:
        t = encoder_dict['enc'].features_report
        t.columns = pd.MultiIndex.from_product([[encoder_dict['method']], t.columns])
        df = pd.concat([df, t], axis=1)

    t = pd.Series(True, index=df.index)
    for c in df.columns:
        if c[-1].endswith('_support'):
            t = t & df[c]
    df['final_supprt'] = t

    df.round(4).to_pickle(os.path.join(report_dst, 'feature_filter.report'))

    _gen_eda_report(df_train, df_test, label, report_dst, gen_report=gen_report,
                    covariate_shift_eva=covariate_shift_eva)


def optimizer_report(all_df, report_dst):
    """
    每轮迭代参数和效果的输出
    :param all_df:
    :param report_dst:
    :return:
    """
    all_df.to_pickle(os.path.join(report_dst, 'optimizer_result.report'))
    """
    Currently (as of Pandas version 0.18), df.to_dict('records') accesses the NumPy array df.values. 
    This property upcasts the dtype of the int column to float so that the array can have a single common dtype. 
    After this point there is no hope of returning the desired result -- all the ints have been converted to floats.
    refers: https://stackoverflow.com/questions/37897527/get-python-pandas-to-dict-with-orient-records-but-without-float-cast
    """
    dic = [{col: getattr(row, col) for col in all_df} for row in all_df.itertuples()]
    simplejson.dump(dic, open(os.path.join(report_dst, 'all_params_result.json'), 'w'), ignore_nan=True,
                    cls=simplejsonEncoder)


def trainer_report(enc, df_train, df_val, test_src, test_data_dst, report_dst):
    """
    训练报告
    :param enc:
    :param df_train:
    :param df_val:
    :param df_test:
    :param test_m:
    :param report_dst:
    :return:
    """
    if test_src is not None:
        df_test = enc.transform(test_src)
        df_test.to_pickle(test_data_dst)
        test_m = 'oot'
    else:
        test_m = 'test'

    df = get_data(test_src)

    try:
        df_label = df.pop(enc.label)
        df_label.name = 'y_true'
    except:
        df_label = pd.DataFrame()

    if enc.enc._estimator_type == 'ruler':
        tmp = enc.enc.get_metrics(df, df_label)
        tmp.round(4).to_pickle(os.path.join(report_dst, 'rule_report.report'))
        tmp = table_to_dict(tmp)
        simplejson.dump(tmp, open(os.path.join(report_dst, 'rule_report.json'), 'w'), ignore_nan=True,
                        cls=simplejsonEncoder)

    try:
        rr = feature_evaluation.ResultReport()
        rr.fit(df_train.y_pred)
        df = rr.transform({'train': df_train, 'val': df_val, test_m: df_test})
        df.to_pickle(os.path.join(report_dst, 'level_report.report'))
        pickle.dump(rr, open(os.path.join(report_dst, 'rr_enc.pkl'), 'wb'))

        eval_plot_path = os.path.join(report_dst, 'trainer_eva_report.png')
        dic_all = eva_plot({'train': [df_train.y_true, df_train.y_pred], 'val': [df_val.y_true, df_val.y_pred],
                            test_m: [df_test.y_true, df_test.y_pred]}, path=eval_plot_path)
        import re
        from collections import defaultdict
        dic = defaultdict(dict)
        token = re.compile('[\d.]+')
        for k in dic_all['roc_curve']:
            for v in dic_all['ks_curve'][k]:
                if v.startswith('KS'):
                    s = float(re.findall(token, v)[0])
                    #             dic[k].append({'ks': s})
                    dic[k]['ks'] = s
        for v in dic_all['roc_curve']['train'].keys():
            k1, k2 = v.split('(')
            k1 = k1.strip()
            s = float(re.findall(token, k2)[0])
            #     dic[k1].append({'auc': s})
            dic[k1]['auc'] = s
        # df = pd.DataFrame.from_dict(dic)
        simplejson.dump(dict(dic), open(os.path.join(report_dst, 'abstract_data.json'), 'w'), ignore_nan=True,
                        cls=simplejsonEncoder)
        simplejson.dump(dic_all, open(os.path.join(report_dst, 'eva_data.json'), 'w'), ignore_nan=True,
                        cls=simplejsonEncoder)

        pd.DataFrame(dic).T.to_pickle(os.path.join(report_dst, 'abstract_data.report'))

        importances = getattr(enc.enc, "feature_importances_", None)
        coef_ = getattr(enc.enc, "coef_", None)
        rules_ = getattr(enc.enc, "rules_", None)
        if not (importances is None and coef_ is None):
            try:
                df_features = \
                    pd.DataFrame(
                        {'feature_names': enc.columns, 'feature_importances': get_feature_importances(enc.enc)})[
                        ['feature_names', 'feature_importances']]
            except Exception as e:
                print(traceback.format_exc())
                df_features = pd.DataFrame()
        elif rules_ is not None:
            try:
                df_features = pd.DataFrame({'rules': enc.enc.rules_})
            except Exception as e:
                print(traceback.format_exc())
                df_features = pd.DataFrame()
        else:
            df_features = pd.DataFrame()

        simplejson.dump(df_features.to_dict(orient='records'),
                        open(os.path.join(report_dst, 'feature_importances.json'), 'w'), ignore_nan=True,
                        cls=simplejsonEncoder)

        pd.DataFrame(df_features).round(4).to_pickle(os.path.join(report_dst, 'feature_importances.report'))
    except Exception as e:
        print(traceback.format_exc())

    report_src = os.path.dirname(os.path.dirname(report_dst))
    reporter_report(report_src)
    return
