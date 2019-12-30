import os
import subprocess
import traceback

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import numpy as np
import pandas as pd
from sklearn import metrics as mr
from sklearn.tree import export_graphviz

from . import feature_evaluation
from collections import defaultdict


def visualize_tree(dt, path=None, feature_names=None):
    """
    决策树可视化
    :param dt: sklearn.tree.DecisionTreeClassifier().fit后的实例
    :param path: 图片保存父路径
    :param feature_names: 特征名，默认取data.columns
    :return:
    """
    if path is None:
        dot_path = 'dt.dot'
        png_path = 'dt.png'
    else:
        dot_path = os.path.join(path, 'dt.dot')
        png_path = os.path.join(path, 'dt.png')

    if feature_names is None:
        feature_names = ['feature_%s' % i for i in range(dt.n_features_)]

    with open(dot_path, 'w') as f:
        export_graphviz(dt, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", dot_path, "-o", png_path]
    try:
        subprocess.check_call(command)
    except Exception as e:
        print(traceback.format_exc())
        print("Could not run dot, ie graphviz, to produce visualization")


def pr_curve(y_true, y_pred, ax, loc='best'):
    p, r, _ = mr.precision_recall_curve(y_true, y_pred)
    f1 = 2 * p * r / (p + r)

    ax.plot(r, p, label='P-R curve')
    ax.plot(r, f1, label='F1 curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall Rate')
    ax.set_ylabel('Precision Rate')
    ax.set_title('P-R curve')
    ax.legend(loc=loc)
    return ax


def roc_line(y_true, y_pred, ax, label='', loc='best'):
    fpr, tpr, _ = mr.roc_curve(y_true, y_pred)
    auc = mr.roc_auc_score(y_true, y_pred)

    ax.plot(fpr, tpr, label=label + ' (auc = %0.2f)' % auc)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.legend(loc=loc)


def roc_curve(data, ax):
    """

    :param data: dict. i.e. dict. i.e. {'model1': [y_true1, y_pred1], 'model2': [y_true2, y_pred2]]}
    :param ax:
    :return:
    """
    for label, v in data.items():
        y_true, y_pred = v[0], v[1]
        roc_line(y_true, y_pred, ax, label=label)
    return ax


def roc_curve_for_demo(data, ax, loc='best'):
    """

    :param data: dict. i.e. dict. i.e. {'model1': [y_true1, y_pred1], 'model2': [y_true2, y_pred2]]}
    :param ax:
    :return:
    """
    for label, v in data.items():
        y_true, y_pred = v[0], v[1]
        fpr, tpr, _ = mr.roc_curve(y_true, y_pred)
        auc = mr.roc_auc_score(y_true, y_pred)
        ax.plot(fpr, tpr, label=label + ' (auc = %0.2f)' % auc)
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC curve')
    ax.legend(loc=loc)
    return ax


# def roc_curve(data, ax):
#     """
#
#     :param data: list. i.e. [[y_true1, y_pred1, 'model1'], [y_true2, y_pred2, 'model2']]]
#     :param ax:
#     :return:
#     """
#     for y_true, y_pred, label in data:
#         roc_line(y_true, y_pred, ax, label=label)
#     return ax



def ks_curve(y_true, y_pred, ax, loc='best', label='', lw=2):
    t = pd.DataFrame({'y_t':list(y_true), 'y_p':list(y_pred)}).sort_values('y_p', ascending=False)
    positive_sum = np.sum(y_true)
    negtive_sum = len(y_true) - positive_sum
    t['tpr'] = t.y_t.cumsum() / positive_sum
    t['fpr'] = (1-t.y_t).cumsum() / negtive_sum
    t['proportion_single'] = t.y_p.map(t.y_p.value_counts(1))
    t = t.drop_duplicates('y_p', keep='first')
    t['proportion'] = t.proportion_single.cumsum()
    neg, pos, thr, prob = np.array([0] + t.fpr.tolist()+[1]), np.array([0] + t.tpr.tolist()+[1]), \
                          np.array([1.0001] + t.y_p.tolist() + [0]), np.array([0] + t.proportion.tolist()+[1])
    ks = pos - neg

    # # roc_curve方式
    # neg, pos, thr = mr.roc_curve(y_true, y_pred)
    # ks = pos - neg
    # t = sorted(y_pred)
    # s = len(y_pred)
    # # method1: 该方法速度太慢
    # # prob = [sum((df['y_pred'] >= i)) / df.shape[0] for i in thr]
    # # prob = [0] + [(s - t.index(i)) / s for i in thr[1:]]
    # # method2: 该方法得到的横坐标比例不精确，尤其当大量样本集中在某一个概率值时，对应横坐标——样本比例会严重错误
    # prob = [1.0 * i / len(pos) for i in range(len(pos))]

    threshold = prob[np.argmax(np.array(pos) - np.array(neg))]
    max_ks = np.max(np.array(pos) - np.array(neg))
    best_thr = thr[np.argmax(np.array(pos) - np.array(neg))]

    ax.plot(prob, pos, label='TPR')
    ax.plot(prob, neg, label='FPR')
    ax.plot(prob, ks, label='KS (%0.2f)' % max_ks)
    ax.plot([threshold, threshold], [0, 1], lw=lw, linestyle='--',
            label=' best passing rate (%0.0f%%)\n(best threshold = %0.2f)' % (100 - threshold * 100, best_thr))
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f'{label} dataset K-S curve')
    ax.legend(loc=loc)
    return ax


# deprecated
def boost_hist_curve(x, y, ax, target=None, max_depth=5, min_samples_leaf=0.01, method='avg', cut_points=None,
                     rotation=55):
    if target is None:
        target = ''
    df = pd.DataFrame()
    if method != 'avg':
        th = feature_evaluation.cut_points(x, y, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        x2 = feature_evaluation.feature_discretion(x, th)
    else:
        if cut_points is None:
            x2 = pd.qcut(np.array(x), q=10, duplicates='drop')
        else:
            x2 = pd.cut(np.array(x), bins=cut_points, include_lowest=True)
    df['x'] = np.array(x2)
    df['y'] = np.array(y)
    t = pd.pivot_table(df, values='y', index='x', aggfunc=[np.mean, len]).reset_index().sort_values('x',
                                                                                                    ascending=False)
    t.columns = ['x', 'y', 'l']

    ax.bar(range(t.shape[0]), t.l, tick_label=t.x, color='lightblue', label='volume', yerr=0.00001, width=0.5)
    ax.set_ylim([0, max(t.l) * 1.2])
    ax.set_ylabel('volume')
    # ax.legend(loc="upper right")
    ax.set_xticklabels(t.x, rotation=rotation)

    ax2 = ax.twinx()

    ax2.plot(range(t.shape[0]), t.y, color='r', label='test dataset')
    ax.set_xticklabels(t.x, rotation=rotation)
    ax2.set_title('sorting ability for %s' % target)
    ax2.set_ylabel('%s Overdue Rate' % target)
    ax2.legend(loc="upper right")


def sorting_ability(y_true, y_pred, ax, cut_points=None, label='', width=0.5, bins=10, loc='best',
                    plt_label='overdue rate'):
    df = sta_groups(y_true, y_pred, bins=bins, cut_points=cut_points)

    ax1 = ax
    ax2 = ax1.twinx()  # 创建第二个坐标轴

    # 条形图
    plt_x = df.y_pred_range
    plt_y = df['count']
    ax2.bar(plt_x, plt_y, width, alpha=0.5)
    upper2 = np.max(plt_y) * 1.2

    # 折线图
    plt_y = round(df.single_overdue_rate, 3) * 100
    ax1.plot(plt_x, plt_y, linestyle='--', label=plt_label, marker='o', markersize=5)
    upper1 = np.max(plt_y) * 1.2

    plt_y = round(df.acc_overdue_rate, 3) * 100
    ax1.plot(plt_x, plt_y, linestyle='--', label='accumulate ' + plt_label, marker='o', markersize=5)

    ax1.set_xlabel('Groups(good -> bad)')  # fontsize使用方法和plt.xlabel()中一样
    ax1.set_ylabel('Percentage(%)')
    ax2.set_ylabel('The number of person')

    #     ax1.set_xlim([0.5, 10.5])
    ax1.set_ylim([0.0, upper1])
    ax1.set_xticks(df.y_pred_range)

    #     ax2.set_xlim([0.5, 10.5])
    ax2.set_ylim([0.0, upper2])
    ax1.legend(loc=loc)
    ax1.set_title(f'{label} dataset sorting ability')

    df.single_overdue_rate = round(df.single_overdue_rate, 3) * 100
    df.acc_overdue_rate = round(df.acc_overdue_rate, 3) * 100
    df = df[['y_pred_range', 'count', 'single_overdue_rate', 'acc_overdue_rate']]
    df.columns = ['x', 'volume', 'single_overdue_rate', 'acc_overdue_rate']
    return df


def plot_acc_od_ps_rc(y_true, y_pred, ax, cut_points=None, bins=10, label='', loc='best', plt_label='overdue rate'):
    df = sta_groups(y_true, y_pred, bins=bins, cut_points=cut_points)
    #### 累积逾期率、通过率和好人召回率图
    plt_x = df.y_pred_range
    # 累积逾期率
    plt_y = round(df.acc_overdue_rate, 3) * 100
    ax.plot(plt_x, plt_y,
            linestyle='--', label='accumulate ' + plt_label,
            marker='o', markersize=4)
    for i, (a, b) in enumerate(zip(plt_x, plt_y)):
        ax.text(a, b + 1, f'{round(b, 3)}%', ha='center', va='bottom', fontsize=8)

    # 通过率
    plt_y = round(df.acc_pass_rate, 3) * 100
    ax.plot(plt_x, plt_y,
            linestyle='--', label='accumulate passing rate',
            marker='o', markersize=4)
    for i, (a, b) in enumerate(zip(plt_x, plt_y)):
        ax.text(a + 0.2, b - 4, f'{round(b, 3)}%', ha='center', va='bottom', fontsize=8)

    # 累积好人召回率
    plt_y = round(df.acc_recall_rate_good, 3) * 100
    ax.plot(plt_x, plt_y,
            linestyle=':', label='accumulate expected person recall rate',
            marker='o', markersize=4)
    for i, (a, b) in enumerate(zip(plt_x, plt_y)):
        ax.text(a, b + 2, f'{round(b, 3)}%', ha='center', va='bottom', fontsize=8)

    ax.set_xlim([0.3, 10.5])
    ax.set_ylim([0.0, 105])
    ax.set_xticks(df.y_pred_range)
    ax.set_xlabel('Groups(good -> bad)');
    ax.set_ylabel('Percentage(%)')
    ax.set_title(f'{label} dataset recall-pass curve')
    ax.legend(loc=loc)
    return ax


def plot_lift(y_true, y_pred, ax, label='', n_cut=100):
    """Plot Lift curve.

    Parameters:
    -----------
    y_true: array_like, true binary labels

    y_pred: array_like, predicted probability estimates

    Returns:
    --------

    """
    y_true = pd.Series(list(y_true))
    y_pred = pd.Series(list(y_pred))

    qcut, ret = pd.qcut(-y_pred, q=n_cut, duplicates='drop', retbins=True)
    ncut = len(qcut.cat.categories)
    while ncut != qcut.nunique():
        qcut, ret = pd.qcut(-y_pred, q=ncut - 1, duplicates='drop', retbins=True)
        ncut = len(qcut.cat.categories)
    qcut_cat = qcut.cat.categories

    t = (qcut.value_counts().sort_index().cumsum() / qcut.shape[0])
    t.index = t.index.astype(str)
    dic = t.to_dict()

    qcut = qcut.astype(str).map(dic)
    overall_resp = np.mean(y_true)
    sample_size = [1 for i in range(len(y_true))]
    qcut_cumresp = pd.DataFrame({'response': y_true}).groupby(qcut).agg(sum).sort_index().cumsum()
    qcut_cumsamp = pd.DataFrame({'count': sample_size}, index=y_true.index).groupby(qcut).agg(sum).sort_index().cumsum()
    lift = qcut_cumsamp.join(qcut_cumresp)
    lift['lift'] = lift['response'] / (lift['count'] * overall_resp)

    ax.plot(lift.index, lift.lift, label='Lift Curve')

    li_o = -1
    for i in [1, 5, 10]:
        li = int(len(lift) * i / 100)
        if li == li_o:
            continue
        try:
            cut_x = lift.index[li]
            cut_y = lift.iloc[li, 2]
            ax.plot([cut_x], [cut_y], 'o', label='Top ' + str(int((100 * cut_x))) + 'pct Lift: %0.1f' % cut_y)
            li_o = li
        except:
            pass

    ax.set_xlabel('Proportion')
    ax.set_ylabel('Lift')
    ax.set_title(f'{label} Lift Curve')
    ax.legend(loc="best")
    return ax


def twinx_to_dict(df, c=None):
    if c is None:
        c = 'Groups(good -> bad)'
    dic_all = {}
    l = [i for i in df.columns if i != 'x']
    for col in l:
        dic = {}
        li = [{'x': x, 'y': y} for x, y in zip(df.x, df[col])]
        dic['values'] = li
        if col == 'volume':
            dic['type'] = 'bar'
            dic['ylabel'] = 'volume'
        else:
            dic['type'] = 'line'
            dic['ylabel'] = 'percentage(%)'
        dic['title'] = 'sorting ability'
        dic['xlabel'] = c
        dic_all[col] = dic
    return dic_all


def ax_to_dict(ax):
    if type(ax) == pd.DataFrame:
        dic_all = twinx_to_dict(ax)
    else:
        dic_all = {}
        for line in ax.lines:
            if line.get_label().startswith('_'):
                continue
            dic = {}
            li = [{'x': x, 'y': y} for x, y in zip(line.get_xdata(), line.get_ydata())]
            if len(li) > 100:
                li = [li[0]] + li[1:-1][::(len(li) // 100)] + [li[-1]]
            dic['values'] = li
            if len(li) < 5:
                dic['type'] = 'scatter'
            else:
                dic['type'] = 'line'
            dic['title'] = ax.get_title()
            dic['xlabel'] = ax.get_xlabel()
            dic['ylabel'] = ax.get_ylabel()
            dic_all[line.get_label()] = dic

    return dic_all


# 临时for demo
def eva_plot(data, bins=10, figsize=(7, 4), plt_label='overdue rate', n_cut=100, path=None, cut_points=None,
             save_fig=True):
    """

    :param data: dict. i.e. dict. i.e. {'model1': [y_true1, y_pred1], 'model2': [y_true2, y_pred2]]}
    :param bins:
    :param figsize:
    :param path:
    :param cut_points:
    :param save_fig:
    :return:
    """
    if cut_points is None:
        for label, v in data.items():
            y_true, y_pred = v[0], v[1]
            if label == 'train':
                _, cut_points = pd.qcut(y_pred, q=10, retbins=True, precision=8, duplicates='drop')
                cut_points = list(cut_points)[1:-1]
                cut_points.append(np.inf)
                cut_points.insert(0, -np.inf)
                break

    fig2 = plt.figure(figsize=(figsize[0] * len(data), figsize[1] * 5))
    spec2 = gridspec.GridSpec(5, len(data))
    ax1 = fig2.add_subplot(spec2[0, 0])

    from collections import defaultdict
    dic_all = defaultdict(dict)

    tmp0 = roc_curve_for_demo(data, ax1)

    i = 0
    for label, v in data.items():
        y_true, y_pred = v[0], v[1]
        ax2 = fig2.add_subplot(spec2[1, i])
        tmp = ks_curve(y_true, y_pred, ax2, label=label)
        dic_all['ks_curve'][label] = ax_to_dict(tmp)
        ax3 = fig2.add_subplot(spec2[2, i])
        tmp = sorting_ability(y_true, y_pred, ax3, bins=bins, cut_points=cut_points, label=label, plt_label=plt_label)
        dic_all['sorting_ability'][label] = ax_to_dict(tmp)
        ax4 = fig2.add_subplot(spec2[3, i])
        tmp = plot_acc_od_ps_rc(y_true, y_pred, ax4, bins=bins, cut_points=cut_points, label=label, plt_label=plt_label)
        dic_all['recall_accuracy'][label] = ax_to_dict(tmp)
        ax5 = fig2.add_subplot(spec2[4, i])
        tmp = plot_lift(y_true, y_pred, ax5, label=label, n_cut=n_cut)
        dic_all['lift_curve'][label] = ax_to_dict(tmp)

        dic_all['roc_curve'][label] = ax_to_dict(tmp0)

        i += 1
    plt.tight_layout()

    if save_fig:
        if path is None:
            if not os.path.exists('reportsource'):
                os.mkdir('reportsource')
            path = 'reportsource/eva_plot.png'
        plt.savefig(path)
        # plt.show()
    return dic_all


def feature_curve(x, y, ax, target=None, max_depth=5, min_samples_leaf=0.01, method='avg', bins=10, cate_thr=20, cut_points=None,
                  x_points=None, rotation=55, ylim=[0, 1], plt_label='Overdue Rate'):
    if target is None:
        target = ''
    df = pd.DataFrame()
    if cut_points is not None:
        x2 = pd.cut(np.array(x), bins=cut_points, include_lowest=True)
    elif x_points is not None:
        x2 = x
    else:
        if x.nunique() > cate_thr:
            if method != 'avg':
                th = feature_evaluation.cut_points(x, y, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                x2 = feature_evaluation.feature_discretion(x, th)
            else:
                x2 = pd.qcut(np.array(x), q=bins, duplicates='drop')
        else:
            x_points = x.unique()
            x2 = x
    df['x'] = np.array(x2)
    df['x'] = df['x']
    df['y'] = np.array(y)
    t = pd.pivot_table(df, values='y', index='x', aggfunc=[np.mean, len]).reset_index().sort_values('x',
                                                                                                    ascending=True)
    t.columns = ['x', 'overdue_rate', 'volume']

    if (cut_points is None) and (x_points is not None):
        t2 = pd.DataFrame(x_points, columns=['x'])
    else:
        t2 = pd.DataFrame(x2.categories, columns=['x'])
    t = t.merge(t2, on='x', how='outer')
    t.iloc[:, 1:] = t.iloc[:, 1:].fillna(0)
    t = t.sort_values('x')

    ax.bar(range(t.shape[0]), t['volume'], tick_label=t.x, label='volume', yerr=0.00001, width=0.5, alpha=0.5)
    ax.set_ylim([0, max(t['volume']) * 1.2])
    ax.set_ylabel('volume')
    # ax.legend(loc="upper right")
    ax.set_xticklabels(t.x, rotation=rotation)

    ax2 = ax.twinx()

    ax2.plot(range(t.shape[0]), t['overdue_rate'], label='overdue rate', linestyle='--', marker='o', markersize=5)
    # ax.set_xticklabels(t['overdue_rate'], rotation=rotation)
    ax2.set_title('%s' % target)
    ax2.set_ylim(ylim)
    ax2.set_ylabel(plt_label)
    ax2.legend(loc="upper right")

    return t


def feature_plot(df_o, y, bins=10, figsize=(7, 4), plt_label='overdue rate', col_num=3, path=None, cut_points=None,
                 ylim=None, save_fig=True):
    """

    :param data: dict. i.e. dict. i.e. {'model1': [y_true1, y_pred1], 'model2': [y_true2, y_pred2]]}
    :param bins:
    :param figsize:
    :param path:
    :param cut_points:
    :param save_fig:
    :return:
    """
    df = df_o.copy()
    if type(y) == str:
        y = df.pop(y)
    row_num = df.shape[1] // col_num
    if df.shape[1] % col_num > 0:
        row_num += 1
    # TODO: 服务器报错valueerror，疑似dpi过大，临时解决
    fig2 = plt.figure(figsize=(figsize[0] * col_num, figsize[1] * row_num))
    spec2 = gridspec.GridSpec(row_num, col_num)

    dic_all = {}
    for i, v in enumerate(df.columns):
        row_n = i // col_num
        col_n = i % col_num
        ax2 = fig2.add_subplot(spec2[row_n, col_n])
        tmp = feature_curve(df[v], y, ax2, target=v, plt_label=plt_label, ylim=ylim)
        dic_all[v] = twinx_to_dict(tmp, v)
    plt.tight_layout()

    if save_fig:
        if path is None:
            if not os.path.exists('reportsource'):
                os.mkdir('reportsource')
            path = 'reportsource/feature_plot.png'
        plt.savefig(path)
    # plt.show()
    return dic_all


def single_feature_plot_by_partition(df_o, y, partition, col, bins=10, figsize=(7,4), plt_label='', col_num=3, path=None, cut_points=None,
                 save_fig=True):
    df = df_o.copy()
    if type(y) == str:
        y = df.pop(y)
    row_num = df[partition].nunique() // col_num
    if df[partition].nunique() % col_num > 0:
        row_num += 1
    fig2 = plt.figure(figsize=(figsize[0] * col_num, figsize[1] * row_num))
    spec2 = gridspec.GridSpec(row_num, col_num)

    dic_all = {}
    for i, v in enumerate(df[partition].unique()):
        row_n = i // col_num
        col_n = i % col_num
        ax2 = fig2.add_subplot(spec2[row_n, col_n])
        tmp = feature_curve(df[col][df[partition]==v], y[df[partition]==v], ax2, target=str(col) + ': ' + str(v), plt_label=plt_label, cut_points=cut_points)
        dic_all[str(col) + ': ' + str(v)] = twinx_to_dict(tmp, str(col) + ': ' + str(v))
    plt.tight_layout()
    
    if save_fig:
        if path is None:
            if not os.path.exists('reportsource'):
                os.mkdir('reportsource')
            path = 'reportsource/%s.png'%(str(col) + '_by_' + str(partition))
        plt.savefig(path)
    # plt.show()
    return dic_all

def feature_plot_by_partition(df_o, y, partition_list, col_list=None, bins=10, figsize=(7,4), plt_label='', col_num=3, dst=None, cut_points=None,
                 save_fig=True):
    if col_list is None:
        col_list = list(df_o.columns)
        if type(y) == str:
            col_list = [c for c in col_list if (c!=y) & (c not in partition_list)]

    if save_fig:
        if dst is None:
            dst = 'reportsource'
        if not os.path.exists(dst):
            os.mkdir(dst)
        for partition in partition_list:
            if not os.path.exists(os.path.join(dst, partition)):
                os.mkdir(os.path.join(dst, partition))            

    dic_all_list = defaultdict(dict)
    for partition in partition_list:
        for col in col_list:
            if df_o[col].nunique() <= 20:
                cut_points = None
            else:
                _, cut_points = pd.qcut(df_o[col], q=10, retbins=True, duplicates='drop')
                cut_points[0] = -np.inf
                cut_points[-1] = np.inf
                cut_points = sorted(list(set(cut_points)))
            path = os.path.join(dst, partition, str(col) + '_by_' + str(partition) + '.png')
            dic_all = single_feature_plot_by_partition(df_o, y, partition, col, bins, figsize, plt_label, col_num, path, cut_points, save_fig)
            dic_all_list[partition][col] = dic_all
    return dic_all_list


def feature_plot_for_contrast(df_o, y, partition, col_list=None, bins=10, cate_thr=20, figsize=(7,4), ylim=[0, 1], plt_label='', col_num=3, path=None, cut_points=None,
                 base_partition='train', save_fig=True):
    if base_partition not in df_o[partition].unique():
        raise ValueError('the value of pararm base_partition is not in dataset!')

    if col_list is None:
        col_list = list(df_o.columns)
        if type(y) == str:
            col_list = [c for c in col_list if (c!=y) & (c != partition)]
            
    df = df_o.copy()
    if type(y) == str:
        y = df.pop(y)
              
    col_num = int(df[partition].nunique())
    row_num = len(col_list)
    fig2 = plt.figure(figsize=(figsize[0] * col_num, figsize[1] * row_num))
    spec2 = gridspec.GridSpec(row_num, col_num)
            
    dic_all = {}
    for j, col in enumerate(col_list):
        row_n = j
        if df[col].nunique() <= cate_thr:
            cut_points = None
            x_points = df[col].unique()
        else:
            _, cut_points = pd.qcut(df[col][df[partition]==base_partition], q=bins, retbins=True, duplicates='drop')
            cut_points[0] = -np.inf
            cut_points[-1] = np.inf
            cut_points = sorted(list(set(cut_points)))
            x_points = None
        for i, v in enumerate(df[partition].unique()):
            col_n = i
            ax2 = fig2.add_subplot(spec2[row_n, col_n])
            tmp = feature_curve(df[col][df[partition]==v], y[df[partition]==v], ax2, target=str(col) + ': ' + str(v), cate_thr=cate_thr, cut_points=cut_points, x_points=x_points, ylim=ylim, plt_label=plt_label)
            dic_all[str(col) + ': ' + str(v)] = twinx_to_dict(tmp, str(col) + ': ' + str(v))
    plt.tight_layout()
    
    if save_fig:
        if path is None:
            if not os.path.exists('reportsource'):
                os.mkdir('reportsource')
            path = 'reportsource/feature_plot_for_contrast.png'
        plt.savefig(path)
    # plt.show()
    return dic_all


def profit_line(y_true, y_pred, ax, label='', cost_model=1, rule_pass=0.95, gain=305, loss=2000, cost_sell=20,
                cost_rule=1, loc='best', bins=100):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.sort_values('y_pred')

    y_pred = list(df.y_pred)
    y_true = list(df.y_true)
    s = len(y_pred)

    # _, _, thr = mr.roc_curve(y_true, y_pred)
    # passrate = [y_pred.index(i) / s for i in thr[1:-1]]
    # delayrate = [np.mean(y_true[:y_pred.index(i)]) for i in thr[1:-1]]

    # 使用以下方法存在一个风险，可能把相同得分的样本分割到两侧
    passrate = []
    delayrate = []
    span = 100 // bins
    for i in range(0, 101, span):
        passrate.append(i / 100)
        if i == 0:
            delayrate.append(0)
        else:
            delayrate.append(np.mean(y_true[: int(i / 100 * s)]))

    profit = [rule_pass * (gain * (1 - r) * p - loss * p * r - cost_model) - cost_sell - cost_rule for p, r in
              zip(passrate, delayrate)]

    cut_points = np.argmax(np.array(profit))
    max_p = passrate[cut_points] * 100
    max_d = delayrate[cut_points] * 100
    max_g = profit[cut_points]
    text = '\nbest pass rate: %.0f%%\nbest overdue rate: %.2f%%\nbest profit: %.2f' % (max_p, max_d, max_g)

    ax.plot(passrate, profit, label=label + text)
    ax.set_xlabel('Passing Rate')
    ax.set_ylabel('Profit')
    ax.set_title('passing rate-profit curve')
    ax.legend(loc=loc)


def profit_curve(data, ax, bins=100):
    """

    :param data: dict. i.e. dict. i.e. {'model1': [y_true1, y_pred1], 'model2': [y_true2, y_pred2]]}
    :param ax:
    :param bins: int. 分成多少箱，计算粒度。最好可以整除100
    :return:
    """
    for label, v in data.items():
        y_true, y_pred = v[0], v[1]
        profit_line(y_true, y_pred, ax, label=label, bins=bins)


def pass_overdue_line(y_true, y_pred, ax, spec_p=None, label='', loc='best', bins=100):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    df = df.sort_values('y_pred')

    y_pred = list(df.y_pred)
    y_true = list(df.y_true)
    s = len(y_pred)

    # 利用roc_curve得到切分点，太细，没有必要
    # _, _, thr = mr.roc_curve(y_true, y_pred)
    # passrate = [y_pred.index(i) / s for i in thr[1:-1]]
    # delayrate = [np.mean(y_true[:y_pred.index(i)]) for i in thr[1:-1]]

    # 按bins切分，但是可能出现同一个值被分到两侧的情况
    passrate = []
    delayrate = []
    span = 100 // bins
    for i in range(0, 101, span):
        passrate.append(i / 100)
        if i == 0:
            delayrate.append(0)
        else:
            delayrate.append(np.mean(y_true[: int(i / 100 * s)]))

    if spec_p is not None:
        spec_d = np.mean(y_true[:int(s * spec_p)]) * 100
        text = '\noverdue rate: %.2f%%' % (spec_d)
        ax.plot([spec_p, spec_p], [0, spec_d / 100], linestyle='--')
    else:
        text = ''

    ax.plot(passrate, delayrate, label=label + text)
    ax.set_xlabel('Passing Rate')
    ax.set_ylabel('Overdue Rate')
    ax.set_title('passing rate-overdue rate curve')
    ax.legend(loc=loc)


def pass_overdue_curve(data, ax, spec_p=None, bins=100):
    """

    :param data: dict. i.e. {'model1': [y_true1, y_pred1], 'model2': [y_true2, y_pred2]]}
    :param ax:
    :param spec_p: float. 指定通过率，则添加对应垂直线，并在label中加入对应的逾期率
    :param bins：int. 分成多少箱，计算粒度。最好可以整除100
    :return:
    """
    for label, v in data.items():
        y_true, y_pred = v[0], v[1]
        pass_overdue_line(y_true, y_pred, ax, label=label, spec_p=spec_p, bins=bins)


def cost_profit_plot(data, bins=1, spec_p=None, figsize=(12, 4.5), path=None, save_fig=False):
    plt.figure(figsize=figsize)
    ax = plt.subplot(121)
    pass_overdue_curve(data, ax, spec_p=spec_p, bins=bins)
    ax = plt.subplot(122)
    profit_curve(data, ax, bins=bins)
    plt.tight_layout()

    if save_fig:
        if not os.path.exists('report'):
            os.mkdir('report')
        if path is None:
            path = 'cost_profit_plot.png'
        plt.savefig(os.path.join('report', path))
    plt.show()


def sta_groups(y_true, y_pred, cut_points=None, bins=10, labels=None):
    if cut_points is not None:
        labels = list(range(1, len(cut_points)))
    df = pd.DataFrame([y_pred, y_true], index=['y_pred', 'y_true']).T
    df['count'] = 1
    df['y_true_opp'] = df['y_true'].replace([0, 1], [1, 0])

    # bins 合并在一起
    if cut_points is not None:
        df['y_pred_range'] = pd.cut(df.y_pred, bins=cut_points, labels=labels, include_lowest=True).astype(int)
    else:
        df['y_pred_range'] = pd.qcut(df.y_pred, q=bins, duplicates='drop')
        df['y_pred_range'] = df['y_pred_range'].cat.remove_unused_categories().cat.codes + 1
    tmp = df.groupby('y_pred_range')

    # 单箱逾期率
    sing_overdue = tmp.mean().reset_index()
    final_df = sing_overdue[['y_pred_range', 'count', 'y_true']]
    final_df.columns = ['y_pred_range', 'count', 'single_overdue_rate']
    final_df['count'] = tmp.sum()['count'].tolist()

    # 累计逾期率
    acc_overdue = tmp.sum().cumsum().reset_index()
    final_df['acc_overdue_rate'] = acc_overdue['y_true'] / acc_overdue['count']

    # 累计好人和坏人召回率
    sum_good_person = df['y_true_opp'].sum()
    sum_bad_person = df['y_true'].sum()
    sum_all_person = len(df)
    final_df['acc_recall_rate_good'] = acc_overdue['y_true_opp'] / sum_good_person
    final_df['acc_recall_rate_bad'] = acc_overdue['y_true'] / sum_bad_person
    final_df['acc_pass_rate'] = acc_overdue['count'] / sum_all_person
    return final_df


if __name__ == '__main__':
    y_t = np.random.randint(0, 2, 100)
    y_p = np.random.uniform(0, 1, 100)
    f, ax = plt.subplots(dpi=100)
    boost_hist_curve(y_p, y_t, ax)
    plt.show()
    # roc_curve(y_t, y_p, ax)
