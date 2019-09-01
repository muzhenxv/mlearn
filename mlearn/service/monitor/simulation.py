import sys
import pandas as pd
import numpy as np
import xlwt
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
mpl.style.use('ggplot')
import prestodb
import os
import math
import json


def extract_nested_json(x):
    """
    args:
        x: json-formated data
    return:
        anti-nested formated data 将多层嵌套的json提取出只有一层的json，返回数据也是json类型
    example:
        df.data.map(extract_nested_json).apply(lambda s:pd.Series(json.loads(s)))
    """
    global_dic = {}
    x = x.replace('\\\\', '\\')
    def json_to_dict(key, value, prefix=''):
        if isinstance(value, dict):
            for k, v in value.items():
                if key and prefix:
                    json_to_dict(k, v, prefix + '_' + key)
                elif key and not prefix:
                    json_to_dict(k, v, key)
                elif not key and prefix:
                    json_to_dict(k, v, prefix)
                else:
                    json_to_dict(k, v, '')
        else:
            if prefix:
                key = prefix + '_' + key
            global_dic[key] = value
    try:
        tmp = json.loads(x)
        json_to_dict('', tmp)
    except:
        global_dic['_ERROR_'] = 1
    return json.dumps(global_dic)


def get_round(s):
    if s is np.nan:
        return s
    elif type(s) == str:
        return s
    return round(s, 3)


def get_new_A(real_14d_odds,Y_new):
    # Step 1
    aim_score = 423.82 - 72.14 * np.log(real_14d_odds)

    # Step 2
    Y_new = pd.Series(Y_new)
    p = np.mean(Y_new)
    real_oot_odds = p/(1-p)
    A_new = aim_score + 72.14 * np.log(real_oot_odds)
    return A_new


def prob_to_score(p, A=473.48, B=72.14):
    """
    args:
        p: 模型输出的概率，对1的概率
        A: 基础分补偿，不用修改
        B: 刻度，不用修改
    return:
        score: 分数，头尾掐掉
    """
    odds = p / (1 - p)
    score = A - B * np.log(odds)
    score = max(350, score)
    score = min(950, score)
    return round(float(score), 4)


def load_table(table):
    hive_lib = HiveLib(user='wang_l')
    df = hive_lib.saveTableToLocal('dm_temp', table, out_dir='datasource', out_type='pkl', overwrite_file=True, delimiter='\x01')
    df = df.set_index('apply_risk_id')
    return df


def get_hive_data(real, oot):
    """
    real_data: 与沙盒模拟数据时间段一致，从线下宽表中取出的数据，必须包含apply_risk_id
    oot_data: 训练模型时用的oot数据
    example:
    online_data_table: pingtai_lk_1108_1110
    oot_data_table: pingtai_lk_oot
    注释：所有表存在dm_temp
    """
    df = load_table(table=real)
    oot = load_table(table=oot)
    return df, oot


def execute(sql):
    #获取presto的连接
    conn=prestodb.dbapi.connect(
        host='10.105.110.176',
        port=8080,
        user='mapeng',
        catalog='hive',
        schema='dwb',
    )
    cur=conn.cursor()
    cur.execute(sql)
    #返回执行结果
    rows=cur.fetchall()
    conn.commit()
    #关闭连接
    conn.close()
    #返回结果
    return rows


def get_sandbox(model_id, dt, task_id=None):
    """
    dt: 模拟的日期，精确到天
    eg: '2018-10-10'
    """

    if task_id is None:
        sql1 = 'select apply_risk_id,request,score from sandbox.sandbox.offline_model_record where offline_model_record.model_id = ' + str(model_id)
        sql2 = 'select ar1.apply_risk_individual_id riskdata_individual_id,ar1.apply_risk_id source_19_apply_risk_id,ar1.apply_risk_type,ar1.apply_risk_created_at,ar1.apply_risk_score,ar1.apply_risk_result,offline_model_record.score from dwb.dwb_r_apply_risk ar1 left join sandbox.sandbox.offline_model_record on offline_model_record.apply_risk_id=ar1.apply_risk_id and offline_model_record.model_id=' + str(model_id) + ' where ar1.apply_risk_source=19 and substr(ar1.apply_risk_created_at,1,10) = ' + '\'' + str(dt) + '\'' + ' and ar1.apply_risk_type=2 and cast(ar1.apply_risk_score as varchar)>\'0\' order by offline_model_record.score desc'
    else:
        sql1 = 'select apply_risk_id,request,score from sandbox.sandbox.offline_model_record where offline_model_record.model_id = ' + str(model_id) + ' and offline_model_record.task_id = ' + str(task_id)
        sql2 = 'select ar1.apply_risk_individual_id riskdata_individual_id,ar1.apply_risk_id source_19_apply_risk_id,ar1.apply_risk_type,ar1.apply_risk_created_at,ar1.apply_risk_score,ar1.apply_risk_result,offline_model_record.score from dwb.dwb_r_apply_risk ar1 left join sandbox.sandbox.offline_model_record on offline_model_record.apply_risk_id=ar1.apply_risk_id and offline_model_record.model_id=' + str(model_id) + ' and offline_model_record.task_id = ' + str(task_id) + ' where ar1.apply_risk_source=19 and substr(ar1.apply_risk_created_at,1,10) = ' + '\'' + str(dt) + '\'' + ' and ar1.apply_risk_type=2 and cast(ar1.apply_risk_score as varchar)>\'0\' order by offline_model_record.score desc'

    df_sandbox = pd.DataFrame(execute(sql1),columns = ['apply_risk_id','request','score'])
    sandbox_score = df_sandbox.score.convert_objects(convert_numeric = True)
    sandbox = df_sandbox.request.map(extract_nested_json).apply(lambda s:pd.Series(json.loads(s)))
    # deal with None value
    for col in sandbox.columns:
        if sandbox[col].dtype == 'O':
            sandbox[col].str.lower().replace(['none', 'null', 'nan'], np.nan)
        else:
            sandbox[col] = sandbox[col]

    for col in sandbox.columns:
        if sandbox[col].dtype == 'O':
            sandbox[col] = sandbox[col].convert_objects(convert_numeric=True)
        else:
            sandbox[col] = sandbox[col]

    sandbox.apply(pd.to_numeric,errors='ignore')
    sandbox['apply_risk_id'] =  df_sandbox.apply_risk_id
    sandbox = sandbox.set_index('apply_risk_id')
    Bscore = pd.DataFrame(execute(sql2), columns = ['risk_data_individual_id','source_19_apply_risk_id','apply_risk_type','apply_risk_created_at','apply_risk_score','apply_risk_result','score'])
    return sandbox, Bscore, sandbox_score




def plot_normal_curve(simu_score, oot_score, real_14d_odds, A, w, pic_path):
    u1 = np.mean(simu_score)
    u2 = np.mean(oot_score)
    sig1 = np.std(simu_score)
    sig2 = np.std(oot_score)

    x_1 = np.linspace(u1 - 6 * sig1, u1 + 6 * sig1, 10000)
    x_2 = np.linspace(u2 - 10 * sig2, u2 + 10 * sig2, 10000)

    f, ax = plt.subplots(dpi=200,nrows=1, ncols=1, sharex=True, sharey=True)
    y_sig1 = np.exp(-(x_1 - u1) ** 2 /(2* sig1 **2))/(math.sqrt(2*math.pi)*sig1)
    y_sig2 = np.exp(-(x_2 - u2) ** 2 / (2 * sig2 ** 2)) / (math.sqrt(2 * math.pi) * sig2)

    ax.plot(x_1, y_sig1, label='score distribution: sandbox data')
    ax.plot(x_2, y_sig2, label='score distribution: oot data')
    ax.set_xlabel('Score')
    ax.set_ylabel('Probability')
    ax.legend(loc='upper right')
    plt.savefig('数据score分布图' +'.png')
    plt.show()

    # empty = pd.DataFrame()
    # empty.to_excel(w, sheet_name='数据score分布图')
    # workbook = w.book
    # worksheet = w.sheets['数据score分布图']
    # worksheet.insert_image('B02', pic_path, {'x_scale': 0.5, 'y_scale': 0.5})
    # workbook.close()


def get_consistency_rate(num, sand, dff, com_col, com_ix, equip):
    if equip is None:
        sandbox1 = pd.DataFrame(sand.ix[com_ix][com_col]).apply(pd.to_numeric, errors = 'ignore').applymap(get_round).fillna(num)
        df1 = pd.DataFrame(dff.ix[com_ix][com_col]).apply(pd.to_numeric, errors = 'ignore').applymap(get_round).fillna(num)

    else:
        sandbox1 = pd.DataFrame(sand.ix[com_ix][com_col].drop([equip], axis = 1)).apply(pd.to_numeric, errors = 'ignore').applymap(get_round).fillna(num)
        df1 = pd.DataFrame(dff.ix[com_ix][com_col].drop([equip], axis = 1)).apply(pd.to_numeric, errors = 'ignore').applymap(get_round).fillna(num)

    if num != 0:
        consistency_rate= (pd.DataFrame((sandbox1 == df1).apply(np.mean), columns = ['实际一致率'])).T

    else:
        consistency_rate= (pd.DataFrame((sandbox1 == df1).apply(np.mean), columns = ['空值填{0}的一致率'.format(num)])).T

    if equip is not None:
        consistency_rate[equip] = pd.Series((str(df.ix[k][equip]).split().sort() == str(sand.ix[k][equip]).split().sort()) for k in com_ix).mean()

    return consistency_rate


def data_process(sand, dff, com_col, com_ix, equip):
    if equip is None:
        sandbox1 = pd.DataFrame(sand.ix[com_ix][com_col]).apply(pd.to_numeric, errors = 'ignore').applymap(get_round).fillna(-999)
        df1 = pd.DataFrame(dff.ix[com_ix][com_col]).apply(pd.to_numeric, errors = 'ignore').applymap(get_round).fillna(-999)
    else:
        sandbox1 = pd.DataFrame(sand.ix[com_ix][com_col].drop([equip], axis = 1)).apply(pd.to_numeric, errors = 'ignore').applymap(get_round).fillna(-999)
        df1 = pd.DataFrame(dff.ix[com_ix][com_col].drop([equip], axis = 1)).apply(pd.to_numeric, errors = 'ignore').applymap(get_round).fillna(-999)
    return sandbox1, df1


def get_difference(sand, dff, com_col, com_ix, equip, w):
    sandbox1, df1 = data_process(sand=sand, dff=dff, com_col=com_col, com_ix=com_ix, equip=equip)
    diff = pd.DataFrame(sandbox1 == df1)
    diff_dict = {}
    for i in com_col:
        if i != equip:
            diff_dict[i] = diff[diff[i] == False].index.tolist()

    for key,values in diff_dict.items():
        if key != equip:
            diff_df = pd.DataFrame()
            diff_df['apply_risk_id'] = values
            diff_df['real_data'] = df[key].loc[values].tolist()
            diff_df['sandbox_data'] = sandbox[key].loc[values].tolist()
            if diff_df.empty == False:
                diff_df['dirt_data'] = diff_df[['real_data','sandbox_data']].apply(lambda x : 'real_data: ' + str(x['real_data']) +' , sandbox_data : '+str(x['sandbox_data']), axis=1 )
                diff_df = diff_df.set_index('apply_risk_id')
                diff_df = pd.DataFrame(diff_df,dtype=str)
                diff[key].loc[values] = diff_df['dirt_data'].loc[values]
    # diff_df.to_excel(w, sheet_name = key, index = True, encoding = 'gbk')
    diff.to_excel(w, sheet_name = '对比结果详情', index = True, encoding = 'gbk')


def get_eda(w, out_dir, test, dff, equip, label=None):
    sandbox1 = test
    df1 = dff
    sys.path.append('/data4/dm_share')
    from mlearn import mlearn
    eda = mlearn.service.reporter.data_reporter._gen_desc_report
    eda_df = eda(df_train_origin=sandbox1, df_test_origin=df1, label=None, report_dst=out_dir, decimals=4, gen_report=True)
    eda1 = pd.DataFrame(eda_df)
    eda1.to_excel(w, sheet_name='探索数据分析', index=True, encoding='gbk')


# Get Comparison Results
def get_result(user, real, test, model_id, task_id, dt, rename_col, real_14d_odds, oot_y_pred, oot_y_pred_col, equipment_app_names):

    """
    equipment_app_names默认为特征系统名equipment_app_name_v2(统一小写)

    数据字典传参示例：
    example: key为模型特征名，value为特征系统名
    rename_col = {
           'jxl_c1m_gre_0_ratio': 'jxl_contact_c1m_then_0_ratio',
           'jxl_c3m_gre_0_ratio': 'jxl_contact_c3m_then_0_ratio',
           'jxl_ccm_gre_0_ratio': 'jxl_contact_cmorning_then_0_ratio',
           'hit_type_lijihuankuan_d30': 'user_statistics_hit_type_lijihuankuan_d30',
           'jxl_coc_gre_0_ratio': 'jxl_contact_calloutcnt_then_0_ratio',
           'hit_type_bangdingshoukuanyinhangka_d30': 'user_statistics_hit_type_bangdingshoukuanyinhangka_d30',
           'jxl_avg_morning': 'jxl_contact_morning_avg',
           'enter_type_querenjiekuan_d30': 'user_statistics_enter_type_querenjiekuan_d30',
           'fbi_score': 'fbi_score_for_model',
           'i011': 'realtime_i011',
           'i061': 'realtime_i061',
           'i301': 'realtime_i301',
           'i601': 'realtime_i601',
           'kuaidi_cnt': 'sms_1m_kuaidi_cnt',
           'yiyuqi_cnt': 'sms_1m_yiyuqi_cnt'
    }
    """

    df, oot = get_hive_data(real=real, oot=test)
    sandbox, Bscore, sandbox_score = get_sandbox(model_id=model_id,task_id=task_id,dt=dt)

    sandbox.columns = [c.lower() for c in sandbox.columns]
    df.columns = [c.lower() for c in df.columns]
    oot.columns = [c.lower() for c in oot.columns]
    rename_col = dict(zip([k.lower() for k in rename_col.keys()], [k.lower() for k in rename_col.values()]))
    df.rename(columns = rename_col, inplace = True)
    oot.rename(columns = rename_col, inplace = True)

    path = os.path.join(os.environ['HOME'], 'reportsource', 'sandbox_data_check_report.xlsx')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    path1 = os.path.join(os.environ['HOME'], 'reportsource')
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    pic_path = os.path.join(os.environ['HOME'], '数据score分布图.png')
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    writer = pd.ExcelWriter(path, engine='openpyxl')


    common_col = sandbox.columns & df.columns
    common_ix = sandbox.index & df.index
    # consistency rate when fill na with -999
    consistency_rate_0 = get_consistency_rate(num=0, sand=sandbox, dff=df, com_col=common_col, com_ix=common_ix, equip=equipment_app_names)
    # consistency rate when fill na with 0
    consistency_rate_999 = get_consistency_rate(num=-999, sand=sandbox, dff=df, com_col=common_col, com_ix=common_ix, equip=equipment_app_names)
    cons = pd.DataFrame(pd.concat([consistency_rate_0.T, consistency_rate_999.T], axis=1))

    cons['由于0和null不一致导致的差异占比'] = cons.iloc[:,0] - cons.iloc[:,1]
    cons = cons.applymap(lambda s: "%.2f%%" % (s * 100))
    cons['是否完全一致'] = (cons.iloc[:,0]==cons.iloc[:,1]).astype(int)
    cons.to_excel(writer, sheet_name = '一致率', index = True, encoding = 'gbk')

    get_difference(sand=sandbox, dff=df, com_col=common_col, com_ix=common_ix, equip=equipment_app_names, w=writer)

    get_eda(w=writer, out_dir=path1, test=oot, dff=df, equip=equipment_app_names)

    # feature mean comparison between real data, oot data and sandbox data
    drop_list = []
    for col in common_col:
        if sandbox[col].convert_objects(convert_numeric=True).dtype == 'O':
            drop_list.append(col)

    sandbox_mean = sandbox.ix[common_ix][common_col].drop(drop_list,axis = 1).describe().T.iloc[:,1]
    df_mean = df.ix[common_ix][common_col].drop(drop_list,axis = 1).describe().T.iloc[:,1]
    oot_mean = oot.drop(drop_list, axis = 1).describe().T.iloc[:,1]
    df_oot_sandbox_mean = pd.concat([sandbox_mean,df_mean,oot_mean], axis = 1)
    df_oot_sandbox_mean.columns = ['sandbox_data_mean','real_data_mean','oot_data_mean']
    df_oot_sandbox_mean.to_excel(writer, sheet_name = '特征均值对比', index = True, encoding = 'gbk')

    # Bscore: 评分清单数据
    Bscore.to_excel(writer, sheet_name = '评分数据清单',index = True, encoding = 'gbk')

    oot_score_df = pd.Series(oot_y_pred[oot_y_pred_col])
    A_new = get_new_A(real_14d_odds=real_14d_odds, Y_new=oot_y_pred[oot_y_pred_col])
    oot_score = oot_score_df.map(lambda s: prob_to_score(s, A = A_new, B = 72.14))
    simu_score = sandbox_score

    # plot normal distribution curve
    plot_normal_curve(simu_score=simu_score, oot_score=oot_score, real_14d_odds=real_14d_odds, A=A_new, w=writer, pic_path=pic_path)

    writer.save()
    writer.close()

if __name__ == '__main__':
    get_result(user='wang_l', real='wl_mutualdebt_model_xk_real', test='wl_mutualdebt_model_xk_oot', model_id=147, task_id=None, dt='2018-12-01',rename_col=rename_col, real_14d_odds=0.066, oot_y_pred=df_pred, oot_y_pred_col='y_pred', equipment_app_names='equipment_app_names_v2')

