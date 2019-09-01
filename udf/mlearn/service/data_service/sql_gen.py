import re
import pandas as pd
import numpy as np
from pyhive import hive


def execute(sql, host='10.66.245.156', port=10000, username='yangkai', passwd="8w8hu2R6QVfXCmB8"):
    # 获取hive的连接
    conn = hive.connect(host=host, port=port, username=username, password=passwd, auth='LDAP')
    df = pd.read_sql(sql, conn)
    conn.close()
    # 返回结果
    return df


# 本处理仅考虑无内部子查询嵌套的sql，不考虑union all
def get_all_table_column(sql):
    def convert_type(s):
        if s.startswith('int') | s.startswith('float'):
            return 'decimal(20,12)'
        else:
            return 'string'

    def infer_type_for_df(df, nan_process=True, convert_dates=False, special_cols=None):
        if nan_process:
            df = df.apply(lambda x: x.astype(str).str.lower()).replace(['null', 'none', 'nan'], np.nan)
            df = df.apply(pd.to_numeric, errors='ignore')
        df.columns = [c.split('.')[-1] for c in df.columns]
        t = pd.DataFrame(df.convert_objects(convert_dates=False).dtypes, columns=['col_type']).astype(str)
        t.col_type = t.col_type.map(convert_type)
        t.index.name = 'col_name'
        t = t.reset_index()
        return t

    def infer_type_for_table(table_name):
        sql = 'select * from %s limit 100000' % table_name
        df = execute(sql)
        t = infer_type_for_df(df)
        t['table'] = table_name
        return t

    # 暂时只考虑全部有表别名，且字段全部都加了表别名指明的情况
    def get_table_alias(sql):
        #         pattern = re.compile('(?<=(from|join) )([\w \.]+?)(?= (join|left|right|where|on|outer|inner))')
        pattern = re.compile('(?<=(from|join) )([\w\.]+? \w+)')
        table_alias_dict = pattern.findall(sql)
        table_alias_dict = {i[1].split(' ')[0]: i[1].split(' ')[1] for i in table_alias_dict}
        df = pd.DataFrame(table_alias_dict, index=['alias']).T.reset_index()
        df.columns = ['table', 'alias']
        df = df.drop_duplicates('table')
        if max(df['alias'].value_counts()) > 1:
            print(df)
            raise ValueError('alias must be different!')
        del t['alias']
        return df

    # 对sql进行归整
    sql = re.sub('[ \n\t]+', ' ', sql.lower()).strip()

    t = get_table_alias(sql)
    df = pd.DataFrame()
    for i in t['table']:
        try:
            df = pd.concat([df, infer_type_for_table(i)], axis=0)
        except:
            pass
    return df


def format_sql(sql, df, blank_n=0):
    def get_table_info(sql, df_all):
        # 只能捕获无嵌套sql，如果有嵌套且表别名重复则会出错
        # 暂时只考虑全部有表别名，且字段全部都加了表别名指明的情况
        pattern = re.compile('(?<=(from|join) )([\w\.]+? \w+)')
        #     (?= (join|left|right|where|on|outer|inner)
        table_alias_dict = pattern.findall(sql)
        table_alias_dict = {i[1].split(' ')[0]: i[1].split(' ')[1] for i in table_alias_dict}
        df = pd.DataFrame(table_alias_dict, index=['alias']).T.reset_index()
        df.columns = ['table', 'alias']
        if df.table.nunique() != df.alias.nunique():
            print(df.drop_duplicates('table'))
            raise ValueError('alias must be different!')

        # 得到所有涉及到的字段， 暂时不考虑sql正则形式，e.g. t1.`apply_risk_id+?`
        alias_regex = '|'.join(table_alias_dict.values())
        pattern = re.compile('((%s)\.\w+)' % alias_regex)
        col_list = pattern.findall(sql)
        df2 = pd.DataFrame(col_list, columns=['col_name', 'alias']).drop_duplicates()
        df2['col_name'] = df2['col_name'].map(lambda s: s.split('.')[1])
        df = df.merge(df2, on='alias', how='right')

        df = df.merge(df_all, on=['col_name', 'table'], how='left')
        df['cast'] = 'cast(' + df['alias'] + '.' + df['col_name'] + ' as ' + df['col_type'] + ')'
        return df

    def cast_sql(sql, df):
        # cast处理。暂时没有考虑原始sql中就使用cast的情况
        for i in df.index:
            try:
                pattern = re.compile('%s(?=\W)' % (df.loc[i, 'alias'] + '.' + df.loc[i, 'col_name']))
                sql = re.sub(pattern, df.loc[i, 'cast'], sql)
            except:
                pass

        # 对于比较运算符中涉及到的类型一致问题，暂时没有处理。e.g. t1.apply = 123, t1.apply = '123'.
        # ...
        return sql

    def format_sql_blank(sql, blank_n=0):
        # sql格式化对齐
        def _gen_key_blank(s, blank_length=20):
            if '\n' in s:
                return '\n' + ' ' * (blank_length - 5 - len(s.strip())) + s.strip()
            else:
                return '\n' + ' ' * (blank_length - len(s)) + s

        base_blank_length = 20
        sub_blank_length = 10
        blank_length = base_blank_length + sub_blank_length * blank_n

        # 如果这里写成16空格，那么紧接其后的就是join，不满足向后查询的正则条件
        sql2 = ' ' + re.sub('[ \n\t]+', ' ', sql.lower()).strip()
        join_postfix = ' \n' + ' ' * (blank_length - 5)
        df_blank = pd.DataFrame({'keyword': ['select', 'drop table', 'create table', 'from', 'join',
                                             'inner' + join_postfix, 'left' + join_postfix, 'outer' + join_postfix,
                                             'right' + join_postfix, 'where', 'on']})
        df_blank['key_blank'] = df_blank['keyword'].map(lambda s: _gen_key_blank(s, blank_length))
        df_blank

        for i in df_blank.index:
            pattern = re.compile('(?<= )%s(?=\W)' % df_blank.loc[i, 'keyword'])
            sql2 = re.sub(pattern, df_blank.loc[i, 'key_blank'], sql2)
        return sql2

    # 本处理仅考虑无内部子查询嵌套的sql，不考虑union all
    sql = re.sub('[ \n\t]+', ' ', sql.lower()).strip()
    df_alias = get_table_info(sql, df)
    sql = cast_sql(sql, df_alias)
    sql = format_sql_blank(sql, blank_n)
    return sql


def get_all_nested_sql(sql):
    def get_nested_sql_start_index(sql, start=0):
        regex1 = 'from ('
        regex2 = 'join ('

        try:
            return min([sql.find(s, start) + len(s) for s in [regex1, regex2] if sql.find(s, start) > -1])
        except:
            return -1

    def get_nested_sql(sql, start_index):
        left = 0
        for i in range(len(sql)):
            if sql[start_index + i] == '(':
                left += 1
            if sql[start_index + i] == ')':
                left -= 1
            if left == -1:
                return (start_index, start_index + i)

    l = []
    start = 0
    while 1:
        start_index = get_nested_sql_start_index(sql, start)
        if start_index == -1:
            break
        start_begin, start = get_nested_sql(sql, start_index)
        l.append((start_begin, start))
    return l


def update_sql(sql, i=0):
    dic = {}
    l = get_all_nested_sql(sql)
    for li in l:
        old_sql = sql[li[0]:li[1]]
        dic[old_sql] = i + 1
        dic_t = update_sql(old_sql, i + 1)
        dic.update(dic_t)
    return dic


import copy


def final_format_sql(sql, df_table_column_all):
    # dic的key为每层sql， value为对应层数
    dic = update_sql(sql)
    dic.update({sql: 0})

    # dic_n的key为每层sql，value为对应临时别名
    dic_n = {}
    for i, (k, v) in enumerate(dic.items()):
        if v >= 1:
            dic_n[k] = 'dm_temp.temp_temp_temp_temp_temp_%s' % i

    nm = max(dic.values())
    # dic_r的key为每层sql，value为对内嵌sql使用临时别名替换后的新sql
    dic_r = {}
    # dic_f的key为内嵌sql使用临时别名替换后的新sql，value为format之后的sql
    dic_f = {}
    while nm > 0:
        replace_sql = []
        pre_sql = []
        for k, v in dic.items():
            if v == nm:
                replace_sql.append(k)
            if v == nm - 1:
                pre_sql.append(k)
            if (v == max(dic.values())) & (nm == v):
                dic_r[k] = k
                dic_f[k] = format_sql(k, df_table_column_all, nm)
        for ps in pre_sql:
            psc = copy.copy(ps)
            for rs in replace_sql:
                psc = psc.replace('(' + rs + ')', dic_n[rs])
                psc = psc.replace('( ' + rs + ' )', dic_n[rs])
            dic_r[ps] = psc
            dic_f[psc] = format_sql(psc, df_table_column_all, nm - 1)
        nm -= 1

    if len(dic_r) == 0:
        return format_sql(sql, df_table_column_all)
    sql_t = dic_f[dic_r[sql]]
    nm = 1
    while nm <= max(dic.values()):
        for k, v in dic.items():
            if v == nm:
                sql_t = sql_t.replace(dic_n[k], '( ' + dic_f[dic_r[k]] + ' )')
        nm += 1
    return sql_t


if __name__ == '__main__':
    sql = """create table dm_temp.xh_lk_kb_base_final
            as
               select t1.apply_risk_id, t1.post_rid, t1.post_aid, t1.overdue_days,
                      t1.biz_report_time, t1.biz_report_expect_at, t1.biz_report_created_at, t1.biz_report_status,
                      t3.apply_risk_created_at, t3.apply_risk_result, t3.apply_status, t3.apply_risk_type,
                      t2.ty2,t2.ty2forqnn, 
                      t2.baidu_panshi_black_match, t2.baidu_panshi_black_score,
                      t2.baidu_panshi_black_count_level1, t2.baidu_panshi_black_count_level2,
                      t2.baidu_panshi_black_count_level3, 
                      t2.baidu_panshi_duotou_name_match, t2.baidu_panshi_duotou_name_score,
                      t2.baidu_panshi_duotou_name_detail_key, t2.baidu_panshi_duotou_name_detail_val,
                      t2.baidu_panshi_duotou_identity_match, t2.baidu_panshi_duotou_identity_score,
                      t2.baidu_panshi_duotou_identity_detail_key, t2.baidu_panshi_duotou_identity_detail_val,
                      t2.baidu_panshi_duotou_phone_match, t2.baidu_panshi_duotou_phone_score,
                      t2.baidu_panshi_duotou_phone_detail_key, t2.baidu_panshi_duotou_phone_detail_val,
                      t2.baidu_panshi_prea_models, t2.baidu_panshi_prea_score,
                      t3.apply_times, t3.age, t3.gender, t3.prov, t3.city,
                      t3.individual_name, t3.individual_mobile, t3.individual_identity, t3.oauth_user_id,
                      t3.apply_risk_score, t3.apply_product_id, t3.apply_code, t3.apply_amount,
                      t4.max_yuqi_day, t4.min_yuqi_day, t4.mean_yuqi_day, t4.latest_yuqi_day, t4.farest_yuqi_day, t4.danqi_num, t4.duoqi_num,
                      t4.latest_borrow_span, t4.farest_created_span, t4.mean_created_span   
                 from (
                               select t1.apply_risk_id
                                 from (
                                        select t6.apply_risk_id
                                          from dm_temp.xh_lk_kb_base_label t6)  t1) t1
            left join dm_temp.xh_lk_kb_base_1 t2
                   on t1.apply_risk_id = t2.apply_risk_id
            left join (
                       select t2.apply_risk_id
                         from dm_temp.xh_lk_kb_base_0 t2) t3
                   on t1.apply_risk_id = t3.apply_risk_id
            left join dm_temp.xh_lk_kb_base_4 t4
                   on t1.apply_risk_id = t4.apply_risk_id
                where t1.apply_risk_id = 'www.baidu.com';
"""

    df_table_column_all = get_all_table_column(sql)
    new_sql = final_format_sql(sql, df_table_column_all)
    print(new_sql)
