import json
from ..interface import *
from .set_params import set_params
import os
from .utils import *

def chaintest(test_path='', label='01d', dst='', time_col='apply_risk_created_at', index_col='apply_risk_id',
              label_col='overdue_days', group_key='level', cate_dict=None, optimizer_f=True,
              app_f=False, sampler_f=False, custom_params=None):
    if dst == '':
        dst = os.path.join(get_rootpath(os.path.abspath(__file__)), 'flow') + '/'
    if cate_dict is None:
        # cate_dict = os.environ['HOME'] + '/' + cate_dict.split('/', 1)[1]
        cate_dict = os.path.join(get_rootpath(os.path.abspath(__file__)), 'materials', 'cate_dict_20180806_v1.pkl')

    if custom_params is None:
        dic_params = set_params(
            test_path, label, dst, time_col, index_col, label_col, group_key, cate_dict, optimizer_f, app_f, sampler_f)
    else:
        dic_params = custom_params

    if sampler_f:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>> start sampler')
        js_str = sampler_ui(json.dumps(dic_params['sampler']))
        if json.loads(js_str)['code'] == 2:
            return js_str

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> start spliter')
    js_str = spliter_ui(json.dumps(dic_params['spliter']))
    if json.loads(js_str)['code'] == 2:
        return js_str

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> start transformer')
    js_str = transformer_ui(json.dumps(dic_params['transformer']))
    if json.loads(js_str)['code'] == 2:
        return js_str

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> start filter')
    js_str = filter_ui(json.dumps(dic_params['filter']))
    if json.loads(js_str)['code'] == 2:
        return js_str

    if optimizer_f:
        print('>>>>>>>>>>>>>>>>>>>>>>>>>> start optimizer')
        js_str = optimizer_ui(json.dumps(dic_params['optimizer']))
        if json.loads(js_str)['code'] == 2:
            return js_str

    print('>>>>>>>>>>>>>>>>>>>>>>>>>> start trainer')
    js_str = trainer_ui(json.dumps(dic_params['trainer']))
    return js_str
