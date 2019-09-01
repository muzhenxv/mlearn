from ..service import data_sampler
import json
import os
import traceback
from .utils import create_dirpath


def sampler_ui(json_str):
    train_data_dst = None
    test_data_dst = None
    report_dst = None
    train_process_dst = None
    try:
        dic = json.loads(json_str)

        dst = dic['out']['dst']
        data_src = dic['ds']['table']
        label = dic['ds']['label']['name']
        dic_p = dic['st']
        group_key = dic_p['group_key']

        train_data_dst, train_process_dst, test_data_dst, report_dst = create_dirpath(dst, 'sampler')

        df = data_sampler(data_src, report_dst, **dic_p)
        df.to_pickle(train_data_dst)

        # if df[label].nunique() == 1:
        #     raise Exception('dataset label with only one value!')

        code = 1
        msg = 'succ'
    except Exception as e:
        msg = traceback.format_exc()
        print(msg)
        code = 2

    return json.dumps({"code": code, "msg": msg, "result": {
        "ds": {"table": train_data_dst},
        "meta": train_process_dst,
        "report": report_dst,
        "params": {"group_key": group_key}}})
