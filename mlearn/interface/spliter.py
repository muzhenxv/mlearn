from ..service import data_spliter
import json
import os
import traceback
from .utils import create_dirpath


def spliter_ui(json_str):
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

        train_data_dst, train_process_dst, test_data_dst, report_dst = create_dirpath(dst, 'spliter')

        train, test = data_spliter(data_src, label, report_dst, **dic_p)
        train.to_pickle(train_data_dst)
        test.to_pickle(test_data_dst)

        if train[label].nunique() == 1:
            raise Exception('train dataset label with only one value!')
        elif test[label].nunique() == 1:
            raise Exception('test dataset label with only one value!')

        code = 1
        msg = 'succ'
    except Exception as e:
        msg = traceback.format_exc()
        print(msg)
        code = 2

    return json.dumps({"code": code, "msg": msg, "result": {
        "ds": {"train": train_data_dst,
               "test": test_data_dst},
        "meta": train_process_dst,
        "report": report_dst}})
