from ..service import FeatureFilter
import json
import pickle
import os
import traceback
import pandas as pd
import dill
from .utils import create_dirpath
from ..service import filter_report


def filter_ui(json_str):
    train_data_dst = None
    train_process_dst = None
    test_data_dst = None
    report_dst = None
    try:
        dic = json.loads(json_str)

        dst = dic['out']['dst']
        train_src = dic['ds']['train']
        test_src = dic['ds']['test']
        label = dic['ds']['label']['name']

        train_data_dst, train_process_dst, test_data_dst, report_dst = create_dirpath(dst, 'filter')

        enc = FeatureFilter(dic['st'])

        df_train = enc.fit_transform(train_src, label, test_src, label)

        dill.dump(enc, open(train_process_dst, 'wb'))
        df_train.to_pickle(train_data_dst)

        if test_src is not None:
            df_test = enc.transform(test_src)
            df_test.to_pickle(test_data_dst)
        else:
            test_data_dst = None

        filter_report(enc, df_train, df_test, label, report_dst)


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
