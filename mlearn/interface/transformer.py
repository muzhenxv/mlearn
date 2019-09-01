from ..service import FeatureTransformer
import json
import pickle
import os
import traceback
import dill
from ..service import transformer_report
import numpy as np
from .utils import create_dirpath

def transformer_ui(json_str):
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
        dic_p = dic['st']

        train_data_dst, train_process_dst, test_data_dst, report_dst = create_dirpath(dst, 'transformer')

        enc = FeatureTransformer(dic_p, dst)
        df_train = enc.fit_transform(train_src, label)

        try:
            df_train.to_pickle(train_data_dst)
        except:
            pickle.dump(df_train, open(train_data_dst, 'wb'))
        dill.dump(enc, open(train_process_dst, 'wb'))

        if test_src is not None:
            df_test = enc.transform(test_src)
            try:
                df_test.to_pickle(test_data_dst)
            except:
                pickle.dump(df_test, open(test_data_dst, 'wb'))
        else:
            test_data_dst = None
            df_test = None

        try:
            transformer_report(df_train, df_test, label, report_dst)
        except Exception as e:
            print(traceback.format_exc())
            pass

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
