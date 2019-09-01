import pandas as pd
import numpy as np
import json
import traceback
import os
import pickle
from ..service import get_data
import dill


def inference_ui(json_str):
    train_data_dst = None
    train_process_dst = None
    test_data_dst = None
    report_dst = None
    try:
        dic = json.loads(json_str)

        test_src = dic['ds']['test']
        meta = dic['ds']['meta']
        dst = dic['out']['dst']

        enc = dill.load(open(meta, 'rb'))
        if 'verbose' in enc.__dict__:
            enc.verbose = False
        test_src = enc.transform(test_src)

        test_src.to_pickle(dst)

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


def inference_ui_online(df, meta_src):
    if type(meta_src) == str:
        enc = dill.load(open(meta_src, 'rb'))
    else:
        enc = meta_src
    if 'verbose' in enc.__dict__:
        enc.verbose = False
    df = enc.transform(df)

    return df
