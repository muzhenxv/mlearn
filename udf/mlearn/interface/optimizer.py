from ..service import ParamsOptimizer
import json
import os
import pickle
import traceback
import shutil
from ..service import optimizer_report

def optimizer_ui(json_str):
    train_data_dst = None
    train_process_dst = None
    test_data_dst = None
    report_dst = None
    try:
        dic = json.loads(json_str)

        dst = dic['out']['dst']
        report_dst = os.path.join(dst, 'report')
        train_dst = os.path.join(dst, 'train')
        test_dst = os.path.join(dst, 'test')
        if not os.path.exists(train_dst):
            os.makedirs(train_dst)
        if not os.path.exists(test_dst):
            os.makedirs(test_dst)
        if not os.path.exists(report_dst):
            os.makedirs(report_dst)

        train_src = dic['ds']['train']
        test_src = dic['ds']['test']
        label = dic['ds']['label']['name']

        train_data_dst = os.path.join(train_dst, os.path.basename(train_src))
        test_data_dst = os.path.join(test_dst, os.path.basename(test_src))
        train_process_dst = os.path.join(train_dst, 'enc.pkl')

        dic_p = dic['st']

        all_df, max_dict, BO = ParamsOptimizer(train_src, label, **dic_p)

        optimizer_report(all_df, report_dst)
        # pickle.dump(BO, open(train_process_dst, 'wb'))

        shutil.copyfile(train_src, train_data_dst)
        shutil.copyfile(test_src, test_data_dst)

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
