from ..service import ModelTrainer
import json
import pickle
import os
import traceback
import dill
from ..service import trainer_report
import numpy as np

def trainer_ui(json_str):
    train_data_dst = None
    train_process_dst = None
    test_data_dst = None
    report_dst = None
    try:
        dic = json.loads(json_str)

        dst = dic['out']['dst']
        train_dst = os.path.join(dst, 'train')
        train_data_dst = os.path.join(train_dst, 'result')
        test_dst = os.path.join(dst, 'test')
        report_dst = os.path.join(dst, 'report')
        if not os.path.exists(train_dst):
            os.makedirs(train_dst)
        if not os.path.exists(test_dst):
            os.makedirs(test_dst)
        if not os.path.exists(train_data_dst):
            os.makedirs(train_data_dst)
        if not os.path.exists(report_dst):
            os.makedirs(report_dst)
        name = 'trainer'
        train_train_dst = os.path.join(train_data_dst, f'{name}_train_result.pkl')
        train_val_dst = os.path.join(train_data_dst, f'{name}_val_result.pkl')
        train_test_dst = os.path.join(train_data_dst, f'{name}_test_result.pkl')
        train_cv_dst = os.path.join(train_data_dst, f'{name}_cv_result.pkl')
        train_process_dst = os.path.join(train_dst, f'{name}_enc.pkl')
        test_data_dst = os.path.join(test_dst, f'{name}_result.pkl')
        mcc_dst = os.path.join(train_dst, f'{name}_mcc.pkl')
        sample_weights_dst = os.path.join(train_dst, f'{name}_sample_weights.pkl')


        train_src = dic['ds']['train']
        test_src = dic['ds']['test']
        label = dic['ds']['label']['name']

        dic_p = dic['st']

        enc = ModelTrainer(**dic_p)
        clf, df_cv, df_test, df_val, df_train, mcc, sample_weights = enc.fit(train_src, label, test_src, label)

        dill.dump(enc, open(train_process_dst, 'wb'))
        df_cv.to_pickle(train_cv_dst)
        df_val.to_pickle(train_val_dst)
        df_train.to_pickle(train_train_dst)
        df_test.to_pickle(train_test_dst)
        pickle.dump(mcc, open(mcc_dst, 'wb'))
        pickle.dump(sample_weights, open(sample_weights_dst, 'wb'))

        # if test_src is not None:
        #     df_test = enc.transform(test_src)
        #     df_test.to_pickle(test_data_dst)
        #     test_m = 'oot'
        # else:
        #     test_data_dst = None
        #     test_m = 'test'
        #
        # if clf._estimator_type == 'ruler':
        #     clf.get_metrics(test_src)

        # trainer_report(enc, df_train, df_val, df_test, test_m, report_dst, test_src)
        trainer_report(enc, df_train, df_val, test_src, test_data_dst, report_dst)

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
