import mlearn
import json
label = '14d'
test_path = "/analyze-server/mlearn_install/flow_0/datasource/xh_20180616_test_mlearn_0.pkl"
dst = '/analyze-server/mlearn_install/flow_0/'

json_str = {
    "ds":{
        "table":test_path,
        "train":None,
        "test":None,
        "label":{"name":label, "type": "number"}
    },
    "out":{"dst":dst+"spliter"},
    "st":{
        "method":"oot",
        "time_col":"biz_report_expect_at",
        "index_col":"apply_risk_id",
        "label_col":"overdue_days",
        "test_size":0.2,
        "random_state":7
    }
}
json_str = json.dumps(json_str)
json_str

mlearn.spliter_ui(json_str)

json_str = {
    "ds":{
        "train":dst+"spliter/train/spliter_result.pkl",
        "test":dst+"spliter/test/spliter_result.pkl",
        "label":{"name":label, "type": "number"}
    },
    "out":{"dst":dst+"transformer"},
    "st":{
        "method":"auto",
        "params":{
            "thr":5
        },
        "verbose": True,
        "cate":[
            {
                "cols":[],
                "encoders":[
                    {
                        "method":"BaseEncoder",
                        "params":{
                            "missing_thr":0.8,
                            "same_thr":0.9,
                            "cate_thr":0.5
                        }
                    },
                    {
                        "method":"CountEncoder",
                        "params":{
                            "unseen_value":1,
                            "log_transform":True,
                            "smoothing":1
                        }
                    }
                ]
            }
        ],
        "cont":[
            {
                "cols":[],
                "encoders":[
                    {
                        "method":"BaseEncoder",
                        "params":{
                            "missing_thr":0.8,
                            "same_thr":0.9,
                            "cate_thr":0.5
                        }
                    },
                    {
                        "method":"ContImputerEncoder",
                        "params":{
                            "missing_values":"NaN",
                            "strategy":"mean",
                            "axis":0,
                            "verbose":0
                        }
                    },
                    {
                        "method":"ContBinningEncoder",
                        "params":{
                            "diff_thr":20,
                            "bins":10,
                            "binning_method":"dt"
                        }
                    },
                    {
                        "method":"WOEEncoder",
                        "params":{
                            "diff_thr":20,
                            "woe_min":-20,
                            "woe_max":20,
                            "nan_thr":0.01
                        }
                    }
                ]
            }
        ]
    }
}
json_str = json.dumps(json_str)

mlearn.transformer_ui(json_str)

json_str = json.dumps({
    "ds":{
        "train":dst+"transformer/train/transformer_result.pkl",
        "test":dst+"transformer/test/transformer_result.pkl",
        "label":{"name":label, "type": "number"}
    },
    "st":{
        "method":"SelectKBest",
        "params":{
            "k":20,
            "score_func":"f_classif"
        }
    },
    "out":{"dst":dst+"filter"}
})
json_str

mlearn.filter_ui(json_str)

json_str = json.dumps({
    "ds": {
        "train": dst+"filter/train/filter_result.pkl",
        "test": dst+"filter/test/filter_result.pkl",
        "label": {
            "name": label,
            "type": "number"
        }
    },
    "out": {
        "dst": dst+"optimizer"
    },
    "st": {
        "opt_method": "BayesianOptimizer",
        "n_folds": 0,
        "test_size": 0.2,
        "score_func": "roc_auc",
        "opt_params": {
            "init_points": 1,
            "n_iter": 1,
            "acq": "ucb",
            "kappa": 2.576,
            "alpha": 10e-5
        },
        "params": {
            "gamma": [0, 1],
            "learning_rate": [0.001, 0.8],
            "max_depth": [2, 8],
#             "min_child_weight": [1, 1],
            "n_estimators": [100, 2000],
            "reg_lambda": [0, 40],
#             "subsample": [0.7, 0.7]
        },
        "method": "XGBClassifier"
    }
}
)
json_str

mlearn.optimizer_ui(json_str)

json_str = json.dumps({
    "ds": {
        "train": dst+"optimizer/train/filter_result.pkl",
        "test": dst+"optimizer/test/filter_result.pkl",
        "label": {"name": label, "type": "number"}
    },
    "out": {"dst": dst+"trainer"},
    "st": {
        "method": "XGBClassifier",
        "n_folds": 0,
        "test_size": 0.2,
        "random_state": 7,
        "oversample": False,
        "verbose": True,
        "params": {'base_score': 0.5,
                   'booster': 'gbtree',
                   'colsample_bylevel': 1,
                   'colsample_bytree': 1,
                   'gamma': 0,
                   'learning_rate': 0.1,
                   'max_delta_step': 0,
                   'max_depth': 3,
                   'min_child_weight': 1,
                   'missing': None,
                   'n_estimators': 1000,
                   'n_jobs': 1,
                   'nthread': None,
                   'objective': 'binary:logistic',
                   'random_state': 0,
                   'reg_alpha': 0,
                   'reg_lambda': 1,
                   'scale_pos_weight': 1,
                   'seed': None,
                   'silent': True,
                   'subsample': 0.7,
                   "verbose": False,
                   "eval_metric": "auc",
                   "early_stopping_rounds": 30}
    }
})
json_str

mlearn.trainer_ui(json_str)

json_str = json.dumps({
    "ds":{
        "test":test_path,
        "meta":dst+"transformer/train/transformer_enc.pkl",
        "label":{"name":label, "type": "number"}
    },
    "out":{"dst":dst+"result.pkl"}
})
json_str

mlearn.inference_ui(json_str)

import pandas as pd
df = pd.read_pickle(test_path)
test_src = mlearn.inference_ui_online(df, dst+"transformer/train/transformer_enc.pkl")
test_src = mlearn.inference_ui_online(test_src, dst+"filter/train/filter_enc.pkl")
test_src = mlearn.inference_ui_online(test_src, dst+"trainer/train/trainer_enc.pkl")
print(test_src)
