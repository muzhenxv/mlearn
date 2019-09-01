import json
import os


def set_params(test_path, label='01d', dst='~/dev/flow_0/', time_col='apply_risk_created_at', index_col='apply_risk_id',
               label_col='overdue_days', group_key='level',
               cate_dict='~/repo/mlearn/mlearndev/materials/cate_dict_20180806_v1.pkl', optimizer_f=True,
               app_f=True, sampler_f=True):
    dst = dst.replace('~/', os.environ['HOME'] + '/')
    cate_dict = cate_dict.replace('~/', os.environ['HOME'] + '/')
    if not os.path.exists(cate_dict):
        cate_dict = cate_dict.replace('mlearndev', 'mlearn')

    sampler_json_str = {
        "ds": {
            "table": test_path,
            "train": None,
            "test": None,
            "label": {"name": label, "type": "number"}
        },
        "out": {"dst": dst + "sampler"},
        "st": {
            "group_key": group_key,
            "group_key_level": True,
            "sort_values": time_col,
            "group_ratio": {'ratio': {0: 0.1, 1: 0.2, 2: 0.5, 3: 0.2}},
            "group_num": 10,
            "base_df": None,
            "base_df_key": "level",
            "get_group_data": None,
            "thr": 0.5
        }
    }

    if sampler_f:
        spliter_json_str = {
            "ds": {
                "table": dst + "sampler/train/sampler_result.pkl",
                "train": None,
                "test": None,
                "label": {"name": label, "type": "number"}
            },
            "out": {"dst": dst + "spliter"},
            "st": {
                "method": "oot",
                "time_col": time_col,
                "index_col": index_col,
                "label_col": label_col,
                "test_size": 0.1,
                "random_state": 7,
                "group_key": group_key
            }
        }
    else:
        spliter_json_str = {
            "ds": {
                "table": test_path,
                "train": None,
                "test": None,
                "label": {"name": label, "type": "number"}
            },
            "out": {"dst": dst + "spliter"},
            "st": {
                "method": "oot",
                "time_col": time_col,
                "index_col": index_col,
                "label_col": label_col,
                "test_size": 0.1,
                "random_state": 7,
                "group_key": None
            }
        }

    if app_f:
        transformer_json_str = {
            "ds": {
                "train": dst + "spliter/train/spliter_result.pkl",
                "test": dst + "spliter/test/spliter_result.pkl",
                "label": {"name": label, "type": "number"}
            },
            "out": {"dst": dst + "transformer"},
            "st": {
                "method": "auto",
                "params": {
                    "thr": 5
                },
                "verbose": True,
                "cate": [
                    {
                        "cols": [],
                        "encoders": [
                            {
                                "method": "BaseEncoder",
                                "params": {
                                    "missing_thr": 0.8,
                                    "same_thr": 0.9,
                                    "cate_thr": 0.5
                                }
                            },
                            {
                                "method": "CountEncoder",
                                "params": {
                                    "unseen_value": 1,
                                    "log_transform": True,
                                    "smoothing": 1
                                }
                            }
                        ]
                    }
                ],
                "cont": [
                    {
                        "cols": [],
                        "encoders": [
                            {
                                "method": "BaseEncoder",
                                "params": {
                                    "missing_thr": 0.8,
                                    "same_thr": 0.9,
                                    "cate_thr": 0.5
                                }
                            },
                            {
                                "method": "ContImputerEncoder",
                                "params": {
                                    "missing_values": "NaN",
                                    "strategy": "mean",
                                    "axis": 0,
                                    "verbose": 0
                                }
                            },
                            {
                                "method": "ReduceGen",
                                "params": {
                                    "method": "KMeans",
                                    "method_params": {
                                        "n_clusters": 5
                                    }
                                }
                            },
                            {
                                "method": "ContBinningEncoder",
                                "params": {
                                    "diff_thr": 20,
                                    "bins": 10,
                                    "binning_method": "dt"
                                }
                            },
                            {
                                "method": "WOEEncoder",
                                "params": {
                                    "diff_thr": 20,
                                    "woe_min": -20,
                                    "woe_max": 20,
                                    "nan_thr": 0.01
                                }
                            }
                        ]
                    }
                ]
                , "custom": [
                    {
                        "cols": ["equipment_app_name"],
                        "encoders": [
                            {
                                "method": "AppCateEncoder_udf",
                                "params": {
                                    "cate_dict": cate_dict,
                                    "delimiter": ",",
                                    "prefix": "app_",
                                    "unknown": "unknown"
                                }
                            }
                        ]
                    }
                ]
            }
        }
    else:
        transformer_json_str = {
            "ds": {
                "train": dst + "spliter/train/spliter_result.pkl",
                "test": dst + "spliter/test/spliter_result.pkl",
                "label": {"name": label, "type": "number"}
            },
            "out": {"dst": dst + "transformer"},
            "st": {
                "method": "auto",
                "params": {
                    "thr": 5
                },
                "verbose": True,
                "cate": [
                    {
                        "cols": [],
                        "encoders": [
                            {
                                "method": "BaseEncoder",
                                "params": {
                                    "missing_thr": 0.8,
                                    "same_thr": 0.9,
                                    "cate_thr": 0.5
                                }
                            },
                            {
                                "method": "CountEncoder",
                                "params": {
                                    "unseen_value": 1,
                                    "log_transform": True,
                                    "smoothing": 1
                                }
                            }
                        ]
                    }
                ],
                "cont": [
                    {
                        "cols": [],
                        "encoders": [
                            {
                                "method": "BaseEncoder",
                                "params": {
                                    "missing_thr": 0.8,
                                    "same_thr": 0.9,
                                    "cate_thr": 0.5
                                }
                            },
                            # {
                            #     "method": "ContImputerEncoder",
                            #     "params": {
                            #         "missing_values": "NaN",
                            #         "strategy": "mean",
                            #         "axis": 0,
                            #         "verbose": 0
                            #     }
                            # },
                            # {
                            #     "method": "ReduceGen",
                            #     "params": {
                            #         "method": "KMeans",
                            #         "method_params": {
                            #             "n_clusters": 5
                            #         }
                            #     }
                            # },
                            # {
                            #     "method": "ContBinningEncoder",
                            #     "params": {
                            #         "diff_thr": 20,
                            #         "bins": 10,
                            #         "binning_method": "dt"
                            #     }
                            # },
                            # {
                            #     "method": "WOEEncoder",
                            #     "params": {
                            #         "diff_thr": 20,
                            #         "woe_min": -20,
                            #         "woe_max": 20,
                            #         "nan_thr": 0.01
                            #     }
                            # }

                        ]
                    }
                ]
            }
        }

    filter_json_str = {
        "ds": {
            "train": dst + "transformer/train/transformer_result.pkl",
            "test": dst + "transformer/test/transformer_result.pkl",
            "label": {"name": label, "type": "number"}
        },
        "st": [
            {
                "method": "StableFilter",
                "params": {
                    "indice_name": "psi",
                    "indice_thr": 0.2
                }},
            {
                "method": "SelectKBestFilter",
                "params": {
                    "n_features_to_select": 60,
                    "k": 60,
                    "score_func": "f_classif"
                }}
        ],
        "out": {"dst": dst + "filter"}
    }

    optimizer_json_str = {
        "ds": {
            "train": dst + "filter/train/filter_result.pkl",
            "test": dst + "filter/test/filter_result.pkl",
            "label": {
                "name": label,
                "type": "number"
            }
        },
        "out": {
            "dst": dst + "optimizer"
        },
        'st': {'n_folds': 0,
               'opt_encoder': {
                   'method': 'BayesianOptimizer',
                   'params': {'acq': 'ucb',
                              'alpha': 0.0001,
                              'init_points': 1,
                              'kappa': 2.576,
                              'n_iter': 1}},
               'estimator': {
                   'method': 'XGBClassifier',
                   'params': {'gamma': [0, 1],
                              'learning_rate': [0.001, 0.8],
                              'max_depth': [2, 8],
                              'n_estimators': [100, 2000],
                              'reg_lambda': [0, 40]}},
               'score_func': 'roc_auc',
               'test_size': 0.2}}

    if optimizer_f:
        name = 'optimizer'
    else:
        name = 'filter'
    trainer_json_str = {
        "ds": {
            "train": dst + name + "/train/filter_result.pkl",
            "test": dst + name + "/test/filter_result.pkl",
            "label": {"name": label, "type": "number"}
        },
        "out": {"dst": dst + "trainer"},
        "st": {
            "test_size": 0,
            "oversample": False,
            "n_folds": 5,
            "random_state": 7,
            "shift_thr": 0.1,
            "reweight": False,
            "reweight_with_label": True,
            "estimator": {
                "method": "XGBClassifier",
                "params": {
                    "colsample_bytree": 0.8,
                    "reg_lambda": 20,
                    "silent": True,
                    "base_score": 0.5,
                    "scale_pos_weight": 1,
                    "eval_metric": "auc",
                    "max_depth": 3,
                    "n_jobs": 1,
                    "early_stopping_rounds": 300,
                    "n_estimators": 1000,
                    "random_state": 0,
                    "reg_alpha": 1,
                    "booster": "gbtree",
                    "objective": "binary:logistic",
                    "verbose": False,
                    "colsample_bylevel": 0.8,
                    "subsample": 0.7,
                    "learning_rate": 0.1,
                    "gamma": 0.5,
                    "max_delta_step": 0,
                    "min_child_weight": 10
                }
            },
            "verbose": True,
            "cut_off_sample_ratio": 0.5,
            'cut_off_use_weights': False
        },
    }

    dic = {'sampler': sampler_json_str, 'spliter': spliter_json_str, 'transformer': transformer_json_str,
           'filter': filter_json_str,
           'optimizer': optimizer_json_str, 'trainer': trainer_json_str}
    return dic
