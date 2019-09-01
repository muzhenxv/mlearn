import json
import os
import copy


def get_rootpath(path):
    return os.path.join(path.rsplit('/mlearn', 1)[0], 'mlearn')


root_path = get_rootpath(os.path.abspath(__file__))

# ds
xk_v4_transformer_result = {'train_src': os.path.join(root_path, 'data/test_data/xk_v4_train_transformer_result.pkl'),
                            'test_src': os.path.join(root_path, 'data/test_data/xk_v4_test_transformer_result.pkl'),
                            'label': '14d',
                            'desc': '特征有缺失值'}

data_for_filter = [xk_v4_transformer_result]

xk_v4_filter_result = {'train_src': os.path.join(root_path, 'data/test_data/xk_v4_train_filter_result.pkl'),
                       'test_src': os.path.join(root_path, 'data/test_data/xk_v4_test_filter_result.pkl'),
                       'label': '14d',
                       'desc': '特征无缺失值'}

data_for_trainer = [xk_v4_filter_result]

xk_v4_ds = {'table': os.path.join(root_path, 'data/test_data/xk_v4_data.pkl'),
            'label': '14d',
            'time_col': 'apply_risk_created_at',
            'label_col': 'overdue_days',
            'index_col': 'apply_risk_id',
            'desc': '特征无缺失值'}

data_for_spliter = [xk_v4_ds]

# spliter_encoders
spliter_encoders = ['oot', 'random']

# estimator encoder

import traceback
# 需要全全引入，不然dir方法难道sklearn的所有子包
from sklearn import *


def get_estimaor():
    classifier_list = []
    filer_estimator_list = []
    module = __import__('sklearn')
    mod_l = dir(module)
    # 这样处理一方面是为了避免多余的搜索，一方面是全量搜索会爆jupyter kernel
    mod_l = [m for m in mod_l if
             m not in ['_ASSUME_FINITE', '__SKLEARN_SETUP__', '__all__', '__builtins__', '__cached__',
                       '__check_build', '__doc__', '__file__', '__loader__', '__name__', '__package__',
                       '__path__', '__spec__', '__version__', '_contextmanager', 'base', 'clone',
                       'config_context', 'exceptions', 'externals', 'get_config', 'logger', 'logging',
                       'os', 're', 'set_config', 'setup_module', 'sys', 'utils', 'warnings', '_isotonic',
                       'datasets', 'cross_validation', 'grid_search', 'learning_curve']]
    for sub in mod_l:
        sub_module_ = getattr(module, sub)
        sub_l = dir(sub_module_)
        for cls in sub_l:
            try:
                class_ = getattr(sub_module_, cls)
                instance = class_()
                if not '_estimator_type' in dir(instance):
                    continue
                if getattr(instance, '_estimator_type') == 'classifier':
                    print(sub)
                    params = instance.get_params()
                    classifier_list.append(
                        {'method': cls, 'params': params})
                    if (('feature_importances_' in dir(instance)) | ('coef_' in dir(instance))):
                        filer_estimator_list.append({'method': cls, 'params': params})
            except Exception as e:
                # print(cls, traceback.format_exc())
                pass
    return classifier_list, filer_estimator_list


xgbclassifier_params = {
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
}

skoperuler_params = {
    "method": "SkopeRuler",
    "params": {'random_state': 7,
               'precision_min': 0.1,
               'recall_min': 0.05}}

rulefitclassifier_params = {'method': 'RuleFitClassifier',
                            'params': {
                                'cv': 3,
                                'exp_rand_tree_size': True,
                                'lin_standardise': True,
                                'lin_trim_quantile': 0.025,
                                'max_rules': 2000,
                                'memory_par': 0.01,
                                'model_type': 'rl',
                                'random_state': None,
                                'rfmode': 'regress'}}

LogisticRegressionEstimator_params = {'method': 'LogisticRegressionEstimator',
                                      'params': {'C': 1.0,
                                                 'class_weight': None,
                                                 'dual': False,
                                                 'fit_intercept': True,
                                                 'intercept_scaling': 1,
                                                 'max_iter': 100,
                                                 'multi_class': 'ovr',
                                                 'n_jobs': 1,
                                                 'penalty': 'l2',
                                                 'random_state': None,
                                                 'solver': 'liblinear',
                                                 'tol': 0.0001,
                                                 'verbose': 0,
                                                 'warm_start': False}}

trainer_estimator_encoders, estimator_encoders = get_estimaor()
estimator_encoders.extend([xgbclassifier_params, LogisticRegressionEstimator_params])

trainer_estimator_encoders.extend(
    [xgbclassifier_params, rulefitclassifier_params, skoperuler_params, LogisticRegressionEstimator_params])

# transformer encoders

baseencoder = {'method': 'BaseEncoder',
               'params': {'cate_thr': 0.5, 'missing_thr': 0.8, 'same_thr': 0.9}}
transformer_encoders = [baseencoder]

# score func

score_func = ['f_classif']

# filter encoder
RFEFilter_encoder = {'method': 'RFEFilter',
                     'params': {
                         'n_features_to_select': 60,
                         'estimator': None}}

SelectKBest_encoder = {'method': 'SelectKBestFilter',
                       'params': {
                           'n_features_to_select': 60,
                           'k': 60,
                           'score_func': None}}

SelectFromModel_encoder = {'method': 'SelectFromModelFilter',
                           'params': {
                               'n_features_to_select': 60,
                               'estimator': None}}

LRFilter_encoder = {'method': 'LRFilter',
                    'params': {
                        'n_features_to_select': 60,
                        'method': 'coef',
                        'estimator': {'method': 'LogisticRegression',
                                      'params': {'C': 1.0,
                                                 'class_weight': None,
                                                 'dual': False,
                                                 'fit_intercept': True,
                                                 'intercept_scaling': 1,
                                                 'max_iter': 100,
                                                 'multi_class': 'ovr',
                                                 'n_jobs': 1,
                                                 'penalty': 'l2',
                                                 'random_state': None,
                                                 'solver': 'liblinear',
                                                 'tol': 0.0001,
                                                 'verbose': 0,
                                                 'warm_start': False}}}}

filter_encoders = [RFEFilter_encoder, SelectKBest_encoder, SelectFromModel_encoder, LRFilter_encoder]

# optimizer encoder

bayes_opt = {
    'method': 'BayesianOptimizer',
    'params': {'acq': 'ucb',
               'alpha': 0.0001,
               'init_points': 1,
               'kappa': 2.576,
               'n_iter': 1}}

optimizer_encoders = [bayes_opt]

# spliter cases
spliter_params_template = {'ds': {'label': {'name': None, 'type': 'number'},
                                  'table': None,
                                  'test': None,
                                  'train': None},
                           'out': {'dst': os.path.join(root_path, 'flow', 'spliter')},
                           'st': {'group_key': None,
                                  'index_col': None,
                                  'label_col': None,
                                  'method': None,
                                  'random_state': 7,
                                  'test_size': 0.25,
                                  'time_col': None}}
spliter_params = []

for ds in data_for_spliter:
    tmp_dict_0 = spliter_params_template.copy()
    tmp_dict_0['ds']['label']['name'] = ds['label']
    tmp_dict_0['ds']['table'] = ds['table']
    tmp_dict_0['st']['index_col'] = ds['index_col']
    tmp_dict_0['st']['time_col'] = ds['time_col']
    tmp_dict_0['st']['label_col'] = ds['label_col']
    for st in spliter_encoders:
        tmp_dict = copy.deepcopy(tmp_dict_0)
        tmp_dict['st']['method'] = st
        spliter_params.append(copy.deepcopy(tmp_dict))

# filter cases
filter_params_template = {'ds': {'label': {'name': None, 'type': 'number'},
                                 'test': None,
                                 'train': None},
                          'out': {'dst': os.path.join(root_path, 'flow', 'filter')},
                          'st': None}
filter_params = []

for ds in data_for_filter:
    tmp_dict_0 = filter_params_template.copy()
    tmp_dict_0['ds']['label']['name'] = ds['label']
    tmp_dict_0['ds']['train'] = ds['train_src']
    tmp_dict_0['ds']['test'] = ds['test_src']
    for st in filter_encoders:
        tmp_dict = copy.deepcopy(tmp_dict_0)
        tmp_dict['st'] = [st]
        if 'estimator' in st['params']:
            if st['params']['estimator'] is not None:
                filter_params.append(copy.deepcopy(tmp_dict))
                continue
            for estimator in estimator_encoders:
                tmp_dict['st'][0]['params']['estimator'] = estimator
                filter_params.append(copy.deepcopy(tmp_dict))
        elif 'score_func' in st['params']:
            for sf in score_func:
                tmp_dict['st'][0]['params']['score_func'] = sf
                filter_params.append(copy.deepcopy(tmp_dict))

# transformer cases
transformer_params_template = {'ds': {'label': {'name': None, 'type': 'number'},
                                      'test': None,
                                      'train': None},
                               'out': {'dst': os.path.join(root_path, 'flow', 'transformer')},
                               'st': {'cate': [{'cols': [],
                                                'encoders': None}],
                                      'cont': [{'cols': [],
                                                'encoders': None}],
                                      'method': 'auto',
                                      'params': {'thr': 5},
                                      'verbose': True}}

transformer_params = []

# for ds in data_transformer_result:
#     tmp_dict_0 = transformer_params_template.copy()
#     tmp_dict_0['ds']['label']['name'] = ds['label']
#     tmp_dict_0['ds']['train'] = ds['train_src']
#     tmp_dict_0['ds']['test'] = ds['test_src']
#     for st in transformer_encoders:
#         tmp_dict = copy.deepcopy(tmp_dict_0)
#         tmp_dict['st'] = [st]
#         if 'estimator' in st['params']:
#             for estimator in estimator_encoders:
#                 tmp_dict['st'][0]['params']['estimator'] = estimator
#                 filter_params.append(copy.deepcopy(tmp_dict))
#         elif 'score_func' in st['params']:
#             for sf in score_func:
#                 tmp_dict['st'][0]['params']['score_func'] = sf
#                 filter_params.append(copy.deepcopy(tmp_dict))

# trainer cases
trainer_params_template = {'ds': {'label': {'name': None, 'type': 'number'},
                                  'test': None,
                                  'train': None},
                           'out': {'dst': os.path.join(root_path, 'flow', 'trainer')},
                           'st': {'estimator': None,
                                  'n_folds': 5,
                                  'oversample': False,
                                  'random_state': 7,
                                  'reweight': False,
                                  'reweight_with_label': False,
                                  'cut_off_use_weights': True,
                                  'cut_off_sample_ratio': 1,
                                  'shift_thr': 0.1,
                                  'test_size': 0,
                                  'verbose': False}}

trainer_params = []

for ds in data_for_trainer:
    tmp_dict_0 = trainer_params_template.copy()
    tmp_dict_0['ds']['label']['name'] = ds['label']
    tmp_dict_0['ds']['train'] = ds['train_src']
    tmp_dict_0['ds']['test'] = ds['test_src']
    for st in trainer_estimator_encoders:
        tmp_dict = copy.deepcopy(tmp_dict_0)
        tmp_dict['st']['estimator'] = st
        trainer_params.append(copy.deepcopy(tmp_dict))

# optimizer cases
optimizer_params_template = {'ds': {'label': {'name': None, 'type': 'number'},
                                    'test': None,
                                    'train': None},
                             'out': {'dst': os.path.join(root_path, 'flow', 'optimizer')},
                             'st': {'n_folds': 0,
                                    'opt_encoder': None,
                                    'estimator': None,
                                    'score_func': 'roc_auc',
                                    'test_size': 0.2}}

optimizer_params = []
for ds in data_for_trainer:
    tmp_dict_0 = optimizer_params_template.copy()
    tmp_dict_0['ds']['label']['name'] = ds['label']
    tmp_dict_0['ds']['train'] = ds['train_src']
    tmp_dict_0['ds']['test'] = ds['test_src']
    for opt_st in optimizer_encoders:
        tmp_dict = copy.deepcopy(tmp_dict_0)
        tmp_dict['st']['opt_encoder'] = opt_st
        for st in trainer_estimator_encoders:
            estimator = copy.deepcopy(st)
            for k, v in st['params'].items():
                if type(v) == int:
                    estimator['params'][k] = [1, (v + 1) * 20]
                elif type(v) == float:
                    estimator['params'][k] = [0, 0.99]
                else:
                    estimator['params'].pop(k)
            tmp_dict['st']['estimator'] = estimator
            optimizer_params.append(copy.deepcopy(tmp_dict))

            # optimizer_params = optimizer_params[:3]
