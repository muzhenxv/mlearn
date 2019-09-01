SUB_DIR = "/Users/muzhen/dev/deepfm_output"

NUM_SPLITS = 3
RANDOM_SEED = 2017

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt

from ...service.data_service import DataParser
from .metrics import gini_norm
from .DeepFM import DeepFM

# if not os.path.exists(SUB_DIR):
#     os.makedirs(SUB_DIR)

dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [32, 32],
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layer_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,
    "random_seed": RANDOM_SEED

}


def run_base_model_dfm(dfTrain, dfTest, dfm_params=dfm_params, n_folds=5):
    data_parser = DataParser()
    Xi_train, Xv_train, y_train, ids_train = data_parser.fit_transform(dfTrain, '01d')
    Xi_test, Xv_test, y_test, ids_test = data_parser.transform(dfTest)

    dfm_params['feature_size'] = data_parser.feat_dim
    dfm_params['field_size'] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)

    _get = lambda x, l: [x[i] for i in l]

    gini_results_cv = np.zeros(n_folds, dtype=float)
    gini_results_epoch_train = np.zeros((n_folds, dfm_params['epoch']), dtype=float)
    gini_results_epoch_valid = np.zeros((n_folds, dfm_params['epoch']), dtype=float)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7)
    for i, (train_idx, valid_idx) in enumerate(skf.split(Xv_train, y_train)):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        y_train_meta[valid_idx, 0] = dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:, 0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(n_folds)

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv" % (clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)
    return gini_results_epoch_train, gini_results_epoch_valid, y_train_meta, y_test_meta

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
        os.path.join(SUB_DIR, filename), index=False, float_format="%.5f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green", "yellow", "black", 'red', 'blue']
    xs = np.arange(1, train_results.shape[1] + 1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d" % (i + 1))
        legends.append("valid-%d" % (i + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(legends)
    plt.savefig(os.path.join(SUB_DIR, "fig/%s.png" % model_name))
    plt.close()


if __name__ == '__main__':
    dfTrain = '..train.pkl'
    dfTest = '..test.pkl'
    y_train_dfm, y_test_dfm = run_base_model_dfm(dfTrain, dfTest)
