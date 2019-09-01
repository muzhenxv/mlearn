import tensorflow as tf
from time import time
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


# x = tf.placeholder(tf.float32, shape=[None, 108])
# y = tf.placeholder(tf.float32, shape=[None])
#
# m = 2
# learning_rate = 0.3
# u = tf.Variable(tf.random_normal([108, m], 0.0, 0.5), name='u')
# w = tf.Variable(tf.random_normal([108, m], 0.0, 0.5), name='w')
#
# U = tf.matmul(x, u)
# p1 = tf.nn.softmax(U)
#
# W = tf.matmul(x, w)
# p2 = tf.nn.sigmoid(W)
#
# pred = tf.reduce_sum(tf.multiply(p1, p2), 1)
#
# cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
# cost = tf.add_n([cost1])
# train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)
# train_x, train_y, test_x, test_y = get_data()
# time_s = time.time()
# result = []
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(0, 10000):
#         f_dict = {x: train_x, y: train_y}
#
#         _, cost_, predict_ = sess.run([train_op, cost, pred], feed_dict=f_dict)
#
#         auc = roc_auc_score(train_y, predict_)
#         time_t = time.time()
#         if epoch % 100 == 0:
#             f_dict = {x: test_x, y: test_y}
#             _, cost_, predict_test = sess.run([train_op, cost, pred], feed_dict=f_dict)
#             test_auc = roc_auc_score(test_y, predict_test)
#             print("%d %ld cost:%f,train_auc:%f,test_auc:%f" % (epoch, (time_t - time_s), cost_, auc, test_auc))
#             result.append([epoch, (time_t - time_s), auc, test_auc])
#
# pd.DataFrame(result, columns=['epoch', 'time', 'train_auc', 'test_auc']).to_csv("data/mlr_" + str(m) + '.csv')


class MLR(BaseEstimator, TransformerMixin):
    def __init__(self, m=2, learning_rate=0.1, batch_size=32, epoch=30, random_seed=7, eval_metric=roc_auc_score,
                 greater_is_better=True, verbose=True, **kwargs):
        self.m = m
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.epoch = epoch
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.verbose = verbose
        self.train_result, self.valid_result = [], []

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self.input = tf.placeholder(tf.float32,
                                        shape=[None, None],
                                        name='input')

            self.label = tf.placeholder(tf.float32, shape=[None], name='label')

            self._initialize_weights()

            self.U = tf.matmul(self.input, self.u)
            self.p1 = tf.nn.softmax(self.U)

            self.W = tf.matmul(self.input, self.w)
            self.p2 = tf.nn.sigmoid(self.W)

            self.pred = tf.reduce_sum(tf.multiply(self.p1, self.p2), 1)

            self.cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred, labels=self.label))
            self.cost = tf.add_n([self.cost1])
            self.train_op = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.cost)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        self.u = tf.Variable(tf.random_normal([self.feature_size, self.m], 0.0, 0.5), name='u')
        self.w = tf.Variable(tf.random_normal([self.feature_size, self.m], 0.0, 0.5), name='w')

    def _get_feature_size(self, X):
        return X.shape[1]

    def get_batch(self, X, y, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < len(y) else len(y)
        return X[start:end], y[start:end]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def evaluate(self, X, y):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :param y: label of each sample in the dataset
        :return: metric of the evaluation
        """
        y_pred = self.predict(X)
        return self.eval_metric(y, y_pred)

    def predict(self, X):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # TODO: 有必要进行分批预测么？
        # dummy y
        dummy_y = [1] * len(X)
        batch_index = 0
        X_batch, y_batch = self.get_batch(X, dummy_y, self.batch_size, batch_index)
        y_pred = None
        while len(X_batch) > 0:
            num_batch = len(y_batch)
            feed_dict = {self.input: X_batch,
                         self.label: y_batch}
            batch_out = self.sess.run(self.pred, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1
            X_batch, y_batch = self.get_batch(X, dummy_y, self.batch_size, batch_index)

        return y_pred

    def fit_on_batch(self, X, y):
        feed_dict = {self.input: X,
                     self.label: y}

        loss, opt = self.sess.run([self.cost, self.train_op], feed_dict=feed_dict)
        return loss

    def fit(self, X_train, y_train, X_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                         indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                         vali_j is the feature value of feature field j of sample i in the training set
                         vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :return: None
        """
        self.feature_size = X_train.shape[1]
        self._init_graph()

        has_valid = X_valid is not None
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(X_train, y_train)
            total_batch = int(len(y_train) / self.batch_size)
            for i in range(total_batch):
                X_batch, y_batch = self.get_batch(X_train, y_train, self.batch_size, i)
                self.fit_on_batch(X_batch, y_batch)

            # evaluate training and validation datasets
            train_result = self.evaluate(X_train, y_train)
            self.train_result.append(train_result)
            if has_valid:
                valid_result = self.evaluate(X_valid, y_valid)
                self.valid_result.append(valid_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                if has_valid:
                    print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, valid_result, time() - t1))
                else:
                    print("[%d] train-result=%.4f [%.1f s]"
                          % (epoch + 1, train_result, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            X_train = X_train + X_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                self.shuffle_in_unison_scary(X_train, y_train)
                total_batch = int(len(y_train) / self.batch_size)
                for i in range(total_batch):
                    X_batch, y_batch = self.get_batch(X_train, y_train,
                                                      self.batch_size, i)
                    self.fit_on_batch(X_batch, y_batch)
                # check
                train_result = self.evaluate(X_train, y_train)
                if abs(train_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                                valid_result[-2] < valid_result[-3] and \
                                valid_result[-3] < valid_result[-4] and \
                                valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                                valid_result[-2] > valid_result[-3] and \
                                valid_result[-3] > valid_result[-4] and \
                                valid_result[-4] > valid_result[-5]:
                    return True
        return False
