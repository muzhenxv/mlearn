from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import tensorflow as tf
import numpy as np
import pandas as pd
from ...service.base_utils import *

class FMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, embedding_size=8, learning_rate=0.01, n_iters=10, batch_size=8, lambda_b=0.001, lambda_w=0.001,
                 lambda_v=0.001, random_seed=7, opt_method='GradientDescent', opt_params=None):
        self.learning_rate = learning_rate
        self.lambda_b = lambda_b
        self.lambda_w = lambda_w
        self.lambda_v = lambda_v
        self.random_seed = random_seed
        self.embedding_size = embedding_size
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.opt_method = opt_method
        self.opt_params = {} if opt_params is None else opt_params

    def _init_weights(self):
        self.w0 = tf.Variable(tf.zeros([1]))
        self.w = tf.Variable(tf.random_normal([1, self.feature_size], mean=0, stddev=0.01))
        self.v = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], mean=0, stddev=0.01))

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)

            self._init_weights()

            self.x = tf.placeholder('float', [None, self.feature_size])
            self.y = tf.placeholder('float', [None, 1])

            y_linear = self.w0 + tf.reduce_sum(tf.multiply(self.w, self.x), axis=1, keep_dims=True)
            y_partial_1 = tf.reduce_sum(tf.pow(tf.matmul(self.x, self.v), 2), axis=1, keep_dims=True)
            y_partial_2 = tf.reduce_sum(tf.matmul(tf.pow(self.x, 2), tf.pow(self.v, 2)), axis=1, keep_dims=True)
            y_partial = 0.5 * (y_partial_1 - y_partial_2)

            self.y_hat = y_linear + y_partial

            # 按公式的标准实现
            # 方法1
            # self.error = tf.reduce_mean(self.y * tf.sigmoid(self.y_hat) + (1 - self.y) * tf.sigmoid(1 - self.y_hat))
            # 方法2
            self.out = tf.nn.sigmoid(self.y_hat)
            self.error = tf.losses.log_loss(self.y, self.out)

            # onehot之后的计算，tf在tf.nn.softmax_cross_entropy_with_logits中有优化，可以避免直接照公式实现可能存在的数值计算问题
            # 不过该函数返回是个二维张量而非标量，直接tf.reduce_mean等于是在标准error上多除了一个label维度，对于二分类就是多乘以系数1/2。
            # y_onehot = tf.one_hot(tf.cast(self.y, 'int32'), 2)
            # 这样得到的不是logits值，这种写法是错的！！！
            # y_hat_onehot = tf.concat([1-self.y_hat, self.y_hat], axis=1)
            # self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=y_hat_onehot))

            # MSE
            # self.error = 0.5 * tf.reduce_sum(tf.pow(self.y - y_hat, 2))

            self.l2_norm = self.lambda_b * tf.reduce_sum(tf.pow(self.w0, 2)) + self.lambda_w * tf.reduce_sum(
                tf.pow(self.w, 2)) + self.lambda_v * tf.reduce_sum(tf.pow(self.v, 2))

            self.loss = tf.add(self.error, self.l2_norm)

            self._optimizer()

            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _optimizer(self):
        if self.opt_method == 'GradientDescent':
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate,
                                                              **self.opt_params).minimize(self.loss)
        elif self.opt_method == 'Adam':
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate, **self.opt_params).minimize(
                self.loss)
        elif self.opt_method == 'Adagrad':
            self.train_op = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, **self.opt_params).minimize(
                self.loss)
        elif self.opt_method == 'Momentum':
            self.train_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, **self.opt_params).minimize(
                self.loss)

    def _batcher(self, X, y=None):
        batch_num = X.shape[0] // self.batch_size
        for i in range(batch_num):
            left_indices = self.batch_size * i
            right_indices = self.batch_size * (i + 1)
            if y is not None:
                yield X[left_indices:right_indices], y[left_indices:right_indices]
            else:
                yield X[left_indices:right_indices], None

    def process_dataset(self, X):
        df = convert_df_type(pd.DataFrame(X))




    def fit(self, X, y):
        self.feature_size = X.shape[1]

        self._init_graph()

        for i in range(self.n_iters):
            for batch_x, batch_y in self._batcher(X, y):
                iter_loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.x: batch_x, self.y: batch_y})
            print('n_iters: ', i, 'loss: ', iter_loss)
        return self

    def predict(self, X, threshold=0.5):
        self.threshold = threshold
        out = self.predict_proba(X)[:, 1]
        return (out > self.threshold).astype(int).flatten()

    def predict_proba(self, X):
        out = self.sess.run(self.out, feed_dict={self.x: X, self.y: [[1]] * self.feature_size})
        out = np.concatenate((1-out, out), axis=1)
        return out

if __name__ == '__main__':
    X = np.random.rand(400, 3)
    y = np.random.randint(0, 2, (400, 1))

    fm = FMClassifier(opt_method='Momentum', opt_params={'momentum': 0.95})
    fm.fit(X, y)
    fm.predict(X)
