import pandas as pd
import numpy as np
import sklearn.metrics as mr
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from collections import Counter
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
import gensim
import tensorflow as tf
import codecs
import collections
import pickle
import os


def prepare_data(filename, seq_length):
    df = pd.read_pickle(filename)
    vocab = pd.read_pickle('vocab.pkl')
    
    contents = []
    labels = []
    for i in df.index:
        contents.append([vocab.get(w, 0) for s in df.loc[i, 'parse_words'] for w in s])
        try:
            labels.append(df.loc[i, 'label'])
        except:
            labels.append(0)
    x_pad = tf.contrib.keras.preprocessing.sequence.pad_sequences(contents, seq_length, padding='post', truncating='post')
    y_pad = tf.contrib.keras.utils.to_categorical(labels, num_classes=3)
    return x_pad, y_pad


def batch_iter(x, y, batch_size=128, shuffle=True):
    if batch_size is None:
        batch_size = len(x)
        
    if shuffle:
        inds = np.random.permutation(np.arange(len(x)))
        x = x[inds]
        y = y[inds]
    
    for i in range(int(np.ceil(len(x) / batch_size))):
        start_ind = i*batch_size
        end_ind = min(i*batch_size+batch_size, len(x))
        x_ = x[start_ind:end_ind]
        y_ = y[start_ind:end_ind]
        yield x_, y_


class TextRNN:
    def __init__(self, config):
        self.config = config
        
        self.rnn()
        
    def rnn(self):
        with tf.name_scope('placeholder'):
            self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
            self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.label_size], name='input_y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.cell_keep_prob = tf.placeholder(tf.float32, name='cell_keep_prob')
            
        with tf.name_scope('embedding'):
            if type(self.config.pre_embedding) == str:
                pre_embedding = pd.read_pickle(self.config.pre_embedding)
            else:
                pre_embedding = self.config.pre_embedding
                
            if pre_embedding is not None:
                tf_pre_embedding = tf.get_variable('embedding', shape=pre_embedding.shape, initializer=tf.constant_initializer(pre_embedding), trainable=self.config.embedding_trainable)
            else:
                tf_pre_embedding = tf.get_variable('embedding', shape=[self.config.vocab_size, self.config.embedding_size], initializer=tf.random_normal([self.config.vocab_size, self.config.embedding_size]), trainable=self.config.embedding_trainable)
                
            self.embedding = tf.nn.embedding_lookup(tf_pre_embedding, self.input_x)
            
        with tf.name_scope(self.config.cell_type):
            if self.config.cell_type == 'rnn':
                cell_func = tf.nn.rnn_cell.RNNCell
            elif self.config.cell_type == 'gru':
                cell_func = tf.nn.rnn_cell.GRUCell
            elif self.config.cell_type == 'lstm':
                cell_func = tf.nn.rnn_cell.LSTMCell
            else:
                raise ValueError("the value of cell_type must be in ['gru', 'lstm', 'rnn']")
                
            def get_cell():
                cell = cell_func(self.config.cell_hidden_dim, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.cell_keep_prob)
                return cell
            
            cells = tf.nn.rnn_cell.MultiRNNCell([get_cell() for _ in range(self.config.rnn_layers)], state_is_tuple=True)
            outputs, states = tf.nn.dynamic_rnn(cells, self.embedding, dtype=tf.float32)
            self.output = outputs[:, -1, :]
            
        with tf.name_scope('fc'):
            fc = tf.nn.dropout(self.output, keep_prob=self.keep_prob)
            w = tf.Variable(tf.truncated_normal([fc.shape[1].value, self.config.label_size], stddev=0.1), name='weights')
            b = tf.Variable(tf.constant(0.1, shape=[self.config.label_size], name='biases'))
            self.logits = tf.nn.bias_add(tf.matmul(fc, w), b)
            
        with tf.name_scope('loss'):
            self.l2_loss = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits)
            self.loss = tf.reduce_mean(loss) + self.l2_loss * self.config.l2_reg_lambda
            
        with tf.name_scope('opt'):
            self.global_step = tf.Variable(tf.constant(0), trainable=False, name='global_step')
            if self.config.opt == 'adam':
                opt_func = tf.train.AdamOptimizer
            opt = opt_func(self.config.learning_rate)
            gradients, var = zip(*opt.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.opt = opt.apply_gradients(zip(gradients, var), global_step=self.global_step)
            
        with tf.name_scope('metric'):
            self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.input_y, 1), tf.argmax(self.logits, 1)), tf.float32))
#             self.macro_f1 = mr.precision_recall_fscore_support(tf.argmax(self.input_y, 1).numpy(), tf.argmax(self.logits, 1).numpy(), average='macro')
            
        return self


class TextConfig:
    embedding_size = 100
    vocab_size = 8000
    pre_embedding = 'embeddings'
    embedding_trainable = True
    
    seq_length = 600
    label_size = 3
    
    keep_prob = 0.5
    cell_keep_prob = 0.5
    
    learning_rate = 0.01
    clip = 6
    l2_reg_lambda = 0.01
    
    num_epochs = 10
    batch_size = 64
    pring_per_batch = 100
    
    opt = 'adam'
    
    cell_type = 'lstm'
    rnn_layers = 1
    cell_hidden_dim = 32
    
    model_name = 'TextRNN'

    tensorboard_dir = f'{model_name}_tensorboard/'
    save_dir = f'{model_name}_checkpoints/'

    train_path = 'train.pkl'
    val_path = 'val.pkl'
    

class TextClassification:
    def __init__(self, config):
        self.config = config

    def train(config):
        tf.reset_default_graph()
        model = eval(config.model_name)(config)
        
        tensorboard_dir = config.tensorboard_dir
        save_dir = config.save_dir
        train_path = config.train_path
        val_path = config.val_path
        
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, 'best_validation')
        
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('accucary', model.acc)
        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(tensorboard_dir)
        saver = tf.train.Saver()
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        
        x_pad, y_pad = prepare_data(train_path, 600)
        
        val_x, val_y = prepare_data(val_path, 600)
        
        best_macro_f1 = 0
        last_improved = 0
        
        for i in range(20):
            gen = batch_iter(x_pad, y_pad, batch_size=128)
            for j, (x_, y_) in enumerate(gen):
                _, train_summary, global_step = sess.run([model.opt, merged_summary, model.global_step], feed_dict={model.input_x: x_, model.input_y: y_, model.keep_prob: 0.5, model.cell_keep_prob: 0.5})
                if j % 10 == 0:
                    writer.add_summary(train_summary, global_step)
                    acc, loss = sess.run([model.acc, model.loss], feed_dict={model.input_x: x_, model.input_y: y_, model.keep_prob: 1, model.cell_keep_prob: 1})
                    val_acc, val_loss, res = sess.run([model.acc, model.loss, model.logits], feed_dict={model.input_x: val_x, model.input_y: val_y, model.keep_prob: 1, model.cell_keep_prob: 1})
                    macro_f1 = mr.f1_score(np.argmax(val_y, 1), np.argmax(res, 1), average='macro')
                    print(f'epoch: {i}, batch: {j}, acc: {acc:.2f} loss : {loss:.2f}, val_acc: {val_acc:.2f}, val_loss: {val_loss:.2f}, macro_f1: {macro_f1:.2f}')

                    if macro_f1 > best_macro_f1:
                        saver.save(sess, save_path)
                        best_macro_f1 = macro_f1
                        last_improved = global_step
                if global_step - last_improved > 200:
                    break
                
    
config = TextConfig()
train(config)