import pandas as pd
import os

def create_dirpath(dst, name):
    train_dst = os.path.join(dst, 'train')
    test_dst = os.path.join(dst, 'test')
    report_dst = os.path.join(dst, 'report')
    if not os.path.exists(train_dst):
        os.makedirs(train_dst)
    if not os.path.exists(test_dst):
        os.makedirs(test_dst)
    if not os.path.exists(report_dst):
        os.makedirs(report_dst)
    train_data_dst = os.path.join(dst, 'train', f'{name}_result.pkl')
    train_process_dst = os.path.join(dst, 'train', f'{name}_enc.pkl')
    test_data_dst = os.path.join(dst, 'test', f'{name}_result.pkl')
    return train_data_dst, train_process_dst, test_data_dst, report_dst
