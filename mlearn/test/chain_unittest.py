# import HtmlTestRunner
import unittest
import json
import os
import time

root_path = os.path.abspath(__file__).rsplit('/mlearn', 1)[0]
import sys

sys.path.append(root_path)
import mlearn
from mlearn.configuration_params import *
from mlearn.test.HTMLTestRunnner import HTMLTestRunner
from mlearn.test.filter_test import *
from mlearn.test.transformer_test import *
from mlearn.test.spliter_test import *
from mlearn.test.trainer_test import *
from mlearn.test.optimizer_test import *

# class FilterTest(unittest.TestCase):  # 继承unittest.TestCase
#     def tearDown(self):
#         # 每个测试用例执行之后做操作
#         pass
#
#     def setUp(self):
#         # 每个测试用例执行之前做操作
#         pass
#
#     # def test_SelectFromModelFilter(self):
#     #     # self.assertEqual(1,1)
#     #     json_str = json.dumps({'ds': {'label': {'name': xk_v4_transformer['label'], 'type': 'number'},
#     #                                   'test': xk_v4_transformer['test_src'],
#     #                                   'train': xk_v4_transformer['train_src']},
#     #                            'out': {'dst': os.path.join(root_path, 'flow', 'spliter')},
#     #                            'st': [{'method': 'RFEFilter',
#     #                                    'params': {
#     #                                        'n_features_to_select': 60,
#     #                                        'estimator': {
#     #                                            'estimator': 'LogisticRegression',
#     #                                            "params": {}}}}]})
#     #     res = mlearn.filter_ui(json_str)
#     #     self.assertEqual(json.loads(res)['code'], 1)
#
#     def test_Filter(self):
#         for dict in filter_params:
#             json_str = json.dumps(dict)
#             print(json_str, '\n')
#             res = mlearn.filter_ui(json_str)
#             print(res)
#             with self.subTest(msg=json_str):
#                 self.assertEqual(json.loads(res)['code'], 1)


if __name__ == '__main__':
    test_suite = unittest.TestSuite()  # 创建一个测试集合
    test_suite.addTest(unittest.makeSuite(FilterTest))
    test_suite.addTest(unittest.makeSuite(TransformerTest))
    # test_suite.addTest(unittest.makeSuite(OptimizerTest))
    test_suite.addTest(unittest.makeSuite(TrainerTest))
    test_suite.addTest(unittest.makeSuite(SpliterTest))

    report_title = 'mlearn用例执行报告'
    desc = ''
    cur_path = os.path.dirname(os.path.abspath(__file__))
    report_file = os.path.join(cur_path, 'reports/%s.html' % (
        report_title + '_' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    if not os.path.exists(os.path.dirname(report_file)):
        os.makedirs(os.path.dirname(report_file))

    with open(report_file, 'wb') as report:
        runner = HTMLTestRunner(stream=report, title=report_title, description=desc)
        runner.run(test_suite)
