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

# import warnings
# warnings.filterwarnings("ignore")

class SpliterTest(unittest.TestCase):  # 继承unittest.TestCase
    def tearDown(self):
        # 每个测试用例执行之后做操作
        pass

    def setUp(self):
        # 每个测试用例执行之前做操作
        pass

    # def test_SelectFromModelFilter(self):
    #     # self.assertEqual(1,1)
    #     json_str = json.dumps({'ds': {'label': {'name': xk_v4_transformer['label'], 'type': 'number'},
    #                                   'test': xk_v4_transformer['test_src'],
    #                                   'train': xk_v4_transformer['train_src']},
    #                            'out': {'dst': os.path.join(root_path, 'flow', 'spliter')},
    #                            'st': [{'method': 'RFEFilter',
    #                                    'params': {
    #                                        'n_features_to_select': 60,
    #                                        'estimator': {
    #                                            'estimator': 'LogisticRegression',
    #                                            "params": {}}}}]})
    #     res = mlearn.filter_ui(json_str)
    #     self.assertEqual(json.loads(res)['code'], 1)

    def test_Spliter(self):
        for dict in spliter_params:
            json_str = json.dumps(dict)
            # print(json_str, '\n')
            res = mlearn.spliter_ui(json_str)
            # print(res, '\n')
            msg = json_str + '\n\n\n\n' + res
            with self.subTest(msg=msg):
                self.assertEqual(json.loads(res)['code'], 1)


if __name__ == '__main__':
    test_suite = unittest.TestSuite()  # 创建一个测试集合
    test_suite.addTest(unittest.makeSuite(SpliterTest))
    # test_suite.addTest(FilterTest('test_SelectFromModelFilter'))  # 测试套件中添加单个测试用例
    # test_suite.addTest(unittest.makeSuite(MyTest))#使用makeSuite方法添加所有的测试方法
    # runner = HtmlTestRunner.HTMLTestRunner(output='example_dir')
    # # 生成执行用例的对象
    # runner.run(test_suite)
    # 执行测试套件

    report_title = 'Spliter用例执行报告'
    desc = ''
    cur_path = os.path.dirname(os.path.abspath(__file__))
    report_file = os.path.join(cur_path, 'reports/%s.html' % (
        report_title + '_' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    if not os.path.exists(os.path.dirname(report_file)):
        os.makedirs(os.path.dirname(report_file))

    with open(report_file, 'wb') as report:
        runner = HTMLTestRunner(stream=report, title=report_title, description=desc)
        runner.run(test_suite)
