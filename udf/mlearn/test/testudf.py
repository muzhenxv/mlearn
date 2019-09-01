import json
import os
import sys

def testudf(js_str):
    path = json.loads(js_str)['st']['udf']
    dirpath = os.path.dirname(path)
    classname = os.path.basename(path).split('.')[0]
    sys.path.append(dirpath)

    mod = __import__(classname)
    clf = getattr(mod, classname, None)()
    res = clf.fit()
    return res

if __name__ == '__main__':
    js_str = json.dumps({'st':{'udf':'/Users/muzhen/dev/udf/custom_udf1.py'}})
    # mlearn.testudf(js_str)