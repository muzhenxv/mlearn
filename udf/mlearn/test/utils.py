import os

def get_rootpath(path):
    return os.path.join(path.rsplit('/mlearn', 1)[0], 'mlearn')

