yum install wget -y
# cd /tmp
# wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz
# tar -xzvf Python-3.6.1.tgz -C /tmp
yum -y update
yum install yum-plugin-ovl gcc zlib* openssl* mysql mysql-devel mysql-lib gcc-c++ bzip2-devel sqlite-devel bzip2 mesa-libGL-devel -y
# cd /tmp/Python-3.6.1
# ./configure --prefix=/usr/local && make && make install
cd /tmp
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh -b -p /home/anaconda3
echo 'export PATH="/home/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
/home/anaconda3/bin/pip install -i https://mirrors.aliyun.com/pypi/simple cmake PyQt5 dill bayesian-optimization category_encoders seaborn graphviz xgboost
cd /home/test
tar -xzf mlearn-*
cd mlean-*.0
/home/anaconda3/bin/python setup.py install
cd /home/test
/home/anaconda3/bin/python mlearn_test.py
