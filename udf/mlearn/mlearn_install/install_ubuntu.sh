apt-get update
apt-get install wget libglib2.0-0 -y
# cd /tmp
# wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz
# tar -xzvf Python-3.6.1.tgz -C /tmp
# cd /tmp/Python-3.6.1
# ./configure --prefix=/usr/local && make && make install
cd /tmp
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh -b -p /home/anaconda3
#echo 'export PATH="/home/anaconda3/bin:$PATH"' >> ~/.bashrc
#source ~/.bashrc
cd /home/test
tar -xzf mlearn-1.0.0.tar.gz
cd mlearn-1.0.0
/home/anaconda3/bin/python setup.py install
