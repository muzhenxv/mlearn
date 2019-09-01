FROM continuumio/anaconda3:5.1.0
#FROM tensorflow/tensorflow:1.9.0-devel-py3
ENV LANG C.UTF-8

RUN mkdir analyze-server
WORKDIR /analyze-server
ADD . /analyze-server

RUN apt-get update
RUN apt-get install -y gcc vim python-dev python3-dev nginx build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        libgl1-mesa-glx
#RUN pip3 install --no-index --find-links=/ae-server/packages -r /ae-server/ae-server-requirements.txt
# RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple gensim
RUN conda config --add channels conda-forge
RUN conda install -c conda-forge libiconv -y
RUN conda update -n base conda -y
RUN conda install -y uwsgi
RUN pip install -i https://mirrors.aliyun.com/pypi/simple -r /analyze-server/analyze-server-requirements.txt
#RUN export LD_LIBLABRARY_PATH = /usr/local/cuda-8.0/lib64:$LD_LIBLABRARY_PATH
ADD uwsgi/analyze-server.conf /etc/nginx/sites-available/default

RUN mkdir -p /data/log/nginx
EXPOSE 80
#ENTRYPOINT ["python3"]
# CMD ["python3", "manage.py", "runserver", "-h", "0.0.0.0", "-p", "5000"]
# CMD ["python3", "manage.py", "runserver"]
CMD ["/bin/bash", "run.sh"]
