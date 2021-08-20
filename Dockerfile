FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-dev python3-pip python3-setuptools libgtk2.0-dev git g++ wget make vim

# Upgrade pip to latest version is necessary, otherwise the default version cannot install tensorflow 2.1.0
RUN pip3 install --upgrade setuptools pip
#for python packages
RUN pip3 install cython \
        numpy>=1.14 \
        wheel \
        opencv-python==3.4.5.20 \
        keras==2.2.4 \
        keras-resnet==0.2.0 \
        h5py \
        matplotlib \
        pillow \
        progressbar2 \
        scipy \
        scikit-learn \
        imutils \
        beautifulsoup4 \
        tk \
        setuptools
# for coco api
RUN pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
#COPY requirements.txt /
#RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install python3.7-tk
COPY . /social_distancing_app
WORKDIR /social_distancing_app
#RUN yes | apt-get install git
#RUN git clone https://github.com/fizyr/keras-retinanet.git
#WORKDIR /social_distancing_app/keras-retinanet
#RUN git checkout 42068ef9e406602d92a1afe2ee7d470f7e9860df
#RUN yes | apt-get install python-setuptools
#RUN yes | apt-get install gcc python3.7-dev
#RUN python setup.py install
# Download source code
WORKDIR /gui
#RUN git clone https://github.com/yytang2012/keras-retinanet.git .
RUN git clone https://github.com/fizyr/keras-retinanet.git 
WORKDIR /gui/keras-retinanet
RUN git checkout 42068ef9e406602d92a1afe2ee7d470f7e9860df
#WORKDIR /gui/keras-retinanet
# Install Dependencies
#RUN pip3 install -r requirements.txt
# Install retinaNet
RUN python3 setup.py build_ext --inplace
RUN pip install . --user
# Download pretrain weights
#RUN wget -P /keras-retinanet/snapshots https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet101_oid_v1.0.0.h5
#RUN wget -P /keras-retinanet/snapshots https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet152_oid_v1.0.0.h5
#RUN wget -P /keras-retinanet/snapshots https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5
#RUN wget -P /keras-retinanet/snapshots https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_oid_v1.0.0.h5
WORKDIR /social_distancing_app/gui
#RUN python3 gui.py
ENTRYPOINT [ "python3" ]
CMD [ "gui.py" ]
# Healthcheck
#HEALTHCHECK CMD pidof python3 || exit 1
#CMD ["/bin/bash"]