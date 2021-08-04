FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install -y swig git gcc libglib2.0-0 build-essential
RUN git clone --depth 1 -b master https://github.com/nuveo/BBAVectors-Oriented-Object-Detection.git

WORKDIR BBAVectors-Oriented-Object-Detection

RUN cd DOTA_devkit && \
    swig -c++ -python polyiou.i && \
    python setup.py build_ext --inplace && \
    cd .. && python -m pip install -e .[infer] && \
    python -m pip install opencv-python-headless==4.5.3.56

RUN apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
