FROM tensorflow/tensorflow:1.9.0-devel-gpu-py3

LABEL maintainer="luozhouyang<stupidme.me.lzy@gmail.com>"

COPY naivenmt/ /root/naivenmt

ENV PYTHONPATH=$PYTHONPATH:/root/naivenmt

WORKDIR /root/naivenmt

RUN pip install -r requirements.txt \
    && rm -rf /root/.cache/pip/*

CMD ["bash"]